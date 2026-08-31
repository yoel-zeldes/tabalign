import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tqdm import tqdm

import pruning_utils
from evaluate_aligned_student import evaluate_aligned_student
from train_xgboost import train_xgboost
from train_xgboost_opt import train_xgboost_opt


def load_results_for_repeat(args, dataset, layer_k, repeat):
    """
    Load evaluate_aligned_student results for all student_n values for a
    given dataset and repeat.

    Returns a dict:
        {student_n: {"teacher_roc_auc": float, "baseline_roc_auc": float, "aligned_roc_auc": float}}
    """
    synthetic_train_dataset = (
        f"{dataset}[synthetic-n_samples_{args.n_samples}-repeat_{repeat}]"
    )

    results = {}
    for student_n in sorted(args.student_n):
        aligner_opt = getattr(args, "aligner_opt", False)
        eval_kwargs = dict(
            eval_dataset=dataset,
            train_dataset=synthetic_train_dataset,
            student_n=student_n,
            layer_k=layer_k,
            n_estimators=args.n_estimators,
            patience=args.patience,
            lr=args.lr,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            repeat=repeat,
            max_epochs=args.max_epochs,
            model=args.model,
            aligner_opt=aligner_opt
        )
        if not evaluate_aligned_student.check_call_in_cache(**eval_kwargs):
            print(
                f"  [SKIP] Missing result for dataset={dataset}, "
                f"student_n={student_n}, repeat={repeat}"
            )
            continue

        metrics = evaluate_aligned_student(**eval_kwargs)["metrics"]
        results[student_n] = {
            "teacher_roc_auc": metrics["teacher_roc_auc"],
            "baseline_roc_auc": metrics["baseline_roc_auc"],
            "aligned_roc_auc": metrics["aligned_roc_auc"],
        }

    return results


def load_xgboost_result_for_repeat(args, dataset, repeat):
    """
    Load train_xgboost (or train_xgboost_opt) result for the full training set (student_n=-1) for a given repeat.
    Returns xgboost_roc_auc or None.
    """
    train_xgboost_func = train_xgboost_opt if args.xgboost_opt else train_xgboost
    xgboost_kwargs = dict(dataset=dataset, student_n=-1, repeat=repeat)
    if not train_xgboost_func.check_call_in_cache(**xgboost_kwargs):
        print(f"  [SKIP] Missing XGBoost result for dataset={dataset}, repeat={repeat}")
        return None
    return train_xgboost_func(**xgboost_kwargs)["metrics"]["xgboost_roc_auc"]


def nanmean(lst):
    arr = [v for v in lst if v is not None and not np.isnan(v)]
    return float(np.mean(arr)) if arr else np.nan


def find_maximal_m(student_ns, baseline_scores, aligned_score, teacher_score=None):
    """
    For a given aligned score at student_n = N, find the effective M via linear interpolation
    between baseline scores and the teacher score (considered as student_n = 1.0).

    If baseline_score(M_i) <= aligned_score < baseline_score(M_{i+1}), then M is linearly
    interpolated between M_i and M_{i+1}.
    If aligned_score >= baseline_score(M_max), returns M_max (e.g. 1.0 if teacher is highest).
    If aligned_score <= baseline_score(M_min), returns M_min.
    If aligned_score is NaN, returns np.nan.
    """
    if aligned_score is None or np.isnan(aligned_score):
        return np.nan

    scores_dict = dict(baseline_scores)
    grid = list(student_ns)
    if teacher_score is not None and not np.isnan(teacher_score) and 1.0 not in scores_dict:
        scores_dict[1.0] = teacher_score
        grid.append(1.0)

    valid_student_ns = [
        sn for sn in sorted(set(grid))
        if sn in scores_dict
        and scores_dict[sn] is not None
        and not np.isnan(scores_dict[sn])
    ]
    if not valid_student_ns:
        return min(student_ns)

    # If aligned score is lower than or equal to the minimum baseline score
    if aligned_score <= scores_dict[valid_student_ns[0]]:
        return valid_student_ns[0]

    # If aligned score is higher than or equal to the maximum baseline / teacher score
    if aligned_score >= scores_dict[valid_student_ns[-1]]:
        return valid_student_ns[-1]

    # Find interval [M_i, M_{i+1}] such that scores_dict[M_i] <= aligned_score <= scores_dict[M_{i+1}]
    for i in range(len(valid_student_ns) - 1, -1, -1):
        m_i = valid_student_ns[i]
        b_i = scores_dict[m_i]
        if b_i <= aligned_score:
            if i == len(valid_student_ns) - 1:
                return m_i
            m_next = valid_student_ns[i + 1]
            b_next = scores_dict[m_next]
            if b_next == b_i:
                return m_i
            alpha = (aligned_score - b_i) / (b_next - b_i)
            alpha = max(0.0, min(1.0, alpha))
            return m_i + alpha * (m_next - m_i)

    return valid_student_ns[0]


def _save_table_png(
    png_path,
    present_datasets,
    short_names,
    student_ns,
    m_matrix,
    avg_m_vals,
    aligned_matrix,
    baseline_matrix,
    teacher_values,
    xgb_values,
):
    """
    Render a results table as a PNG showing effective M for each dataset and N.
    Cells are:
      - gray background if baseline score >= teacher score on that dataset
      - green background if aligned score > XGBoost score on that dataset
      rows = datasets (+ Average row)
      cols = Dataset | N=... -> M | Mean Gain (M - N)
    """
    n_students = len(student_ns)
    col_headers = ["Dataset"] + [f"N={sn:.2g}" for sn in student_ns] + ["Mean Gain"]
    n_cols = len(col_headers)

    HEADER_BG = "#2c3e50"
    HEADER_FG = "white"
    ROW_ODD = "#eef2f7"
    ROW_EVEN = "white"
    HIGHLIGHT_BG = "#c8e6c9"  # soft light green background
    RED_BG = "#ffcdd2"        # soft light red background for M <= N
    GRAY_BG = "#d5d8dc"       # gray background for baseline >= teacher
    RULE_CLR = "#5d6d7e"
    THIN_CLR = "#d0d3d4"

    cell_text = []
    cell_bg_colors = []

    for di, short in enumerate(short_names):
        row_text = [short]
        row_bgs = [None]
        gains = []
        xgb_val = xgb_values[di]
        teacher_val = teacher_values[di]

        for si, sn in enumerate(student_ns):
            m_val = m_matrix[di, si]
            aligned_val = aligned_matrix[di, si]
            baseline_val = baseline_matrix[di, si]
            is_gray = (
                not np.isnan(baseline_val)
                and not np.isnan(teacher_val)
                and baseline_val >= teacher_val
            )

            if is_gray:
                row_text.append("")
                row_bgs.append(GRAY_BG)
            elif np.isnan(m_val):
                row_text.append("–")
                row_bgs.append(None)
            else:
                row_text.append(f"{m_val:.3g}")
                if m_val < sn:
                    row_bgs.append(RED_BG)
                else:
                    beats_xgb = (
                        not np.isnan(aligned_val)
                        and not np.isnan(xgb_val)
                        and aligned_val > xgb_val
                    )
                    row_bgs.append(HIGHLIGHT_BG if beats_xgb else None)
                gains.append(m_val - sn)

        if gains:
            mean_gain = float(np.mean(gains))
            prefix = "+" if mean_gain > 0 else ""
            row_text.append(f"{prefix}{mean_gain:.3f}")
            row_bgs.append(None)
        else:
            row_text.append("–")
            row_bgs.append(None)

        cell_text.append(row_text)
        cell_bg_colors.append(row_bgs)

    # Average row
    avg_row_text = ["Average"]
    avg_row_bgs = [None]
    avg_gains = []

    for si, sn in enumerate(student_ns):
        v = avg_m_vals[si]
        valid_indices = [
            di for di in range(len(short_names))
            if not np.isnan(aligned_matrix[di, si])
            and not (
                not np.isnan(baseline_matrix[di, si])
                and not np.isnan(teacher_values[di])
                and baseline_matrix[di, si] >= teacher_values[di]
            )
        ]
        valid_aligned = [aligned_matrix[di, si] for di in valid_indices]
        valid_xgb = [xgb_values[di] for di in valid_indices if not np.isnan(xgb_values[di])]

        avg_aligned = float(np.mean(valid_aligned)) if valid_aligned else np.nan
        col_xgb_avg = float(np.mean(valid_xgb)) if valid_xgb else np.nan

        if np.isnan(v):
            avg_row_text.append("–")
            avg_row_bgs.append(None)
        else:
            avg_row_text.append(f"{v:.3f}")
            if v < sn:
                avg_row_bgs.append(RED_BG)
            else:
                beats_xgb_avg = (
                    not np.isnan(avg_aligned)
                    and not np.isnan(col_xgb_avg)
                    and avg_aligned > col_xgb_avg
                )
                avg_row_bgs.append(HIGHLIGHT_BG if beats_xgb_avg else None)
            avg_gains.append(v - sn)

    if avg_gains:
        mean_avg_gain = float(np.mean(avg_gains))
        prefix = "+" if mean_avg_gain > 0 else ""
        avg_row_text.append(f"{prefix}{mean_avg_gain:.3f}")
        avg_row_bgs.append(None)
    else:
        avg_row_text.append("–")
        avg_row_bgs.append(None)

    n_rows = len(cell_text)

    # Layout geometry
    FONT_SIZE = 9
    HEAD_FONT = 9
    PAD_X = 0.25
    ROW_H = 0.30
    HEAD_H = 0.38
    CHAR_W = 0.090

    def text_w(s, bold=False):
        return len(s) * CHAR_W * (1.05 if bold else 1.0) + 2 * PAD_X

    col_widths = []
    for ci in range(n_cols):
        w = text_w(col_headers[ci], bold=True)
        for ri in range(n_rows):
            w = max(w, text_w(cell_text[ri][ci]))
        w = max(w, text_w(avg_row_text[ci]))
        col_widths.append(w)

    CAPTION_H = 0.72
    fig_w = sum(col_widths) + 0.05
    fig_h = HEAD_H + n_rows * ROW_H + ROW_H + 0.50 + CAPTION_H

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Title
    ax.text(
        fig_w / 2,
        fig_h - 0.12,
        "Effective Baseline Student Fraction M for Aligned N",
        ha="center",
        va="top",
        fontsize=10,
        color=HEADER_BG,
        fontweight="bold",
    )

    top = fig_h - 0.42

    # Header row
    x = 0.0
    for ci, (hdr, cw) in enumerate(zip(col_headers, col_widths)):
        rect = plt.Rectangle((x, top - HEAD_H), cw, HEAD_H, facecolor=HEADER_BG, edgecolor="none")
        ax.add_patch(rect)
        tx = (x + PAD_X) if ci == 0 else (x + cw / 2)
        ha = "left" if ci == 0 else "center"
        ax.text(
            tx,
            top - HEAD_H / 2,
            hdr,
            ha=ha,
            va="center",
            fontsize=HEAD_FONT,
            color=HEADER_FG,
            fontweight="bold",
        )
        x += cw

    ax.hlines(top, 0, fig_w, colors=RULE_CLR, linewidth=1.8)
    ax.hlines(top - HEAD_H, 0, fig_w, colors=RULE_CLR, linewidth=1.2)

    # Data rows
    for ri in range(n_rows):
        row_top = top - HEAD_H - ri * ROW_H
        bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
        rect = plt.Rectangle((0, row_top - ROW_H), fig_w, ROW_H, facecolor=bg, edgecolor="none")
        ax.add_patch(rect)
        if ri > 0:
            ax.hlines(row_top, 0, fig_w, colors=THIN_CLR, linewidth=0.4)

        x = 0.0
        for ci, cw in enumerate(col_widths):
            bg_color = cell_bg_colors[ri][ci]
            if bg_color is not None:
                cell_rect = plt.Rectangle((x, row_top - ROW_H), cw, ROW_H, facecolor=bg_color, edgecolor="none")
                ax.add_patch(cell_rect)

            txt = cell_text[ri][ci]
            color = "#1c2833"
            ha = "left" if ci == 0 else "center"
            tx = (x + PAD_X) if ci == 0 else (x + cw / 2)
            ax.text(
                tx,
                row_top - ROW_H / 2,
                txt,
                ha=ha,
                va="center",
                fontsize=FONT_SIZE,
                color=color,
                fontweight="bold" if bg_color is not None else "normal",
            )
            x += cw

    # Separator before average row
    avg_top = top - HEAD_H - n_rows * ROW_H
    ax.hlines(avg_top, 0, fig_w, colors=RULE_CLR, linewidth=1.5)

    # Average row
    AVG_BG = "#dce8f5"
    rect = plt.Rectangle((0, avg_top - ROW_H), fig_w, ROW_H, facecolor=AVG_BG, edgecolor="none")
    ax.add_patch(rect)
    x = 0.0
    for ci, cw in enumerate(col_widths):
        bg_color = avg_row_bgs[ci]
        if bg_color is not None:
            cell_rect = plt.Rectangle((x, avg_top - ROW_H), cw, ROW_H, facecolor=bg_color, edgecolor="none")
            ax.add_patch(cell_rect)

        txt = avg_row_text[ci]
        color = "#1c2833"
        ha = "left" if ci == 0 else "center"
        tx = (x + PAD_X) if ci == 0 else (x + cw / 2)
        ax.text(
            tx,
            avg_top - ROW_H / 2,
            txt,
            ha=ha,
            va="center",
            fontsize=FONT_SIZE,
            color=color,
            fontweight="bold",
        )
        x += cw

    bottom = avg_top - ROW_H
    ax.hlines(bottom, 0, fig_w, colors=RULE_CLR, linewidth=1.8)
    ax.vlines(col_widths[0], bottom, top, colors=THIN_CLR, linewidth=0.6)

    # ── Caption below table explaining colors ─────────────────────────────
    sw_w, sw_h = 0.18, 0.12
    cap_x = 0.10

    # Line 1: Green swatch & explanation
    y1 = bottom - 0.18
    rect_green = plt.Rectangle((cap_x, y1 - sw_h), sw_w, sw_h, facecolor=HIGHLIGHT_BG, edgecolor="#81c784", linewidth=0.8)
    ax.add_patch(rect_green)
    ax.text(
        cap_x + sw_w + 0.08, y1 - sw_h / 2,
        "Green: Aligned student score > XGBoost score on that dataset (matched-subset average in Average row)",
        ha="left", va="center", fontsize=8.5, color="#2c3e50"
    )

    # Line 2: Red swatch & explanation
    y2 = bottom - 0.36
    rect_red = plt.Rectangle((cap_x, y2 - sw_h), sw_w, sw_h, facecolor=RED_BG, edgecolor="#e57373", linewidth=0.8)
    ax.add_patch(rect_red)
    ax.text(
        cap_x + sw_w + 0.08, y2 - sw_h / 2,
        "Red: Value < N (aligned student score < non-aligned baseline score at size N)",
        ha="left", va="center", fontsize=8.5, color="#2c3e50"
    )

    # Line 3: Gray swatch & explanation
    y3 = bottom - 0.54
    rect_gray = plt.Rectangle((cap_x, y3 - sw_h), sw_w, sw_h, facecolor=GRAY_BG, edgecolor="#95a5a6", linewidth=0.8)
    ax.add_patch(rect_gray)
    ax.text(
        cap_x + sw_w + 0.08, y3 - sw_h / 2,
        "Gray: Non-aligned baseline score \u2265 Teacher score (omitted from Mean Gain and Average row)",
        ha="left", va="center", fontsize=8.5, color="#2c3e50"
    )

    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot aligned student sample efficiency: maps each fractional student_n N to maximal baseline M (score(M) <= aligned_score(N))."
    )
    parser.add_argument("--dataset", type=str, nargs="+", default=["breast_cancer"])
    parser.add_argument("--student_n", type=pruning_utils.parse_student_n, nargs="+", default=[20], help="Student training sizes")
    parser.add_argument("--layer", type=int, default=11, help="The layer to evaluate")
    parser.add_argument("--n_estimators", type=int, default=None,
                        help="Number of estimators (default: 8 for tabpfn, 32 for tabfm).")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples used.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for aligner training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for aligner training.")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[],
                        help="Hidden layer multipliers for MLP aligner. Empty = linear.")
    parser.add_argument("--output", type=str, default=None, help="Path to save the figure. Defaults to auto-generated name.")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of OpenML repeats to aggregate over.")
    parser.add_argument("--max_epochs", type=int, default=None, help="If specified, look up results where max_epochs was used.")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use.")
    parser.add_argument("--xgboost_opt", action="store_true", help="Look up results from train_xgboost_opt.py instead of train_xgboost.py.")
    parser.add_argument("--aligner_opt", action="store_true", help="Look up results from aligners trained with hyperparameter optimization.")
    args = parser.parse_args()
    if args.n_estimators is None:
        args.n_estimators = 32 if args.model == "tabfm" else 8

    # Validate that all student_n values are fractional < 1
    for sn in args.student_n:
        if not (isinstance(sn, float) and 0.0 < sn < 1.0):
            raise ValueError(f"student_n must be a fraction strictly between 0 and 1, got {sn}")

    # Expand "tabarena" shorthand
    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    # ------------------------------------------------------------------ #
    # Gather results — average across repeats
    # ------------------------------------------------------------------ #
    all_data = {}  # all_data[dataset][student_n] = {metric: mean_value}
    xgb_data = {}  # xgb_data[dataset][-1] = mean

    for dataset in tqdm(datasets, desc="Loading results"):
        accum = {}  # student_n -> {metric: [values across repeats]}
        xgb_vals = []

        for repeat in range(args.n_repeats):
            res = load_results_for_repeat(args, dataset, args.layer, repeat)
            for student_n, entry in res.items():
                if student_n not in accum:
                    accum[student_n] = {"teacher_roc_auc": [], "baseline_roc_auc": [], "aligned_roc_auc": []}
                for k in accum[student_n]:
                    accum[student_n][k].append(entry[k])

            xgb_val = load_xgboost_result_for_repeat(args, dataset, repeat)
            if xgb_val is not None:
                xgb_vals.append(xgb_val)

        if accum:
            all_data[dataset] = {
                sn: {k: nanmean(vs) for k, vs in entry.items()}
                for sn, entry in accum.items()
            }
            xgb_data[dataset] = {-1: nanmean(xgb_vals)} if xgb_vals else {}

    if not all_data:
        print("No results found. Make sure evaluate_aligned_student.py has been run first.")
        return

    present_datasets = list(all_data.keys())
    student_ns = sorted(args.student_n)
    n_datasets = len(present_datasets)
    n_students = len(student_ns)

    # ------------------------------------------------------------------ #
    # Compute effective M and aligned score matrix for each dataset and each N
    # ------------------------------------------------------------------ #
    m_matrix = np.full((n_datasets, n_students), np.nan)
    aligned_matrix = np.full((n_datasets, n_students), np.nan)
    baseline_matrix = np.full((n_datasets, n_students), np.nan)
    teacher_values = np.full(n_datasets, np.nan)
    xgb_values = np.full(n_datasets, np.nan)

    for di, dataset in enumerate(present_datasets):
        ds_data = all_data[dataset]
        baseline_scores = {
            sn: ds_data[sn]["baseline_roc_auc"]
            for sn in student_ns
            if sn in ds_data and ds_data[sn]["baseline_roc_auc"] is not None
        }

        teacher_list = [
            ds_data[sn]["teacher_roc_auc"]
            for sn in student_ns
            if sn in ds_data and ds_data[sn].get("teacher_roc_auc") is not None and not np.isnan(ds_data[sn]["teacher_roc_auc"])
        ]
        teacher_val = float(np.mean(teacher_list)) if teacher_list else np.nan
        teacher_values[di] = teacher_val

        for si, sn in enumerate(student_ns):
            if sn not in ds_data:
                continue
            aligned_score = ds_data[sn].get("aligned_roc_auc")
            baseline_score = ds_data[sn].get("baseline_roc_auc")

            if aligned_score is not None:
                aligned_matrix[di, si] = aligned_score
                m_matrix[di, si] = find_maximal_m(student_ns, baseline_scores, aligned_score, teacher_score=teacher_val)
            if baseline_score is not None:
                baseline_matrix[di, si] = baseline_score

        xgb_values[di] = xgb_data.get(dataset, {}).get(-1, np.nan)

    # Compute average M across datasets for each N (excluding cases where baseline >= teacher)
    avg_m_vals = []
    for si in range(n_students):
        valid_ms = [
            m_matrix[di, si] for di in range(n_datasets)
            if not np.isnan(m_matrix[di, si])
            and not (
                not np.isnan(baseline_matrix[di, si])
                and not np.isnan(teacher_values[di])
                and baseline_matrix[di, si] >= teacher_values[di]
            )
        ]
        avg_m_vals.append(float(np.mean(valid_ms)) if valid_ms else np.nan)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    short_names = [d.split("/")[-1] if "/" in d else d for d in present_datasets]
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, n_datasets)))

    fig, ax = plt.subplots(figsize=(8, 6))

    # 1. Baseline parity line (y = x)
    min_n = min(student_ns)
    max_n = max(student_ns)
    ax.plot(
        [min_n, max_n],
        [min_n, max_n],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        alpha=0.7,
        label="Baseline ($M = N$)",
        zorder=1,
    )

    # 2. Individual dataset lines
    for di, (dataset, short) in enumerate(zip(present_datasets, short_names)):
        valid_indices = [si for si in range(n_students) if not np.isnan(m_matrix[di, si])]
        if not valid_indices:
            continue
        xs = [student_ns[si] for si in valid_indices]
        ys = [m_matrix[di, si] for si in valid_indices]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.2,
            alpha=0.5,
            color=colors[di % len(colors)],
            label=short,
            zorder=2,
        )

    # 3. Average line across all datasets (bold, drawn LAST with highest zorder)
    valid_avg_indices = [si for si in range(n_students) if not np.isnan(avg_m_vals[si])]
    if valid_avg_indices:
        avg_xs = [student_ns[si] for si in valid_avg_indices]
        avg_ys = [avg_m_vals[si] for si in valid_avg_indices]
        ax.plot(
            avg_xs,
            avg_ys,
            marker="s",
            markersize=7,
            linewidth=3.0,
            color="black",
            label="Average",
            zorder=10,
        )

    # Formatting axes and labels
    ax.set_xlabel("Aligned Student Fraction $N$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Effective Baseline Student Fraction $M$", fontsize=11, fontweight="bold")

    pad = 0.03 * (max_n - min_n) if max_n > min_n else 0.05
    ax.set_xlim(min_n - pad, max_n + pad)
    ax.set_ylim(min_n - pad, 1.0 + pad)

    ax.set_xticks(student_ns)
    y_ticks = sorted(set(student_ns + [1.0]))
    ax.set_yticks(y_ticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))

    ax.grid(True, linestyle=":", alpha=0.5, zorder=0)

    repeat_suffix = f" (mean of {args.n_repeats} repeats)" if args.n_repeats > 1 else ""
    aligned_label = "Aligned (Opt)" if args.aligner_opt else "Aligned"
    ax.set_title(
        f"{aligned_label} Student Sample Efficiency (layer={args.layer}){repeat_suffix}\n"
        f"Estimators={args.n_estimators}, Patience={args.patience}",
        fontsize=11,
        fontweight="bold",
    )

    # Legend outside to the right if many datasets
    if n_datasets > 10:
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize=8, framealpha=0.9)
    else:
        ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if args.output:
        output_path = args.output
    else:
        output_path = pruning_utils.create_filename_from_args(
            args,
            script_name="plot_aligned_sample_efficiency"
        )
        output_path = f"{output_path}/graph.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")

    # ------------------------------------------------------------------ #
    # Table PNG
    # ------------------------------------------------------------------ #
    if "graph.png" in output_path:
        table_path = output_path.replace("graph.png", "table.png")
    elif output_path.endswith(".png"):
        table_path = output_path[:-4] + "_table.png"
    else:
        table_path = f"{output_path}_table.png"

    _save_table_png(
        table_path,
        present_datasets,
        short_names,
        student_ns,
        m_matrix,
        avg_m_vals,
        aligned_matrix,
        baseline_matrix,
        teacher_values,
        xgb_values,
    )
    print(f"Table PNG saved to {table_path}")


if __name__ == "__main__":
    main()
