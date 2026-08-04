import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pruning_utils
from tqdm import tqdm


def load_results_for_repeat(args, dataset, all_datasets, layer_k, repeat):
    """
    Load evaluate_aligned_student JSON results for all student_n values for a
    given dataset and repeat.

    Returns a dict:
        {student_n: {"teacher_roc_auc": float, "baseline_roc_auc": float, "aligned_roc_auc": float}}
    """
    if args.train_on_rest:
        training_datasets = [ds for ds in all_datasets if ds != dataset]
    else:
        training_datasets = [dataset]

    synthetic_train_datasets = [
        f"{ds}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-repeat_{repeat}-use_tabpfn_{args.use_tabpfn}]"
        for ds in training_datasets
    ]

    results = {}
    for student_n in sorted(args.student_n):
        path_args = {
                "eval_dataset": dataset,
                "train_dataset": synthetic_train_datasets,
                "student_n": student_n,
                "layer_k": layer_k,
                "n_estimators": args.n_estimators,
                "patience": args.patience,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "per_token": args.per_token,
                "hidden_layers": args.hidden_layers,
                "predict_residual": args.predict_residual,
                "repeat": repeat,
                "output_dir": args.output_dir,
                "use_feature_stats": args.use_feature_stats,
                "max_epochs": args.max_epochs,
                "model": args.model,
        }
        if args.loss_beta is not None:
            path_args["loss_beta"] = args.loss_beta
        if getattr(args, "clip_grad", None) is not None:
            path_args["clip_grad"] = args.clip_grad

        k_result_path = pruning_utils.create_filename_from_args(
            path_args,
            script_name="evaluate_aligned_student",
            extension=".json"
        )

        if not os.path.exists(k_result_path):
            print(
                f"  [SKIP] Missing result for dataset={dataset}, "
                f"student_n={student_n}, repeat={repeat}: {k_result_path}"
            )
            continue

        with open(k_result_path, "r") as f:
            metrics = json.load(f)["metrics"]

        results[student_n] = {
            "teacher_roc_auc": metrics["teacher_roc_auc"],
            "baseline_roc_auc": metrics["baseline_roc_auc"],
            "aligned_roc_auc": metrics["aligned_roc_auc"],
        }

    return results


def load_xgboost_result_for_repeat(args, dataset, repeat):
    """
    Load train_xgboost JSON result for the full training set (student_n=-1) for a given repeat.
    Returns xgboost_roc_auc or None.
    """
    path = pruning_utils.create_filename_from_args(
        {
            "dataset": dataset,
            "student_n": -1,
            "repeat": repeat,
            "output_dir": args.output_dir,
        },
        script_name="train_xgboost",
        extension=".json",
    )
    if not os.path.exists(path):
        print(f"  [SKIP] Missing XGBoost result for dataset={dataset}, repeat={repeat}: {path}")
        return None
    with open(path, "r") as f:
        metrics = json.load(f)["metrics"]
    return metrics["xgboost_roc_auc"]


def normalize(value, baseline, teacher):
    """Normalize so that baseline -> 0 and teacher -> 1."""
    denom = teacher - baseline
    if abs(denom) < 1e-12:
        return 0.0
    return (value - baseline) / denom


def nanmean(lst):
    arr = [v for v in lst if v is not None and not np.isnan(v)]
    return float(np.mean(arr)) if arr else np.nan


def _save_table_png(
    png_path,
    present_datasets,
    short_names,
    student_ns,
    teacher_values,
    teacher_std,
    xgb_full_raw,
    xgb_full_raw_std,
    baseline_values,
    baseline_std,
    raw_values,
    raw_std,
):
    """
    Render an arxiv-style results table as a PNG using matplotlib.
      rows  = datasets
      cols  = Teacher | XGBoost | Student N=... | Aligned N=...
    Values shown as mean ± std. Best mean per row is bold.
    """
    n_students = len(student_ns)

    # ── Column headers ────────────────────────────────────────────────────
    method_headers = ["Teacher", "XGBoost"]
    for sn in student_ns:
        method_headers.append(f"Student N={sn}")
    for sn in student_ns:
        method_headers.append(f"Aligned N={sn}")
    col_headers = ["Dataset"] + method_headers
    n_cols = len(col_headers)

    # ✓ column indices — none (checkmark is embedded in the Aligned cell text)
    check_col_indices = set()

    # ── Build cell text ───────────────────────────────────────────────────
    def fmt(mean, std):
        if np.isnan(mean):
            return "–"
        if std > 0:
            return f"{mean:.3f} \u00b1{std:.3f}"
        return f"{mean:.3f}"

    cell_text = []   # list of rows, each row is list of strings
    check_mask = []  # parallel bool: True means show ✓ in that cell
    gray_rows  = set()  # row indices where Teacher < XGBoost (gray background)

    for di, short in enumerate(short_names):
        xgb_val = xgb_full_raw[di]
        teacher_val = teacher_values[di]

        row_text  = [short]
        row_check = [False]

        # Teacher — mark row gray if Teacher < XGBoost
        if (not np.isnan(teacher_val) and not np.isnan(xgb_val)
                and teacher_val < xgb_val):
            gray_rows.add(di)
        row_text.append(fmt(teacher_val, teacher_std[di]))
        row_check.append(False)
        # XGBoost
        row_text.append(fmt(xgb_val, xgb_full_raw_std[di])); row_check.append(False)
        # Student N=...
        for si in range(n_students):
            row_text.append(fmt(baseline_values[di, si], baseline_std[di, si]))
            row_check.append(False)
        # Aligned N=... (with inline ✓ when it beats XGBoost & matching Student)
        for si in range(n_students):
            aligned_val = raw_values[di, si]
            student_val = baseline_values[di, si]
            beats = (not np.isnan(aligned_val)
                     and (np.isnan(xgb_val) or aligned_val > xgb_val)
                     and (np.isnan(student_val) or aligned_val > student_val))
            row_text.append(fmt(aligned_val, raw_std[di, si]))
            row_check.append(beats)

        cell_text.append(row_text)
        check_mask.append(row_check)

    # ── Average row ───────────────────────────────────────────────────────
    # Step 1: collect column-wise means
    avg_vals = {}   # ci -> float mean (or nan)
    for ci in range(1, n_cols):
        vals = []
        for ri in range(len(cell_text)):
            raw = cell_text[ri][ci].split("\u00b1")[0].strip()
            try:
                vals.append(float(raw))
            except ValueError:
                pass
        avg_vals[ci] = float(np.mean(vals)) if vals else np.nan

    # Step 2: build text and check mask for the average row
    # Column layout: Teacher(1) XGBoost(2) Student_0(3)..Student_{n-1}(2+n)
    #                Aligned_0(3+n)..Aligned_{n-1}(2+2n)
    xgb_avg = avg_vals.get(2, np.nan)
    teacher_avg = avg_vals.get(1, np.nan)
    avg_row_text  = ["Average"]
    avg_row_check = [False]
    avg_row_gray  = (not np.isnan(teacher_avg) and not np.isnan(xgb_avg)
                     and teacher_avg < xgb_avg)
    for ci in range(1, n_cols):
        v = avg_vals.get(ci, np.nan)
        avg_row_text.append(f"{v:.3f}" if not np.isnan(v) else "\u2013")
        is_aligned_col = (ci >= 3 + n_students)
        if is_aligned_col:
            si = ci - (3 + n_students)
            student_avg = avg_vals.get(3 + si, np.nan)
            beats = (not np.isnan(v)
                     and (np.isnan(xgb_avg) or v > xgb_avg)
                     and (np.isnan(student_avg) or v > student_avg))
            avg_row_check.append(beats)
        else:
            avg_row_check.append(False)

    n_rows = len(cell_text)


    # ── Measure column widths from content (character-based estimate) ──────
    FONT_SIZE = 9
    HEAD_FONT = 9
    PAD_X     = 0.28   # horizontal padding per side (inches)
    ROW_H     = 0.30   # data row height (inches)
    HEAD_H    = 0.38   # header row height (inches)
    CHAR_W    = 0.090  # inches per character at 9pt

    def text_w(s, bold=False):
        return len(s) * CHAR_W * (1.05 if bold else 1.0) + 2 * PAD_X

    col_widths = []
    CHECK_COL_W = 0.35   # fixed narrow width for ✓ columns
    for ci in range(n_cols):
        if ci in check_col_indices:
            col_widths.append(CHECK_COL_W)
        else:
            w = text_w(col_headers[ci], bold=True)
            for ri in range(n_rows):
                w = max(w, text_w(cell_text[ri][ci]))
            col_widths.append(w)

    fig_w = sum(col_widths) + 0.05
    fig_h = HEAD_H + n_rows * ROW_H + ROW_H + 0.50   # +ROW_H for average row, +0.50 for title

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # ── Colors ────────────────────────────────────────────────────────────
    HEADER_BG = "#2c3e50"
    HEADER_FG = "white"
    ROW_ODD   = "#eef2f7"
    ROW_EVEN  = "white"
    CHECK_FG  = "#1e8449"   # green for ✓
    RULE_CLR  = "#5d6d7e"
    THIN_CLR  = "#d0d3d4"

    # Title
    ax.text(fig_w / 2, fig_h - 0.12,
            "ROC-AUC results (mean \u00b1 std). Green = Aligned beats XGBoost & Student.",
            ha="center", va="top", fontsize=10,
            color=HEADER_BG, fontweight="bold")

    # Top of header row (below title)
    top = fig_h - 0.42

    # ── Header row ────────────────────────────────────────────────────────
    x = 0.0
    for ci, (hdr, cw) in enumerate(zip(col_headers, col_widths)):
        rect = plt.Rectangle((x, top - HEAD_H), cw, HEAD_H,
                              facecolor=HEADER_BG, edgecolor="none")
        ax.add_patch(rect)
        ha = "left"
        tx = x + PAD_X
        ax.text(tx, top - HEAD_H / 2, hdr,
                ha=ha, va="center", fontsize=HEAD_FONT,
                color=HEADER_FG, fontweight="bold")
        x += cw

    # Heavy top rule and header-bottom rule
    ax.hlines(top,          0, fig_w, colors=RULE_CLR, linewidth=1.8)
    ax.hlines(top - HEAD_H, 0, fig_w, colors=RULE_CLR, linewidth=1.2)

    ROW_GRAY  = "#d5d8dc"   # light gray for Teacher < XGBoost rows

    # ── Data rows ─────────────────────────────────────────────────────────
    for ri in range(n_rows):
        row_top = top - HEAD_H - ri * ROW_H
        if ri in gray_rows:
            bg = ROW_GRAY
        else:
            bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
        rect = plt.Rectangle((0, row_top - ROW_H), fig_w, ROW_H,
                              facecolor=bg, edgecolor="none")
        ax.add_patch(rect)
        if ri > 0:
            ax.hlines(row_top, 0, fig_w, colors=THIN_CLR, linewidth=0.4)

        x = 0.0
        for ci, cw in enumerate(col_widths):
            txt      = cell_text[ri][ci]
            is_check = check_mask[ri][ci]
            color    = CHECK_FG if is_check else "#1c2833"
            ha       = "left" if ci == 0 else "center"
            tx       = (x + PAD_X) if ci == 0 else (x + cw / 2)
            ax.text(tx, row_top - ROW_H / 2, txt,
                    ha=ha, va="center",
                    fontsize=FONT_SIZE,
                    color=color,
                    fontweight="bold" if is_check else "normal")
            x += cw

    # ── Thick separator before average row ────────────────────────────────
    avg_top = top - HEAD_H - n_rows * ROW_H
    ax.hlines(avg_top, 0, fig_w, colors=RULE_CLR, linewidth=1.5)

    # ── Average row ───────────────────────────────────────────────────────
    AVG_BG = ROW_GRAY if avg_row_gray else "#dce8f5"   # gray if Teacher avg < XGBoost avg, else light blue
    rect = plt.Rectangle((0, avg_top - ROW_H), fig_w, ROW_H,
                          facecolor=AVG_BG, edgecolor="none")
    ax.add_patch(rect)
    x = 0.0
    for ci, cw in enumerate(col_widths):
        txt      = avg_row_text[ci]
        is_check = avg_row_check[ci]
        color    = CHECK_FG if is_check else "#1c2833"
        ha       = "left" if ci == 0 else "center"
        tx       = (x + PAD_X) if ci == 0 else (x + cw / 2)
        ax.text(tx, avg_top - ROW_H / 2, txt,
                ha=ha, va="center",
                fontsize=FONT_SIZE,
                color=color,
                fontweight="bold")
        x += cw

    # Heavy bottom rule
    bottom = avg_top - ROW_H
    ax.hlines(bottom, 0, fig_w, colors=RULE_CLR, linewidth=1.8)

    # Thin vertical separator after dataset column
    ax.vlines(col_widths[0], bottom, top, colors=THIN_CLR, linewidth=0.6)

    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser(
        description="Plot aggregated evaluate_aligned_student results as a normalized histogram"
    )
    parser.add_argument("--dataset", type=str, nargs="+", default=["breast_cancer"])
    parser.add_argument("--student_n", type=int, nargs="+", default=[20], help="Student training sizes")
    parser.add_argument("--layer", type=int, default=11, help="The layer to evaluate")
    parser.add_argument("--per_token", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples used.")
    parser.add_argument("--use_tabpfn", action="store_true", help="Use TabPFN to generate synthetic data.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for aligner training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for aligner training.")
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[])
    parser.add_argument("--predict_residual", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default=None, help="Path to save the figure. Defaults to auto-generated name.")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of OpenML repeats to aggregate over.")
    parser.add_argument("--train_on_rest", action="store_true",
                        help="If set, look up results where the aligner was trained on all datasets except the one "
                             "being evaluated (leave-one-out). Otherwise, look up results where the aligner was "
                             "trained on the same dataset (default).")
    parser.add_argument("--use_feature_stats", action="store_true",
                        help="Look up results where the aligner was conditioned on per-feature statistics.")
    parser.add_argument("--loss_beta", type=float, default=None,
                        help="If specified, look up results produced with train_activation_aligner_v2 using this KL weight. "
                             "If unspecified, look up results from the original train_activation_aligner.")
    parser.add_argument("--clip_grad", type=float, default=None, help="If specified, look up results where gradient clipping was used.")
    parser.add_argument("--max_epochs", type=int, default=None, help="If specified, look up results where max_epochs was used.")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use.")
    args = parser.parse_args()

    # Expand "tabarena" shorthand
    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    # ------------------------------------------------------------------ #
    # Gather results — average across repeats, keep per-repeat for CI
    # ------------------------------------------------------------------ #
    # all_data[dataset][student_n] = {metric: mean_value}
    # all_data_raw[dataset][student_n] = {metric: [per-repeat values]}
    all_data = {}
    all_data_raw = {}
    xgb_data = {}       # xgb_data[dataset][-1] = mean
    xgb_data_raw = {}   # xgb_data_raw[dataset] = [per-repeat values]

    for dataset in tqdm(datasets, desc="Loading results"):
        accum = {}   # student_n -> {metric: [values across repeats]}
        xgb_vals = []

        for repeat in range(args.n_repeats):
            res = load_results_for_repeat(args, dataset, datasets, args.layer, repeat)
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
            all_data_raw[dataset] = accum
            xgb_data[dataset] = {-1: nanmean(xgb_vals)} if xgb_vals else {}
            xgb_data_raw[dataset] = xgb_vals

    if not all_data:
        print("No results found. Make sure evaluate_aligned_student.py has been run first.")
        return

    present_datasets = list(all_data.keys())
    student_ns = sorted(args.student_n)

    # ------------------------------------------------------------------ #
    # Build value arrays
    # ------------------------------------------------------------------ #
    raw_values      = np.full((len(present_datasets), len(student_ns)), np.nan)
    raw_std         = np.full((len(present_datasets), len(student_ns)), np.nan)
    baseline_values = np.full((len(present_datasets), len(student_ns)), np.nan)
    baseline_std    = np.full((len(present_datasets), len(student_ns)), np.nan)
    teacher_values  = np.full(len(present_datasets), np.nan)
    teacher_std     = np.full(len(present_datasets), np.nan)
    xgb_full_raw    = np.full(len(present_datasets), np.nan)
    xgb_full_raw_std= np.full(len(present_datasets), np.nan)

    for di, dataset in enumerate(present_datasets):
        for si, sn in enumerate(student_ns):
            if sn not in all_data[dataset]:
                continue
            entry     = all_data[dataset][sn]
            entry_raw = all_data_raw[dataset][sn]
            teacher   = entry["teacher_roc_auc"]
            baseline  = entry["baseline_roc_auc"]
            aligned   = entry["aligned_roc_auc"]
            raw_values[di, si]      = aligned
            raw_std[di, si]         = float(np.std(entry_raw["aligned_roc_auc"])) if len(entry_raw["aligned_roc_auc"]) > 1 else 0.0
            baseline_values[di, si] = baseline
            baseline_std[di, si]    = float(np.std(entry_raw["baseline_roc_auc"])) if len(entry_raw["baseline_roc_auc"]) > 1 else 0.0
            if np.isnan(teacher_values[di]):
                teacher_values[di] = teacher
                teacher_std[di]    = float(np.std(entry_raw["teacher_roc_auc"])) if len(entry_raw["teacher_roc_auc"]) > 1 else 0.0

        xgb_full_val = xgb_data.get(dataset, {}).get(-1, np.nan)
        xgb_raw_list = xgb_data_raw.get(dataset, [])
        if not np.isnan(xgb_full_val):
            xgb_full_raw[di]      = xgb_full_val
            xgb_full_raw_std[di]  = float(np.std(xgb_raw_list)) if len(xgb_raw_list) > 1 else 0.0

    # ------------------------------------------------------------------ #
    # Count datasets where aligned beats full-training-set XGBoost
    # ------------------------------------------------------------------ #
    beats_full_xgb = []
    for si, sn in enumerate(student_ns):
        count_better = 0
        count_valid = 0
        for di, dataset in enumerate(present_datasets):
            aligned_val = raw_values[di, si]
            xgb_full_val = xgb_full_raw[di]
            baseline_val = baseline_values[di, si]
            if not np.isnan(aligned_val) and not np.isnan(xgb_full_val):
                count_valid += 1
                if aligned_val > xgb_full_val and (np.isnan(baseline_val) or aligned_val > baseline_val):
                    count_better += 1
        beats_full_xgb.append((count_better, count_valid))

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    n_datasets = len(present_datasets)
    n_students = len(student_ns)

    group_width = 0.5
    x           = np.arange(n_datasets)
    colors      = plt.cm.tab10(np.linspace(0, 0.9, n_students))
    short_names = [d.split("/")[-1] if "/" in d else d for d in present_datasets]

    repeat_suffix = f" (mean of {args.n_repeats} repeats)" if args.n_repeats > 1 else ""

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 1.2), 6))

    # ── Raw metric ───────────────────────────────────────────────────────
    def draw_ci(ax, x_center, mean, std, color, zorder=6):
        """Draw a ±1 std vertical error bar with caps at x_center."""
        if std <= 0:
            return
        cap_width = slot_w * 0.2
        ax.vlines(x_center, mean - std, mean + std, color=color, linewidth=1.2, alpha=0.6, zorder=zorder)
        ax.hlines([mean - std, mean + std], x_center - cap_width, x_center + cap_width,
                  color=color, linewidth=1.2, alpha=0.6, zorder=zorder)

    # Layout: Teacher | XGBoost | (Aligned_N | Baseline_N) * n_students
    # All slots equal width within group_width.
    n_slots = 2 + n_students * 2  # teacher + xgb + (aligned+baseline) per student
    slot_w  = group_width / n_slots
    half_line = slot_w * 0.45

    def slot_center(di, slot_idx):
        """x-center of slot_idx (0-based) within dataset di's group."""
        group_start = x[di] - group_width / 2 + slot_w / 2
        return group_start + slot_idx * slot_w

    XGB_FULL_COLOR = "#1a5c1a"

    # checkmarks_per_dataset[di] = list of si indices that earned a ✓
    checkmarks_per_dataset = {di: [] for di in range(n_datasets)}

    for di in range(n_datasets):
        # ── Teacher (slot 0) ─────────────────────────────────────────────
        tv = teacher_values[di]
        if not np.isnan(tv):
            cx = slot_center(di, 0)
            ax.plot([cx - half_line, cx + half_line], [tv, tv],
                    color="red", linewidth=2.0, linestyle="-", zorder=4,
                    label="Teacher" if di == 0 else "_nolegend_")
            draw_ci(ax, cx, tv, teacher_std[di], "red", zorder=5)

        # ── XGBoost (slot 1) ─────────────────────────────────────────────
        xv = xgb_full_raw[di]
        if not np.isnan(xv):
            cx = slot_center(di, 1)
            ax.plot([cx - half_line, cx + half_line], [xv, xv],
                    color=XGB_FULL_COLOR, linewidth=2.0, linestyle="-", zorder=5,
                    label="XGBoost" if di == 0 else "_nolegend_")
            draw_ci(ax, cx, xv, xgb_full_raw_std[di], XGB_FULL_COLOR, zorder=6)

        # ── Per-student: Aligned then Baseline (slots 2+) ────────────────
        for si, sn in enumerate(student_ns):
            baseline_slot = 2 + si * 2
            aligned_slot  = 2 + si * 2 + 1
            count_better, count_valid = beats_full_xgb[si]

            av = raw_values[di, si]
            bv = baseline_values[di, si]
            if not np.isnan(av):
                cx = slot_center(di, aligned_slot)
                ax.plot([cx - half_line, cx + half_line], [av, av],
                        color=colors[si], linewidth=2.0, linestyle="-", zorder=3,
                        label=f"Aligned N={sn} ({count_better}/{count_valid})" if di == 0 else "_nolegend_")
                draw_ci(ax, cx, av, raw_std[di, si], colors[si])
                # Track ✓ if aligned mean beats both XGBoost and student baseline
                xv_check = xgb_full_raw[di]
                if (not np.isnan(xv_check) and av > xv_check
                        and (np.isnan(bv) or av > bv)):
                    checkmarks_per_dataset[di].append(si)

            if not np.isnan(bv):
                cx = slot_center(di, baseline_slot)
                ax.plot([cx - half_line, cx + half_line], [bv, bv],
                        color=colors[si], linewidth=1.5, linestyle="--", zorder=3,
                        label=f"Student N={sn}" if di == 0 else "_nolegend_")
                draw_ci(ax, cx, bv, baseline_std[di, si], colors[si], zorder=4)

    ax.axhline(y=0, color="black", linewidth=1.0, zorder=4)
    ax.set_ylabel("Raw Metric (ROC-AUC)")
    ax.set_title(
        f"Aligned Student Performance (layer={args.layer}){repeat_suffix}\n"
        f"Estimators={args.n_estimators}, Patience={args.patience}"
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.02), borderaxespad=0)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=-45, ha="left", fontsize=9)
    ax.set_xlabel("Dataset")

    plt.tight_layout()

    # Draw ✓ marks below each dataset's x-tick label, stacked, colored by student
    fig.canvas.draw()  # needed to get tick label positions
    renderer = fig.canvas.get_renderer()
    for di, si_list in checkmarks_per_dataset.items():
        if not si_list:
            continue
        tick_label = ax.get_xticklabels()[di]
        bbox = tick_label.get_window_extent(renderer=renderer)
        # Convert bottom of tick label bbox from display coords to axes coords
        inv = ax.transAxes.inverted()
        x_ax, y_ax = inv.transform((bbox.x0 + bbox.width / 2, bbox.y0))
        line_height = 0.03  # spacing between stacked checkmarks in axes fraction
        for rank, si in enumerate(si_list):
            ax.text(
                x_ax, y_ax - rank * line_height,
                "✓",
                fontsize=18, va="top", ha="center",
                color=colors[si],
                transform=ax.transAxes,
                clip_on=False,
            )

    # (tight_layout already called above)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if args.output:
        output_path = args.output
    else:
        output_path = pruning_utils.create_filename_from_args(
            vars(args),
            script_name="plot_aligned_results"
        )
        output_path = f'{output_path}/graph.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")

    # ------------------------------------------------------------------ #
    # Table PNG (arxiv style)
    # ------------------------------------------------------------------ #
    table_path = output_path.replace("graph.png", "table.png")
    _save_table_png(
        table_path,
        present_datasets,
        short_names,
        student_ns,
        teacher_values,
        teacher_std,
        xgb_full_raw,
        xgb_full_raw_std,
        baseline_values,
        baseline_std,
        raw_values,
        raw_std,
    )
    print(f"Table PNG saved to {table_path}")


if __name__ == "__main__":
    main()
