"""Benchmark evaluation and visualization for aligned student models.

Generates publication-ready artifacts across TabArena classification tasks:
1. Sample Efficiency Curve (Effective M vs. Student N with reference diagonal)
2. Summary Metrics Table (Win Rate, Wilcoxon p-value, Relative Gap Closed %, Mean M Gain, Beats XGBoost)
3. Per-Dataset Sample Efficiency Table (Detailed per-dataset M values with XGBoost and Teacher comparisons)
4. Relative Gap Closed Curve (Distillation efficiency % vs. N)
5. Win Rate Bar Chart (Pairwise win rate % vs. baseline with significance indicators)
"""

import argparse
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from scipy import stats

import pruning_utils
from cache_utils import OUTPUT_DIR
from evaluate_aligned_student import evaluate_aligned_student
from train_xgboost import train_xgboost
from train_xgboost_opt import train_xgboost_opt


# --------------------------------------------------------------------------- #
# 1. Metric Utilities & Interpolation
# --------------------------------------------------------------------------- #

def nanmean(lst):
    """Compute mean ignoring None and NaN values."""
    arr = [v for v in lst if v is not None and not np.isnan(v)]
    return float(np.mean(arr)) if arr else np.nan


def find_maximal_m(student_ns, baseline_scores, aligned_score, teacher_score=None):
    """Find effective baseline student fraction M for an aligned score via linear interpolation.

    Linearly interpolates within the baseline curve (and optional teacher at N=1.0).
    Extrapolates linearly when aligned_score lies outside the baseline range.
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
        if sn in scores_dict and scores_dict[sn] is not None and not np.isnan(scores_dict[sn])
    ]
    if not valid_student_ns:
        return min(student_ns)

    # Extrapolate below minimum baseline
    if aligned_score <= scores_dict[valid_student_ns[0]]:
        if len(valid_student_ns) >= 2:
            m0, m1 = valid_student_ns[0], valid_student_ns[1]
            b0, b1 = scores_dict[m0], scores_dict[m1]
            if b1 != b0:
                return m0 + (aligned_score - b0) / (b1 - b0) * (m1 - m0)
        return valid_student_ns[0]

    # Extrapolate above maximum baseline / teacher
    if aligned_score >= scores_dict[valid_student_ns[-1]]:
        if len(valid_student_ns) >= 2:
            m_last, m_prev = valid_student_ns[-1], valid_student_ns[-2]
            b_last, b_prev = scores_dict[m_last], scores_dict[m_prev]
            if b_last != b_prev:
                return m_last + (aligned_score - b_last) / (b_last - b_prev) * (m_last - m_prev)
        return valid_student_ns[-1]

    # Interpolate within interval [M_i, M_{i+1}]
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


def _compute_slice_metrics(a_vals, b_vals, m_vals, gc_vals, xgb_vals, sn=None):
    """Compute summary metrics (Win Rate, Wilcoxon, Gap Closed, M, XGBoost) for a slice of evaluations."""
    total = len(a_vals)
    if total == 0:
        return None

    wins = int(np.sum(a_vals > b_vals))
    win_rate = (wins / total) * 100.0

    diffs = a_vals - b_vals
    if total >= 5 and np.any(diffs != 0):
        try:
            res = stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
            pval_str = f"p={res.pvalue:.4f}" if res.pvalue >= 0.0001 else "p<0.0001"
        except Exception:
            pval_str = "n/a"
    else:
        pval_str = "n/a"

    valid_gc = [g for g in gc_vals if not np.isnan(g)]
    mean_gc = float(np.mean(valid_gc)) if valid_gc else np.nan
    median_gc = float(np.median(valid_gc)) if valid_gc else np.nan

    valid_m = [m for m in m_vals if not np.isnan(m)]
    mean_m = float(np.mean(valid_m)) if valid_m else np.nan
    mean_gain = (mean_m - sn) if (sn is not None and not np.isnan(mean_m)) else np.nan

    xgb_comparisons = [
        (a_vals[i] > xgb_vals[i])
        for i in range(total)
        if not np.isnan(xgb_vals[i])
    ]
    beats_xgb_cnt = int(np.sum(xgb_comparisons))
    xgb_total = len(xgb_comparisons)
    beats_xgb_str = (
        f"{beats_xgb_cnt}/{xgb_total} ({beats_xgb_cnt/xgb_total*100:.1f}%)"
        if xgb_total > 0 else "–"
    )

    return {
        "wins": wins,
        "total": total,
        "win_rate": win_rate,
        "pval_str": pval_str,
        "mean_gc": mean_gc,
        "median_gc": median_gc,
        "mean_m": mean_m,
        "mean_gain": mean_gain,
        "beats_xgb_cnt": beats_xgb_cnt,
        "beats_xgb_total": xgb_total,
        "beats_xgb_str": beats_xgb_str,
        "valid_gc": valid_gc,
        "valid_m": valid_m,
    }


# --------------------------------------------------------------------------- #
# 2. Data Loading & Matrix Assembly
# --------------------------------------------------------------------------- #

def load_results_for_repeat(args, dataset, layer_k, repeat):
    """Load evaluate_aligned_student results for all student_n values for a given repeat."""
    synthetic_train = f"{dataset}[synthetic-n_samples_{args.n_samples}-repeat_{repeat}]"
    results = {}

    for student_n in sorted(args.student_n):
        eval_kwargs = dict(
            eval_dataset=dataset,
            train_dataset=synthetic_train,
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
            aligner_opt=getattr(args, "aligner_opt", False),
        )
        if not evaluate_aligned_student.check_call_in_cache(**eval_kwargs):
            continue

        metrics = evaluate_aligned_student(**eval_kwargs)["metrics"]
        results[student_n] = {
            "teacher_roc_auc": metrics["teacher_roc_auc"],
            "baseline_roc_auc": metrics["baseline_roc_auc"],
            "aligned_roc_auc": metrics["aligned_roc_auc"],
        }
    return results


def load_xgboost_result_for_repeat(args, dataset, repeat):
    """Load XGBoost result on the full training set (student_n=-1) for a given repeat."""
    train_func = train_xgboost_opt if args.xgboost_opt else train_xgboost
    kwargs = dict(dataset=dataset, student_n=-1, repeat=repeat)
    if not train_func.check_call_in_cache(**kwargs):
        return None
    return train_func(**kwargs)["metrics"]["xgboost_roc_auc"]


def load_benchmark_data(args, datasets, student_ns, layer):
    """Gather evaluation data across datasets and repeats, returning structured matrices."""
    all_data = {}
    xgb_data = {}

    for ds in datasets:
        accum = {}
        xgb_vals = []
        for repeat in range(args.n_repeats):
            res = load_results_for_repeat(args, ds, layer, repeat)
            for sn, entry in res.items():
                if sn not in accum:
                    accum[sn] = {"teacher_roc_auc": [], "baseline_roc_auc": [], "aligned_roc_auc": []}
                for k in accum[sn]:
                    accum[sn][k].append(entry[k])

            xgb_val = load_xgboost_result_for_repeat(args, ds, repeat)
            if xgb_val is not None:
                xgb_vals.append(xgb_val)

        if accum:
            all_data[ds] = {
                sn: {k: nanmean(vs) for k, vs in entry.items()}
                for sn, entry in accum.items()
            }
            xgb_data[ds] = {-1: nanmean(xgb_vals)} if xgb_vals else {}

    if not all_data:
        return None

    present_datasets = list(all_data.keys())
    n_datasets = len(present_datasets)
    n_students = len(student_ns)

    m_matrix = np.full((n_datasets, n_students), np.nan)
    aligned_matrix = np.full((n_datasets, n_students), np.nan)
    baseline_matrix = np.full((n_datasets, n_students), np.nan)
    teacher_values = np.full(n_datasets, np.nan)
    xgb_values = np.full(n_datasets, np.nan)

    for di, ds in enumerate(present_datasets):
        ds_data = all_data[ds]
        baseline_scores = {
            sn: ds_data[sn]["baseline_roc_auc"]
            for sn in student_ns
            if sn in ds_data and ds_data[sn]["baseline_roc_auc"] is not None
        }

        teacher_list = [
            ds_data[sn]["teacher_roc_auc"]
            for sn in student_ns
            if sn in ds_data and ds_data[sn].get("teacher_roc_auc") is not None
            and not np.isnan(ds_data[sn]["teacher_roc_auc"])
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
                m_matrix[di, si] = find_maximal_m(
                    student_ns, baseline_scores, aligned_score, teacher_score=teacher_val
                )
            if baseline_score is not None:
                baseline_matrix[di, si] = baseline_score

        xgb_values[di] = xgb_data.get(ds, {}).get(-1, np.nan)

    # Teacher superiority condition: True if Baseline < Teacher
    valid_mask = np.zeros((n_datasets, n_students), dtype=bool)
    for di in range(n_datasets):
        for si in range(n_students):
            b_val = baseline_matrix[di, si]
            t_val = teacher_values[di]
            a_val = aligned_matrix[di, si]
            if not np.isnan(a_val) and not np.isnan(b_val) and not np.isnan(t_val) and b_val < t_val:
                valid_mask[di, si] = True

    # Relative gap closed: (aligned - baseline) / (teacher - baseline) * 100%
    gap_closed_matrix = np.full((n_datasets, n_students), np.nan)
    for di in range(n_datasets):
        for si in range(n_students):
            if valid_mask[di, si]:
                denom = teacher_values[di] - baseline_matrix[di, si]
                if abs(denom) > 1e-9:
                    gap_closed_matrix[di, si] = (
                        (aligned_matrix[di, si] - baseline_matrix[di, si]) / denom * 100.0
                    )

    return {
        "present_datasets": present_datasets,
        "m_matrix": m_matrix,
        "aligned_matrix": aligned_matrix,
        "baseline_matrix": baseline_matrix,
        "teacher_values": teacher_values,
        "xgb_values": xgb_values,
        "valid_mask": valid_mask,
        "gap_closed_matrix": gap_closed_matrix,
    }


# --------------------------------------------------------------------------- #
# 3. Plot Functions
# --------------------------------------------------------------------------- #

def plot_sample_efficiency_curve(output_path, student_ns, m_matrix, valid_mask, model_name="tabpfn", layer=23):
    """Plot Effective Baseline Fraction M vs. Aligned Student Fraction N."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    n_datasets = m_matrix.shape[0]

    # Diagonal parity reference line (M = N)
    min_val, max_val = min(student_ns), max(student_ns)
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        linestyle="--", color="#7f8c8d", linewidth=2.0, alpha=0.85,
        label="No gain ($M = N$)", zorder=2,
    )

    # Translucent dataset trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, n_datasets)))
    for di in range(n_datasets):
        ds_ns, ds_ms = [], []
        for si, sn in enumerate(student_ns):
            if valid_mask[di, si] and not np.isnan(m_matrix[di, si]):
                ds_ns.append(sn)
                ds_ms.append(m_matrix[di, si])
        if len(ds_ns) > 1:
            ax.plot(ds_ns, ds_ms, color=colors[di % len(colors)], alpha=0.20, linewidth=1.0, zorder=3)

    # Average M curve with ±1 SEM error band
    avg_m_vals, sem_m_vals, valid_sns = [], [], []
    for si, sn in enumerate(student_ns):
        vals = [m_matrix[di, si] for di in range(n_datasets) if valid_mask[di, si] and not np.isnan(m_matrix[di, si])]
        if vals:
            valid_sns.append(sn)
            avg_m_vals.append(float(np.mean(vals)))
            sem_m_vals.append(float(stats.sem(vals)) if len(vals) > 1 else 0.0)

    if valid_sns:
        avg_m_arr = np.array(avg_m_vals)
        sem_m_arr = np.array(sem_m_vals)
        ax.fill_between(
            valid_sns, avg_m_arr - sem_m_arr, avg_m_arr + sem_m_arr,
            color="#2980b9", alpha=0.20, label="Mean $\\pm 1$ SEM", zorder=4,
        )
        ax.plot(
            valid_sns, avg_m_arr,
            marker="o", markersize=7, linewidth=2.8, color="#1b4f72",
            label="Aligned Student (Mean)", zorder=5,
        )

        for sn, m_val in zip(valid_sns, avg_m_arr):
            gain = m_val - sn
            if gain > 0:
                ax.annotate(
                    f"+{gain:.2f}", xy=(sn, m_val), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold", color="#145a32",
                )

    ax.set_xlabel("Aligned Student Fraction ($N$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Effective Baseline Student Fraction ($M$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Sample Efficiency Curve: Effective Baseline $M$ vs. Student $N$\n"
        f"Model: {model_name.upper()} | Layer: {layer}",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xticks(student_ns)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.grid(True, linestyle=":", alpha=0.55, zorder=1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_summary_metrics_table(
    output_path,
    student_ns,
    aligned_matrix,
    baseline_matrix,
    teacher_values,
    xgb_values,
    m_matrix,
    valid_mask,
    gap_closed_matrix,
):
    """Render a compact publication-ready summary table combining Win Rate, Wilcoxon, and Gap Closed."""
    headers = [
        "Student Budget (N)",
        "Win Rate vs Base",
        "Wilcoxon p-val",
        "Mean Gap Closed",
        "Median Gap Closed",
        "Effective M",
        "Mean Gain (M-N)",
        "Beats Full XGB",
    ]

    table_data = []
    n_datasets = aligned_matrix.shape[0]

    pooled_a, pooled_b, pooled_gc, pooled_gains = [], [], [], []
    total_beats_xgb, total_xgb = 0, 0

    for si, sn in enumerate(student_ns):
        valid_indices = [di for di in range(n_datasets) if valid_mask[di, si]]
        if not valid_indices:
            continue

        a_slice = aligned_matrix[valid_indices, si]
        b_slice = baseline_matrix[valid_indices, si]
        m_slice = m_matrix[valid_indices, si]
        gc_slice = gap_closed_matrix[valid_indices, si]
        xgb_slice = xgb_values[valid_indices]

        m_dict = _compute_slice_metrics(a_slice, b_slice, m_slice, gc_slice, xgb_slice, sn=sn)
        if not m_dict:
            continue

        table_data.append([
            f"N = {sn:.2g}",
            f"{m_dict['wins']}/{m_dict['total']} ({m_dict['win_rate']:.1f}%)",
            m_dict["pval_str"],
            f"{m_dict['mean_gc']:+.1f}%" if not np.isnan(m_dict["mean_gc"]) else "–",
            f"{m_dict['median_gc']:+.1f}%" if not np.isnan(m_dict["median_gc"]) else "–",
            f"{m_dict['mean_m']:.3f}" if not np.isnan(m_dict["mean_m"]) else "–",
            f"{'+' if m_dict['mean_gain'] > 0 else ''}{m_dict['mean_gain']:.3f}" if not np.isnan(m_dict["mean_gain"]) else "–",
            m_dict["beats_xgb_str"],
        ])

        pooled_a.extend(a_slice)
        pooled_b.extend(b_slice)
        pooled_gc.extend(m_dict["valid_gc"])
        pooled_gains.extend([m - sn for m in m_dict["valid_m"]])
        total_beats_xgb += m_dict["beats_xgb_cnt"]
        total_xgb += m_dict["beats_xgb_total"]

    # Overall Summary Row
    if pooled_a:
        pa, pb = np.array(pooled_a), np.array(pooled_b)
        o_wins = int(np.sum(pa > pb))
        o_tot = len(pa)
        o_wr = o_wins / o_tot * 100.0

        try:
            o_res = stats.wilcoxon(pa, pb, alternative="two-sided")
            o_pval = f"p={o_res.pvalue:.4f}" if o_res.pvalue >= 0.0001 else "p<0.0001"
        except Exception:
            o_pval = "n/a"

        o_mgc = float(np.mean(pooled_gc)) if pooled_gc else np.nan
        o_medgc = float(np.median(pooled_gc)) if pooled_gc else np.nan
        o_gain = float(np.mean(pooled_gains)) if pooled_gains else np.nan
        o_xgb = f"{total_beats_xgb}/{total_xgb} ({total_beats_xgb/total_xgb*100:.1f}%)" if total_xgb > 0 else "–"

        table_data.append([
            "Overall (Pooled)",
            f"{o_wins}/{o_tot} ({o_wr:.1f}%)",
            o_pval,
            f"{o_mgc:+.1f}%" if not np.isnan(o_mgc) else "–",
            f"{o_medgc:+.1f}%" if not np.isnan(o_medgc) else "–",
            "—",
            f"{'+' if o_gain > 0 else ''}{o_gain:.3f}" if not np.isnan(o_gain) else "–",
            o_xgb,
        ])

    n_rows = len(table_data)
    col_widths = [1.35, 1.45, 1.15, 1.25, 1.25, 1.05, 1.20, 1.45]
    fig, ax = plt.subplots(figsize=(sum(col_widths) + 0.4, (n_rows + 2) * 0.38 + 0.8), dpi=150)
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        colWidths=[w / sum(col_widths) for w in col_widths],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.6)

    for (ri, ci), cell in table.get_celld().items():
        cell.set_edgecolor("#bdc3c7")
        cell.set_linewidth(0.8)
        if ri == 0:
            cell.set_facecolor("#2c3e50")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif ri == n_rows:
            cell.set_facecolor("#d5f5e3")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#fdfefe" if ri % 2 == 1 else "#f2f4f4")

    ax.set_title(
        "Aligned Student Benchmark Summary (Pairwise Win Rates & Distillation Metrics)",
        fontsize=11.5, fontweight="bold", pad=16,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sample_efficiency_table(
    output_path,
    present_datasets,
    student_ns,
    m_matrix,
    aligned_matrix,
    baseline_matrix,
    teacher_values,
    xgb_values,
):
    """Render full per-dataset breakdown table showing effective M for each dataset and student N."""
    short_names = [d.split("/")[-1] if "/" in d else d for d in present_datasets]
    col_headers = ["Dataset"] + [f"N={sn:.2g}" for sn in student_ns] + ["Mean Gain"]
    n_cols = len(col_headers)

    HEADER_BG, ROW_ODD, ROW_EVEN = "#2c3e50", "#eef2f7", "white"
    HIGHLIGHT_BG, RED_BG, GRAY_BG = "#c8e6c9", "#ffcdd2", "#d5d8dc"
    RULE_CLR, THIN_CLR = "#5d6d7e", "#d0d3d4"

    cell_text, cell_bg_colors = [], []

    for di, short in enumerate(short_names):
        row_text = [short]
        row_bgs = [None]
        gains = []
        xgb_val, teacher_val = xgb_values[di], teacher_values[di]

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
        else:
            row_text.append("–")
        row_bgs.append(None)

        cell_text.append(row_text)
        cell_bg_colors.append(row_bgs)

    # Average row calculation
    avg_m_vals = []
    for si, sn in enumerate(student_ns):
        valid_ms = [
            m_matrix[di, si] for di in range(len(short_names))
            if not np.isnan(m_matrix[di, si])
            and not (
                not np.isnan(baseline_matrix[di, si])
                and not np.isnan(teacher_values[di])
                and baseline_matrix[di, si] >= teacher_values[di]
            )
        ]
        avg_m_vals.append(float(np.mean(valid_ms)) if valid_ms else np.nan)

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
    else:
        avg_row_text.append("–")
    avg_row_bgs.append(None)

    # Layout geometry
    n_rows = len(cell_text)
    FONT_SIZE, HEAD_FONT = 9, 9
    PAD_X, ROW_H, HEAD_H, CHAR_W = 0.25, 0.30, 0.38, 0.090

    def text_w(s, bold=False):
        return len(s) * CHAR_W * (1.05 if bold else 1.0) + 2 * PAD_X

    col_widths = [
        max(
            text_w(col_headers[ci], bold=True),
            max(text_w(cell_text[ri][ci]) for ri in range(n_rows)),
            text_w(avg_row_text[ci]),
        )
        for ci in range(n_cols)
    ]

    CAPTION_H = 0.72
    fig_w = sum(col_widths) + 0.05
    fig_h = HEAD_H + n_rows * ROW_H + ROW_H + 0.50 + CAPTION_H

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(
        fig_w / 2, fig_h - 0.12,
        "Effective Baseline Student Fraction M for Aligned N",
        ha="center", va="top", fontsize=10, color=HEADER_BG, fontweight="bold",
    )

    top = fig_h - 0.42

    # Headers
    x = 0.0
    for ci, (hdr, cw) in enumerate(zip(col_headers, col_widths)):
        ax.add_patch(plt.Rectangle((x, top - HEAD_H), cw, HEAD_H, facecolor=HEADER_BG, edgecolor="none"))
        tx = (x + PAD_X) if ci == 0 else (x + cw / 2)
        ha = "left" if ci == 0 else "center"
        ax.text(tx, top - HEAD_H / 2, hdr, ha=ha, va="center", fontsize=HEAD_FONT, color="white", fontweight="bold")
        x += cw

    ax.hlines(top, 0, fig_w, colors=RULE_CLR, linewidth=1.8)
    ax.hlines(top - HEAD_H, 0, fig_w, colors=RULE_CLR, linewidth=1.2)

    # Data rows
    for ri in range(n_rows):
        row_top = top - HEAD_H - ri * ROW_H
        bg = ROW_ODD if ri % 2 == 0 else ROW_EVEN
        ax.add_patch(plt.Rectangle((0, row_top - ROW_H), fig_w, ROW_H, facecolor=bg, edgecolor="none"))
        if ri > 0:
            ax.hlines(row_top, 0, fig_w, colors=THIN_CLR, linewidth=0.4)

        x = 0.0
        for ci, cw in enumerate(col_widths):
            bg_color = cell_bg_colors[ri][ci]
            if bg_color is not None:
                ax.add_patch(plt.Rectangle((x, row_top - ROW_H), cw, ROW_H, facecolor=bg_color, edgecolor="none"))

            tx = (x + PAD_X) if ci == 0 else (x + cw / 2)
            ha = "left" if ci == 0 else "center"
            ax.text(
                tx, row_top - ROW_H / 2, cell_text[ri][ci],
                ha=ha, va="center", fontsize=FONT_SIZE, color="#1c2833",
                fontweight="bold" if bg_color is not None else "normal",
            )
            x += cw

    # Average row
    avg_top = top - HEAD_H - n_rows * ROW_H
    ax.hlines(avg_top, 0, fig_w, colors=RULE_CLR, linewidth=1.5)
    ax.add_patch(plt.Rectangle((0, avg_top - ROW_H), fig_w, ROW_H, facecolor="#dce8f5", edgecolor="none"))

    x = 0.0
    for ci, cw in enumerate(col_widths):
        bg_color = avg_row_bgs[ci]
        if bg_color is not None:
            ax.add_patch(plt.Rectangle((x, avg_top - ROW_H), cw, ROW_H, facecolor=bg_color, edgecolor="none"))
        tx = (x + PAD_X) if ci == 0 else (x + cw / 2)
        ha = "left" if ci == 0 else "center"
        ax.text(
            tx, avg_top - ROW_H / 2, avg_row_text[ci],
            ha=ha, va="center", fontsize=FONT_SIZE, color="#1c2833", fontweight="bold",
        )
        x += cw

    bottom = avg_top - ROW_H
    ax.hlines(bottom, 0, fig_w, colors=RULE_CLR, linewidth=1.8)
    ax.vlines(col_widths[0], bottom, top, colors=THIN_CLR, linewidth=0.6)

    # Caption Swatches
    sw_w, sw_h = 0.18, 0.12
    cap_x = 0.10

    y1 = bottom - 0.18
    ax.add_patch(plt.Rectangle((cap_x, y1 - sw_h), sw_w, sw_h, facecolor=HIGHLIGHT_BG, edgecolor="#81c784", linewidth=0.8))
    ax.text(cap_x + sw_w + 0.08, y1 - sw_h / 2, "Green: Aligned student score > XGBoost score on full dataset", ha="left", va="center", fontsize=8.5, color="#2c3e50")

    y2 = bottom - 0.36
    ax.add_patch(plt.Rectangle((cap_x, y2 - sw_h), sw_w, sw_h, facecolor=RED_BG, edgecolor="#e57373", linewidth=0.8))
    ax.text(cap_x + sw_w + 0.08, y2 - sw_h / 2, "Red: Value < N (aligned student score < baseline student score at size N)", ha="left", va="center", fontsize=8.5, color="#2c3e50")

    y3 = bottom - 0.54
    ax.add_patch(plt.Rectangle((cap_x, y3 - sw_h), sw_w, sw_h, facecolor=GRAY_BG, edgecolor="#95a5a6", linewidth=0.8))
    ax.text(cap_x + sw_w + 0.08, y3 - sw_h / 2, "Gray: Baseline score >= Teacher score (omitted from Mean Gain and Average row)", ha="left", va="center", fontsize=8.5, color="#2c3e50")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_gap_closed_curve(output_path, student_ns, gap_closed_matrix, valid_mask, model_name="tabpfn", layer=23):
    """Plot Mean Relative Gap Closed (%) vs. Student Fraction N."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    n_datasets = gap_closed_matrix.shape[0]

    ax.axhline(0, linestyle="--", color="#95a5a6", linewidth=1.5, label="Baseline Student (0%)")
    ax.axhline(100, linestyle=":", color="#27ae60", linewidth=1.5, label="Teacher Performance (100%)")

    mean_gcs, sem_gcs, valid_sns = [], [], []
    for si, sn in enumerate(student_ns):
        vals = [gap_closed_matrix[di, si] for di in range(n_datasets) if valid_mask[di, si] and not np.isnan(gap_closed_matrix[di, si])]
        if vals:
            valid_sns.append(sn)
            mean_gcs.append(float(np.mean(vals)))
            sem_gcs.append(float(stats.sem(vals)) if len(vals) > 1 else 0.0)

    if valid_sns:
        mean_arr = np.array(mean_gcs)
        sem_arr = np.array(sem_gcs)
        ax.fill_between(valid_sns, mean_arr - sem_arr, mean_arr + sem_arr, color="#e67e22", alpha=0.20, label="Mean $\\pm 1$ SEM")
        ax.plot(valid_sns, mean_arr, marker="s", markersize=7, linewidth=2.5, color="#d35400", label="Mean Gap Closed (%)", zorder=4)

        for sn, gc in zip(valid_sns, mean_arr):
            ax.annotate(
                f"{gc:.1f}%", xy=(sn, gc), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8.5,
                fontweight="bold", color="#b9770e",
            )

    ax.set_xlabel("Aligned Student Fraction ($N$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Relative Gap Closed (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Distillation Efficiency: Relative Gap Closed (%)\nModel: {model_name.upper()} | Layer: {layer}",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xticks(student_ns)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_win_rate_bar_chart(output_path, student_ns, aligned_matrix, baseline_matrix, valid_mask, model_name="tabpfn", layer=23):
    """Plot Pairwise Win Rate (%) vs. Baseline at each student fraction N."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    n_datasets = aligned_matrix.shape[0]

    win_rates, p_vals, valid_sns = [], [], []

    for si, sn in enumerate(student_ns):
        valid_indices = [di for di in range(n_datasets) if valid_mask[di, si]]
        if not valid_indices:
            continue
        a_vals = aligned_matrix[valid_indices, si]
        b_vals = baseline_matrix[valid_indices, si]
        wins = int(np.sum(a_vals > b_vals))
        total = len(valid_indices)
        win_rates.append(wins / total * 100.0)
        valid_sns.append(sn)

        if total >= 5 and np.any(a_vals != b_vals):
            try:
                res = stats.wilcoxon(a_vals, b_vals, alternative="two-sided")
                p_vals.append(res.pvalue)
            except Exception:
                p_vals.append(1.0)
        else:
            p_vals.append(1.0)

    x = np.arange(len(valid_sns))
    bars = ax.bar(x, win_rates, width=0.55, color="#2980b9", edgecolor="#1b4f72", linewidth=1.2, zorder=3)
    ax.axhline(50, linestyle="--", color="#c0392b", linewidth=1.8, label="50% Parity (Random)", zorder=2)

    for bar, wr, p in zip(bars, win_rates, p_vals):
        sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        label = f"{wr:.1f}%\n{sig_str}" if sig_str else f"{wr:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            label, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1a5276",
        )

    ax.set_xlabel("Aligned Student Fraction ($N$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Win Rate vs. Baseline (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Pairwise Win Rate vs. Baseline Student\nModel: {model_name.upper()} | Layer: {layer} (*p<0.05, **p<0.01, ***p<0.001)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sn:.2g}" for sn in valid_sns])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", alpha=0.55, axis="y", zorder=1)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def stack_plots_vertically(image_paths, output_path, padding=40, bg_color=(255, 255, 255)):
    """Stack multiple plot images vertically into a single consolidated image."""
    images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
    if not images:
        return None

    max_w = max(img.width for img in images)
    canvas_w = max_w + 2 * padding
    total_h = sum(img.height for img in images) + padding * (len(images) + 1)

    combined = Image.new("RGB", (canvas_w, total_h), bg_color)
    curr_y = padding
    for img in images:
        curr_x = (canvas_w - img.width) // 2
        combined.paste(img, (curr_x, curr_y))
        curr_y += img.height + padding

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.save(output_path, quality=95)
    return output_path


# --------------------------------------------------------------------------- #
# 4. Master Orchestrator
# --------------------------------------------------------------------------- #

def plot(
    dataset,
    student_n,
    layer,
    n_estimators,
    n_samples,
    patience,
    lr,
    batch_size,
    hidden_layers,
    output,
    n_repeats,
    max_epochs,
    model,
    xgboost_opt,
    aligner_opt,
):
    """Generate all benchmark evaluation plots and tables.

    Returns:
        Dict mapping artifact names to their file paths, or None if no data.
    """
    if isinstance(dataset, str):
        dataset = [dataset]

    args_obj = argparse.Namespace(
        dataset=dataset,
        student_n=student_n,
        layer=layer,
        n_estimators=n_estimators,
        n_samples=n_samples,
        patience=patience,
        lr=lr,
        batch_size=batch_size,
        hidden_layers=hidden_layers,
        output=output,
        n_repeats=n_repeats,
        max_epochs=max_epochs,
        model=model,
        xgboost_opt=xgboost_opt,
        aligner_opt=aligner_opt,
    )

    data = load_benchmark_data(args_obj, dataset, sorted(student_n), layer)
    if data is None:
        return None

    clean_args = {k: v for k, v in vars(args_obj).items() if k != "output"}
    clean_args["dataset"] = sorted(clean_args["dataset"]) if isinstance(clean_args.get("dataset"), list) else clean_args.get("dataset")
    clean_args["student_n"] = sorted(clean_args["student_n"]) if isinstance(clean_args.get("student_n"), list) else clean_args.get("student_n")

    if output is None:
        cache_name = pruning_utils.create_filename_from_args(
            clean_args,
            output_dir_arg_name="output",
            script_name="plot_aligned_benchmarks",
        )
        out_dir = os.path.join(OUTPUT_DIR, cache_name)
    elif output.endswith(".png"):
        out_dir = os.path.dirname(output)
    else:
        out_dir = output

    os.makedirs(out_dir, exist_ok=True)

    curve_path = os.path.join(out_dir, "sample_efficiency_curve.png")
    summary_path = os.path.join(out_dir, "summary_metrics_table.png")
    detail_table_path = os.path.join(out_dir, "sample_efficiency_table.png")
    legacy_table_path = os.path.join(out_dir, "table.png")
    gap_closed_path = os.path.join(out_dir, "gap_closed_curve.png")
    win_rate_path = os.path.join(out_dir, "win_rate_bar_chart.png")
    dashboard_path = os.path.join(out_dir, "benchmarks.png")

    # Generate all plots
    plot_sample_efficiency_curve(
        curve_path, sorted(student_n), data["m_matrix"], data["valid_mask"],
        model_name=model, layer=layer,
    )

    plot_summary_metrics_table(
        summary_path, sorted(student_n), data["aligned_matrix"], data["baseline_matrix"],
        data["teacher_values"], data["xgb_values"], data["m_matrix"],
        data["valid_mask"], data["gap_closed_matrix"],
    )

    plot_sample_efficiency_table(
        detail_table_path, data["present_datasets"], sorted(student_n),
        data["m_matrix"], data["aligned_matrix"], data["baseline_matrix"],
        data["teacher_values"], data["xgb_values"],
    )
    if os.path.exists(detail_table_path) and detail_table_path != legacy_table_path:
        shutil.copyfile(detail_table_path, legacy_table_path)

    plot_gap_closed_curve(
        gap_closed_path, sorted(student_n), data["gap_closed_matrix"],
        data["valid_mask"], model_name=model, layer=layer,
    )

    plot_win_rate_bar_chart(
        win_rate_path, sorted(student_n), data["aligned_matrix"],
        data["baseline_matrix"], data["valid_mask"],
        model_name=model, layer=layer,
    )

    # Consolidate into a single scrollable dashboard image
    stack_plots_vertically(
        [
            summary_path,
            curve_path,
            gap_closed_path,
            win_rate_path,
            detail_table_path,
        ],
        dashboard_path,
    )

    return {
        "Benchmark Dashboard": dashboard_path,
        "Sample Efficiency Curve": curve_path,
        "Summary Metrics Table": summary_path,
        "Detailed Dataset Table": detail_table_path,
        "Legacy Table": legacy_table_path,
        "Gap Closed Curve": gap_closed_path,
        "Win Rate Bar Chart": win_rate_path,
    }
