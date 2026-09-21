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
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from scipy import stats

import pruning_utils
from cache_utils import OUTPUT_DIR
from consts import (
    TABFM_DEFAULT_LR,
    TABPFN_DEFAULT_LR,
    TABFM_DEFAULT_N_ESTIMATORS,
    TABPFN_DEFAULT_N_ESTIMATORS,
    DEFAULT_N_SAMPLES,
)
from evaluate_aligned_student import evaluate_aligned_student
from train_xgboost import train_xgboost
from train_xgboost_opt import train_xgboost_opt

# --------------------------------------------------------------------------- #
# 0. Publication Palette & Style
# --------------------------------------------------------------------------- #

# Imperial Violet & Warm Gold, plus neutral slates/charcoal.
# `*_soft` are soft tints used for secondary series, `*_dark` for accents.
PALETTE = {
    "tabpfn": "#592D86",        # Imperial Violet
    "tabpfn_soft": "#B49DCB",   # Soft Lavender Tint
    "tabpfn_dark": "#3B1861",
    "tabfm": "#D1A11F",         # Warm Gold / Yellow-Ochre
    "tabfm_soft": "#ECD38D",    # Soft Gold Tint
    "tabfm_dark": "#8C6A0C",
    "neutral": "#C8CED4",       # light silver (unaligned baseline)
    "xgboost": "#203A6A",       # midnight indigo (full-data XGBoost)
    "ink": "#26282B",           # primary text / axes
    "muted": "#7A828A",         # secondary text, annotations
    "reference": "#AEB6BD",     # parity / chance guide lines
    "grid": "#E7E9EB",
}


def _apply_paper_style():
    """Apply publication-quality matplotlib style.

    Minimal chrome: hairline axes, light solid horizontal grids, frameless
    legends, muted ink, and no heavy bold titles.
    """
    import matplotlib as mpl
    style = {
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'mathtext.fontset': 'cm',
        'text.color': PALETTE['ink'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'axes.edgecolor': PALETTE['muted'],
        'axes.labelcolor': PALETTE['ink'],
        'axes.labelsize': 9.5,
        'axes.titlesize': 10,
        'axes.titleweight': 'normal',
        'axes.titlelocation': 'left',
        'axes.titlepad': 8,
        'axes.titlecolor': PALETTE['ink'],
        'axes.axisbelow': True,
        'grid.color': PALETTE['grid'],
        'grid.linestyle': '-',
        'grid.linewidth': 0.6,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'xtick.color': PALETTE['muted'],
        'ytick.color': PALETTE['muted'],
        'xtick.labelcolor': PALETTE['ink'],
        'ytick.labelcolor': PALETTE['ink'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'lines.linewidth': 1.7,
        'lines.markersize': 4.2,
        'patch.linewidth': 0.0,
        'legend.frameon': False,
        'legend.fontsize': 8.5,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.6,
        'legend.labelspacing': 0.4,
        'legend.columnspacing': 1.3,
        'legend.borderpad': 0.0,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
    }
    mpl.rcParams.update(style)


MODEL_CONFIGS = {
    "tabpfn": {
        "lr": TABPFN_DEFAULT_LR,
        "n_estimators": TABPFN_DEFAULT_N_ESTIMATORS,
        "name": "TabPFN v3",
        "tex_name": r"\textbf{TabPFN~v3}",
        "short": "PFN",
        "color": PALETTE["tabpfn"],
        "soft_color": PALETTE["tabpfn_soft"],
        "edge_color": PALETTE["tabpfn_dark"],
        "marker": "o",
        "description": r"24 blocks, 8 estimators, $\eta = 10^{-3}$",
    },
    "tabfm": {
        "lr": TABFM_DEFAULT_LR,
        "n_estimators": TABFM_DEFAULT_N_ESTIMATORS,
        "name": "TabFM v1.0",
        "tex_name": r"\textbf{TabFM~v1.0}",
        "short": "FM",
        "color": PALETTE["tabfm"],
        "soft_color": PALETTE["tabfm_soft"],
        "edge_color": PALETTE["tabfm_dark"],
        "marker": "s",
        "description": r"24 layers, 1 estimator, $\eta = 10^{-4}$",
    },
}


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
    Caps M to valid physical boundaries [0.0, 1.0].
    """
    if aligned_score is None or np.isnan(aligned_score):
        return np.nan

    scores_dict = dict(baseline_scores)
    grid = list(student_ns)
    if teacher_score is not None and not np.isnan(teacher_score) and 1.0 not in scores_dict:
        valid_b = [v for v in scores_dict.values() if v is not None and not np.isnan(v)]
        max_b = max(valid_b) if valid_b else -np.inf
        if teacher_score >= max_b:
            scores_dict[1.0] = teacher_score
            grid.append(1.0)

    valid_student_ns = [
        sn for sn in sorted(set(grid))
        if sn in scores_dict and scores_dict[sn] is not None and not np.isnan(scores_dict[sn])
    ]
    if not valid_student_ns:
        return float(np.clip(min(student_ns) if student_ns else 0.0, 0.0, 1.0))

    # Extrapolate below minimum baseline
    if aligned_score <= scores_dict[valid_student_ns[0]]:
        m0 = valid_student_ns[0]
        if m0 <= 0.0:
            return 0.0
        if len(valid_student_ns) >= 2:
            m1 = valid_student_ns[1]
            b0, b1 = scores_dict[m0], scores_dict[m1]
            if b1 > b0:
                raw_m = m0 + (aligned_score - b0) / (b1 - b0) * (m1 - m0)
                return float(np.clip(raw_m, 0.0, 1.0))
        return float(np.clip(m0, 0.0, 1.0))

    # Extrapolate above maximum baseline / teacher
    if aligned_score >= scores_dict[valid_student_ns[-1]]:
        m_last = valid_student_ns[-1]
        if m_last >= 1.0:
            return 1.0
        if len(valid_student_ns) >= 2:
            m_prev = valid_student_ns[-2]
            b_last, b_prev = scores_dict[m_last], scores_dict[m_prev]
            if b_last > b_prev:
                raw_m = m_last + (aligned_score - b_last) / (b_last - b_prev) * (m_last - m_prev)
                return float(np.clip(raw_m, 0.0, 1.0))
        return float(np.clip(m_last, 0.0, 1.0))

    # Interpolate within interval [M_i, M_{i+1}]
    for i in range(len(valid_student_ns) - 1, -1, -1):
        m_i = valid_student_ns[i]
        b_i = scores_dict[m_i]
        if b_i <= aligned_score:
            if i == len(valid_student_ns) - 1:
                return float(np.clip(m_i, 0.0, 1.0))
            m_next = valid_student_ns[i + 1]
            b_next = scores_dict[m_next]
            if b_next <= b_i:
                return float(np.clip(m_i, 0.0, 1.0))
            alpha = (aligned_score - b_i) / (b_next - b_i)
            alpha = max(0.0, min(1.0, alpha))
            raw_m = m_i + alpha * (m_next - m_i)
            return float(np.clip(raw_m, 0.0, 1.0))

    return float(np.clip(valid_student_ns[0], 0.0, 1.0))


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

    ranks_a, ranks_b, ranks_xgb = [], [], []
    for i in range(total):
        if not np.isnan(xgb_vals[i]) and not np.isnan(a_vals[i]) and not np.isnan(b_vals[i]):
            r = stats.rankdata([-a_vals[i], -b_vals[i], -xgb_vals[i]])
            ranks_a.append(float(r[0]))
            ranks_b.append(float(r[1]))
            ranks_xgb.append(float(r[2]))

    if ranks_a:
        mean_rank_a = float(np.mean(ranks_a))
        mean_rank_b = float(np.mean(ranks_b))
        mean_rank_xgb = float(np.mean(ranks_xgb))
        avg_rank_str = f"{mean_rank_a:.2f} / {mean_rank_b:.2f} / {mean_rank_xgb:.2f}"
    else:
        avg_rank_str = "–"

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
        "avg_rank_str": avg_rank_str,
        "valid_ranks_a": ranks_a,
        "valid_ranks_b": ranks_b,
        "valid_ranks_xgb": ranks_xgb,
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_combined_curves(
    output_path,
    student_ns,
    m_matrix,
    gap_closed_matrix,
    valid_mask,
    model_name="tabpfn",
    layer=23,
):
    """Plot side-by-side 2-panel figure: (a) Sample Efficiency M vs N and (b) Relative Gap Closed % vs N."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=150)
    n_datasets = m_matrix.shape[0]

    # --- Panel (a): Sample Efficiency ---
    min_val, max_val = min(student_ns), max(student_ns)
    ax1.plot(
        [min_val, max_val], [min_val, max_val],
        linestyle="--", color="#7f8c8d", linewidth=2.0, alpha=0.85,
        label="Parity ($M = N$)", zorder=2,
    )

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, n_datasets)))
    for di in range(n_datasets):
        ds_ns, ds_ms = [], []
        for si, sn in enumerate(student_ns):
            if valid_mask[di, si] and not np.isnan(m_matrix[di, si]):
                ds_ns.append(sn)
                ds_ms.append(m_matrix[di, si])
        if len(ds_ns) > 1:
            ax1.plot(ds_ns, ds_ms, color=colors[di % len(colors)], alpha=0.20, linewidth=1.0, zorder=3)

    avg_m_vals, sem_m_vals, valid_sns_m = [], [], []
    for si, sn in enumerate(student_ns):
        vals = [m_matrix[di, si] for di in range(n_datasets) if valid_mask[di, si] and not np.isnan(m_matrix[di, si])]
        if vals:
            valid_sns_m.append(sn)
            avg_m_vals.append(float(np.mean(vals)))
            sem_m_vals.append(float(stats.sem(vals)) if len(vals) > 1 else 0.0)

    if valid_sns_m:
        avg_m_arr = np.array(avg_m_vals)
        sem_m_arr = np.array(sem_m_vals)
        ax1.fill_between(
            valid_sns_m, avg_m_arr - sem_m_arr, avg_m_arr + sem_m_arr,
            color="#2980b9", alpha=0.20, label="Mean $\\pm 1$ SEM", zorder=4,
        )
        ax1.plot(
            valid_sns_m, avg_m_arr,
            marker="o", markersize=7, linewidth=2.8, color="#1b4f72",
            label="Aligned Student", zorder=5,
        )
        for sn, m_val in zip(valid_sns_m, avg_m_arr):
            gain = m_val - sn
            if gain > 0:
                ax1.annotate(
                    f"+{gain:.2f}", xy=(sn, m_val), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    fontweight="bold", color="#145a32",
                )

    ax1.set_xlabel("Aligned Student Fraction ($N$)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Effective Baseline Fraction ($M$)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_title("(a) Sample Efficiency: Effective Baseline $M$ vs. Budget $N$", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xticks(student_ns)
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax1.grid(True, linestyle=":", alpha=0.55, zorder=1)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=9.0)

    # --- Panel (b): Relative Gap Closed ---
    ax2.axhline(0, linestyle="--", color="#95a5a6", linewidth=1.5, label="Baseline Student (0%)")
    ax2.axhline(100, linestyle=":", color="#27ae60", linewidth=1.5, label="Teacher Performance (100%)")

    median_gcs, q25_gcs, q75_gcs, valid_sns_gc = [], [], [], []
    for si, sn in enumerate(student_ns):
        vals = [
            gap_closed_matrix[di, si]
            for di in range(n_datasets)
            if valid_mask[di, si] and not np.isnan(gap_closed_matrix[di, si])
        ]
        if vals:
            valid_sns_gc.append(sn)
            median_gcs.append(float(np.median(vals)))
            q25_gcs.append(float(np.percentile(vals, 25)))
            q75_gcs.append(float(np.percentile(vals, 75)))

    if valid_sns_gc:
        med_arr = np.array(median_gcs)
        q25_arr = np.array(q25_gcs)
        q75_arr = np.array(q75_gcs)
        ax2.fill_between(valid_sns_gc, q25_arr, q75_arr, color="#e67e22", alpha=0.20, label="IQR (25th–75th %ile)")
        ax2.plot(valid_sns_gc, med_arr, marker="s", markersize=7, linewidth=2.5, color="#d35400", label="Median Gap Closed (%)", zorder=4)

        for sn, gc in zip(valid_sns_gc, med_arr):
            ax2.annotate(
                f"{gc:.1f}%", xy=(sn, gc), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8.5,
                fontweight="bold", color="#b9770e",
            )

    ax2.set_xlabel("Aligned Student Fraction ($N$)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Relative Gap Closed (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_title("(b) Distillation Efficiency: Relative Gap Closed (%)", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xticks(student_ns)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g"))
    ax2.grid(True, linestyle=":", alpha=0.55)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.0)

    fig.suptitle(
        f"Aligned Student Scaling & Distillation Curves (Model: {model_name.upper()} | Layer: {layer})",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
        "Avg Rank (A/B/XGB)",
        "Mean Gap Closed",
        "Median Gap Closed",
        "Effective M",
        "Mean Gain (M-N)",
    ]

    table_data = []
    n_datasets = aligned_matrix.shape[0]

    pooled_a, pooled_b, pooled_gc, pooled_gains = [], [], [], []
    pooled_ranks_a, pooled_ranks_b, pooled_ranks_xgb = [], [], []
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
            m_dict["avg_rank_str"],
            f"{m_dict['mean_gc']:+.1f}%" if not np.isnan(m_dict["mean_gc"]) else "–",
            f"{m_dict['median_gc']:+.1f}%" if not np.isnan(m_dict["median_gc"]) else "–",
            f"{m_dict['mean_m']:.3f}" if not np.isnan(m_dict["mean_m"]) else "–",
            f"{'+' if m_dict['mean_gain'] > 0 else ''}{m_dict['mean_gain']:.3f}" if not np.isnan(m_dict["mean_gain"]) else "–",
        ])

        pooled_a.extend(a_slice)
        pooled_b.extend(b_slice)
        pooled_gc.extend(m_dict["valid_gc"])
        pooled_gains.extend([m - sn for m in m_dict["valid_m"]])
        pooled_ranks_a.extend(m_dict["valid_ranks_a"])
        pooled_ranks_b.extend(m_dict["valid_ranks_b"])
        pooled_ranks_xgb.extend(m_dict["valid_ranks_xgb"])
        total_beats_xgb += m_dict["beats_xgb_cnt"]
        total_xgb += m_dict["beats_xgb_total"]

    # Overall Summary Row
    if pooled_a:
        pa, pb = np.array(pooled_a), np.array(pooled_b)
        o_wins = int(np.sum(pa > pb))
        o_tot = len(pa)
        o_wr = o_wins / o_tot * 100.0

        dataset_mean_diffs = []
        for di in range(n_datasets):
            diffs_d = [
                aligned_matrix[di, si] - baseline_matrix[di, si]
                for si in range(len(student_ns))
                if valid_mask[di, si] and not np.isnan(aligned_matrix[di, si]) and not np.isnan(baseline_matrix[di, si])
            ]
            if diffs_d:
                dataset_mean_diffs.append(float(np.mean(diffs_d)))

        diff_arr = np.array(dataset_mean_diffs)
        if len(diff_arr) >= 5 and np.any(diff_arr != 0):
            try:
                o_res = stats.wilcoxon(diff_arr, alternative="two-sided")
                o_pval = f"p={o_res.pvalue:.4f}" if o_res.pvalue >= 0.0001 else "p<0.0001"
            except Exception:
                o_pval = "n/a"
        else:
            o_pval = "n/a"

        o_mgc = float(np.mean(pooled_gc)) if pooled_gc else np.nan
        o_medgc = float(np.median(pooled_gc)) if pooled_gc else np.nan
        o_gain = float(np.mean(pooled_gains)) if pooled_gains else np.nan
        o_xgb = f"{total_beats_xgb}/{total_xgb} ({total_beats_xgb/total_xgb*100:.1f}%)" if total_xgb > 0 else "–"

        if pooled_ranks_a:
            o_rank_a = float(np.mean(pooled_ranks_a))
            o_rank_b = float(np.mean(pooled_ranks_b))
            o_rank_xgb = float(np.mean(pooled_ranks_xgb))
            o_rank_str = f"{o_rank_a:.2f} / {o_rank_b:.2f} / {o_rank_xgb:.2f}"
        else:
            o_rank_str = "–"

        table_data.append([
            "Overall",
            o_rank_str,
            f"{o_mgc:+.1f}%" if not np.isnan(o_mgc) else "–",
            f"{o_medgc:+.1f}%" if not np.isnan(o_medgc) else "–",
            "—",
            f"{'+' if o_gain > 0 else ''}{o_gain:.3f}" if not np.isnan(o_gain) else "–",
        ])

    if not table_data:
        table_data = [["No valid evaluations"] + ["–"] * (len(headers) - 1)]

    n_rows = len(table_data)
    col_widths = [1.35, 1.65, 1.25, 1.25, 1.05, 1.20]
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
        "Aligned Student Benchmark Summary (Win Rates, Average Ranks & Distillation Metrics)\n"
        "[A = Aligned Student, B = Baseline Student, XGB = Full XGBoost (100% Training Budget)]",
        fontsize=11.0, fontweight="bold", pad=16,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
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
    valid_mask=None,
):
    """Render full per-dataset breakdown table showing effective M for each dataset and student N."""
    n_datasets = len(present_datasets)
    n_students = len(student_ns)

    if valid_mask is None:
        valid_mask = np.zeros((n_datasets, n_students), dtype=bool)
        for di in range(n_datasets):
            for si in range(n_students):
                b_val = baseline_matrix[di, si]
                t_val = teacher_values[di]
                a_val = aligned_matrix[di, si]
                if not np.isnan(a_val) and not np.isnan(b_val) and not np.isnan(t_val) and b_val < t_val:
                    valid_mask[di, si] = True

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
            elif not valid_mask[di, si] or np.isnan(m_val):
                row_text.append("–")
                row_bgs.append(None)
            else:
                row_text.append(f"{m_val:.2f}")
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
            row_text.append(f"{prefix}{mean_gain:.2f}")
        else:
            row_text.append("–")
        row_bgs.append(None)

        cell_text.append(row_text)
        cell_bg_colors.append(row_bgs)

    # Average row calculation
    avg_m_vals = []
    for si, sn in enumerate(student_ns):
        valid_ms = [
            m_matrix[di, si] for di in range(n_datasets)
            if valid_mask[di, si] and not np.isnan(m_matrix[di, si])
        ]
        avg_m_vals.append(float(np.mean(valid_ms)) if valid_ms else np.nan)

    avg_row_text = ["Average"]
    avg_row_bgs = [None]
    avg_gains = []

    for si, sn in enumerate(student_ns):
        v = avg_m_vals[si]
        paired_indices = [
            di for di in range(n_datasets)
            if valid_mask[di, si]
            and not np.isnan(aligned_matrix[di, si])
            and not np.isnan(xgb_values[di])
        ]
        if paired_indices:
            avg_aligned = float(np.mean([aligned_matrix[di, si] for di in paired_indices]))
            col_xgb_avg = float(np.mean([xgb_values[di] for di in paired_indices]))
        else:
            avg_aligned = np.nan
            col_xgb_avg = np.nan

        if np.isnan(v):
            avg_row_text.append("–")
            avg_row_bgs.append(None)
        else:
            avg_row_text.append(f"{v:.2f}")
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
        avg_row_text.append(f"{prefix}{mean_avg_gain:.2f}")
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
    """Plot Median Relative Gap Closed (%) vs. Student Fraction N with IQR shading."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    n_datasets = gap_closed_matrix.shape[0]

    ax.axhline(0, linestyle="--", color="#95a5a6", linewidth=1.5, label="Baseline Student (0%)")
    ax.axhline(100, linestyle=":", color="#27ae60", linewidth=1.5, label="Teacher Performance (100%)")

    median_gcs, q25_gcs, q75_gcs, valid_sns = [], [], [], []
    for si, sn in enumerate(student_ns):
        vals = [
            gap_closed_matrix[di, si]
            for di in range(n_datasets)
            if valid_mask[di, si] and not np.isnan(gap_closed_matrix[di, si])
        ]
        if vals:
            valid_sns.append(sn)
            median_gcs.append(float(np.median(vals)))
            q25_gcs.append(float(np.percentile(vals, 25)))
            q75_gcs.append(float(np.percentile(vals, 75)))

    if valid_sns:
        med_arr = np.array(median_gcs)
        q25_arr = np.array(q25_gcs)
        q75_arr = np.array(q75_gcs)
        ax.fill_between(valid_sns, q25_arr, q75_arr, color="#e67e22", alpha=0.20, label="IQR (25th–75th %ile)")
        ax.plot(valid_sns, med_arr, marker="s", markersize=7, linewidth=2.5, color="#d35400", label="Median Gap Closed (%)", zorder=4)

        for sn, gc in zip(valid_sns, med_arr):
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
        sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(
            bar.get_x() + bar.get_width() / 2, min(wr, 100.0) + 1.0,
            f"({sig_str})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1a5276",
            clip_on=False,
        )

    ax.set_xlabel("Aligned Student Fraction ($N$)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Win Rate vs. Baseline (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"Pairwise Win Rate vs. Baseline Student\nModel: {model_name.upper()} | Layer: {layer} (*p<0.05, **p<0.01, ***p<0.001)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sn:.2g}" for sn in valid_sns])
    ax.set_ylim(0, 100)
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.save(output_path, quality=95)
    return output_path


# --------------------------------------------------------------------------- #
# 4. Multi-Model Combined Visualizations & Tables
# --------------------------------------------------------------------------- #

def plot_multi_model_scaling_curves(
    output_path,
    models_data,
    student_ns,
    layer=23,
):
    """Plot side-by-side 2-panel figure combining multiple models:
    (a) Sample Efficiency M vs alpha with SEM bands for all models, and
    (b) Distillation Efficiency (Median Relative Gap Closed %) vs alpha for all models.
    """
    _apply_paper_style()
    x_max = max(student_ns) if max(student_ns) <= 0.55 else 1.0
    x_step = 0.1 if x_max <= 0.55 else 0.2

    def _percentile_curve(matrix, mask):
        """Median / IQR of `matrix` across datasets, per context fraction."""
        n_ds = matrix.shape[0]
        xs, med, q25, q75 = [], [], [], []
        for si, sn in enumerate(student_ns):
            vals = [matrix[di, si] for di in range(n_ds) if mask[di, si] and not np.isnan(matrix[di, si])]
            if vals:
                xs.append(sn)
                med.append(float(np.median(vals)))
                q25.append(float(np.percentile(vals, 25)))
                q75.append(float(np.percentile(vals, 75)))
        return xs, np.array(med), np.array(q25), np.array(q75)

    curves = {}
    for m_key, data in models_data.items():
        if data is None:
            continue
        cfg = MODEL_CONFIGS.get(m_key, {
            "name": m_key.upper(), "color": PALETTE["tabpfn"], "marker": "o", "short": m_key
        })
        curves[m_key] = {
            "label": re.sub(r"\s+v\d+(\.\d+)*", "", cfg["name"]),
            "color": cfg["color"],
            "marker": cfg.get("marker", "o"),
            "m": _percentile_curve(data["m_matrix"], data["valid_mask"]),
            "gc": _percentile_curve(data["gap_closed_matrix"], data["valid_mask"]),
        }

    # Full [0, 1] range with equal data scaling: the panel is 2:1 for x_max=0.5,
    # which makes the parity line render at exactly 45 degrees.
    y_max = 1.0
    aspect = y_max / x_max

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 4.4), dpi=300)
    ax1.set_box_aspect(aspect)
    ax2.set_box_aspect(aspect)

    # --- Panel (a): Sample Efficiency ---
    parity_line, = ax1.plot(
        [0, x_max], [0, x_max],
        linestyle=(0, (5, 4)), color=PALETTE["reference"], linewidth=0.9, zorder=2,
    )

    def _draw_band(ax, xs, q25, q75, color):
        """Soft IQR band with hairline boundaries (readable where bands overlap)."""
        ax.fill_between(xs, q25, q75, color=color, alpha=0.10, linewidth=0, zorder=3)
        ax.plot(xs, q25, color=color, alpha=0.45, linewidth=0.5, zorder=3)
        ax.plot(xs, q75, color=color, alpha=0.45, linewidth=0.5, zorder=3)

    for c in curves.values():
        xs, med, q25, q75 = c["m"]
        if not xs:
            continue
        _draw_band(ax1, xs, q25, q75, c["color"])
        ax1.plot(
            xs, med, marker=c["marker"], color=c["color"], label=c["label"],
            markeredgecolor="white", markeredgewidth=0.7, zorder=5,
        )

    ax1.set_xlabel(r"Context fraction ($\alpha$)", labelpad=5)
    ax1.set_ylabel(r"Effective baseline fraction ($M_\alpha$)", labelpad=5)
    ax1.set_title(r"\textbf{(a)} Sample efficiency")
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, y_max)
    ax1.set_xticks(np.arange(0, x_max + 1e-5, x_step))
    ax1.set_yticks(np.arange(0, y_max + 1e-5, 0.2 if y_max > 0.5 else 0.1))
    ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax1.grid(True, axis="y", zorder=0)
    ax1.legend(loc="upper left", borderaxespad=0.2)

    # --- Panel (b): Distillation Efficiency ---
    for c in curves.values():
        xs, med, q25, q75 = c["gc"]
        if not xs:
            continue
        _draw_band(ax2, xs, q25, q75, c["color"])
        ax2.plot(
            xs, med, marker=c["marker"], color=c["color"], label=c["label"],
            markeredgecolor="white", markeredgewidth=0.7, zorder=5,
        )

    # Teacher reference: 100% of the baseline-to-teacher gap recovered
    ax2.axhline(100, linestyle=(0, (5, 4)), color=PALETTE["reference"], linewidth=0.9, zorder=2)
    ax2.annotate(
        "teacher", xy=(0.02 * x_max, 100), xytext=(0, -3),
        textcoords="offset points", ha="left", va="top",
        fontsize=7.0, color=PALETTE["muted"],
    )

    ax2.set_xlabel(r"Context fraction ($\alpha$)", labelpad=5)
    ax2.set_ylabel(r"Relative gap closed (\%)", labelpad=5)
    ax2.set_title(r"\textbf{(b)} Distillation efficiency")
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(np.arange(0, x_max + 1e-5, x_step))
    ax2.set_yticks(np.arange(0, 101, 25))
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.grid(True, axis="y", zorder=0)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.08)

    # The 2:1 aspect ratio ensures the diagonal parity reference line is at 45 degrees.
    fig.canvas.draw()
    parity_angle = 45.0
    ax1.annotate(
        r"parity ($M_\alpha = \alpha$)",
        xy=(0.62 * x_max, 0.62 * x_max), xytext=(4, -3),
        textcoords="offset points", rotation=parity_angle, rotation_mode="anchor",
        ha="left", va="top", fontsize=7.0, color=PALETTE["muted"],
    )
    parity_line.set_zorder(4)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path



def plot_multi_model_win_rate_bar_chart(
    output_path,
    models_data,
    student_ns,
    layer=23,
):
    """Plot grouped bar chart comparing win rates of multiple models side-by-side:
    both vs. Baseline Student and vs. Full-Data XGBoost (4 bars per alpha for 2 models).
    """
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.0), dpi=300)
    active_models = [m for m in ["tabpfn", "tabfm"] if m in models_data and models_data[m] is not None]
    if not active_models:
        active_models = [m for m in models_data if models_data[m] is not None]
    if not active_models:
        plt.close(fig)
        return None

    # Series to plot: for each model, (vs_base, vs_xgb).
    # Comparator is encoded by tint (solid = baseline student, light = XGBoost),
    # so the chart stays readable without hatching.
    series_list = []
    for m_key in active_models:
        data = models_data[m_key]
        cfg = MODEL_CONFIGS.get(m_key, {
            "name": m_key.upper(),
            "color": PALETTE["tabpfn"],
            "soft_color": PALETTE["tabpfn_soft"],
            "short": m_key,
        })
        model_display_name = re.sub(r"\s+v\d+(\.\d+)*", "", cfg["name"])
        # Series 1: Aligned vs Baseline
        series_list.append({
            "model_key": m_key,
            "target": "base",
            "label_prefix": f"{model_display_name} vs.\\ baseline",
            "color": cfg["color"],
        })
        # Series 2: Aligned vs XGBoost (same hue, lighter tint)
        series_list.append({
            "model_key": m_key,
            "target": "xgb",
            "label_prefix": f"{model_display_name} vs.\\ XGBoost",
            "color": cfg.get("soft_color", PALETTE["neutral"]),
        })

    n_series = len(series_list)
    x = np.arange(len(student_ns))
    total_width = 0.78
    width = total_width / n_series

    for si_idx, s_info in enumerate(series_list):
        m_key = s_info["model_key"]
        target = s_info["target"]
        data = models_data[m_key]
        a_mat = data["aligned_matrix"]
        b_mat = data["baseline_matrix"]
        xgb_vals = data.get("xgb_values", np.full(a_mat.shape[0], np.nan))
        v_mask = data["valid_mask"]
        n_ds = a_mat.shape[0]

        win_rates, sig_strs, slice_wins, slice_totals = [], [], [], []
        for si, sn in enumerate(student_ns):
            if target == "base":
                valid_indices = [di for di in range(n_ds) if v_mask[di, si]]
            else:
                valid_indices = [di for di in range(n_ds) if v_mask[di, si] and not np.isnan(xgb_vals[di])]

            if not valid_indices:
                win_rates.append(0.0)
                sig_strs.append("")
                continue

            a_vals = a_mat[valid_indices, si]
            if target == "base":
                comp_vals = b_mat[valid_indices, si]
            else:
                comp_vals = xgb_vals[valid_indices]

            wins = int(np.sum(a_vals > comp_vals))
            tot = len(valid_indices)
            slice_wins.append(wins)
            slice_totals.append(tot)
            wr = (wins / tot) * 100.0
            win_rates.append(wr)

            if tot >= 5 and np.any(a_vals != comp_vals):
                try:
                    res = stats.wilcoxon(a_vals, comp_vals, alternative="two-sided")
                    p = res.pvalue
                    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                except Exception:
                    sig = "n.s."
            else:
                sig = "n.s."
            sig_strs.append(sig)

        overall_wr = (sum(slice_wins) / sum(slice_totals) * 100.0) if slice_totals else 0.0
        offset = (si_idx - (n_series - 1) / 2) * width
        bars = ax.bar(
            x + offset, win_rates, width * 0.92,
            label=f"{s_info['label_prefix']} ({overall_wr:.1f}\\%)",
            color=s_info["color"], linewidth=0.0, zorder=3,
        )

        for bar, wr, sig in zip(bars, win_rates, sig_strs):
            if wr > 0 and sig:
                ax.annotate(
                    sig,
                    xy=(bar.get_x() + bar.get_width() / 2, min(wr, 100.0)),
                    xytext=(0, 2.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.0,
                    color=PALETTE["muted"] if sig != "n.s." else PALETTE["reference"],
                    clip_on=False,
                )

    # Chance reference
    ax.axhline(50, color=PALETTE["reference"], linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "chance", xy=(1.0, 50), xycoords=("axes fraction", "data"),
        xytext=(3, 1), textcoords="offset points",
        ha="left", va="bottom", fontsize=6.8, color=PALETTE["muted"],
        annotation_clip=False,
    )

    ax.set_xlabel(r"Context fraction ($\alpha$)", labelpad=6)
    ax.set_ylabel("Win rate (\\%)", labelpad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sn:.2g}" for sn in student_ns])
    ax.set_xlim(-0.6, len(student_ns) - 0.4)
    ax.set_ylim(0, 104)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.grid(True, axis="y", zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(2, n_series),
        fontsize=7.4, handlelength=1.1, handleheight=0.8,
        columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_multi_model_summary_table(
    output_tex_path,
    output_png_path,
    models_data,
    student_ns,
    layer=23,
):
    """Generate publication-quality multi-model summary table in both LaTeX and PNG formats."""
    active_models = [m for m in models_data if models_data[m] is not None]
    if not active_models:
        return None, None

    headers_tex = [
        r"$\alpha$",
        r"Avg Rank (A/B/XGB)",
        r"Mean GC",
        r"Med GC",
        r"Effective $M_\alpha$",
        r"Gain",
    ]

    latex_lines = [
        r"\begin{table}[t]",
        r"\caption{Summary benchmark evaluation across 38 TabArena classification datasets for TabPFN and TabFM, averaged over 5 repeats. Columns report: context fraction $\alpha$; 3-way average rank (lower is better) among the Aligned student (A), unaligned Baseline student (B), and full-context XGBoost (XGB); mean and median relative gap closed (GC); mean effective baseline sample fraction ($M_\alpha$); and mean effective data gain ($M_\alpha - \alpha$).}",
        r"\label{tab:benchmark_results}",
        r"\begin{center}",
        r"\small",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        " & ".join(headers_tex) + r" \\",
    ]

    png_rows = []
    png_headers = [
        "Budget (N)", "Avg Rank (A/B/X)",
        "Mean GC", "Med GC", "Effective M", "Gain",
    ]

    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for mi, m_key in enumerate(active_models):
        data = models_data[m_key]
        cfg = MODEL_CONFIGS.get(m_key, {
            "name": m_key.upper(),
            "tex_name": m_key.upper(),
            "description": "",
        })
        desc = f" ({cfg['description']})" if cfg.get("description") else ""
        panel_label = f"{panel_letters[mi % len(panel_letters)]} {cfg['tex_name']}{desc}"

        latex_lines.append(r"\midrule")
        latex_lines.append(r"\multicolumn{6}{c}{\textbf{" + panel_label + r"}} \\")
        latex_lines.append(r"\midrule")

        png_rows.append([f"--- {cfg['name']}{desc} ---"] + [""] * (len(png_headers) - 1))

        a_mat = data["aligned_matrix"]
        b_mat = data["baseline_matrix"]
        m_mat = data["m_matrix"]
        gc_mat = data["gap_closed_matrix"]
        xgb_vals = data["xgb_values"]
        v_mask = data["valid_mask"]
        n_ds = a_mat.shape[0]

        pooled_a, pooled_b, pooled_gc, pooled_gains = [], [], [], []
        pooled_ranks_a, pooled_ranks_b, pooled_ranks_xgb = [], [], []
        total_beats_xgb, total_xgb = 0, 0

        for si, sn in enumerate(student_ns):
            valid_indices = [di for di in range(n_ds) if v_mask[di, si]]
            if not valid_indices:
                continue

            a_slice = a_mat[valid_indices, si]
            b_slice = b_mat[valid_indices, si]
            m_slice = m_mat[valid_indices, si]
            gc_slice = gc_mat[valid_indices, si]
            xgb_slice = xgb_vals[valid_indices]

            m_dict = _compute_slice_metrics(a_slice, b_slice, m_slice, gc_slice, xgb_slice, sn=sn)
            if not m_dict:
                continue

            mgc_str = f"{'+' if m_dict['mean_gc'] > 0 else ''}{m_dict['mean_gc']:.1f}\\%" if not np.isnan(m_dict["mean_gc"]) else "---"
            medgc_str = f"{'+' if m_dict['median_gc'] > 0 else ''}{m_dict['median_gc']:.1f}\\%" if not np.isnan(m_dict["median_gc"]) else "---"
            effm_str = f"{m_dict['mean_m']:.3f}" if not np.isnan(m_dict["mean_m"]) else "---"
            gain_str = f"{'+' if m_dict['mean_gain'] > 0 else ''}{m_dict['mean_gain']:.3f}" if not np.isnan(m_dict["mean_gain"]) else "---"

            latex_lines.append(
                f"{sn:.2g} & {m_dict['avg_rank_str']} & {mgc_str} & {medgc_str} & {effm_str} & {gain_str} \\\\"
            )

            png_rows.append([
                f"N={sn:.2g}",
                m_dict["avg_rank_str"],
                f"{'+' if m_dict['mean_gc'] > 0 else ''}{m_dict['mean_gc']:.1f}%" if not np.isnan(m_dict["mean_gc"]) else "–",
                f"{'+' if m_dict['median_gc'] > 0 else ''}{m_dict['median_gc']:.1f}%" if not np.isnan(m_dict["median_gc"]) else "–",
                effm_str,
                gain_str,
            ])

            pooled_a.extend(a_slice)
            pooled_b.extend(b_slice)
            pooled_gc.extend(m_dict["valid_gc"])
            pooled_gains.extend([m - sn for m in m_dict["valid_m"]])
            pooled_ranks_a.extend(m_dict["valid_ranks_a"])
            pooled_ranks_b.extend(m_dict["valid_ranks_b"])
            pooled_ranks_xgb.extend(m_dict["valid_ranks_xgb"])
            total_beats_xgb += m_dict["beats_xgb_cnt"]
            total_xgb += m_dict["beats_xgb_total"]

        # Overall row for this model
        if pooled_a:
            pa, pb = np.array(pooled_a), np.array(pooled_b)
            o_wins = int(np.sum(pa > pb))
            o_tot = len(pa)
            o_wr = o_wins / o_tot * 100.0

            dataset_mean_diffs = []
            for di in range(n_ds):
                diffs_d = [
                    a_mat[di, si] - b_mat[di, si]
                    for si in range(len(student_ns))
                    if v_mask[di, si] and not np.isnan(a_mat[di, si]) and not np.isnan(b_mat[di, si])
                ]
                if diffs_d:
                    dataset_mean_diffs.append(float(np.mean(diffs_d)))
            diff_arr = np.array(dataset_mean_diffs)
            if len(diff_arr) >= 5 and np.any(diff_arr != 0):
                try:
                    o_res = stats.wilcoxon(diff_arr, alternative="two-sided")
                    o_pval = f"p={o_res.pvalue:.4f}" if o_res.pvalue >= 0.0001 else "p<0.0001"
                except Exception:
                    o_pval = "n/a"
            else:
                o_pval = "n/a"

            o_mgc = float(np.mean(pooled_gc)) if pooled_gc else np.nan
            o_medgc = float(np.median(pooled_gc)) if pooled_gc else np.nan
            o_gain = float(np.mean(pooled_gains)) if pooled_gains else np.nan
            o_xgb = f"{total_beats_xgb}/{total_xgb} ({total_beats_xgb/total_xgb*100:.1f}%)" if total_xgb > 0 else "–"

            if pooled_ranks_a:
                o_rank_str = f"{float(np.mean(pooled_ranks_a)):.2f} / {float(np.mean(pooled_ranks_b)):.2f} / {float(np.mean(pooled_ranks_xgb)):.2f}"
            else:
                o_rank_str = "–"

            o_rank_tex = f"\\textbf{{{o_rank_str}}}"
            o_mgc_tex = f"\\textbf{{{'+' if o_mgc > 0 else ''}{o_mgc:.1f}\\%}}" if not np.isnan(o_mgc) else "---"
            o_medgc_tex = f"\\textbf{{{'+' if o_medgc > 0 else ''}{o_medgc:.1f}\\%}}" if not np.isnan(o_medgc) else "---"
            o_gain_tex = f"\\textbf{{{'+' if o_gain > 0 else ''}{o_gain:.3f}}}" if not np.isnan(o_gain) else "---"

            latex_lines.append(r"\midrule")
            latex_lines.append(
                f"\\textbf{{Overall}} & {o_rank_tex} & {o_mgc_tex} & {o_medgc_tex} & --- & {o_gain_tex} \\\\"
            )

            png_rows.append([
                "Overall",
                o_rank_str,
                f"{'+' if o_mgc > 0 else ''}{o_mgc:.1f}%" if not np.isnan(o_mgc) else "–",
                f"{'+' if o_medgc > 0 else ''}{o_medgc:.1f}%" if not np.isnan(o_medgc) else "–",
                "—",
                f"{'+' if o_gain > 0 else ''}{o_gain:.3f}" if not np.isnan(o_gain) else "–",
            ])

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{center}",
        r"\end{table}",
    ])

    if output_tex_path:
        os.makedirs(os.path.dirname(output_tex_path) or ".", exist_ok=True)
        with open(output_tex_path, "w") as f:
            f.write("\n".join(latex_lines) + "\n")

    # Render PNG
    if output_png_path and png_rows:
        n_r = len(png_rows)
        col_w = [1.35, 1.65, 1.25, 1.25, 1.05, 1.20]
        fig, ax = plt.subplots(figsize=(sum(col_w) + 0.4, (n_r + 2) * 0.36 + 0.8), dpi=150)
        ax.axis("off")

        table = ax.table(
            cellText=png_rows,
            colLabels=png_headers,
            colWidths=[w / sum(col_w) for w in col_w],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.0)
        table.scale(1, 1.5)

        for (ri, ci), cell in table.get_celld().items():
            cell.set_edgecolor("#bdc3c7")
            cell.set_linewidth(0.8)
            if ri == 0:
                cell.set_facecolor("#2c3e50")
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                row_val = png_rows[ri - 1][0]
                if "---" in row_val:
                    cell.set_facecolor("#34495e")
                    cell.get_text().set_color("white")
                    cell.get_text().set_fontweight("bold")
                elif row_val == "Overall":
                    cell.set_facecolor("#d5f5e3")
                    cell.get_text().set_fontweight("bold")
                else:
                    cell.set_facecolor("#fdfefe" if ri % 2 == 1 else "#f2f4f4")

        ax.set_title(
            f"Multi-Model Aligned Student Benchmark Summary (Layer {layer})\n"
            "[A = Aligned Student, B = Baseline Student, XGB = Full XGBoost (100% Training Budget)]",
            fontsize=11.0, fontweight="bold", pad=16,
        )
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_png_path) or ".", exist_ok=True)
        plt.savefig(output_png_path, bbox_inches="tight")
        plt.close(fig)

    return output_tex_path, output_png_path


def plot_multi_model_avg_rank_histogram(
    output_path,
    models_data,
    student_ns,
    layer=23,
):
    """Plot side-by-side 2-panel figure showing 3-way Average Rank histogram across context fractions alpha:
    (a) TabPFN and (b) TabFM, comparing Aligned student, Baseline student, and full-data XGBoost.
    """
    _apply_paper_style()
    active_models = [m for m in ["tabpfn", "tabfm"] if m in models_data and models_data[m] is not None]
    if not active_models:
        active_models = [m for m in models_data if models_data[m] is not None]
    if not active_models:
        return None

    n_models = len(active_models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(3.7 * n_models, 2.9), dpi=300, sharey=True, squeeze=False
    )
    axes = axes.flatten()

    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for mi, m_key in enumerate(active_models):
        ax = axes[mi]
        data = models_data[m_key]
        cfg = MODEL_CONFIGS.get(m_key, {
            "name": m_key.upper(),
            "tex_name": m_key.upper(),
            "color": PALETTE["tabpfn"],
            "edge_color": PALETTE["tabpfn_dark"],
        })
        model_display_name = re.sub(r"\s+v\d+(\.\d+)*", "", cfg["name"])

        a_mat = data["aligned_matrix"]
        b_mat = data["baseline_matrix"]
        xgb_vals = data["xgb_values"]
        v_mask = data["valid_mask"]
        n_ds = a_mat.shape[0]

        ranks_a_means, ranks_b_means, ranks_xgb_means = [], [], []

        for si, sn in enumerate(student_ns):
            valid_indices = [di for di in range(n_ds) if v_mask[di, si]]
            if not valid_indices:
                ranks_a_means.append(np.nan)
                ranks_b_means.append(np.nan)
                ranks_xgb_means.append(np.nan)
                continue

            a_slice = a_mat[valid_indices, si]
            b_slice = b_mat[valid_indices, si]
            m_slice = data["m_matrix"][valid_indices, si]
            gc_slice = data["gap_closed_matrix"][valid_indices, si]
            xgb_slice = xgb_vals[valid_indices]

            m_dict = _compute_slice_metrics(a_slice, b_slice, m_slice, gc_slice, xgb_slice, sn=sn)
            if m_dict and m_dict["valid_ranks_a"]:
                ranks_a_means.append(float(np.mean(m_dict["valid_ranks_a"])))
                ranks_b_means.append(float(np.mean(m_dict["valid_ranks_b"])))
                ranks_xgb_means.append(float(np.mean(m_dict["valid_ranks_xgb"])))
            else:
                ranks_a_means.append(np.nan)
                ranks_b_means.append(np.nan)
                ranks_xgb_means.append(np.nan)

        x = np.arange(len(student_ns))
        width = 0.25

        # Tonal encoding: aligned student in the model hue, comparators in neutrals
        bar_configs = [
            {
                "values": ranks_a_means,
                "offset": -width - 0.015,
                "label": "Aligned student",
                "color": cfg["color"],
            },
            {
                "values": ranks_b_means,
                "offset": 0.0,
                "label": "Baseline student",
                "color": PALETTE["neutral"],
            },
            {
                "values": ranks_xgb_means,
                "offset": width + 0.015,
                "label": "Full-data XGBoost",
                "color": PALETTE["xgboost"],
            },
        ]

        for b_cfg in bar_configs:
            ax.bar(
                x + b_cfg["offset"],
                b_cfg["values"],
                width,
                label=b_cfg["label"],
                color=b_cfg["color"],
                linewidth=0.0,
                zorder=3,
            )

        ax.set_xlabel(r"Context fraction ($\alpha$)", labelpad=6)
        if mi == 0:
            ax.set_ylabel("Average rank (lower is better)", labelpad=6)
        ax.set_title(rf"\textbf{{{panel_letters[mi % len(panel_letters)]}}} {model_display_name}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sn:.1f}" for sn in student_ns])
        ax.set_xlim(-0.6, len(student_ns) - 0.4)
        ax.set_ylim(1.0, 3.45)
        ax.set_yticks(np.arange(1.0, 3.01, 0.5))
        ax.grid(True, axis="y", zorder=0)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper left", fontsize=7.8, labelspacing=0.35, borderaxespad=0.1)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.10)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_multi_model_sample_efficiency_table(
    output_tex_path,
    output_png_path=None,
    models_data=None,
    student_ns=None,
):
    """Generate per-dataset sample efficiency comparison table across multiple models."""
    active_models = [m for m in ["tabpfn", "tabfm"] if m in models_data and models_data[m] is not None]
    if not active_models:
        active_models = [m for m in models_data if models_data[m] is not None]
    if not active_models:
        return None, None

    # Collect union of all datasets
    all_datasets_set = set()
    for m in active_models:
        all_datasets_set.update(models_data[m]["present_datasets"])
    all_datasets = sorted(list(all_datasets_set))
    n_ds = len(all_datasets)
    n_models = len(active_models)

    # 1. LaTeX landscape table
    latex = [
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\caption{Per-dataset effective baseline sample fraction $M_\alpha$ and Mean Gain ($M_\alpha - \alpha$) across 38 TabArena classification datasets for TabPFN (PFN) and TabFM (FM). Shaded green cells indicate effective samples ($M_\alpha > \alpha$ or Mean Gain $> 0$). Dashes (---) denote slices excluded by the teacher superiority filter (Section~\ref{sec:eval_protocol}).}",
        r"\label{tab:combined_per_dataset}",
        r"\begin{center}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.85}",
        r"\begin{tabular}{@{}l" + "c" * (n_models * (len(student_ns) + 1)) + r"@{}}",
        r"\toprule",
    ]

    h1 = [r"Dataset"]
    for sn in student_ns:
        h1.append(r"\multicolumn{" + str(n_models) + r"}{c}{$" + f"{sn:.2g}" + r"$}")
    h1.append(r"\multicolumn{" + str(n_models) + r"}{c}{Mean Gain}")
    latex.append(" & ".join(h1) + r" \\")

    h2 = [""]
    for _ in range(len(student_ns) + 1):
        for m in active_models:
            short_name = MODEL_CONFIGS.get(m, {}).get("short", m.upper())
            h2.append(r"\textbf{" + short_name + r"}")
    latex.append(" & ".join(h2) + r" \\")
    latex.append(r"\midrule")

    # Dataset rows
    ds_gains = {m: [] for m in active_models}
    for ds in all_datasets:
        short_ds = ds.split("/")[-1] if "/" in ds else ds
        row_tex = [short_ds.replace("_", r"\_")]
        ds_m_gains = {}

        for sn_i, sn in enumerate(student_ns):
            for m in active_models:
                m_data = models_data[m]
                if ds in m_data["present_datasets"]:
                    di = m_data["present_datasets"].index(ds)
                    v_mask = m_data["valid_mask"][di, sn_i]
                    m_val = m_data["m_matrix"][di, sn_i]
                    if v_mask and not np.isnan(m_val):
                        val_str = f"{m_val:.2f}"
                        if m_val > sn:
                            row_tex.append(f"\\cellcolor{{lightgreen}}{val_str}")
                        else:
                            row_tex.append(val_str)
                        ds_m_gains.setdefault(m, []).append(m_val - sn)
                    else:
                        row_tex.append("---")
                else:
                    row_tex.append("---")

        for m in active_models:
            g_list = ds_m_gains.get(m, [])
            if g_list:
                mg = float(np.mean(g_list))
                ds_gains[m].append(mg)
                val_str = f"{'+' if mg > 0 else ''}{mg:.2f}"
                if mg > 0:
                    row_tex.append(f"\\cellcolor{{lightgreen}}{val_str}")
                else:
                    row_tex.append(val_str)
            else:
                row_tex.append("---")

        latex.append(" & ".join(row_tex) + r" \\")

    # Average row
    latex.append(r"\midrule")
    avg_row = [r"\textbf{Average}"]
    for sn_i, sn in enumerate(student_ns):
        for m in active_models:
            m_data = models_data[m]
            col_vals = []
            for di in range(len(m_data["present_datasets"])):
                if m_data["valid_mask"][di, sn_i] and not np.isnan(m_data["m_matrix"][di, sn_i]):
                    col_vals.append(m_data["m_matrix"][di, sn_i])
            if col_vals:
                avg_m = float(np.mean(col_vals))
                if avg_m > sn:
                    avg_row.append(f"\\cellcolor{{lightgreen}}\\textbf{{{avg_m:.2f}}}")
                else:
                    avg_row.append(f"\\textbf{{{avg_m:.2f}}}")
            else:
                avg_row.append("---")

    for m in active_models:
        if ds_gains[m]:
            omg = float(np.mean(ds_gains[m]))
            val_str = f"{'+' if omg > 0 else ''}{omg:.2f}"
            if omg > 0:
                avg_row.append(f"\\cellcolor{{lightgreen}}\\textbf{{{val_str}}}")
            else:
                avg_row.append(f"\\textbf{{{val_str}}}")
        else:
            avg_row.append("---")

    latex.append(" & ".join(avg_row) + r" \\")
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{table}",
        r"\end{landscape}",
    ])

    if output_tex_path:
        os.makedirs(os.path.dirname(output_tex_path) or ".", exist_ok=True)
        with open(output_tex_path, "w") as f:
            f.write("\n".join(latex) + "\n")

    # For PNG: use first active model for detail table PNG
    if output_png_path and active_models:
        first_m = active_models[0]
        fd = models_data[first_m]
        plot_sample_efficiency_table(
            output_png_path,
            fd["present_datasets"],
            student_ns,
            fd["m_matrix"],
            fd["aligned_matrix"],
            fd["baseline_matrix"],
            fd["teacher_values"],
            fd["xgb_values"],
            valid_mask=fd["valid_mask"],
        )

    return output_tex_path, output_png_path


def plot_multi_model_dashboard(output_path, summary_png, scaling_png, sample_eff_png):
    """Stack summary table, scaling curves, and sample efficiency table into a master dashboard."""
    return stack_plots_vertically(
        [summary_png, scaling_png, sample_eff_png],
        output_path,
    )


# --------------------------------------------------------------------------- #
# 5. Master Orchestrator
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
    """Generate all benchmark evaluation plots and tables for one or more models.

    Args:
        dataset: str or list of datasets (or "tabarena" for all 38 datasets).
        student_n: list of student context fractions (e.g. [0.1, ..., 0.9]).
        layer: transformer layer for alignment hook.
        n_estimators: number of estimators (or None to use model-specific defaults).
        n_samples: number of synthetic samples for aligner training.
        patience: early stopping patience.
        lr: learning rate (or None to use model-specific defaults).
        batch_size: batch size.
        hidden_layers: hidden layers for aligner MLP.
        output: output directory or filepath.
        n_repeats: number of random splits.
        max_epochs: max epochs.
        model: str or list of strings (e.g. "tabpfn", "tabfm", or ["tabpfn", "tabfm"]).
        xgboost_opt: whether to use tuned XGBoost.
        aligner_opt: whether to use tuned aligner.

    Returns:
        Dict mapping artifact names to their file paths, or None/empty dict if no data.
    """
    if isinstance(dataset, str):
        dataset = [dataset]

    expanded_datasets = []
    for d in dataset:
        if d == "tabarena":
            expanded_datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            expanded_datasets.append(d)
    dataset = expanded_datasets

    # Normalize model argument to list of model strings
    if isinstance(model, str):
        if "," in model:
            models = [m.strip() for m in model.split(",")]
        else:
            models = [model]
    elif isinstance(model, (list, tuple)):
        models = list(model)
    else:
        models = ["tabpfn"]

    sorted_sns = sorted(student_n)

    # Determine out_dir
    if output is None:
        clean_model_str = "_".join(models)
        out_dir = os.path.join(OUTPUT_DIR, f"benchmarks_{clean_model_str}")
    elif output.endswith(".png") or output.endswith(".tex"):
        out_dir = os.path.dirname(output) or "."
    else:
        out_dir = output
    os.makedirs(out_dir, exist_ok=True)

    models_data = {}
    for m in models:
        m_lr = lr if lr is not None else MODEL_CONFIGS.get(m, {}).get("lr", 1e-3)
        m_estimators = n_estimators if n_estimators is not None else MODEL_CONFIGS.get(m, {}).get("n_estimators", 8)

        args_obj = argparse.Namespace(
            dataset=dataset,
            student_n=sorted_sns,
            layer=layer,
            n_estimators=m_estimators,
            n_samples=n_samples,
            patience=patience,
            lr=m_lr,
            batch_size=batch_size,
            hidden_layers=hidden_layers,
            output=output,
            n_repeats=n_repeats,
            max_epochs=max_epochs,
            model=m,
            xgboost_opt=xgboost_opt,
            aligner_opt=aligner_opt,
        )
        data = load_benchmark_data(args_obj, dataset, sorted_sns, layer)
        if data is not None:
            models_data[m] = data

    if not models_data:
        return {}

    scaling_curves_path = os.path.join(out_dir, "scaling_curves.png")
    win_rate_path = os.path.join(out_dir, "win_rate_bar_chart.png")
    avg_rank_path = os.path.join(out_dir, "avg_rank_histogram.png")
    detail_tex_path = os.path.join(out_dir, "sample_efficiency_table.tex")

    # Generate scaling curves (Figure 3 in paper)
    plot_multi_model_scaling_curves(scaling_curves_path, models_data, sorted_sns, layer=layer)

    # Generate grouped win rate bar chart (Figure 2 in paper)
    plot_multi_model_win_rate_bar_chart(win_rate_path, models_data, sorted_sns, layer=layer)

    # Generate 3-way average rank histogram (Figure 4 in paper)
    plot_multi_model_avg_rank_histogram(avg_rank_path, models_data, sorted_sns, layer=layer)

    # Generate per-dataset sample efficiency table (Table 1 in paper)
    generate_multi_model_sample_efficiency_table(detail_tex_path, None, models_data, sorted_sns)

    return {
        "Scaling Curves": scaling_curves_path,
        "Win Rate Bar Chart": win_rate_path,
        "Average Rank Histogram": avg_rank_path,
        "Sample Efficiency Table LaTeX": detail_tex_path,
    }


# --------------------------------------------------------------------------- #
# 6. Command Line Entry Point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation and visualization for aligned student models."
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["tabpfn", "tabfm"],
        help="One or more models to evaluate (e.g. --model tabpfn tabfm).",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["tabarena"],
        help="Datasets to evaluate (or 'tabarena' for all 38 datasets).",
    )
    parser.add_argument(
        "--student-n",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Student context fractions to evaluate.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=23,
        help="Transformer layer index for alignment hook.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of estimators (defaults to 8 for tabpfn, 1 for tabfm).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (defaults to 1e-3 for tabpfn, 1e-4 for tabfm).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of synthetic samples for aligner training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for aligner training.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="*",
        default=[],
        help="Hidden layer dimensions for aligner MLP (empty for linear).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of random split repeats.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs for aligner training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory or file path for artifacts.",
    )
    parser.add_argument(
        "--xgboost-opt",
        action="store_true",
        help="Use hyperparameter-tuned XGBoost baseline.",
    )
    parser.add_argument(
        "--aligner-opt",
        action="store_true",
        help="Use tuned aligner hyperparameters.",
    )
    args = parser.parse_args()

    results = plot(
        dataset=args.dataset,
        student_n=args.student_n,
        layer=args.layer,
        n_estimators=args.n_estimators,
        n_samples=args.n_samples,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_layers=args.hidden_layers,
        output=args.output,
        n_repeats=args.n_repeats,
        max_epochs=args.max_epochs,
        model=args.model,
        xgboost_opt=args.xgboost_opt,
        aligner_opt=args.aligner_opt,
    )
    print("\nBenchmark Evaluation Complete! Generated artifacts:")
    for k, v in (results or {}).items():
        print(f"  {k}: {v}")
