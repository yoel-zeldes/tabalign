import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pruning_utils
from tqdm import tqdm


def load_results_for_repeat(args, dataset, layer_k, repeat):
    """
    Load evaluate_aligned_student JSON results for all student_n values for a
    given dataset and repeat.

    Returns a dict:
        {student_n: {"teacher_roc_auc": float, "baseline_roc_auc": float, "aligned_roc_auc": float}}
    """
    results = {}
    for student_n in sorted(args.student_n):
        k_result_path = pruning_utils.create_filename_from_args(
            {
                "eval_dataset": dataset,
                "train_dataset": f"{dataset}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-repeat_{repeat}-use_tabpfn_{args.use_tabpfn}]",
                "student_n": student_n,
                "layer_k": layer_k,
                "n_estimators": args.n_estimators,
                "patience": args.patience,
                "lr": 1e-3,
                "batch_size": 512,
                "per_token": args.per_token,
                "hidden_layers": args.hidden_layers,
                "predict_residual": args.predict_residual,
                "repeat": repeat,
                "output_dir": args.output_dir
            },
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
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[])
    parser.add_argument("--predict_residual", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default=None, help="Path to save the figure. Defaults to auto-generated name.")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of OpenML repeats to aggregate over.")
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
                # Add ✓ if aligned mean beats both XGBoost and student baseline
                xv = xgb_full_raw[di]
                if (not np.isnan(xv) and av > xv
                        and (np.isnan(bv) or av > bv)):
                    ax.text(cx + half_line + slot_w * 0.05, av, "✓",
                            fontsize=9, va="center", ha="left", zorder=7,
                            color=colors[si], clip_on=True)

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

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    if args.output:
        output_path = args.output
    else:
        output_path = pruning_utils.create_filename_from_args(
            vars(args),
            script_name="plot_aligned_results",
            extension=".png",
            makedirs=True,
        )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
