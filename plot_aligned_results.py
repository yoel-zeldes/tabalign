import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pruning_utils
from tqdm import tqdm


def load_results(args, dataset, layer_k):
    """
    Load evaluate_aligned_student JSON results for all (student_n, layer_k) combos
    for a given dataset.

    Returns a dict:
        {
            student_n: {
                "teacher_roc_auc": float,
                "baseline_roc_auc": float,
                "aligned_roc_auc": float,
            }
        }
    """
    results = {}
    for student_n in sorted(args.student_n):
        k_result_path = pruning_utils.create_filename_from_args(
            {
                "eval_dataset": dataset,
                "train_dataset": f"{dataset}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-use_tabpfn_{args.use_tabpfn}]",
                "student_n": student_n,
                "layer_k": layer_k,
                "n_estimators": args.n_estimators,
                "patience": args.patience,
                "lr": 1e-3,
                "batch_size": 512,
                "per_token": args.per_token,
                "hidden_layers": args.hidden_layers,
                "predict_residual": args.predict_residual,
                "output_dir": args.output_dir
            },
            script_name="evaluate_aligned_student",
            extension=".json"
        )

        if not os.path.exists(k_result_path):
            print(
                f"  [SKIP] Missing result for dataset={dataset}, "
                f"student_n={student_n}: {k_result_path}"
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


def load_xgboost_results(args, dataset):
    """
    Load train_xgboost JSON result for the full training set (student_n=-1).

    Returns a dict: {-1: xgboost_roc_auc}
    """
    results = {}
    path = pruning_utils.create_filename_from_args(
        {
            "dataset": dataset,
            "student_n": -1,
            "output_dir": args.output_dir,
        },
        script_name="train_xgboost",
        extension=".json",
    )
    if not os.path.exists(path):
        print(f"  [SKIP] Missing XGBoost result for dataset={dataset}, student_n=-1: {path}")
        return results
    with open(path, "r") as f:
        metrics = json.load(f)["metrics"]
    results[-1] = metrics["xgboost_roc_auc"]
    return results


def normalize(value, baseline, teacher):
    """Normalize so that baseline -> 0 and teacher -> 1."""
    denom = teacher - baseline
    if abs(denom) < 1e-12:
        return 0.0
    return (value - baseline) / denom


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
    args = parser.parse_args()

    # Expand "tabarena" shorthand
    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    # ------------------------------------------------------------------ #
    # Gather results
    # ------------------------------------------------------------------ #
    # all_data[dataset][student_n] = {"teacher_roc_auc", "baseline_roc_auc", "aligned_roc_auc"}
    all_data = {}
    xgb_data = {}  # xgb_data[dataset][student_n] = xgboost_roc_auc
    for dataset in tqdm(datasets, desc="Loading results"):
        res = load_results(args, dataset, args.layer)
        if res:
            all_data[dataset] = res
            xgb_data[dataset] = load_xgboost_results(args, dataset)

    if not all_data:
        print("No results found. Make sure evaluate_aligned_student.py has been run first.")
        return

    present_datasets = list(all_data.keys())
    student_ns = sorted(args.student_n)

    # ------------------------------------------------------------------ #
    # Build raw + normalized value arrays
    # ------------------------------------------------------------------ #
    # Shape: [n_datasets, n_student_ns]
    norm_values     = np.full((len(present_datasets), len(student_ns)), np.nan)
    raw_values      = np.full((len(present_datasets), len(student_ns)), np.nan)
    baseline_values = np.full((len(present_datasets), len(student_ns)), np.nan)
    teacher_values  = np.full(len(present_datasets), np.nan)
    xgb_full_raw    = np.full(len(present_datasets), np.nan)              # student_n=-1, raw
    xgb_full_norm   = np.full((len(present_datasets), len(student_ns)), np.nan)  # normalized per student_n baseline

    for di, dataset in enumerate(present_datasets):
        for si, sn in enumerate(student_ns):
            if sn not in all_data[dataset]:
                continue
            entry = all_data[dataset][sn]
            teacher  = entry["teacher_roc_auc"]
            baseline = entry["baseline_roc_auc"]
            aligned  = entry["aligned_roc_auc"]
            raw_values[di, si]      = aligned
            baseline_values[di, si] = baseline
            norm_values[di, si]     = normalize(aligned, baseline, teacher)
            if np.isnan(teacher_values[di]):
                teacher_values[di] = teacher

        # Full-training-set XGBoost (student_n=-1)
        # Raw value: one per dataset (stored for the spanning line in the raw subplot).
        # Normalized: computed per student_n, using that student_n's own baseline/teacher,
        # so the height varies correctly across datasets.
        xgb_full_val = xgb_data.get(dataset, {}).get(-1, np.nan)
        if not np.isnan(xgb_full_val):
            xgb_full_raw[di] = xgb_full_val
            for si, sn in enumerate(student_ns):
                if sn in all_data[dataset]:
                    entry = all_data[dataset][sn]
                    xgb_full_norm[di, si] = normalize(
                        xgb_full_val, entry["baseline_roc_auc"], entry["teacher_roc_auc"]
                    )

    # ------------------------------------------------------------------ #
    # Count datasets where aligned beats full-training-set XGBoost
    # ------------------------------------------------------------------ #
    # For each student_n, count datasets where aligned_roc_auc > xgb_full_roc_auc
    beats_full_xgb = []  # list of (count_better, count_valid) per student_n
    for si, sn in enumerate(student_ns):
        count_better = 0
        count_valid = 0
        for di, dataset in enumerate(present_datasets):
            aligned_val = raw_values[di, si]
            xgb_full_val = xgb_full_raw[di]
            if not np.isnan(aligned_val) and not np.isnan(xgb_full_val):
                count_valid += 1
                if aligned_val > xgb_full_val:
                    count_better += 1
        beats_full_xgb.append((count_better, count_valid))

    # Count datasets where teacher > full-training-set XGBoost
    teacher_beats_xgb = 0
    teacher_xgb_valid = 0
    for di in range(len(present_datasets)):
        t_val = teacher_values[di]
        x_val = xgb_full_raw[di]
        if not np.isnan(t_val) and not np.isnan(x_val):
            teacher_xgb_valid += 1
            if t_val > x_val:
                teacher_beats_xgb += 1

    # ------------------------------------------------------------------ #
    # Plot — 2-row grid (raw on top, normalized on bottom)
    # ------------------------------------------------------------------ #
    n_datasets = len(present_datasets)
    n_students = len(student_ns)

    group_width = 0.5
    bar_width   = group_width / n_students
    x           = np.arange(n_datasets)
    colors      = plt.cm.tab10(np.linspace(0, 0.9, n_students))
    short_names = [d.split("/")[-1] if "/" in d else d for d in present_datasets]

    fig, (ax_raw, ax_norm) = plt.subplots(
        2, 1,
        figsize=(max(8, n_datasets * 1.2), 10),
        sharex=True,
    )

    # ── Top subplot: raw metric ──────────────────────────────────────────
    for si, sn in enumerate(student_ns):
        offsets = (si - (n_students - 1) / 2) * bar_width
        vals = raw_values[:, si]
        count_better, count_valid = beats_full_xgb[si]
        bars = ax_raw.bar(
            x + offsets,
            vals,
            width=bar_width * 0.9,
            label=f"N={sn} ({count_better}/{count_valid})",
            color=colors[si],
            alpha=0.85,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax_raw.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90 if n_datasets > 6 else 0,
                    clip_on=True,
                )

    # XGBoost colors
    XGB_FULL_COLOR = "#1a5c1a"  # darker green for full-training-set line

    for di in range(n_datasets):
        xgb_full_val = xgb_full_raw[di]
        if not np.isnan(xgb_full_val):
            ax_raw.plot(
                [x[di] - group_width / 2, x[di] + group_width / 2],
                [xgb_full_val, xgb_full_val],
                color=XGB_FULL_COLOR, linewidth=2.0, linestyle="--", zorder=5,
                label="XGBoost" if di == 0 else "_nolegend_",
            )



    # Per-dataset teacher and baseline markers
    for di in range(n_datasets):
        if not np.isnan(teacher_values[di]):
            ax_raw.plot(
                [x[di] - group_width / 2, x[di] + group_width / 2],
                [teacher_values[di], teacher_values[di]],
                color="red", linewidth=1.5, zorder=4,
                label="Teacher" if di == 0 else "_nolegend_",
            )
        for si in range(n_students):
            bv = baseline_values[di, si]
            if not np.isnan(bv):
                offset = (si - (n_students - 1) / 2) * bar_width
                ax_raw.plot(
                    [x[di] + offset - bar_width * 0.45, x[di] + offset + bar_width * 0.45],
                    [bv, bv],
                    color="#ff7f0e", linewidth=1.2, linestyle="--", zorder=4,
                    label="Baseline" if (di == 0 and si == 0) else "_nolegend_",
                )

    ax_raw.axhline(y=0, color="black", linewidth=1.0, zorder=4)
    ax_raw.set_ylabel("Raw Metric (ROC-AUC)")
    ax_raw.set_title(
        f"Aligned Student Performance (layer={args.layer})\n"
        f"Estimators={args.n_estimators}, Patience={args.patience}"
    )
    ax_raw.legend(loc="upper right")
    ax_raw.grid(axis="y", alpha=0.3, zorder=0)
    ax_raw.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax_norm.axhline(y=1.0, color="red",  linestyle="--", linewidth=1.5, label=f"Teacher ({teacher_beats_xgb}/{teacher_xgb_valid})", zorder=4)
    ax_norm.axhline(y=0.0, color="#ff7f0e", linestyle=":",  linewidth=1.5, label="Baseline", zorder=4)

    # ── Bottom subplot: normalized metric ────────────────────────────────
    all_norm = np.concatenate([norm_values.ravel(), xgb_full_norm.ravel()])
    y_cap   = 2.0  if np.nanmax(all_norm) >  2.0 else None
    y_floor = -2.0 if np.nanmin(all_norm) < -2.0 else None

    for si, sn in enumerate(student_ns):
        offsets = (si - (n_students - 1) / 2) * bar_width
        vals = norm_values[:, si]
        count_better, count_valid = beats_full_xgb[si]
        bars = ax_norm.bar(
            x + offsets,
            vals,
            width=bar_width * 0.9,
            label=f"N={sn} ({count_better}/{count_valid})",
            color=colors[si],
            alpha=0.85,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and (y_cap is None or val <= y_cap):
                ax_norm.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90 if n_datasets > 6 else 0,
                    clip_on=True,
                )

    # Full-training-set XGBoost: one tick per student_n bar, normalized against that student_n's
    # own baseline/teacher so the height varies correctly across datasets.
    # Drawn first (lower zorder) so per-N student ticks appear on top.
    first_full_label_drawn = False
    for di in range(n_datasets):
        for si, sn in enumerate(student_ns):
            xgb_full_val = xgb_full_norm[di, si]
            if not np.isnan(xgb_full_val):
                offset = (si - (n_students - 1) / 2) * bar_width
                label = "XGBoost" if not first_full_label_drawn else "_nolegend_"
                first_full_label_drawn = True
                ax_norm.plot(
                    [x[di] + offset - bar_width * 0.45, x[di] + offset + bar_width * 0.45],
                    [xgb_full_val, xgb_full_val],
                    color=XGB_FULL_COLOR, linewidth=2.0, linestyle="--", zorder=5,
                    label=label,
                )



    if y_cap is not None:
        ax_norm.set_ylim(top=y_cap)
    if y_floor is not None:
        ax_norm.set_ylim(bottom=y_floor)

    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(short_names, rotation=-45, ha="left", fontsize=9)
    ax_norm.set_ylabel("Normalized Metric\n(0 = baseline, 1 = teacher)")
    ax_norm.set_xlabel("Dataset")
    ax_norm.set_title("Normalized")
    ax_norm.legend(loc="upper right")
    ax_norm.grid(axis="y", alpha=0.3, zorder=0)
    ax_norm.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

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
