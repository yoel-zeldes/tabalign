import argparse
import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import pruning_utils
from tqdm import tqdm

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_k_result_path(args, dataset, student_n, k, repeat):
    synthetic_training_dataset = (
        f"{dataset}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-repeat_{repeat}]"
    )

    path_args = {
            "eval_dataset": dataset,
            "train_dataset": synthetic_training_dataset,
            "student_n": student_n,
            "layer_k": k,
            "n_estimators": args.n_estimators,
            "patience": args.patience,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "hidden_layers": args.hidden_layers,
            "repeat": repeat,
            "output_dir": args.output_dir,
            "max_epochs": args.max_epochs,
            "model": args.model,
    }
    if args.aligner_opt:
        path_args["aligner_opt"] = True

    k_result_path = pruning_utils.create_filename_from_args(
        path_args,
        script_name="evaluate_aligned_student",
        extension=".json"
    )
    if not os.path.exists(k_result_path):
        cmd = [
            "./venv/bin/python", "run_pipeline.py",
            "--dataset", dataset,
            "--student_n", str(student_n),
            "--layer_k", str(k),
            "--n_estimators", str(args.n_estimators),
            "--n_samples", str(args.n_samples),
            "--hidden_layers", *[str(h) for h in args.hidden_layers],
            "--patience", str(args.patience),
            "--lr", str(args.lr),
            "--batch_size", str(args.batch_size),
            "--repeat", str(repeat),
            "--output_dir", args.output_dir
        ]
        if args.force:
            cmd.append("--force_create_synthetic_dataset")
        if args.max_epochs is not None:
            cmd.extend(["--max_epochs", str(args.max_epochs)])
        if getattr(args, "aligner_opt", False):
            cmd.append("--aligner_opt")
        cmd.extend(["--model", args.model])

        run_command(cmd)
    return k_result_path

def get_xgboost_result_path(args, dataset, student_n, repeat):
    path_args = {
        "dataset": dataset,
        "student_n": student_n,
        "repeat": repeat,
        "output_dir": args.output_dir,
    }
    if args.xgboost_opt:
        script_name = "train_xgboost_opt"
        path_args["n_trials"] = 1000
        path_args["timeout"] = 600
        path_args["n_jobs"] = -1
        path_args["cv_folds"] = 5
        path_args["seed"] = 42
    else:
        script_name = "train_xgboost"

    xgboost_result_path = pruning_utils.create_filename_from_args(
        path_args,
        script_name=script_name,
        extension=".json",
    )
    if not os.path.exists(xgboost_result_path):
        cmd = [
            "./venv/bin/python", f"{script_name}.py",
            "--dataset", dataset,
            "--student_n", str(student_n),
            "--repeat", str(repeat),
            "--output_dir", args.output_dir,
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)
    return xgboost_result_path

def run_dataset(args, dataset):
    output_path = pruning_utils.create_filename_from_args(
        {**vars(args), "dataset": dataset},
        script_name="sweep_pipeline_layers",
        extension=".png",
        makedirs=True,
    )

    X_train, _, _, _ = pruning_utils.load_data(dataset, repeat=0)
    train_size = len(X_train)

    n_repeats = args.n_repeats

    # Collect per-repeat metrics.
    # Structure: {student_n: {"aligned": [r0, r1, ...], "baseline": [r0, r1, ...],
    #                          "teacher": [r0, r1, ...], "xgboost": [r0, r1, ...]}}
    per_student = {}

    for student_n in tqdm(sorted(args.student_n), desc="Student sizes"):
        if pruning_utils.resolve_student_n(student_n, train_size) > train_size:
            print(f">>> sweep_pipeline_layers: Skipping student_n={student_n} (training set size is only {train_size})")
            continue

        aligned_per_repeat = []
        baseline_per_repeat = []
        teacher_per_repeat = []
        xgboost_per_repeat = []

        for repeat in tqdm(range(n_repeats), desc=f"Repeats (N={student_n})", leave=False):
            # Best aligned score across layers for this repeat
            repeat_aligned = []
            repeat_baseline = None
            repeat_teacher = None

            for k in tqdm(args.layers, desc=f"Layers (N={student_n}, repeat={repeat})", leave=False):
                k_result_path = get_k_result_path(args, dataset, student_n, k, repeat)
                with open(k_result_path, "r") as f:
                    metrics = json.load(f)["metrics"]

                repeat_aligned.append(metrics["aligned_roc_auc"])
                if repeat_baseline is None:
                    repeat_baseline = metrics["baseline_roc_auc"]
                if repeat_teacher is None:
                    repeat_teacher = metrics["teacher_roc_auc"]

            # Use the best layer's aligned score for this repeat
            if repeat_aligned:
                aligned_per_repeat.append(max(repeat_aligned))
            if repeat_baseline is not None:
                baseline_per_repeat.append(repeat_baseline)
            if repeat_teacher is not None:
                teacher_per_repeat.append(repeat_teacher)

            # XGBoost for this repeat
            xgb_path = get_xgboost_result_path(args, dataset, student_n=-1, repeat=repeat)
            with open(xgb_path, "r") as f:
                xgb_metrics = json.load(f)["metrics"]
            xgboost_per_repeat.append(xgb_metrics["xgboost_roc_auc"])

        per_student[student_n] = {
            "aligned": aligned_per_repeat,
            "baseline": baseline_per_repeat,
            "teacher": teacher_per_repeat,
            "xgboost": xgboost_per_repeat,
        }

    if not per_student:
        print(f"No results for dataset {dataset}, skipping plot.")
        return

    # ── Boxplot ──────────────────────────────────────────────────────────
    student_ns = sorted(per_student.keys())
    n_students = len(student_ns)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_students))

    fig, ax = plt.subplots(figsize=(max(6, n_students * 2.5), 6))

    group_gap = 1.0          # gap between student_n groups
    box_width = 0.15
    # 4 boxes per group: aligned, baseline, teacher, xgboost
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * box_width * 2

    positions_all = []
    labels_all = []

    for gi, student_n in enumerate(student_ns):
        center = gi * group_gap
        data = per_student[student_n]
        color = colors[gi]

        series = [
            ("Aligned (Opt)" if args.aligner_opt else "Aligned", data["aligned"], color, {}),
            ("Baseline", data["baseline"], "#ff7f0e", {"linestyle": "dashed"}),
            ("Teacher", data["teacher"], "red", {}),
            ("XGBoost (Opt)" if args.xgboost_opt else "XGBoost", data["xgboost"], "#1a5c1a", {}),
        ]

        for (label, vals, col, _), offset in zip(series, offsets):
            pos = center + offset
            bp = ax.boxplot(
                vals,
                positions=[pos],
                widths=box_width * 1.6,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=col, alpha=0.6),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color=col),
                capprops=dict(color=col),
                flierprops=dict(marker="o", color=col, markersize=4),
            )
            positions_all.append(pos)
            labels_all.append(f"N={student_n}\n{label}" if gi == 0 else "")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(
            facecolor=colors[gi],
            alpha=0.7,
            label=f"N={sn} Aligned (Opt)" if args.aligner_opt else f"N={sn} Aligned",
        )
        for gi, sn in enumerate(student_ns)
    ]
    legend_elements += [
        Patch(facecolor="#ff7f0e", alpha=0.7, label="Baseline"),
        Patch(facecolor="red", alpha=0.7, label="Teacher"),
        Patch(facecolor="#1a5c1a", alpha=0.7, label="XGBoost (Opt)" if args.xgboost_opt else "XGBoost"),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xticks([gi * group_gap for gi in range(n_students)])
    ax.set_xticklabels([f"N={sn}" for sn in student_ns])
    ax.set_ylabel("Test Metric")
    ax.set_title(f"Aligners {dataset} (Estimators={args.n_estimators}, Repeats={n_repeats})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"\nSweep plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Sweep pipeline over multiple layers")
    parser.add_argument("--dataset", type=str, nargs='+', default=["breast_cancer"])
    parser.add_argument("--student_n", type=pruning_utils.parse_student_n, nargs='+', default=[20], help="Student training sizes")
    parser.add_argument("--layers", type=int, nargs='+', default=[1, 2, 5, 8, 9, 10, 11])
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples to generate.")
    parser.add_argument("--patience", type=int, default=10, help="Stop aligner training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for aligner training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for aligner training.")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[],
                        help="Hidden layer multipliers for MLP aligner. Empty = linear.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--force", action="store_true", help="Force re-running the pipeline")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of OpenML repeats to run (each uses a different random split).")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum number of training epochs. None = unlimited (rely on patience).")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn",
                        help="Model architecture to use.")
    parser.add_argument("--xgboost_opt", action="store_true",
                        help="Use train_xgboost_opt.py (hyperparameter-tuned XGBoost) instead of train_xgboost.py.")
    parser.add_argument("--aligner_opt", action="store_true",
                        help="Use hyperparameter optimization for aligner training.")
    args = parser.parse_args()

    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    for dataset in tqdm(datasets, desc="Sweeping datasets"):
        run_dataset(args, dataset)

if __name__ == "__main__":
    main()
