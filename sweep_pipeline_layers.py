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

def get_k_result_path(args, dataset, all_datasets, student_n, k, repeat):
    if args.train_on_rest:
        training_datasets = [ds for ds in all_datasets if ds != dataset]
    else:
        training_datasets = [dataset]

    synthetic_training_datasets = [
        f"{ds}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-repeat_{repeat}-use_tabpfn_{args.use_tabpfn}]"
        for ds in training_datasets
    ]

    path_args = {
            "eval_dataset": dataset,
            "train_dataset": synthetic_training_datasets,
            "student_n": student_n,
            "layer_k": k,
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
    }
    if args.loss_beta is not None:
        path_args["loss_beta"] = args.loss_beta
    if args.clip_grad is not None:
        path_args["clip_grad"] = args.clip_grad

    k_result_path = pruning_utils.create_filename_from_args(
        path_args,
        script_name="evaluate_aligned_student",
        extension=".json"
    )
    if not os.path.exists(k_result_path):
        cmd = [
            "./venv/bin/python", "run_pipeline.py",
            "--training_datasets", *training_datasets,
            "--test_dataset", dataset,
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
        if args.per_token:
            cmd.append("--per_token")
        if args.force:
            cmd.append("--force_create_synthetic_dataset")
        if args.use_tabpfn:
            cmd.append("--use_tabpfn")
        if args.predict_residual:
            cmd.append("--predict_residual")
        if args.use_feature_stats:
            cmd.append("--use_feature_stats")
        if args.loss_beta is not None:
            cmd.extend(["--loss_beta", str(args.loss_beta)])
        if args.clip_grad is not None:
            cmd.extend(["--clip_grad", str(args.clip_grad)])
        if args.max_epochs is not None:
            cmd.extend(["--max_epochs", str(args.max_epochs)])

        run_command(cmd)
    return k_result_path

def get_xgboost_result_path(args, dataset, student_n, repeat):
    xgboost_result_path = pruning_utils.create_filename_from_args(
        {
            "dataset": dataset,
            "student_n": student_n,
            "repeat": repeat,
            "output_dir": args.output_dir,
        },
        script_name="train_xgboost",
        extension=".json",
    )
    if not os.path.exists(xgboost_result_path):
        cmd = [
            "./venv/bin/python", "train_xgboost.py",
            "--dataset", dataset,
            "--student_n", str(student_n),
            "--repeat", str(repeat),
            "--output_dir", args.output_dir,
        ]
        if args.force:
            cmd.append("--force")
        run_command(cmd)
    return xgboost_result_path

def run_dataset(args, dataset, all_datasets):
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
        if student_n > train_size:
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
                k_result_path = get_k_result_path(args, dataset, all_datasets, student_n, k, repeat)
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
            ("Aligned",  data["aligned"],  color,       {}),
            ("Baseline", data["baseline"], "#ff7f0e",   {"linestyle": "dashed"}),
            ("Teacher",  data["teacher"],  "red",       {}),
            ("XGBoost",  data["xgboost"], "#1a5c1a",   {}),
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
    legend_elements = [Patch(facecolor=colors[gi], alpha=0.7, label=f"N={sn} Aligned") for gi, sn in enumerate(student_ns)]
    legend_elements += [
        Patch(facecolor="#ff7f0e", alpha=0.7, label="Baseline"),
        Patch(facecolor="red",     alpha=0.7, label="Teacher"),
        Patch(facecolor="#1a5c1a", alpha=0.7, label="XGBoost"),
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
    parser.add_argument("--student_n", type=int, nargs='+', default=[20], help="Student training sizes")
    parser.add_argument("--layers", type=int, nargs='+', default=[1, 2, 5, 8, 9, 10, 11])
    parser.add_argument("--per_token", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples to generate.")
    parser.add_argument("--use_tabpfn", action="store_true", help="Use TabPFN to generate synthetic data.")
    parser.add_argument("--patience", type=int, default=10, help="Stop aligner training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for aligner training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for aligner training.")
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[])
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--force", action="store_true", help="Force re-running the pipeline")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of OpenML repeats to run (each uses a different random split).")
    parser.add_argument("--train_on_rest", action="store_true",
                        help="If set, train the aligner on all datasets except the one being evaluated (leave-one-out). "
                             "Otherwise, train on the same dataset being evaluated (default).")
    parser.add_argument("--use_feature_stats", action="store_true",
                        help="Condition the aligner on per-feature statistics (mean, std, min, max, median) "
                             "from the teacher's training data.")
    parser.add_argument("--loss_beta", type=float, default=None,
                        help="If specified, use train_activation_aligner_v2 with this KL-divergence weight. "
                             "If unspecified, use the original train_activation_aligner (MSE only).")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="Clip gradient norm to this value. Passed to train_activation_aligner_v2.py (only works if loss_beta is specified).")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Maximum number of training epochs. None = unlimited (rely on patience).")
    args = parser.parse_args()

    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    for dataset in tqdm(datasets, desc="Sweeping datasets"):
        run_dataset(args, dataset, all_datasets=datasets)

if __name__ == "__main__":
    main()
