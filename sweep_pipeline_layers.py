import argparse
import os
import subprocess
import json
import matplotlib.pyplot as plt
import pruning_utils
from tqdm import tqdm

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_k_result_path(args, dataset, student_n, k):
    k_result_path = pruning_utils.create_filename_from_args(
        {
            "eval_dataset": dataset,
            "train_dataset": f"{dataset}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-use_tabpfn_{args.use_tabpfn}]",
            "student_n": student_n,
            "layer_k": k,
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
        cmd = [
            "./venv/bin/python", "run_pipeline.py",
            "--dataset", dataset,
            "--student_n", str(student_n),
            "--layer_k", str(k),
            "--n_estimators", str(args.n_estimators),
            "--n_samples", str(args.n_samples),
            "--hidden_layers", *[str(h) for h in args.hidden_layers],
            "--patience", str(args.patience),
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
        
        run_command(cmd)
    return k_result_path

def get_xgboost_result_path(args, dataset, student_n):
    xgboost_result_path = pruning_utils.create_filename_from_args(
        {
            "dataset": dataset,
            "student_n": student_n,
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
    plt.clf()
    teacher_roc_auc = None
    majority_vote_roc_auc = None
    n_unique_labels = None

    X_train, _, _, _ = pruning_utils.load_data(dataset)
    train_size = len(X_train)

    for student_n in tqdm(sorted(args.student_n), desc="Student sizes"):
        if student_n > train_size:
            print(f">>> sweep_pipeline_layers: Skipping student_n={student_n} (training set size is only {train_size})")
            continue
        baseline_roc_auc = None
        aligned_roc_aucs = []
        
        for k in tqdm(args.layers, desc=f"Layers (N={student_n})", leave=False):
            k_result_path = get_k_result_path(args, dataset, student_n, k)
            with open(k_result_path, "r") as f:
                metrics = json.load(f)["metrics"]
                
            aligned_roc_aucs.append(metrics["aligned_roc_auc"])
            
            if teacher_roc_auc is None:
                teacher_roc_auc = metrics["teacher_roc_auc"]
            if baseline_roc_auc is None:
                baseline_roc_auc = metrics["baseline_roc_auc"]
            if majority_vote_roc_auc is None and "majority_vote_roc_auc" in metrics:
                majority_vote_roc_auc = metrics["majority_vote_roc_auc"]
            if n_unique_labels is None and "n_unique_labels" in metrics:
                n_unique_labels = metrics["n_unique_labels"]
            assert teacher_roc_auc == metrics["teacher_roc_auc"], f"Teacher ROC AUC changed between runs: {teacher_roc_auc} != {metrics['teacher_roc_auc']} (file: {k_result_path})"
            assert baseline_roc_auc == metrics["baseline_roc_auc"], f"Baseline ROC AUC changed between runs: {baseline_roc_auc} != {metrics['baseline_roc_auc']} (file: {k_result_path})"

        line, = plt.plot(args.layers[:len(aligned_roc_aucs)], aligned_roc_aucs, marker='o', label=f'Aligned Student (N={student_n})')
        color = line.get_color()
        plt.axhline(y=baseline_roc_auc, color=color, linestyle=':', alpha=0.5, label=f'Baseline Student (N={student_n}, {baseline_roc_auc:.2f})')

        xgboost_result_path = get_xgboost_result_path(args, dataset, student_n)
        with open(xgboost_result_path, "r") as f:
            xgb_metrics = json.load(f)["metrics"]
        xgb_roc_auc = xgb_metrics["xgboost_roc_auc"]
        plt.axhline(y=xgb_roc_auc, color=color, linestyle='--', linewidth=1.2,
                    label=f'XGBoost (N={student_n}, {xgb_roc_auc:.2f})')

    plt.axhline(y=teacher_roc_auc, color='red', linestyle='--', linewidth=2, label=f'Teacher ({teacher_roc_auc:.2f})')
    plt.axhline(y=majority_vote_roc_auc, color='gray', linestyle='-.', linewidth=2, label=f'Majority Vote ({majority_vote_roc_auc:.2f})')

    xgboost_full_result_path = get_xgboost_result_path(args, dataset, student_n=-1)
    with open(xgboost_full_result_path, "r") as f:
        xgb_full_metrics = json.load(f)["metrics"]
    xgb_full_roc_auc = xgb_full_metrics["xgboost_roc_auc"]
    plt.axhline(y=xgb_full_roc_auc, color='darkgreen', linestyle='--', linewidth=2,
                label=f'XGBoost Full ({xgb_full_roc_auc:.2f})')


    # Clip y-axis so a small majority-vote value (when metric is -log_loss)
    # doesn't dwarf the interesting lines.
    if majority_vote_roc_auc < teacher_roc_auc - 3:
        plt.ylim(top=teacher_roc_auc * 0.7, bottom=teacher_roc_auc * 3)

    plt.xlabel('Layer K')
    plt.ylabel('Test Metric')
    n_labels_str = f", Labels={n_unique_labels}" if n_unique_labels is not None else ""
    plt.title(f'Aligners {dataset} (Estimators={args.n_estimators}{n_labels_str})')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[])
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--force", action="store_true", help="Force re-running the pipeline")
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
