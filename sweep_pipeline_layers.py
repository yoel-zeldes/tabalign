import argparse
import os
import subprocess
import json
import matplotlib.pyplot as plt
import glob
import pruning_utils
from tqdm import tqdm

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_result_path(args, dataset, student_n, k):
    glob_args = {
        "eval_dataset": dataset,
        "train_dataset": "*",
        "student_n": student_n,
        "layer_k": k,
        "n_estimators": args.n_estimators,
        "epochs": "*",
        "lr": "*",
        "batch_size": "*",
        "per_token": args.per_token,
        "aligners_dir": "*",
        "output_dir": "results/evaluation" 
    }
    result_pattern = pruning_utils.create_filename_from_args(
        glob_args, 
        script_name="evaluate_aligned_student", 
        extension=".json"
    )
    result_paths = glob.glob(result_pattern)
    assert len(result_paths) <= 1, f"Expected 1 result file for layer {k}, found {len(result_paths)}: {result_paths}"
    return result_paths[0] if result_paths else None

def run_dataset(args, dataset):
    plt.clf()
    teacher_acc = None
    majority_vote_acc = None
    
    for student_n in tqdm(sorted(args.student_n), desc="Student sizes"):
        baseline_acc = None
        aligned_accs = []
        
        for k in tqdm(args.layers, desc=f"Layers (N={student_n})", leave=False):
            if not get_result_path(args, dataset, student_n, k):
                cmd = [
                    "./venv/bin/python", "run_pipeline.py",
                    "--dataset", dataset,
                    "--student_n", str(student_n),
                    "--layer_k", str(k),
                    "--n_estimators", str(args.n_estimators),
                ]
                if args.per_token:
                    cmd.append("--per_token")
                
                run_command(cmd)
                
            with open(get_result_path(args, dataset, student_n, k), "r") as f:
                metrics = json.load(f)["metrics"]
                
            aligned_accs.append(metrics["aligned_acc"])
            
            if teacher_acc is None:
                teacher_acc = metrics["teacher_acc"]
            if baseline_acc is None:
                baseline_acc = metrics["baseline_acc"]
            if majority_vote_acc is None and "majority_vote_acc" in metrics:
                majority_vote_acc = metrics["majority_vote_acc"]
            assert teacher_acc == metrics["teacher_acc"], f"Teacher accuracy changed between runs: {teacher_acc} != {metrics['teacher_acc']}"
            assert baseline_acc == metrics["baseline_acc"], f"Baseline accuracy changed between runs: {baseline_acc} != {metrics['baseline_acc']}"

        line, = plt.plot(args.layers[:len(aligned_accs)], aligned_accs, marker='o', label=f'Aligned Student (N={student_n})')
        color = line.get_color()
        plt.axhline(y=baseline_acc, color=color, linestyle=':', alpha=0.5, label=f'Baseline Student (N={student_n}, {baseline_acc:.2f})')

    if teacher_acc is not None:
        plt.axhline(y=teacher_acc, color='red', linestyle='--', linewidth=2, label=f'Teacher ({teacher_acc:.2f})')
    if majority_vote_acc is not None:
        plt.axhline(y=majority_vote_acc, color='gray', linestyle='-.', linewidth=2, label=f'Majority Vote ({majority_vote_acc:.2f})')

    plt.xlabel('Layer K')
    plt.ylabel('Test Accuracy')
    plt.title(f'Aligners {dataset} (Estimators={args.n_estimators})')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, pruning_utils.make_filename_safe(f"sweep_{dataset}.png"))
    plt.savefig(output_path)
    print(f"\nSweep plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Sweep pipeline over multiple layers")
    parser.add_argument("--dataset", type=str, nargs='+', default=["breast_cancer"])
    parser.add_argument("--student_n", type=int, nargs='+', default=[20], help="Student training sizes")
    parser.add_argument("--layers", type=int, nargs='+', default=[1, 2, 5, 8, 9, 10, 11])
    parser.add_argument("--per_token", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results/sweep_pipeline_layers")
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
