import argparse
import subprocess
import os
import sys

def run_command(cmd):
    full_cmd = [sys.executable] + [str(arg) for arg in cmd]
    print(f"Running: {' '.join(full_cmd)}")
    result = subprocess.run(full_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(full_cmd)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run activation alignment pipeline")
    parser.add_argument("--dataset", type=str, default="breast_cancer", help="Base dataset name")
    parser.add_argument("--student_n", type=int, default=10, help="Number of examples for student")
    parser.add_argument("--layer_k", type=int, default=2, help="Layer index to extract activations from")
    parser.add_argument("--n_estimators", type=int, default=8, help="Number of TabPFN estimators")
    parser.add_argument("--per_token", action="store_true", help="Use per-token alignment")
    parser.add_argument("--hidden_layers", type=int, nargs='*', default=[], help="Hidden layer sizes for MLP aligner. Empty = linear.")
    parser.add_argument("--patience", type=int, default=10, help="Stop aligner training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples to generate.")
    parser.add_argument("--use_tabpfn", action="store_true", help="Use TabPFN to generate synthetic data.")
    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--force_create_synthetic_dataset", action="store_true", help="Force creating synthetic dataset")
    parser.add_argument("--force_extract", action="store_true", help="Force extracting activations")
    parser.add_argument("--force_train", action="store_true", help="Force training aligner")
    args = parser.parse_args()

    # 1. Create Synthetic Dataset
    print(">>> Step 1: Creating Synthetic Dataset")
    create_cmd = [
        "create_synthetic_dataset.py",
        "--dataset", args.dataset,
        "--n_samples", args.n_samples,
        "--output_dir", args.output_dir
    ]
    if args.use_tabpfn:
        create_cmd.append("--use_tabpfn")
    if args.force_create_synthetic_dataset:
        args.force_extract = True
        args.force_train = True
        create_cmd.append("--force")
    run_command(create_cmd)
    
    synthetic_dataset = f"{args.dataset}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-use_tabpfn_{args.use_tabpfn}]"
    
    # 2. Extract Teacher Activations
    print("\n\n*****************\n\n>>> Step 2: Extracting Teacher Activations")
    cmd = [
        "extract_activations.py",
        "--dataset", synthetic_dataset,
        "--student_n", -1,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir
    ]
    if args.force_extract:
        args.force_train = True
        cmd.append("--force")
    run_command(cmd)

    # 3. Extract Student Activations
    print("\n\n*****************\n\n>>> Step 3: Extracting Student Activations")
    cmd = [
        "extract_activations.py",
        "--dataset", synthetic_dataset,
        "--student_n", args.student_n,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir
    ]
    if args.force_extract:
        cmd.append("--force")
    run_command(cmd)
    
    # 4. Train Aligner
    print("\n\n*****************\n\n>>> Step 4: Training Aligner")
    train_cmd = [
        "train_activation_aligner.py",
        "--dataset", synthetic_dataset,
        "--student_n", args.student_n,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir,
        "--patience", args.patience
    ]
    if args.per_token:
        train_cmd.append("--per_token")
    if args.hidden_layers:
        train_cmd.extend(["--hidden_layers"] + args.hidden_layers)
    if args.force_train:
        train_cmd.append("--force")
    run_command(train_cmd)

    # 5. Evaluate Aligned Student
    print("\n\n*****************\n\n>>> Step 5: Evaluating Aligned Student")
    cmd = [
        "evaluate_aligned_student.py",
        "--eval_dataset", args.dataset,
        "--train_dataset", synthetic_dataset,
        "--student_n", args.student_n,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir,
        "--patience", args.patience
    ]
    if args.per_token:
        cmd.append("--per_token")
    if args.hidden_layers:
        cmd.extend(["--hidden_layers"] + args.hidden_layers)
    run_command(cmd)

    print("\n>>> Pipeline complete!")

if __name__ == "__main__":
    main()
