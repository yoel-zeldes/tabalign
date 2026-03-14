import argparse
import subprocess
import os
import sys
from pruning_utils import create_filename_from_args

def run_command(cmd):
    full_cmd = [sys.executable] + [str(arg) for arg in cmd]
    print(f"Running: {' '.join(full_cmd)}")
    result = subprocess.run(full_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(full_cmd)}")
        sys.exit(1)

def _create_synthetic_dataset_path(dataset, n_samples, output_dir, repeat, use_tabpfn):
    return create_filename_from_args({
        "dataset": dataset,
        "n_samples": n_samples,
        "output_dir": output_dir,
        "repeat": repeat,
        "use_tabpfn": use_tabpfn,
    }, script_name="create_synthetic_dataset", extension=".csv")

def _extract_activations_path(dataset, student_n, layer_k, n_estimators, output_dir, repeat, use_feature_stats=False):
    return create_filename_from_args({
        "dataset": dataset,
        "student_n": student_n,
        "layer_k": layer_k,
        "n_estimators": n_estimators,
        "repeat": repeat,
        "output_dir": output_dir,
        "use_feature_stats": use_feature_stats,
    }, script_name="extract_activations", extension=".pt")

def main():
    parser = argparse.ArgumentParser(description="Run activation alignment pipeline")
    parser.add_argument("--training_datasets", type=str, nargs='+', required=True,
                        help="One or more datasets to train the aligner on (synthetic data will be generated for each).")
    parser.add_argument("--test_dataset", type=str, required=True,
                        help="Dataset to evaluate the aligned student on (real data, no synthetic generation).")
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
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--use_feature_stats", action="store_true", help="Condition the aligner on per-feature statistics from the teacher's training data.")
    args = parser.parse_args()

    synthetic_datasets = [
        f"{dataset}[synthetic-n_samples_{args.n_samples}-output_dir_{args.output_dir}-repeat_{args.repeat}-use_tabpfn_{args.use_tabpfn}]"
        for dataset in args.training_datasets
    ]

    # Steps 1-3: For each training dataset, create synthetic data and extract activations
    for dataset, synthetic_dataset in zip(args.training_datasets, synthetic_datasets):
        # 1. Create Synthetic Dataset
        synthetic_path = _create_synthetic_dataset_path(dataset, args.n_samples, args.output_dir, args.repeat, args.use_tabpfn)
        if args.force_create_synthetic_dataset or not os.path.exists(synthetic_path):
            print(f"\n\n*****************\n\n>>> Step 1: Creating Synthetic Dataset for '{dataset}'")
            create_cmd = [
                "create_synthetic_dataset.py",
                "--dataset", dataset,
                "--n_samples", args.n_samples,
                "--output_dir", args.output_dir,
                "--repeat", args.repeat
            ]
            if args.use_tabpfn:
                create_cmd.append("--use_tabpfn")
            if args.force_create_synthetic_dataset:
                args.force_extract = True
                args.force_train = True
                create_cmd.append("--force")
            run_command(create_cmd)
        else:
            print(f">>> Step 1: Skipping (synthetic dataset already exists at {synthetic_path})")

        # 2. Extract Teacher Activations
        teacher_act_path = _extract_activations_path(synthetic_dataset, -1, args.layer_k, args.n_estimators, args.output_dir, args.repeat, use_feature_stats=args.use_feature_stats)
        if args.force_extract or not os.path.exists(teacher_act_path):
            print(f"\n\n*****************\n\n>>> Step 2: Extracting Teacher Activations for '{dataset}'")
            cmd = [
                "extract_activations.py",
                "--dataset", synthetic_dataset,
                "--student_n", -1,
                "--layer_k", args.layer_k,
                "--n_estimators", args.n_estimators,
                "--output_dir", args.output_dir,
                "--repeat", args.repeat
            ]
            if args.force_extract:
                args.force_train = True
                cmd.append("--force")
            if args.use_feature_stats:
                cmd.append("--use_feature_stats")
            run_command(cmd)
        else:
            print(f">>> Step 2: Skipping (teacher activations already exist at {teacher_act_path})")

        # 3. Extract Student Activations
        student_act_path = _extract_activations_path(synthetic_dataset, args.student_n, args.layer_k, args.n_estimators, args.output_dir, args.repeat, use_feature_stats=args.use_feature_stats)
        if args.force_extract or not os.path.exists(student_act_path):
            print(f"\n\n*****************\n\n>>> Step 3: Extracting Student Activations for '{dataset}'")
            cmd = [
                "extract_activations.py",
                "--dataset", synthetic_dataset,
                "--student_n", args.student_n,
                "--layer_k", args.layer_k,
                "--n_estimators", args.n_estimators,
                "--output_dir", args.output_dir,
                "--repeat", args.repeat
            ]
            if args.force_extract:
                cmd.append("--force")
            if args.use_feature_stats:
                cmd.append("--use_feature_stats")
            run_command(cmd)
        else:
            print(f">>> Step 3: Skipping (student activations already exist at {student_act_path})")

    # 4. Train Aligner (on all training datasets combined)
    print(f"\n\n*****************\n\n>>> Step 4: Training Aligner on {len(synthetic_datasets)} dataset(s)")
    train_cmd = [
        "train_activation_aligner.py",
        "--dataset", *synthetic_datasets,
        "--student_n", args.student_n,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir,
        "--patience", args.patience,
        "--repeat", args.repeat
    ]
    if args.per_token:
        train_cmd.append("--per_token")
    if args.hidden_layers:
        train_cmd.extend(["--hidden_layers"] + args.hidden_layers)
    if args.predict_residual:
        train_cmd.append("--predict_residual")
    if args.use_feature_stats:
        train_cmd.append("--use_feature_stats")
    if args.force_train:
        train_cmd.append("--force")
    run_command(train_cmd)

    # 5. Evaluate Aligned Student on the test dataset
    print(f"\n\n*****************\n\n>>> Step 5: Evaluating Aligned Student on '{args.test_dataset}'")
    cmd = [
        "evaluate_aligned_student.py",
        "--eval_dataset", args.test_dataset,
        "--train_dataset", *synthetic_datasets,
        "--student_n", args.student_n,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir,
        "--patience", args.patience,
        "--repeat", args.repeat
    ]
    if args.per_token:
        cmd.append("--per_token")
    if args.hidden_layers:
        cmd.extend(["--hidden_layers"] + args.hidden_layers)
    if args.predict_residual:
        cmd.append("--predict_residual")
    if args.use_feature_stats:
        cmd.append("--use_feature_stats")
    run_command(cmd)

    print("\n>>> Pipeline complete!")

if __name__ == "__main__":
    main()
