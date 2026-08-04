"""Orchestrator script for the tabular model activation alignment pipeline.

This pipeline trains an alignment network (aligner) to map intermediate layer activations
of a data-constrained "student" TabPFN model towards those of a full-data "teacher" model,
improving student prediction performance on downstream tasks.

Pipeline Stages:
1. Synthetic Test Data Generation (create_synthetic_dataset.py):
   Generates unlabeled synthetic feature vectors (X_test) derived from the training dataset's
   feature distributions using TabPFN unsupervised generative models.

2. Teacher Activation Extraction (extract_activations.py):
   Runs the teacher model (using full training context X_train, y_train) on the synthetic test
   feature vectors (X_test) to extract intermediate transformer layer activations at layer_k.

3. Student Activation Extraction (extract_activations.py):
   Runs the student model (using limited training context of student_n samples) on the synthetic
   test feature vectors (X_test) to extract intermediate layer activations at layer_k (skipped
   when using V2 aligner with loss_beta, which computes student activations dynamically).

4. Aligner Training (train_activation_aligner.py / train_activation_aligner_v2.py):
   Trains a linear or MLP aligner model to map student layer activations to teacher activations
   (or predict residuals). If `loss_beta` is specified, uses V2 aligner with combined MSE and
   KL-divergence loss; otherwise uses V1 aligner with MSE loss.

5. Evaluation (evaluate_aligned_student.py):
   Evaluates the student model with the trained aligner injected via forward hooks at the specified
   layer on a held-out real test dataset to measure performance improvements.
"""

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

def _extract_activations_path(dataset, student_n, layer_k, n_estimators, output_dir, repeat, use_feature_stats=False, model="tabpfn"):
    return create_filename_from_args({
        "dataset": dataset,
        "student_n": student_n,
        "layer_k": layer_k,
        "n_estimators": n_estimators,
        "repeat": repeat,
        "output_dir": output_dir,
        "use_feature_stats": use_feature_stats,
        "model": model,
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for aligner training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for aligner training.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples to generate.")
    parser.add_argument("--use_tabpfn", action="store_true", help="Use TabPFN to generate synthetic data.")
    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--force_create_synthetic_dataset", action="store_true", help="Force creating synthetic dataset")
    parser.add_argument("--force_extract", action="store_true", help="Force extracting activations")
    parser.add_argument("--force_train", action="store_true", help="Force training aligner")
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--use_feature_stats", action="store_true", help="Condition the aligner on per-feature statistics from the teacher's training data.")
    parser.add_argument("--loss_beta", type=float, default=None, help="If specified, use train_activation_aligner_v2 with this KL-divergence weight (MSE + loss_beta*KL). If unspecified, use the original train_activation_aligner (MSE only).")
    parser.add_argument("--clip_grad", type=float, default=None, help="Clip gradient norm to this value. None = no clipping.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of training epochs. None = unlimited (rely on patience).")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use (tabpfn or tabfm, default: tabpfn).")
    args = parser.parse_args()

    if args.model == "tabpfn" and args.n_estimators > 1:
        raise ValueError("multiple estimators are not supported for now")

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
        teacher_act_path = _extract_activations_path(
            synthetic_dataset,
            student_n=-1,
            layer_k=args.layer_k,
            n_estimators=args.n_estimators,
            output_dir=args.output_dir,
            repeat=args.repeat,
            use_feature_stats=args.use_feature_stats,
            model=args.model,
        )
        if args.force_extract or not os.path.exists(teacher_act_path):
            print(f"\n\n*****************\n\n>>> Step 2: Extracting Teacher Activations for '{dataset}'")
            cmd = [
                "extract_activations.py",
                "--dataset", synthetic_dataset,
                "--student_n", -1,
                "--layer_k", args.layer_k,
                "--n_estimators", args.n_estimators,
                "--output_dir", args.output_dir,
                "--repeat", args.repeat,
                "--model", args.model,
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
        if args.loss_beta is None:
            student_act_path = _extract_activations_path(
                synthetic_dataset,
                args.student_n,
                args.layer_k,
                args.n_estimators,
                args.output_dir,
                args.repeat,
                use_feature_stats=args.use_feature_stats,
                model=args.model,
            )
            if args.force_extract or not os.path.exists(student_act_path):
                print(f"\n\n*****************\n\n>>> Step 3: Extracting Student Activations for '{dataset}'")
                cmd = [
                    "extract_activations.py",
                    "--dataset", synthetic_dataset,
                    "--student_n", args.student_n,
                    "--layer_k", args.layer_k,
                    "--n_estimators", args.n_estimators,
                    "--output_dir", args.output_dir,
                    "--repeat", args.repeat,
                    "--model", args.model,
                ]
                if args.force_extract:
                    cmd.append("--force")
                if args.use_feature_stats:
                    cmd.append("--use_feature_stats")
                run_command(cmd)
            else:
                print(f">>> Step 3: Skipping (student activations already exist at {student_act_path})")
        else:
            print(f">>> Step 3: Skipping (loss_beta is None, no need for student activations)")

    # 4. Train Aligner (on all training datasets combined)
    train_args = [
        "--dataset", *synthetic_datasets,
        "--student_n", args.student_n,
        "--layer_k", args.layer_k,
        "--n_estimators", args.n_estimators,
        "--output_dir", args.output_dir,
        "--patience", args.patience,
        "--lr", args.lr,
        "--batch_size", args.batch_size,
        "--repeat", args.repeat,
        "--model", args.model,
    ]
    if args.loss_beta is not None:
        # Use V2 aligner (MSE + KL loss)
        print(f"\n\n*****************\n\n>>> Step 4: Training Aligner V2 (loss_beta={args.loss_beta}) on {len(synthetic_datasets)} dataset(s)")
        train_cmd = [
            "train_activation_aligner_v2.py",
            "--loss_beta", str(args.loss_beta),
            *train_args
        ]
        if args.clip_grad is not None:
            train_cmd.extend(["--clip_grad", str(args.clip_grad)])
    else:
        # Use original V1 aligner (MSE only)
        print(f"\n\n*****************\n\n>>> Step 4: Training Aligner on {len(synthetic_datasets)} dataset(s)")
        train_cmd = [
            "train_activation_aligner.py",
            *train_args
        ]
        if args.clip_grad is not None:
            raise ValueError("clip_grad is not supported for V1 aligner")
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
    if args.max_epochs is not None:
        train_cmd.extend(["--max_epochs", str(args.max_epochs)])
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
        "--lr", args.lr,
        "--batch_size", args.batch_size,
        "--repeat", args.repeat,
        "--model", args.model,
    ]
    if args.per_token:
        cmd.append("--per_token")
    if args.hidden_layers:
        cmd.extend(["--hidden_layers"] + args.hidden_layers)
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

    print("\n>>> Pipeline complete!")

if __name__ == "__main__":
    main()
