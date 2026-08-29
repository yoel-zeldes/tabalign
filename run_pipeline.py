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
   test feature vectors (X_test) to extract intermediate layer activations at layer_k.

4. Aligner Training (train_activation_aligner.py):
   Trains a linear or MLP aligner model to map student layer activations to teacher activations
   (or predict residuals).

5. Evaluation (evaluate_aligned_student.py):
   Evaluates the student model with the trained aligner injected via forward hooks at the specified
   layer on a held-out real test dataset to measure performance improvements.
"""

import argparse
from pruning_utils import parse_student_n
from evaluate_aligned_student import evaluate_aligned_student


def run_pipeline(
    dataset,
    student_n=10,
    layer_k=2,
    n_estimators=8,
    n_samples=10000,
    repeat=0,
    model="tabpfn",
    patience=10,
    lr=1e-3,
    batch_size=2048,
    hidden_layers=None,
    max_epochs=None,
    aligner_opt=False,
):
    synthetic_dataset = (
        f"{dataset}[synthetic-n_samples_{n_samples}-repeat_{repeat}]"
    )

    return evaluate_aligned_student(
        eval_dataset=dataset,
        train_dataset=synthetic_dataset,
        student_n=student_n,
        layer_k=layer_k,
        n_estimators=n_estimators,
        patience=patience,
        lr=lr,
        batch_size=batch_size,
        hidden_layers=hidden_layers,
        repeat=repeat,
        max_epochs=max_epochs,
        model=model,
        aligner_opt=aligner_opt,
    )


def main():
    parser = argparse.ArgumentParser(description="Run activation alignment pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name for training aligner on and evaluating on the test subset.")
    parser.add_argument("--student_n", type=parse_student_n, default=10, help="Number of examples for student")
    parser.add_argument("--layer_k", type=int, default=2, help="Layer index to extract activations from")
    parser.add_argument("--n_estimators", type=int, default=8, help="Number of TabPFN estimators")
    parser.add_argument("--hidden_layers", type=int, nargs='*', default=[], help="Hidden layer multipliers for MLP aligner. Empty = linear.")
    parser.add_argument("--patience", type=int, default=10, help="Stop aligner training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for aligner training.")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for aligner training.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of synthetic samples to generate.")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of training epochs. None = unlimited (rely on patience).")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use (tabpfn or tabfm, default: tabpfn).")
    parser.add_argument("--aligner_opt", "--opt", action="store_true", dest="aligner_opt", help="Use hyperparameter optimization when training aligner.")
    args = parser.parse_args()

    result = run_pipeline(
        dataset=args.dataset,
        student_n=args.student_n,
        layer_k=args.layer_k,
        n_estimators=args.n_estimators,
        n_samples=args.n_samples,
        repeat=args.repeat,
        model=args.model,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_layers=args.hidden_layers,
        max_epochs=args.max_epochs,
        aligner_opt=args.aligner_opt,
    )
    print("\n>>> Pipeline complete!")

if __name__ == "__main__":
    main()
