"""Diagnostic activation patching experiment for TabPFN.

This module implements an **activation patching** (also known as causal tracing)
experiment that validates the core hypothesis behind the Activation Alignment
project: that injecting a teacher model's intermediate representations into a
data-constrained student model can recover teacher-level performance.

Motivation
----------
Before training a learned aligner (see ``train_activation_aligner.py``), this
script answers a prerequisite question: *if we had perfect alignment* (i.e. we
directly copy teacher activations into the student), how much performance could
we recover, and at which transformer layers does patching matter most?

The results produced here serve as an **upper bound** on what a trained aligner
can achieve and help identify the most impactful layers to target during aligner
training.

How It Works
------------
For each transformer layer *k* (0 ... L-1):

1. **Capture** — Run the teacher model (fitted on the full training set) on the
   test data and record the activations at the output of layer *k* using a
   forward hook (``capture_hook``).
2. **Patch** — Run the student model (fitted on a small subset of training
   examples) on the same test data, but replace its layer-*k* output with the
   captured teacher activations using a forward hook (``patch_hook``).
3. **Evaluate** — Measure the patched student's ROC-AUC (or -log_loss for
   multiclass) and accuracy on the test set.

By sweeping over all layers, the experiment produces a curve showing how
patched performance varies with layer depth, revealing where teacher and student
representations diverge most.
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pruning_utils import (
    load_data,
    fit_model,
    create_student_training_set,
    calculate_roc_auc,
    predict_from_probabilities,
    parse_student_n,
    resolve_student_n,
    make_filename_safe,
    TABARENA_NAME_TO_TASK_ID,
)
import experiment_utils


def capture_hook(module, input, output, captured_storage):
    """Hook to capture test example activations from teacher model."""
    captured_storage['test_activations'] = output.detach().clone()


def patch_hook(module, input, output, patch_tensor):
    """Hook to inject teacher activations into student model."""
    output.copy_(patch_tensor)
    return output


def run_patching_experiment(
    dataset_name,
    X_train,
    X_test,
    y_train,
    y_test,
    teacher,
    teacher_acc,
    teacher_roc_auc,
    student_n,
    n_estimators=8,
):
    resolved_n = resolve_student_n(student_n, len(X_train))
    print(f"Fitting Student model (N={student_n} -> {resolved_n} examples)...")
    student_X_train, student_y_train = create_student_training_set(X_train, y_train, student_n)
    student = fit_model(
        student_X_train,
        student_y_train,
        n_estimators=n_estimators,
        assure_feature_tokens_are_static=True,
    )
    
    student_probs = student.predict_proba(X_test)
    student_preds = predict_from_probabilities(student, student_probs)
    student_acc = float((student_preds == y_test).mean())
    student_roc_auc = float(calculate_roc_auc(y_test, student_probs))
        
    print(f"Dataset {dataset_name}, Student N={student_n} ({resolved_n} ex) - Student Acc: {student_acc:.4f} (AUC: {student_roc_auc:.4f}) | Teacher Acc: {teacher_acc:.4f} (AUC: {teacher_roc_auc:.4f})")
    
    n_layers = len(student.executor_.models[0].transformer_encoder.layers)
    patched_roc_aucs = []
    patched_accuracies = []

    # Patch at each layer K
    for k in tqdm(range(n_layers), desc=f"Patching layers (N={student_n})"):
        # 1. Capture teacher activations at layer K
        captured = {}
        handles = []
        for m_idx, m in enumerate(teacher.executor_.models):
            layer = m.transformer_encoder.layers[k]
            h = layer.register_forward_hook(
                lambda mod, inp, out, midx=m_idx: capture_hook(mod, inp, out, captured.setdefault(midx, {}))
            )
            handles.append(h)
            
        with torch.no_grad():
            teacher.predict_proba(X_test)
        
        for h in handles:
            h.remove()
            
        # 2. Patch into Student at layer K
        handles = []
        for m_idx, m in enumerate(student.executor_.models):
            layer = m.transformer_encoder.layers[k]
            pt = captured[m_idx]['test_activations']
            h = layer.register_forward_hook(
                lambda mod, inp, out, pt=pt: patch_hook(mod, inp, out, pt)
            )
            handles.append(h)
            
        with torch.no_grad():
            probs = student.predict_proba(X_test)
            preds = predict_from_probabilities(student, probs)
            patched_accuracies.append(float((preds == y_test).mean()))
            patched_roc_aucs.append(float(calculate_roc_auc(y_test, probs)))
            
        for h in handles:
            h.remove()

    return {
        "layers": list(range(n_layers)),
        "patched_roc_aucs": patched_roc_aucs,
        "patched_accuracies": patched_accuracies,
        "teacher_acc": teacher_acc,
        "teacher_roc_auc": teacher_roc_auc,
        "student_acc": student_acc,
        "student_roc_auc": student_roc_auc,
        "student_n": student_n,
        "resolved_student_n": resolved_n,
    }


def run_dataset_experiment(args, dataset):
    print(f"\n=== Dataset: {dataset} ===")
    X_train, X_test, y_train, y_test = load_data(
        dataset, repeat=args.repeat, max_num_examples=args.max_num_examples
    )

    n_unique_labels = len(np.unique(y_test))
    metric_label = "ROC AUC" if n_unique_labels == 2 else "-Log Loss"

    print(f"Fitting Teacher model (full training set: {len(X_train)} examples)...")
    teacher = fit_model(
        X_train, y_train, n_estimators=args.n_estimators,
        assure_feature_tokens_are_static=True
    )
    teacher_probs = teacher.predict_proba(X_test)
    teacher_preds = predict_from_probabilities(teacher, teacher_probs)
    teacher_acc = float((teacher_preds == y_test).mean())
    teacher_roc_auc = float(calculate_roc_auc(y_test, teacher_probs))

    majority_label = np.bincount(y_train).argmax()
    majority_vote_acc = float((majority_label == y_test).mean())
    majority_vote_probs = np.zeros((len(y_test), teacher_probs.shape[1]))
    majority_vote_probs[:, majority_label] = 1.0
    majority_vote_roc_auc = float(calculate_roc_auc(y_test, majority_vote_probs))

    print(f"Teacher Baseline Accuracy: {teacher_acc:.4f} ({metric_label}: {teacher_roc_auc:.4f})")
    print(f"Majority Vote Accuracy: {majority_vote_acc:.4f} ({metric_label}: {majority_vote_roc_auc:.4f})")

    all_results = []
    plt.figure(figsize=(12, 7))

    for student_n in tqdm(sorted(set(args.student_n)), desc="Student sizes"):
        results = run_patching_experiment(
            dataset_name=dataset,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            teacher=teacher,
            teacher_acc=teacher_acc,
            teacher_roc_auc=teacher_roc_auc,
            student_n=student_n,
            n_estimators=args.n_estimators,
        )
        all_results.append(results)

        line, = plt.plot(results["layers"], results["patched_roc_aucs"], marker='o', label=f'Patched Student (N={student_n})')
        color = line.get_color()
        plt.axhline(y=results["student_roc_auc"], color=color, linestyle=':', alpha=0.5, label=f'Student Baseline (N={student_n}, {results["student_roc_auc"]:.2f})')

    plt.axhline(y=teacher_roc_auc, color='red', linestyle='--', linewidth=2, label=f'Teacher ({teacher_roc_auc:.2f})')

    # Y-axis scaling that avoids distortion from extreme multiclass log-loss on majority vote
    all_scores = [teacher_roc_auc] + [r["student_roc_auc"] for r in all_results] + [val for r in all_results for val in r["patched_roc_aucs"]]
    min_score = min(all_scores)
    max_score = max(all_scores)
    margin = (max_score - min_score) * 0.1 if max_score > min_score else 0.05

    if majority_vote_roc_auc >= min_score - 2 * margin:
        plt.axhline(y=majority_vote_roc_auc, color='gray', linestyle='-.', linewidth=2, label=f'Majority Vote ({majority_vote_roc_auc:.2f})')
        plt.ylim(min(min_score - margin, majority_vote_roc_auc - margin / 2), max_score + margin)
    else:
        plt.ylim(min_score - margin, max_score + margin)

    plt.xlabel('Patching Layer K')
    plt.ylabel(f'Test {metric_label}')
    plt.title(f'Activation Patching: Teacher -> Student\n({dataset}, Estimators={args.n_estimators}, Repeat={args.repeat})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_dataset = make_filename_safe(dataset)
    repeat_suffix = f"_repeat_{args.repeat}" if args.repeat != 0 else ""
    base_name = f"patching_results_{safe_dataset}{repeat_suffix}"

    plot_path = os.path.join(args.output_dir, f"{base_name}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    print(f"Results plot saved to {plot_path}")

    dataset_json_path = os.path.join(args.output_dir, f"{base_name}.json")
    with open(dataset_json_path, "w") as f:
        json.dump({
            "config": vars(args),
            "dataset": dataset,
            "results": all_results,
        }, f, indent=4)
    print(f"Dataset results saved to {dataset_json_path}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Activation Patching Experiment for TabPFN")
    parser.add_argument("--dataset", type=str, nargs='+', default=["tabarena"],
                        help="Dataset(s) to evaluate on. E.g. 'tabarena/credit-g', 'tabarena', or multiple datasets.")
    parser.add_argument("--student_n", type=parse_student_n, nargs='+', default=[0.1, 0.5],
                        help="Number of training examples for student (ints >= 1 or fractions between 0 and 1).")
    parser.add_argument("--n_estimators", type=int, default=8,
                        help="Number of estimators.")
    parser.add_argument("--repeat", type=int, default=0,
                        help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--max_num_examples", type=int, default=1000,
                        help="Maximum number of training examples to load.")
    parser.add_argument("--output_dir", type=str, default="results/activation_patching")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    all_dataset_results = {}
    for dataset in (tqdm(datasets, desc="Datasets") if len(datasets) > 1 else datasets):
        all_dataset_results[dataset] = run_dataset_experiment(args, dataset)

    experiment_utils.save_results(config=vars(args), results=all_dataset_results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
