import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pruning_utils import load_data, fit_model, create_student_training_set, calculate_roc_auc, predict_from_probabilities
import experiment_utils

def capture_hook(module, input, output, captured_storage, eval_pos):
    """
    Hook to capture test example activations from teacher model.
    """
    captured_storage['test_activations'] = output.detach().clone()

def patch_hook(module, input, output, patch_tensor):
    """
    Hook to inject test example activations into student model.
    """
    output.copy_(patch_tensor)
    return output

def run_patching_experiment(dataset_name, X_train, X_test, y_train, y_test, teacher, teacher_acc, teacher_roc_auc, student_n, n_estimators=8):
    print(f"Fitting Student model (first {student_n} examples)...")
    student_X_train, student_y_train = create_student_training_set(X_train, y_train, student_n)
    student = fit_model(student_X_train, student_y_train, n_estimators=n_estimators, assure_feature_tokens_are_static=True)
    
    student_probs = student.predict_proba(X_test)
    student_preds = predict_from_probabilities(student, student_probs)
    student_acc = (student_preds == y_test).mean()
    
    student_roc_auc = calculate_roc_auc(y_test, student_probs)
        
    print(f"Dataset {dataset_name}, Student N={student_n} - Student Acc: {student_acc:.4f} (AUC: {student_roc_auc:.4f}) | Teacher Acc: {teacher_acc:.4f} (AUC: {teacher_roc_auc:.4f})")
    
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
                lambda mod, inp, out, midx=m_idx: capture_hook(mod, inp, out, captured.setdefault(midx, {}), len(X_train))
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
            patched_accuracies.append((preds == y_test).mean())
            patched_roc_aucs.append(calculate_roc_auc(y_test, probs))
            
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
        "student_n": student_n
    }

def main():
    parser = argparse.ArgumentParser(description="Activation Patching Experiment for TabPFN")
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--student_n", type=int, nargs='+', default=[10], help="Number of training examples for student (can provide multiple)")
    parser.add_argument("--n_estimators", type=int, default=8, help="Number of TabPFN estimators")
    parser.add_argument("--output_dir", type=str, default="results/activation_patching")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=== Dataset: {args.dataset} ===")
    X_train, X_test, y_train, y_test = load_data(args.dataset)
    
    print("Fitting Teacher model (full training set)...")
    teacher = fit_model(X_train, y_train, n_estimators=args.n_estimators, assure_feature_tokens_are_static=True)
    teacher_probs = teacher.predict_proba(X_test)
    teacher_preds = predict_from_probabilities(teacher, teacher_probs)
    teacher_acc = (teacher_preds == y_test).mean()
    
    teacher_roc_auc = calculate_roc_auc(y_test, teacher_probs)
        
    majority_label = np.bincount(y_train).argmax()
    majority_vote_acc = float((majority_label == y_test).mean())
    majority_vote_probs = np.zeros((len(y_test), teacher_probs.shape[1]))
    majority_vote_probs[:, majority_label] = 1.0
    majority_vote_roc_auc = calculate_roc_auc(y_test, majority_vote_probs)

    print(f"Teacher Baseline Accuracy: {teacher_acc:.4f} (AUC: {teacher_roc_auc:.4f})")
    print(f"Majority Vote Accuracy: {majority_vote_acc:.4f} (AUC: {majority_vote_roc_auc:.4f})")

    all_results = []
    plt.figure(figsize=(12, 7))
    
    student_n_list = sorted({min(n, len(X_train)) for n in args.student_n})
    for student_n in tqdm(student_n_list, desc="Student sizes"):
        results = run_patching_experiment(
            args.dataset, X_train, X_test, y_train, y_test, 
            teacher, teacher_acc, teacher_roc_auc, student_n, n_estimators=args.n_estimators
        )
        all_results.append(results)
        
        line, = plt.plot(results["layers"], results["patched_roc_aucs"], marker='o', label=f'Patched Student (N={student_n})')
        color = line.get_color()
        plt.axhline(y=results["student_roc_auc"], color=color, linestyle=':', alpha=0.5, label=f'Student Baseline (N={student_n}, {results["student_roc_auc"]:.2f})')

    plt.axhline(y=teacher_roc_auc, color='red', linestyle='--', linewidth=2, label=f'Teacher ({teacher_roc_auc:.2f})')
    plt.axhline(y=majority_vote_roc_auc, color='gray', linestyle='-.', linewidth=2, label=f'Majority Vote ({majority_vote_roc_auc:.2f})')
    
    plt.xlabel('Patching Layer K')
    plt.ylabel('Test ROC AUC')
    plt.title(f'Activation Patching: Teacher -> Student\n({args.dataset}, Estimators={args.n_estimators})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, f"patching_results_{args.dataset}.png")
    plt.savefig(plot_path)
    print(f"Results plot saved to {plot_path}")
    
    experiment_utils.save_results(config=vars(args), results=all_results, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
