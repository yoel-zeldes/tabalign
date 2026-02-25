import argparse
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pruning_utils import load_data, fit_model
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

def run_patching_experiment(dataset_name, X_train, X_test, y_train, y_test, teacher, teacher_acc, student_n, n_estimators=8):
    print(f"Fitting Student model (first {student_n} examples)...")
    student_X_train = X_train[:student_n]
    student_y_train = y_train[:student_n]
    student = fit_model(student_X_train, student_y_train, n_estimators=n_estimators, assure_feature_tokens_are_static=True)
    
    student_acc = (student.predict(X_test) == y_test).mean()
    print(f"Dataset {dataset_name}, Student N={student_n} - Student Acc: {student_acc:.4f} (Teacher: {teacher_acc:.4f})")
    
    n_layers = len(student.executor_.models[0].transformer_encoder.layers)
    results = []

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
            preds = student.predict(X_test)
            acc = (preds == y_test).mean()
            
        for h in handles:
            h.remove()
            
        results.append(acc)

    return {
        "layers": list(range(n_layers)),
        "patched_accuracies": results,
        "teacher_acc": teacher_acc,
        "student_acc": student_acc,
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
    teacher_acc = (teacher.predict(X_test) == y_test).mean()
    print(f"Teacher Baseline Accuracy: {teacher_acc:.4f}")

    all_results = []
    plt.figure(figsize=(12, 7))
    
    student_n_list = {min(n, len(X_train)) for n in args.student_n}
    for student_n in tqdm(student_n_list, desc="Student sizes"):
        results = run_patching_experiment(
            args.dataset, X_train, X_test, y_train, y_test, 
            teacher, teacher_acc, student_n, n_estimators=args.n_estimators
        )
        all_results.append(results)
        
        line, = plt.plot(results["layers"], results["patched_accuracies"], marker='o', label=f'Patched Student (N={student_n})')
        color = line.get_color()
        plt.axhline(y=results["student_acc"], color=color, linestyle=':', alpha=0.5, label=f'Student Baseline (N={student_n}, {results["student_acc"]:.2f})')

    plt.axhline(y=teacher_acc, color='red', linestyle='--', linewidth=2, label=f'Teacher ({teacher_acc:.2f})')
    
    plt.xlabel('Patching Layer K')
    plt.ylabel('Test Accuracy')
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
