import os
import json
import argparse
import torch
import torch.nn as nn
from pruning_utils import load_data, fit_model

class AlignedHook:
    def __init__(self, aligner_model):
        self.aligner_model = aligner_model
    
    def __call__(self, module, input, output):
        # output shape: (batch, tokens, hidden)
        orig_shape = output.shape
        x = output.view(-1, orig_shape[-1])
        aligned_x = self.aligner_model(x)
        return aligned_x.view(orig_shape)

def get_predictions(model, X_test, aligner_models=None, layer_k=None):
    handles = []
    if aligner_models is not None:
        for estimator_idx, estimator in enumerate(model.executor_.models):
            layer = estimator.transformer_encoder.layers[layer_k]
            hook = AlignedHook(aligner_models[estimator_idx])
            handles.append(layer.register_forward_hook(hook))
    
    with torch.no_grad():
        preds = model.predict(X_test)
        
    for h in handles:
        h.remove()
    return preds

def validate_metadata(metadata, dataset, student_n, layer_k, n_estimators):
    """Ensure aligner metadata matches the evaluation configuration."""
    if metadata["dataset"] != dataset:
         raise ValueError(f"Dataset mismatch: Aligner trained for {metadata['dataset']} but evaluating on {dataset}")
    if metadata["student_n"] != student_n:
         raise ValueError(f"Student N mismatch: Aligner trained for N={metadata['student_n']} but evaluating on N={student_n}")
    if metadata["layer_k"] != layer_k:
         raise ValueError(f"Layer K mismatch: Aligner trained for K={metadata['layer_k']} but evaluating on K={layer_k}")
    if metadata["n_estimators"] != n_estimators:
         raise ValueError(f"Estimators mismatch: Aligner has {metadata['n_estimators']} models but evaluating with n_estimators={n_estimators}")

def load_aligner_models(aligner_data):
    """Load linear aligner models from saved state dicts."""
    aligner_models = {}
    for est_idx, s_dict in aligner_data["estimator_idx_to_aligner"].items():
        hidden_dim = s_dict["weight"].shape[0]
        model = nn.Linear(hidden_dim, hidden_dim)
        model.load_state_dict(s_dict)
        model.eval()
        aligner_models[est_idx] = model
    return aligner_models

def calc_metrics(teacher_preds, baseline_preds, aligned_preds, y_test):
    """Calculate metrics and return them as a dictionary."""
    baseline_fidelity = (baseline_preds == teacher_preds).mean()
    aligned_fidelity = (aligned_preds == teacher_preds).mean()
    
    metrics = {
        "baseline_fidelity": float(baseline_fidelity),
        "aligned_fidelity": float(aligned_fidelity)
    }
    
    teacher_acc = (teacher_preds == y_test).mean()
    baseline_acc = (baseline_preds == y_test).mean()
    aligned_acc = (aligned_preds == y_test).mean()
    
    metrics.update({
        "teacher_acc": float(teacher_acc),
        "baseline_acc": float(baseline_acc),
        "aligned_acc": float(aligned_acc)
    })
    
    print(f"\nResults (Accuracy vs Real Labels):")
    print(f"Teacher: {teacher_acc:.4f}")
    print(f"Baseline Student: {baseline_acc:.4f}")
    print(f"Aligned Student: {aligned_acc:.4f}")
        
    return metrics

def save_results(args, metrics):
    """Save metrics and configuration to a JSON file."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_data = {
        "config": vars(args),
        "metrics": metrics
    }
    
    safe_name = args.dataset.replace(" ", "_")
    filename = f"results_{safe_name}_N{args.student_n}_K{args.layer_k}_E{args.n_estimators}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=4)
    
    return filepath

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate aligned student model")
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--student_n", type=int, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--aligner_path", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading aligner models from {args.aligner_path}...")
    aligner_data = torch.load(args.aligner_path)
    
    validate_metadata(aligner_data["metadata"], args.dataset, args.student_n, args.layer_k, args.n_estimators)
    
    print("Loading data and fitting models...")
    X_train, X_test, y_train, y_test = load_data(args.dataset)
    
    print("Fitting Teacher model for reference...")
    teacher = fit_model(X_train, y_train, n_estimators=args.n_estimators, fingerprint=False, assure_num_tokens_is_static=True)
    teacher_preds = teacher.predict(X_test)
    
    print(f"Fitting Student model (N={args.student_n}, E={args.n_estimators})...")
    student_X_train = X_train[:args.student_n]
    student_y_train = y_train[:args.student_n]
    student = fit_model(student_X_train, student_y_train, n_estimators=args.n_estimators, fingerprint=False, assure_num_tokens_is_static=True)
    
    print("Evaluating Baseline Student...")
    baseline_preds = get_predictions(student, X_test)
    
    print("Evaluating Aligned Student...")
    aligned_preds = get_predictions(student, X_test, load_aligner_models(aligner_data), args.layer_k)
    
    metrics = calc_metrics(teacher_preds, baseline_preds, aligned_preds, y_test)
    
    save_path = save_results(args, metrics)
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()
