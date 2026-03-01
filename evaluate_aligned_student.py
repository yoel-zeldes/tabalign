import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pruning_utils import load_data, fit_model, create_filename_from_args, create_student_training_set

class AlignedHook:
    def __init__(self, aligner_model, per_token=False):
        self.aligner_model = aligner_model
        self.per_token = per_token
    
    def __call__(self, module, input, output):
        # output shape: (1, batch, tokens, hidden)
        if not self.per_token:
            orig_shape = output.shape
            x = output.view(-1, orig_shape[-1])
            aligned_x = self.aligner_model(x)
            return aligned_x.view(orig_shape)
        else:
            # self.aligner_model is a dict: {token_idx: nn.Linear}
            # Iterate over the token dimension (index 2)
            aligned_token_list = []
            for token_idx in range(output.shape[2]):
                token_activations = output[:, :, token_idx, :]  # shape: (1, batch, tokens, hidden)
                aligned_token = self.aligner_model[token_idx](token_activations)
                aligned_token_list.append(aligned_token.unsqueeze(2))
            return torch.cat(aligned_token_list, dim=2)

def get_predictions(model, X_test, aligner_models=None, layer_k=None, per_token=False):
    handles = []
    if aligner_models is not None:
        for estimator_idx, estimator in enumerate(model.executor_.models):
            layer = estimator.transformer_encoder.layers[layer_k]
            hook = AlignedHook(aligner_models[estimator_idx], per_token=per_token)
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

def _create_aligner_model(state_dict):
    """Create and initialize a linear aligner model from a state dict."""
    hidden_dim = state_dict["weight"].shape[0]
    model = nn.Linear(hidden_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_aligner_models(aligner_data):
    """Load linear aligner models from saved state dicts."""
    aligner_models = {}
    per_token = aligner_data["metadata"]["per_token"]
    
    for est_idx, data in aligner_data["estimator_idx_to_aligner"].items():
        if per_token:
            aligner_models[est_idx] = {
                token_idx: _create_aligner_model(s_dict)
                for token_idx, s_dict in data.items()
            }
        else:
            aligner_models[est_idx] = _create_aligner_model(data)
            
    return aligner_models, per_token

def calc_metrics(teacher_preds, baseline_preds, aligned_preds, y_train, y_test):
    """Calculate metrics and return them as a dictionary."""
    baseline_fidelity = (baseline_preds == teacher_preds).mean()
    aligned_fidelity = (aligned_preds == teacher_preds).mean()
    
    majority_label = np.bincount(y_train).argmax()
    majority_vote_acc = (majority_label == y_test).mean()
    
    metrics = {
        "majority_vote_acc": float(majority_vote_acc),
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
    print(f"Majority Vote: {majority_vote_acc:.4f}")
    print(f"Teacher: {teacher_acc:.4f}")
    print(f"Baseline Student: {baseline_acc:.4f}")
    print(f"Aligned Student: {aligned_acc:.4f}")
        
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate aligned student model")
    parser.add_argument("--eval_dataset", type=str, default="breast_cancer")
    parser.add_argument("--train_dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument("--student_n", type=int, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--aligners_dir", type=str, default="results/aligners")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--per_token", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    return parser.parse_args()

def _warn_if_constant_predictions(model_name, preds, y_train):
    if len(np.unique(preds)) == 1:
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"WARNING: {model_name} predicts the same class ({preds[0]}) for all {len(preds)} test examples! Train labels: {train_dist}")

def main():
    args = parse_args()

    aligner_path = create_filename_from_args({
        "dataset": args.train_dataset,
        "student_n": args.student_n,
        "layer_k": args.layer_k,
        "n_estimators": args.n_estimators,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "per_token": args.per_token,
        "output_dir": args.aligners_dir
    }, script_name="train_activation_aligner", exclude_args=["aligners_dir"], extension=".pt")

    print(f"Loading aligner models from {aligner_path}...")
    aligner_data = torch.load(aligner_path)
    
    validate_metadata(aligner_data["metadata"], args.train_dataset, args.student_n, args.layer_k, args.n_estimators)
    
    print("Loading data and fitting models...")
    X_train, X_test, y_train, y_test = load_data(args.eval_dataset)
    
    print("Fitting Teacher model for reference...")
    teacher = fit_model(X_train, y_train, n_estimators=args.n_estimators, assure_feature_tokens_are_static=True)
    teacher_preds = teacher.predict(X_test)
    
    print(f"Fitting Student model (N={args.student_n}, E={args.n_estimators})...")
    student_X_train, student_y_train = create_student_training_set(X_train, y_train, args.student_n)
    student = fit_model(student_X_train, student_y_train, n_estimators=args.n_estimators, assure_feature_tokens_are_static=True)
    
    print("Evaluating Baseline Student...")
    baseline_preds = get_predictions(student, X_test)
    _warn_if_constant_predictions("Baseline student", baseline_preds, student_y_train)
    
    print("Evaluating Aligned Student...")
    aligner_models, per_token = load_aligner_models(aligner_data)
    aligned_preds = get_predictions(student, X_test, aligner_models, args.layer_k, per_token=per_token)
    _warn_if_constant_predictions("Aligned student", aligned_preds, student_y_train)
    
    metrics = calc_metrics(teacher_preds, baseline_preds, aligned_preds, y_train, y_test)
    
    output_data = {
        "config": vars(args),
        "metrics": metrics
    }
    filepath = create_filename_from_args(args, extension=".json", makedirs=True)
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nResults saved to {filepath}")
    
if __name__ == "__main__":
    main()
