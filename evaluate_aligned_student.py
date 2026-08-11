import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pruning_utils import load_data, fit_model, create_filename_from_args, create_student_training_set, calculate_roc_auc, predict_from_probabilities, append_feature_stats, get_device, fill_nans, parse_student_n
from train_activation_aligner import build_aligner_model
from extract_activations import compute_feature_stats

class AlignedHook:
    def __init__(self, aligner_model, per_token=False, predict_residual=False, feature_stats=None):
        self.aligner_model = aligner_model
        self.per_token = per_token
        self.predict_residual = predict_residual
        self.feature_stats = feature_stats  # [n_features, n_stats] or None
    
    def __call__(self, module, input, output):
        # output shape: (1, batch, tokens, hidden) in tabpfn and (1, batch, hidden) in tabfm
        if not self.per_token:
            orig_shape = output.shape
            x = output.view(-1, orig_shape[-1]).float()
            if self.feature_stats is not None:
                x = append_feature_stats(x, self.feature_stats)
            result = self.aligner_model(x).to(output.dtype).view(orig_shape)
        else:
            # self.aligner_model is a dict: {token_idx: model}
            # Iterate over the token dimension (index 2)
            token_result_list = []
            for token_idx in range(output.shape[2]):
                token_activations = output[:, :, token_idx, :].float()  # shape: (1, batch, hidden)
                result = self.aligner_model[token_idx](token_activations).to(output.dtype)
                token_result_list.append(result.unsqueeze(2))
            result = torch.cat(token_result_list, dim=2)
        
        if self.predict_residual:
            result += output
        return result

def get_predictions_and_probabilities(model, X_test, aligner_models=None, layer_k=None, per_token=False, predict_residual=False, feature_stats=None, model_type="tabpfn"):
    handles = []
    if aligner_models is not None:
        if model_type == "tabfm":
            layer = model.model.icl_predictor.tf_icl.blocks[layer_k]
            hook = AlignedHook(aligner_models[0], per_token=per_token, predict_residual=predict_residual, feature_stats=feature_stats)
            handles.append(layer.register_forward_hook(hook))
        else:
            for estimator_idx, estimator in enumerate(model.executor_.models):
                layer = estimator.transformer_encoder.layers[layer_k]
                hook = AlignedHook(aligner_models[estimator_idx], per_token=per_token, predict_residual=predict_residual, feature_stats=feature_stats)
                handles.append(layer.register_forward_hook(hook))
    
    with torch.no_grad():
        probs = model.predict_proba(X_test)
        preds = predict_from_probabilities(model, probs)
        
    for h in handles:
        h.remove()
    return preds, probs

def validate_metadata(metadata, train_datasets, student_n, layer_k, n_estimators):
    """Ensure aligner metadata matches the evaluation configuration."""
    students_metadata = metadata.get("students_metadata")
    trained_datasets = [m["dataset"] for m in students_metadata]
    if trained_datasets != list(train_datasets):
        raise ValueError(f"Dataset mismatch: Aligner trained on {trained_datasets} but got {list(train_datasets)}")
    ref = students_metadata[0]
    if ref["student_n"] != student_n:
         raise ValueError(f"Student N mismatch: Aligner trained for N={ref['student_n']} but evaluating on N={student_n}")
    if ref["layer_k"] != layer_k:
         raise ValueError(f"Layer K mismatch: Aligner trained for K={ref['layer_k']} but evaluating on K={layer_k}")
    if ref["n_estimators"] != n_estimators:
         raise ValueError(f"Estimators mismatch: Aligner has {ref['n_estimators']} models but evaluating with n_estimators={n_estimators}")

def _create_aligner_model(state_dict, hidden_layers=None, predict_residual=False, n_stats=0):
    """Create and initialize an aligner model from a state dict."""
    if hidden_layers is None:
        hidden_layers = []
    first_key = next(k for k in state_dict if 'weight' in k)
    input_dim = state_dict[first_key].shape[1]
    model = build_aligner_model(
        input_dim=input_dim,
        output_dim=input_dim - n_stats,
        hidden_layers=hidden_layers,
        predict_residual=predict_residual
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_aligner_models(aligner_data):
    """Load aligner models from saved state dicts."""
    aligner_models = {}
    per_token = aligner_data["metadata"]["per_token"]
    hidden_layers = aligner_data["metadata"].get("hidden_layers", [])
    predict_residual = aligner_data["metadata"].get("predict_residual", False)
    n_stats = aligner_data["metadata"].get("n_stats", 0)
    
    for est_idx, data in aligner_data["estimator_idx_to_aligner"].items():
        if per_token:
            aligner_models[est_idx] = {
                token_idx: _create_aligner_model(s_dict, hidden_layers, predict_residual=predict_residual)
                for token_idx, s_dict in data.items()
            }
        else:
            aligner_models[est_idx] = _create_aligner_model(data, hidden_layers, predict_residual=predict_residual, n_stats=n_stats)
            
    return aligner_models, per_token, predict_residual

def calc_metrics(teacher_preds, baseline_preds, aligned_preds, teacher_probs, baseline_probs, aligned_probs, y_train, y_test):
    """Calculate metrics and return them as a dictionary."""
    baseline_fidelity = (baseline_preds == teacher_preds).mean()
    aligned_fidelity = (aligned_preds == teacher_preds).mean()
    
    majority_label = np.bincount(y_train).argmax()
    majority_vote_acc = (majority_label == y_test).mean()
    
    metrics = {
        "n_unique_labels": int(len(np.unique(y_test))),
        "majority_vote_acc": float(majority_vote_acc),
        "baseline_fidelity": float(baseline_fidelity),
        "aligned_fidelity": float(aligned_fidelity)
    }
    
    teacher_acc = (teacher_preds == y_test).mean()
    baseline_acc = (baseline_preds == y_test).mean()
    aligned_acc = (aligned_preds == y_test).mean()
    
    teacher_roc_auc = calculate_roc_auc(y_test, teacher_probs)
    baseline_roc_auc = calculate_roc_auc(y_test, baseline_probs)
    aligned_roc_auc = calculate_roc_auc(y_test, aligned_probs)

    majority_vote_probs = np.zeros((len(y_test), teacher_probs.shape[1]))
    majority_vote_probs[:, majority_label] = 1.0
    majority_vote_roc_auc = calculate_roc_auc(y_test, majority_vote_probs)

    metrics.update({
        "teacher_acc": float(teacher_acc),
        "baseline_acc": float(baseline_acc),
        "aligned_acc": float(aligned_acc),
        "teacher_roc_auc": float(teacher_roc_auc),
        "baseline_roc_auc": float(baseline_roc_auc),
        "aligned_roc_auc": float(aligned_roc_auc),
        "majority_vote_roc_auc": float(majority_vote_roc_auc)
    })
    
    print(f"\nResults (Accuracy vs Real Labels):")
    print(f"Majority Vote: {majority_vote_acc:.4f} (AUC: {majority_vote_roc_auc:.4f})")
    print(f"Teacher: {teacher_acc:.4f} (AUC: {teacher_roc_auc:.4f})")
    print(f"Baseline Student: {baseline_acc:.4f} (AUC: {baseline_roc_auc:.4f})")
    print(f"Aligned Student: {aligned_acc:.4f} (AUC: {aligned_roc_auc:.4f})")
        
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate aligned student model")
    parser.add_argument("--eval_dataset", type=str, default="breast_cancer")
    parser.add_argument("--train_dataset", type=str, nargs='+', default=["tabarena/Amazon_employee_access[synthetic]"],
                        help="One or more synthetic dataset names the aligner was trained on.")
    parser.add_argument("--student_n", type=parse_student_n, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--per_token", action="store_true")
    parser.add_argument("--hidden_layers", type=int, nargs='*', default=[], help="Hidden layer sizes for MLP aligner. Empty = linear.")
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--use_feature_stats", action="store_true", help="Use per-feature statistics conditioning (must match how the aligner was trained).")
    parser.add_argument("--loss_beta", type=float, default=None, help="If specified, look up aligner trained by v2 with this KL weight. If unspecified, look up the original v1 aligner.")
    parser.add_argument("--clip_grad", type=float, default=None, help="Clip gradient norm. None = no clipping.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of training epochs. None = unlimited (rely on patience).")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use.")
    return parser.parse_args()

def _warn_if_constant_predictions(model_name, preds, y_train):
    if len(np.unique(preds)) == 1:
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"WARNING: {model_name} predicts the same class ({preds[0]}) for all {len(preds)} test examples! Train labels: {train_dist}")

def main():
    args = parse_args()

    aligner_args = {
        "dataset": args.train_dataset,
        "student_n": args.student_n,
        "layer_k": args.layer_k,
        "n_estimators": args.n_estimators,
        "patience": args.patience,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_layers": args.hidden_layers,
        "predict_residual": args.predict_residual,
        "repeat": args.repeat,
        "output_dir": args.output_dir,
        "use_feature_stats": args.use_feature_stats,
        "max_epochs": args.max_epochs,
        "model": args.model,
        "per_token": args.per_token
    }
    if args.loss_beta is not None:
        aligner_args["loss_beta"] = args.loss_beta
        aligner_script_name = "train_activation_aligner_v2"
        aligner_args["clip_grad"] = args.clip_grad
    else:
        aligner_script_name = "train_activation_aligner"
        if args.clip_grad is not None:
            raise ValueError("clip_grad is not supported for V1 aligner")

    aligner_path = create_filename_from_args(
        aligner_args, script_name=aligner_script_name, extension=".pt"
    )

    print(f"Loading aligner models from {aligner_path}...")
    aligner_data = torch.load(aligner_path)
    
    validate_metadata(aligner_data["metadata"], args.train_dataset, args.student_n, args.layer_k, args.n_estimators)

    print("Loading data and fitting models...")
    X_train, X_test, y_train, y_test, cat_indices = load_data(args.eval_dataset, repeat=args.repeat, return_cat_indices=True)

    if args.model == "tabfm":
        X_test = fill_nans(X_test)

    if args.use_feature_stats and args.model != "tabfm":
        feature_stats = compute_feature_stats(X_train, y_train, cat_indices=cat_indices)
    else:
        feature_stats = None
    
    print("Fitting Teacher model for reference...")
    teacher = fit_model(X_train, y_train, n_estimators=args.n_estimators, assure_feature_tokens_are_static=True,
                        token_per_feature=args.use_feature_stats, model=args.model)
    teacher_probs = teacher.predict_proba(X_test)
    teacher_preds = predict_from_probabilities(teacher, teacher_probs)
    
    print(f"Fitting Student model (N={args.student_n}, E={args.n_estimators})...")
    student_X_train, student_y_train = create_student_training_set(X_train, y_train, args.student_n)
    student = fit_model(student_X_train, student_y_train, n_estimators=args.n_estimators, assure_feature_tokens_are_static=True,
                        token_per_feature=args.use_feature_stats, model=args.model,
                        model_preprocessor=teacher if args.model == "tabfm" else None)
    
    print("Evaluating Baseline Student...")
    baseline_preds, baseline_probs = get_predictions_and_probabilities(student, X_test, model_type=args.model)
    _warn_if_constant_predictions("Baseline student", baseline_preds, student_y_train)
    
    print("Evaluating Aligned Student...")
    aligner_models, per_token, predict_residual = load_aligner_models(aligner_data)
    device = get_device()
    for est_idx, model_or_dict in aligner_models.items():
        if isinstance(model_or_dict, dict):
            for k in model_or_dict:
                model_or_dict[k] = model_or_dict[k].to(device)
        else:
            aligner_models[est_idx] = model_or_dict.to(device)
    aligned_preds, aligned_probs = get_predictions_and_probabilities(
        student, X_test, aligner_models, args.layer_k,
        per_token=per_token, predict_residual=predict_residual, feature_stats=feature_stats,
        model_type=args.model
    )
    _warn_if_constant_predictions("Aligned student", aligned_preds, student_y_train)
    
    metrics = calc_metrics(teacher_preds, baseline_preds, aligned_preds, teacher_probs, baseline_probs, aligned_probs, y_train, y_test)
    
    output_data = {
        "config": vars(args),
        "metrics": metrics
    }
    exclude_args = []
    if args.loss_beta is None:
        exclude_args.append("loss_beta")
    if args.clip_grad is None:
        exclude_args.append("clip_grad")
    filepath = create_filename_from_args(args, extension=".json", makedirs=True, exclude_args=exclude_args)
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nResults saved to {filepath}")
    
if __name__ == "__main__":
    main()
