from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
from consts import TABFM_DEFAULT_N_ESTIMATORS, TABPFN_DEFAULT_N_ESTIMATORS
from data_utils import fill_nans
from pruning_utils import (
    load_data,
    fit_model,
    create_student_training_set,
    calculate_roc_auc,
    predict_from_probabilities,
    get_device,
    get_transformer_layer,
    memory,
)
from train_activation_aligner import build_aligner_model, train_aligner

def apply_alignment(acts, aligner_model):
    orig_shape = acts.shape
    x = acts.view(-1, orig_shape[-1]).float()
    result = aligner_model(x).to(acts.dtype).view(orig_shape)
    return acts + result  # the aligner predicts residuals


@contextmanager
def track_estimator(model, hook):
    """Sets the active estimator index on the hook during TabPFN inference.

    Wraps the executor's _call_model to read the explicit cache_index kwarg,
    which identifies which ensemble member is being forwarded.
    """
    orig = model.executor_._call_model

    def wrapped(*args, **kwargs):
        hook.current_estimator_idx = kwargs["cache_index"]
        return orig(*args, **kwargs)

    model.executor_._call_model = wrapped
    try:
        yield
    finally:
        model.executor_._call_model = orig


class TabPFNAlignedHook:
    """Forward hook applying residual alignment to TabPFN transformer layer activations.

    Ensemble members are forwarded sequentially (dim 0 = 1 per call) through the shared
    model using cached KV states. The active estimator index is set externally by
    track_estimator via the executor's explicit cache_index. Output is a tuple (x_BRE, kv_entry).
    """

    def __init__(self, aligner_models):
        self.aligner_models = aligner_models
        self.current_estimator_idx = 0

    def __call__(self, module, inp, output):
        x_BRE, kv_entry = output
        aligner = self.aligner_models[self.current_estimator_idx]
        return (apply_alignment(x_BRE, aligner), kv_entry)


class TabFMAlignedHook:
    """Forward hook applying residual alignment to TabFM transformer layer activations.

    Tracks estimator index internally via a counter incremented by the batch dimension.
    Sequential ordering is guaranteed by TabFM's _batch_forward, which iterates
    over np.array_split chunks synchronously.
    """

    def __init__(self, aligner_models):
        self.aligner_models = aligner_models
        self.current_estimator_idx = 0

    def __call__(self, module, inp, output):
        num_estimators_in_batch = output.shape[0]
        aligned_slices = [
            apply_alignment(
                output[estimator_idx : estimator_idx + 1],
                self.aligner_models[self.current_estimator_idx + estimator_idx],
            )
            for estimator_idx in range(num_estimators_in_batch)
        ]
        self.current_estimator_idx += num_estimators_in_batch
        return torch.cat(aligned_slices, dim=0)


def get_predictions_and_probabilities(model, X_test, aligner_models=None, layer_k=None, model_type="tabpfn"):
    hook_handle = None
    if aligner_models is not None:
        layer = get_transformer_layer(model, layer_k, model_type)
        if model_type == "tabfm":
            hook = TabFMAlignedHook(aligner_models)
        else:
            hook = TabPFNAlignedHook(aligner_models)
        hook_handle = layer.register_forward_hook(hook)

    with torch.no_grad():
        if model_type == "tabpfn" and aligner_models is not None:
            with track_estimator(model, hook):
                probs = model.predict_proba(X_test)
        else:
            probs = model.predict_proba(X_test)
        preds = predict_from_probabilities(model, probs)
        
    if hook_handle is not None:
        hook_handle.remove()
    return preds, probs

def validate_metadata(metadata, train_dataset, student_n, layer_k, n_estimators):
    """Ensure aligner metadata matches the evaluation configuration."""
    student_metadata = metadata.get("student_metadata")
    if student_metadata["dataset"] != train_dataset:
        raise ValueError(f"Dataset mismatch: Aligner trained on {student_metadata['dataset']} but got {train_dataset}")
    if student_metadata["student_n"] != student_n:
         raise ValueError(f"Student N mismatch: Aligner trained for N={student_metadata['student_n']} but evaluating on N={student_n}")
    if student_metadata["layer_k"] != layer_k:
         raise ValueError(f"Layer K mismatch: Aligner trained for K={student_metadata['layer_k']} but evaluating on K={layer_k}")
    if student_metadata["n_estimators"] != n_estimators:
         raise ValueError(f"Estimators mismatch: Aligner has {student_metadata['n_estimators']} models but evaluating with n_estimators={n_estimators}")

def _create_aligner_model(state_dict, hyperparams, n_stats=0):
    """Create and initialize an aligner model from a state dict."""
    first_key = next(k for k in state_dict if 'weight' in k)
    input_dim = state_dict[first_key].shape[1]
    model = build_aligner_model(
        input_dim=input_dim,
        output_dim=input_dim - n_stats,
        hyperparams=hyperparams
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_aligner_models(aligner_data):
    """Load aligner models from saved state dicts."""
    aligner_models = {}
    hyperparams = aligner_data["hyperparams"]
    n_stats = aligner_data["metadata"].get("n_stats", 0)
    
    for est_idx, data in aligner_data["estimator_idx_to_aligner"].items():
        aligner_models[est_idx] = _create_aligner_model(
            data,
            n_stats=n_stats,
            hyperparams=hyperparams,
        )
        
    return aligner_models

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

def _warn_if_constant_predictions(model_name, preds, y_train):
    if len(np.unique(preds)) == 1:
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        print(f"WARNING: {model_name} predicts the same class ({preds[0]}) for all {len(preds)} test examples! Train labels: {train_dist}")


@memory.cache
def evaluate_aligned_student(eval_dataset, train_dataset, student_n, layer_k,
                              n_estimators=None, patience=10, lr=1e-3, batch_size=2048,
                              hidden_layers=None, repeat=0, max_epochs=None,
                              model="tabpfn", aligner_opt=False):
    if hidden_layers is None:
        hidden_layers = []
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS


    print(f"Loading aligner models...")
    aligner_data = train_aligner(
        dataset=train_dataset,
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
        opt=aligner_opt,
    )

    validate_metadata(aligner_data["metadata"], train_dataset, student_n, layer_k, n_estimators)

    print("Loading data and fitting models...")
    X_train, X_test, y_train, y_test, cat_indices = load_data(eval_dataset, repeat=repeat, return_cat_indices=True)

    if model == "tabfm":
        X_test = fill_nans(X_test)

    print("Fitting Teacher model for reference...")
    teacher = fit_model(X_train, y_train, n_estimators=n_estimators, model=model)
    teacher_probs = teacher.predict_proba(X_test)
    teacher_preds = predict_from_probabilities(teacher, teacher_probs)

    print(f"Fitting Student model (N={student_n}, E={n_estimators})...")
    student_X_train, student_y_train = create_student_training_set(X_train, y_train, student_n)
    student = fit_model(student_X_train, student_y_train, n_estimators=n_estimators,
                        model=model, model_preprocessor=teacher)

    print("Evaluating Baseline Student...")
    baseline_preds, baseline_probs = get_predictions_and_probabilities(student, X_test, model_type=model)
    _warn_if_constant_predictions("Baseline student", baseline_preds, student_y_train)

    print("Evaluating Aligned Student...")
    aligner_models = load_aligner_models(aligner_data)
    device = get_device()
    for est_idx, m in aligner_models.items():
        aligner_models[est_idx] = m.to(device)
    aligned_preds, aligned_probs = get_predictions_and_probabilities(
        student, X_test, aligner_models, layer_k, model_type=model
    )
    _warn_if_constant_predictions("Aligned student", aligned_preds, student_y_train)

    metrics = calc_metrics(teacher_preds, baseline_preds, aligned_preds, teacher_probs, baseline_probs, aligned_probs, y_train, y_test)

    return {
        "config": {
            "eval_dataset": eval_dataset,
            "train_dataset": train_dataset,
            "student_n": student_n,
            "layer_k": layer_k,
            "n_estimators": n_estimators,
            "patience": patience,
            "lr": lr,
            "batch_size": batch_size,
            "hidden_layers": hidden_layers,
            "repeat": repeat,
            "max_epochs": max_epochs,
            "model": model,
            "aligner_opt": aligner_opt,
        },
        "metrics": metrics,
    }
