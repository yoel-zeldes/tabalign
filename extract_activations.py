import gc
import time
import torch
from data_utils import fill_nans
from pruning_utils import (
    load_data,
    fit_model,
    create_student_training_set,
    get_transformer_layer,
    _get_model_preprocessor_state,
    memory,
)


class ActivationCaptureHook:
    """Forward hook to capture test token activations during inference.

    Differences between models:
    - TabFM: All ensemble members are forwarded in a single batch (dim 0 = n_estimators).
      Output is a Tensor of shape (n_estimators, total_tokens, hidden_dim).
    - TabPFN: Ensemble members are forwarded sequentially (dim 0 = 1 per call)
      through the shared model using cached KV states. Output is a tuple (x_BRE, kv_entry).
    In both cases, accumulating per-call outputs and concatenating along dim 0 yields
    a unified tensor of shape (n_estimators, n_test_tokens, hidden_dim).
    """

    def __init__(self, n_test_tokens):
        self.n_test_tokens = n_test_tokens
        self.activations = []

    def __call__(self, module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        self.activations.append(x[:, -self.n_test_tokens:, :].detach().clone().float().cpu())

    def get_captured_activations(self):
        all_activations = torch.cat(self.activations, dim=0)
        return {
            estimator_idx: all_activations[estimator_idx]
            for estimator_idx in range(all_activations.shape[0])
        }


@memory.cache
def get_teacher_preprocessor(dataset, repeat=0, model="tabpfn", n_estimators=8):
    X_train, _, y_train, _ = load_data(dataset, repeat=repeat)
    fitted_model = fit_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        model=model,
        fit_mode="fit_preprocessors",
    )
    state = _get_model_preprocessor_state(fitted_model)
    del fitted_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


@memory.cache
def extract_activations(dataset, student_n, layer_k, n_estimators=8, repeat=0, model="tabpfn"):
    start_time = time.time()
    print(f"Loading data: {dataset}")
    X_train, X_test, y_train, y_test, cat_indices = load_data(dataset, repeat=repeat, return_cat_indices=True)

    if student_n < 0:
        use_X_train = X_train
        use_y_train = y_train
        teacher_model_preprocessor = None
    else:
        use_X_train, use_y_train = create_student_training_set(X_train, y_train, student_n)
        teacher_model_preprocessor = get_teacher_preprocessor(
            dataset=dataset,
            repeat=repeat,
            model=model,
            n_estimators=n_estimators,
        )

    fitted_model = fit_model(
        use_X_train, use_y_train, n_estimators=n_estimators,
        model=model,
        model_preprocessor=teacher_model_preprocessor,
    )

    layer = get_transformer_layer(fitted_model, layer_k, model)
    hook = ActivationCaptureHook(len(X_test))
    hook_handle = layer.register_forward_hook(hook)

    if model == "tabfm":
        X_test = fill_nans(X_test)
    print(f"Running inference on test set (size {len(X_test)})...")
    with torch.no_grad():
        probs = fitted_model.predict_proba(X_test)

    activations = hook.get_captured_activations()

    assert len(activations) == n_estimators, (
        f"Expected {n_estimators} estimators in activations, got {len(activations)}"
    )
    for i in range(n_estimators):
        assert activations[i].shape[0] == len(X_test), (
            f"Expected activation shape ({len(X_test)}, hidden_dim) for estimator {i}, "
            f"but got {activations[i].shape}"
        )

    hook_handle.remove()
    del fitted_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extraction_time = time.time() - start_time

    return {
        "metadata": {
            "dataset": dataset,
            "student_n": student_n,
            "layer_k": layer_k,
            "n_estimators": n_estimators,
            "repeat": repeat,
            "model": model,
            "extraction_time": extraction_time,
        },
        "activations": activations,
        "probs": probs,
    }
