import torch
from pruning_utils import (
    load_data,
    fit_model,
    create_student_training_set,
    get_device,
    fill_nans,
    _get_model_preprocessor_state,
    memory,
)


def capture_hook(module, input, output, captured_storage, model_idx, n_test_tokens):
    test_acts = output[:, -n_test_tokens:, :].detach().clone().float()
    if model_idx not in captured_storage:
        captured_storage[model_idx] = []
    captured_storage[model_idx].append(test_acts)


@memory.cache
def get_teacher_preprocessor(dataset, repeat=0, model="tabpfn", n_estimators=8):
    X_train, _, y_train, _ = load_data(dataset, repeat=repeat)
    fitted_model = fit_model(X_train, y_train, n_estimators=n_estimators, model=model)
    return _get_model_preprocessor_state(fitted_model)


@memory.cache
def extract_activations(dataset, student_n, layer_k, n_estimators=8, repeat=0, model="tabpfn"):
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

    captured = {}
    handles = []
    current_batch_n = [0]

    if model == "tabfm":
        layer = fitted_model.model.icl_predictor.tf_icl.blocks[layer_k]
        h = layer.register_forward_hook(
            lambda mod, inp, out, stor=captured, idx=0: capture_hook(mod, inp, out, stor, idx, current_batch_n[0])
        )
        handles.append(h)
    else:
        executor = fitted_model.executor_
        device = get_device()
        underlying_models = []
        if hasattr(executor, 'model_caches'):
            underlying_models = [executor.model_caches[em.config._model_index].get(device) for em in executor.ensemble_members]
        elif hasattr(executor, 'models'):
            underlying_models = executor.models
        else:
            raise RuntimeError("Cannot extract models from executor type")

        for i, m in enumerate(underlying_models):
            layer = m.transformer_encoder.layers[layer_k]
            h = layer.register_forward_hook(
                lambda mod, inp, out, stor=captured, idx=i: capture_hook(mod, inp, out, stor, idx, current_batch_n[0])
            )
            handles.append(h)

    if model == "tabfm":
        X_test = fill_nans(X_test)
    print(f"Running inference on test set (size {len(X_test)})...")
    current_batch_n[0] = len(X_test)
    with torch.no_grad():
        probs = fitted_model.predict_proba(X_test)

    for h in handles:
        h.remove()

    concatenated_captured = {}
    for idx, act_list in captured.items():
        concatenated_captured[idx] = torch.cat(act_list, dim=1)

    data_to_save = {
        "metadata": {
            "dataset": dataset,
            "student_n": student_n,
            "layer_k": layer_k,
            "n_estimators": n_estimators,
            "repeat": repeat,
            "model": model,
        },
        "activations": concatenated_captured,
        "probs": probs,
    }

    return data_to_save
