import argparse
import os
import torch
import numpy as np
from scipy import stats as scipy_stats
from pruning_utils import load_data, fit_model, create_filename_from_args, create_student_training_set, get_device, save_tabfm_preprocessor, load_tabfm_preprocessor, fill_nans

def capture_hook(module, input, output, captured_storage, model_idx, n_test_tokens):
    test_acts = output[:, -n_test_tokens:, :].detach().clone().float()
    if model_idx not in captured_storage:
        captured_storage[model_idx] = []
    captured_storage[model_idx].append(test_acts)

def compute_feature_stats(X, y, cat_indices):
    """Compute per-feature statistics from training data, including the label.
    
    Returns a tensor of shape [n_features + 1, 18] with columns:
        [mean, std, skewness, kurtosis, iqr,
         p0(=min), p10, p20, p30, p40, p50(=median), p60, p70, p80, p90, p100(=max),
         is_categorical, nan_frequency]
    """
    X_with_y = np.column_stack([X, y])

    percentiles = np.stack(
        [np.nanpercentile(X_with_y, q, axis=0) for q in range(0, 101, 10)],
        axis=1,
    )  # [n_features + 1, 11]
    iqr = np.nanpercentile(X_with_y, 75, axis=0) - np.nanpercentile(X_with_y, 25, axis=0)

    stat_rows = [
        np.nanmean(X_with_y, axis=0),
        np.nanstd(X_with_y, axis=0),
        np.nan_to_num(scipy_stats.skew(X_with_y, axis=0, nan_policy='omit'), nan=0.0),
        np.nan_to_num(scipy_stats.kurtosis(X_with_y, axis=0, nan_policy='omit'), nan=0.0),
        iqr,
        *[percentiles[:, i] for i in range(percentiles.shape[1])],
    ]

    n_cols = X_with_y.shape[1]
    is_categorical = np.zeros(n_cols, dtype=np.float32)
    is_categorical[list(cat_indices)] = 1.0
    # The label is always categorical (it's a classification target)
    is_categorical[-1] = 1.0
    stat_rows.append(is_categorical)

    nan_frequency = np.isnan(X_with_y.astype(float)).mean(axis=0).astype(np.float32)
    stat_rows.append(nan_frequency)

    stats = np.stack(stat_rows, axis=1)
    result = torch.tensor(stats, dtype=torch.float32)

    print(f"  feature_stats shape: {result.shape}  feature_stats: {result}")

    # Replace any remaining NaN with 0 (e.g. all-NaN columns have no meaningful stats)
    result = torch.nan_to_num(result, nan=0.0)

    return result

def main():
    parser = argparse.ArgumentParser(description="Extract TabPFN activations")
    parser.add_argument("--dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument("--student_n", type=int, default=10, help="Number of examples for student. Negative for full (teacher).")
    parser.add_argument("--layer_k", type=int, default=2, help="Layer index to extract activations from.")
    parser.add_argument("--n_estimators", type=int, default=8, help="Number of TabPFN estimators.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--force", action="store_true", help="Force extraction even if output exists.")
    parser.add_argument("--use_feature_stats", action="store_true",
                        help="Use features_per_group=1 for 1:1 feature-to-token mapping (needed for feature stats conditioning).")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use.")
    args = parser.parse_args()

    output_path = create_filename_from_args(args, extension=".pt", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> extract_activations: Skipping (Output already exists at {output_path})")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data: {args.dataset}")
    X_train, X_test, y_train, y_test, cat_indices = load_data(args.dataset, repeat=args.repeat, return_cat_indices=True)
    
    if args.student_n < 0:
        use_X_train = X_train
        use_y_train = y_train
        n_label = "full"
    else:
        use_X_train, use_y_train = create_student_training_set(X_train, y_train, args.student_n)
        n_label = f"N{args.student_n}"
        
    # For TabFM student extraction, load the teacher's fitted model to reuse its
    # preprocessors (encoding, scaling, filtering). This ensures teacher and student
    # produce activations in the same feature space.
    teacher_model_preprocessor = None
    if args.model == "tabfm" and args.student_n >= 0:
        teacher_args = dict(vars(args), student_n=-1)
        teacher_act_path = create_filename_from_args(
            teacher_args, script_name="extract_activations", extension=".pt"
        )
        teacher_model_preprocessor_path = teacher_act_path.replace('.pt', '.teacher_model_preprocessor.pkl')
        teacher_model_preprocessor = load_tabfm_preprocessor(teacher_model_preprocessor_path)

    model = fit_model(
        use_X_train, use_y_train, n_estimators=args.n_estimators,
        assure_feature_tokens_are_static=True,
        token_per_feature=args.use_feature_stats,
        model=args.model,
        model_preprocessor=teacher_model_preprocessor,
    )

    # For TabFM teacher extraction, save the fitted model so student can reuse preprocessors
    if args.model == "tabfm" and args.student_n < 0:
        teacher_model_preprocessor_path = output_path.replace('.pt', '.teacher_model_preprocessor.pkl')
        save_tabfm_preprocessor(model, teacher_model_preprocessor_path)
    
    captured = {}
    handles = []
    current_batch_n = [0]
    
    if args.model == "tabfm":
        layer = model.model.icl_predictor.tf_icl.blocks[args.layer_k]
        h = layer.register_forward_hook(
            lambda mod, inp, out, stor=captured, idx=0: capture_hook(mod, inp, out, stor, idx, current_batch_n[0])
        )
        handles.append(h)
    else:
        executor = model.executor_
        device = get_device()
        underlying_models = []
        if hasattr(executor, 'model_caches'):
            underlying_models = [executor.model_caches[em.config._model_index].get(device) for em in executor.ensemble_members]
        elif hasattr(executor, 'models'):
            underlying_models = executor.models
        else:
            raise RuntimeError("Cannot extract models from executor type")

        for i, m in enumerate(underlying_models):
            layer = m.transformer_encoder.layers[args.layer_k]
            h = layer.register_forward_hook(
                lambda mod, inp, out, stor=captured, idx=i: capture_hook(mod, inp, out, stor, idx, current_batch_n[0])
            )
            handles.append(h)
    
    if args.model == "tabfm":
        X_test = fill_nans(X_test)
    print(f"Running inference on test set (size {len(X_test)})...")
    current_batch_n[0] = len(X_test)
    with torch.no_grad():
        probs = model.predict_proba(X_test)
        
    for h in handles:
        h.remove()

    concatenated_captured = {}
    for idx, act_list in captured.items():
        concatenated_captured[idx] = torch.cat(act_list, dim=1)

    print("Computing feature statistics from training data...")
    feature_stats = compute_feature_stats(use_X_train, use_y_train, cat_indices=cat_indices)
        
    data_to_save = {
        "metadata": vars(args),
        "activations": concatenated_captured,
        "feature_stats": feature_stats,
        "probs": probs,
    }
    
    torch.save(data_to_save, output_path)
    print(f"Activations saved to {output_path}")

if __name__ == "__main__":
    main()
