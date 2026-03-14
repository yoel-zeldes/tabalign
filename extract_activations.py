import argparse
import os
import torch
import numpy as np
from scipy import stats as scipy_stats
from pruning_utils import load_data, fit_model, create_filename_from_args, create_student_training_set

def capture_hook(module, input, output, captured_storage, model_idx):
    captured_storage[model_idx] = output.detach().clone()

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
        
    model = fit_model(
        use_X_train, use_y_train, n_estimators=args.n_estimators,
        assure_feature_tokens_are_static=True,
        token_per_feature=args.use_feature_stats,
    )
    
    captured = {}
    handles = []
    
    for i, m in enumerate(model.executor_.models):
        layer = m.transformer_encoder.layers[args.layer_k]
        h = layer.register_forward_hook(
            lambda mod, inp, out, stor=captured, idx=i: capture_hook(mod, inp, out, stor, idx)
        )
        handles.append(h)
    
    print(f"Running inference on test set (size {len(X_test)})...")
    with torch.no_grad():
        model.predict_proba(X_test)
        
    for h in handles:
        h.remove()

    print("Computing feature statistics from training data...")
    feature_stats = compute_feature_stats(use_X_train, use_y_train, cat_indices=cat_indices)
        
    data_to_save = {
        "metadata": vars(args),
        "activations": captured,
        "feature_stats": feature_stats,
    }
    
    torch.save(data_to_save, output_path)
    print(f"Activations saved to {output_path}")

if __name__ == "__main__":
    main()
