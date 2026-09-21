"""
Random Subset Uncertainty Estimation for TabPFN
================================================
Idea: repeatedly sample random subsets of the training set, fit TabPFN on each,
and get a probability prediction for the test set. The variance / entropy across
runs acts as an uncertainty signal.

Produces:
  - A calibration curve: uncertainty buckets vs. actual error rate.
  - Prints summary statistics.
  - Saves results to the standard results/ directory via experiment_utils.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless

from tqdm import trange
from sklearn.calibration import calibration_curve

import experiment_utils
from data_utils import load_data
from model_utils import fit_model


# ---------------------------------------------------------------------------
# Core method
# ---------------------------------------------------------------------------

def predict_proba_single_subset(X_train, y_train, X_test, subset_size, global_classes, rng):
    """Fit TabPFN on a random subset and return predicted probabilities.

    Output is always padded to len(global_classes) columns, even if the subset
    doesn't contain every class (which would otherwise cause shape mismatches).
    global_classes: sorted array of all class labels in the full training set.
    """
    indices = rng.choice(len(X_train), subset_size, replace=False)
    classifier = fit_model(X_train[indices], y_train[indices])
    raw_proba = classifier.predict_proba(X_test)       # (n_test, n_subset_classes)
    subset_classes = classifier.classes_               # actual label values seen in subset

    n_classes = len(global_classes)

    # If all classes were seen, return directly
    if raw_proba.shape[1] == n_classes:
        return raw_proba

    # Map subset class labels to column indices in the global ordering
    col_indices = np.searchsorted(global_classes, subset_classes)
    full_proba = np.zeros((X_test.shape[0], n_classes), dtype=raw_proba.dtype)
    full_proba[:, col_indices] = raw_proba
    return full_proba


def random_subset_uncertainty(
    X_train, y_train, X_test, subset_size, num_runs, seed=0
):
    """
    Run the random-subset ensemble.

    Returns
    -------
    mean_proba : np.ndarray, shape (n_test, n_classes)
        Mean predicted probability across all runs.
    uncertainty : np.ndarray, shape (n_test,)
        Predictive entropy of the mean distribution (higher = more uncertain).
    all_probas : np.ndarray, shape (num_runs, n_test, n_classes)
        Full array of predicted probabilities.
    """
    rng = np.random.RandomState(seed)
    global_classes = np.sort(np.unique(y_train))
    all_probas = []

    for _ in trange(num_runs, desc="Random subset runs"):
        proba = predict_proba_single_subset(
            X_train, y_train, X_test, subset_size, global_classes, rng
        )
        all_probas.append(proba)

    all_probas = np.stack(all_probas, axis=0)          # (num_runs, n_test, n_classes)
    mean_proba = all_probas.mean(axis=0)               # (n_test, n_classes)

    # Prediction disagreement rate:
    # For each test sample, each run votes for a class (argmax).
    # Disagreement = 1 - (fraction of runs agreeing with the majority vote).
    all_preds = all_probas.argmax(axis=2)               # (num_runs, n_test)
    from scipy.stats import mode
    majority_vote, majority_count = mode(all_preds, axis=0, keepdims=False)
    uncertainty = 1.0 - (majority_count / num_runs)     # (n_test,)

    return mean_proba, uncertainty, all_probas


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration_curve(uncertainty, y_pred, y_true, n_bins, output_path):
    """
    Calibration curve: bin test examples by uncertainty, plot error rate per bin.

    A perfectly calibrated method produces a monotonically increasing curve.
    """
    # Sort by uncertainty
    sort_idx = np.argsort(uncertainty)
    uncertainty_sorted = uncertainty[sort_idx]
    errors_sorted = (y_pred[sort_idx] != y_true[sort_idx]).astype(float)

    # Bin
    bins = np.array_split(np.arange(len(uncertainty_sorted)), n_bins)
    bin_centers = [uncertainty_sorted[b].mean() for b in bins]
    bin_error_rates = [errors_sorted[b].mean() for b in bins]
    bin_sizes = [len(b) for b in bins]

    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        bin_centers,
        bin_error_rates,
        c=bin_sizes,
        cmap="viridis",
        s=80,
        zorder=3,
        label="Bins (color = # examples)",
    )
    ax.plot(bin_centers, bin_error_rates, color="steelblue", linewidth=1.5, alpha=0.6)

    # Trend line via linear fit
    z = np.polyfit(bin_centers, bin_error_rates, deg=1)
    x_fit = np.linspace(min(bin_centers), max(bin_centers), 200)
    ax.plot(x_fit, np.polyval(z, x_fit), "r--", linewidth=1.5, label="Linear trend")

    plt.colorbar(scatter, ax=ax, label="# examples in bin")

    ax.set_xlabel("Prediction Disagreement Rate (uncertainty)", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("Calibration Curve: Uncertainty vs. Error Rate", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Calibration curve saved to {output_path}")

    return bin_centers, bin_error_rates


def plot_confidence_vs_accuracy(mean_proba, y_true, n_bins, output_path):
    """
    Classic reliability diagram: confidence of the winning class vs. actual accuracy.
    Uses sklearn's calibration_curve for comparison.
    """
    max_proba = mean_proba.max(axis=1)           # confidence in top prediction
    y_pred = mean_proba.argmax(axis=1)
    correct = (y_pred == y_true).astype(float)

    # sklearn calibration_curve expects binary; use "correct vs. not" framing
    fraction_of_positives, mean_predicted_value = calibration_curve(
        correct, max_proba, n_bins=n_bins, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="steelblue",
        label="Random-subset ensemble",
    )
    ax.set_xlabel("Mean Confidence (max class probability)", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Reliability diagram saved to {output_path}")


# ---------------------------------------------------------------------------
# Selective prediction curve
# ---------------------------------------------------------------------------

def plot_selective_prediction(
    ensemble_uncertainty,
    naive_uncertainty,
    y_pred,
    y_true,
    output_path,
):
    """
    Selective prediction (accuracy vs. coverage) for two uncertainty methods.

    Both methods evaluate the SAME predictions (from a full-data forward pass).
    Samples are sorted by uncertainty (ascending).  At each coverage level k
    (keeping the k most-confident samples), we compute accuracy.  A better
    uncertainty method produces a curve that rises faster.
    """

    def _acc_vs_coverage(uncertainty, y_pred, y_true):
        order = np.argsort(uncertainty)  # most confident first
        correct = (y_pred[order] == y_true[order]).astype(float)
        cumulative_acc = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        coverage = np.arange(1, len(correct) + 1) / len(correct)
        return coverage, cumulative_acc

    cov_ens, acc_ens = _acc_vs_coverage(ensemble_uncertainty, y_pred, y_true)
    cov_naive, acc_naive = _acc_vs_coverage(naive_uncertainty, y_pred, y_true)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cov_ens, acc_ens, linewidth=2, label="Ensemble uncertainty (random subsets)", color="#2563eb")
    ax.plot(cov_naive, acc_naive, linewidth=2, label="Naive confidence (single model)", color="#dc2626", linestyle="--")

    # Overall accuracy reference line
    overall_acc = (y_pred == y_true).mean()
    ax.axhline(overall_acc, color="gray", linestyle=":", linewidth=1, label=f"Full-coverage accuracy ({overall_acc:.2f})")

    ax.set_xlabel("Coverage (fraction of test set retained)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Selective Prediction: Accuracy vs. Coverage", fontsize=13)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(max(0, overall_acc - 0.15), 1.01)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Selective prediction curve saved to {output_path}")

    # Compute AUC advantage (area between curves)
    from numpy import trapz
    auc_ens = trapz(acc_ens, cov_ens)
    auc_naive = trapz(acc_naive, cov_naive)
    return auc_ens, auc_naive


# ---------------------------------------------------------------------------
# Baseline: full training set, single prediction
# ---------------------------------------------------------------------------

def baseline_full_train(X_train, y_train, X_test):
    """TabPFN on the full training set (single run, no uncertainty)."""
    classifier = fit_model(X_train, y_train)
    return classifier.predict_proba(X_test)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(
    dataset_name,
    subset_size,
    num_runs,
    n_bins,
    seed,
    output_dir,
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Dataset: {dataset_name} ===")
    X_train, X_test, y_train, y_test = load_data(dataset_name)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    if subset_size is None:
        subset_size = min(max(1, len(X_train) // 2), 500)
        print(f"subset_size not specified, using {subset_size} (min(half train, 500))")

    if subset_size >= len(X_train):
        raise ValueError(
            f"subset_size ({subset_size}) must be < train size ({len(X_train)})"
        )

    # --- Full-data forward pass (for predictions) ---
    print("Running full-data forward pass for predictions...")
    full_proba = baseline_full_train(X_train, y_train, X_test)
    y_pred = full_proba.argmax(axis=1)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy (full data):      {accuracy:.4f}")

    # --- Random-subset ensemble (for uncertainty only) ---
    mean_proba, uncertainty, all_probas = random_subset_uncertainty(
        X_train, y_train, X_test,
        subset_size=subset_size,
        num_runs=num_runs,
        seed=seed,
    )

    # --- Naive confidence uncertainty ---
    naive_uncertainty = 1.0 - full_proba.max(axis=1)   # higher = more uncertain

    # --- Calibration curve (ensemble uncertainty vs full-data predictions) ---
    cal_path = os.path.join(output_dir, f"{dataset_name}_calibration_curve.png")
    bin_centers, bin_error_rates = plot_calibration_curve(
        uncertainty, y_pred, y_test, n_bins, cal_path
    )

    # --- Reliability diagram (full-data probabilities) ---
    rel_path = os.path.join(output_dir, f"{dataset_name}_reliability_diagram.png")
    plot_confidence_vs_accuracy(full_proba, y_test, n_bins, rel_path)

    # --- Selective prediction curve (ensemble vs naive, same predictions) ---
    sel_path = os.path.join(output_dir, f"{dataset_name}_selective_prediction.png")
    auc_ens, auc_naive = plot_selective_prediction(
        ensemble_uncertainty=uncertainty,
        naive_uncertainty=naive_uncertainty,
        y_pred=y_pred,
        y_true=y_test,
        output_path=sel_path,
    )
    print(f"Selective AUC — Ensemble: {auc_ens:.4f},  Naive: {auc_naive:.4f}")

    # --- Spearman correlation: uncertainty vs. error (same predictions) ---
    from scipy.stats import spearmanr
    is_error = (y_pred != y_test).astype(float)
    rho_ens, pval_ens = spearmanr(uncertainty, is_error)
    print(f"Spearman ρ (ensemble):  {rho_ens:.4f}  (p={pval_ens:.4g})")

    rho_naive, pval_naive = spearmanr(naive_uncertainty, is_error)
    print(f"Spearman ρ (naive):     {rho_naive:.4f}  (p={pval_naive:.4g})")

    results = {
        "dataset_name": dataset_name,
        "subset_size": int(subset_size),
        "num_runs": int(num_runs),
        "accuracy": float(accuracy),
        "spearman_rho_ensemble": float(rho_ens),
        "spearman_pval_ensemble": float(pval_ens),
        "spearman_rho_naive": float(rho_naive),
        "spearman_pval_naive": float(pval_naive),
        "selective_auc_ensemble": float(auc_ens),
        "selective_auc_naive": float(auc_naive),
        "bin_centers": [float(v) for v in bin_centers],
        "bin_error_rates": [float(v) for v in bin_error_rates],
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Random-subset uncertainty estimation for TabPFN"
    )
    parser.add_argument("--dataset_name", type=str, default="breast_cancer",
                        help="Dataset(s) (comma-separated): breast_cancer,wine,iris,digits")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Number of training examples per subset (default: half of train)")
    parser.add_argument("--num_runs", type=int, default=50,
                        help="Number of random subsets to sample")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for calibration curve")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/random_subset_uncertainty")
    args = parser.parse_args()

    dataset_names = [d.strip() for d in args.dataset_name.split(",")]
    
    # Iterate over datasets with progress bar
    for d_name in tqdm(dataset_names, desc="Datasets"):
        config = vars(args).copy()
        config["dataset_name"] = d_name
        
        try:
            results = run_experiment(**config)
            experiment_utils.save_results(config=config, results=results)
        except Exception as e:
            print(f"Error processing dataset {d_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    from tqdm import tqdm
    main()
