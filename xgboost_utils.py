import argparse
from typing import Any, Dict, Tuple

import numpy as np

from pruning_utils import (
    calculate_roc_auc,
    create_student_training_set,
    load_data,
)


def load_xgboost_data(
    dataset: str,
    repeat: int = 0,
    student_n: int | float = 10,
    seed: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads dataset and optionally creates a student training subset."""
    X_train, X_test, y_train, y_test = load_data(dataset, repeat=repeat)
    if student_n > 0:
        X_train, y_train = create_student_training_set(
            X_train, y_train, student_n, seed=seed
        )
    return X_train, X_test, y_train, y_test


def get_xgboost_objective_and_metric(n_classes: int) -> Tuple[str, str]:
    """Returns objective and evaluation metric for XGBoost based on the number of classes."""
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    return objective, eval_metric


def calc_metrics(y_probs: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Calculates and prints classification metrics matching evaluate_aligned_student.py format."""
    y_preds = np.argmax(y_probs, axis=1)
    acc = (y_preds == y_test).mean()
    roc_auc = calculate_roc_auc(y_test, y_probs)

    majority_label = np.bincount(y_train).argmax()
    majority_vote_acc = (majority_label == y_test).mean()
    majority_vote_probs = np.zeros_like(y_probs)
    majority_vote_probs[:, majority_label] = 1.0
    majority_vote_roc_auc = calculate_roc_auc(y_test, majority_vote_probs)

    metrics = {
        "n_unique_labels": int(len(np.unique(y_test))),
        "majority_vote_acc": float(majority_vote_acc),
        "majority_vote_roc_auc": float(majority_vote_roc_auc),
        "xgboost_acc": float(acc),
        "xgboost_roc_auc": float(roc_auc),
    }

    print("\nResults (Accuracy vs Real Labels):")
    print(f"Majority Vote: {majority_vote_acc:.4f} (AUC: {majority_vote_roc_auc:.4f})")
    print(f"XGBoost: {acc:.4f} (AUC: {roc_auc:.4f})")

    return metrics
