import argparse
import json
import math
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from optuna.samplers import RandomSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from pruning_utils import calculate_roc_auc, create_filename_from_args, parse_student_n
from xgboost_utils import calc_metrics, get_xgboost_objective_and_metric, load_xgboost_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and hyperparameter-tune an XGBoost model using Optuna."
    )
    parser.add_argument("--dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument(
        "--student_n",
        type=parse_student_n,
        default=10,
        help="Number of training examples. Use -1 to train on the full training set.",
    )
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="OpenML repeat index (different repeats use different random splits).",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1000,
        help="Number of Optuna trials to run during hyperparameter search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Time budget in seconds for the Optuna study.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers for Optuna search (-1 for all available CPU cores).",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of inner cross-validation folds for hyperparameter evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Optuna sampler, CV splits, and XGBoost.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force search and training even if output already exists.",
    )
    return parser.parse_args()


def suggest_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggests XGBoost hyperparameters following TabPFN-v2"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 4000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.2, 1.0),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", math.exp(-16), math.exp(5), log=True
        ),
        "alpha": trial.suggest_float("alpha", math.exp(-16), math.exp(2), log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", math.exp(-16), math.exp(2), log=True
        ),
        "gamma": trial.suggest_float("gamma", math.exp(-16), math.exp(2), log=True),
    }


def get_cv_splits(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Generates cross-validation splits, adapting gracefully to small sample sizes."""
    _, class_counts = np.unique(y, return_counts=True)
    min_count = min(class_counts)

    # StratifiedKFold requires each class to have at least n_splits samples, so
    # cap the number of splits at min_count so small training sets don't crash.
    effective_splits = min(n_splits, min_count)
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))

    return splits


def run_optuna_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    n_trials: int,
    timeout: int | None,
    n_jobs: int,
    cv_folds: int,
    objective_name: str,
    eval_metric: str,
    seed: int,
) -> Tuple[Dict[str, Any], float, int, float]:
    """Runs Optuna hyperparameter optimization using CV on the training data."""
    splits = get_cv_splits(X_train, y_train, n_splits=cv_folds, seed=seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        hyperparams = suggest_hyperparams(trial)
        hyperparams["objective"] = objective_name
        hyperparams["eval_metric"] = eval_metric
        hyperparams["verbosity"] = 0
        hyperparams["random_state"] = seed
        hyperparams["n_jobs"] = 1  # single-thread per fold to avoid oversubscribing when Optuna runs parallel trials

        fold_scores = []
        for train_idx, val_idx in splits:
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            model = XGBClassifier(**hyperparams)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            y_val_probs = model.predict_proba(X_fold_val)

            fold_scores.append(calculate_roc_auc(y_fold_val, y_val_probs))

        return float(np.mean(fold_scores))

    sampler = RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
    duration = time.time() - start_time

    best_hyperparams = dict(study.best_params)
    best_score = float(study.best_value)
    completed_trials = len(study.trials)

    return best_hyperparams, best_score, completed_trials, duration


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = create_filename_from_args(args, extension=".json", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> train_xgboost_opt: Skipping (Output already exists at {output_path})")
        return

    best_hyperparams_path = create_filename_from_args(
        args,
        exclude_args=["repeat"],
        extension=".best_hyperparams.json",
        makedirs=True,
    )

    print(f"Loading data for dataset: {args.dataset}")
    X_train, X_test, y_train, y_test = load_xgboost_data(
        args.dataset, repeat=args.repeat, student_n=args.student_n, seed=2
    )

    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)} (student_n={args.student_n})")

    n_classes = len(np.unique(y_train))
    objective_name, eval_metric = get_xgboost_objective_and_metric(n_classes)

    if os.path.exists(best_hyperparams_path) and not args.force:
        print(f"Found existing best hyperparameters at {best_hyperparams_path}. Loading and skipping search...")
        with open(best_hyperparams_path, "r") as f:
            best_hyperparams_data = json.load(f)
    else:
        print(f"Starting Optuna hyperparameter search ({args.n_trials} trials, {args.cv_folds}-fold CV)...")
        best_hyperparams, best_cv_score, completed_trials, duration = run_optuna_search(
            X_train=X_train,
            y_train=y_train,
            n_classes=n_classes,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            cv_folds=args.cv_folds,
            objective_name=objective_name,
            eval_metric=eval_metric,
            seed=args.seed,
        )

        print(f"Done hyperparameter search in {duration:.2f}s ({completed_trials} trials).")
        print(f"Best CV score: {best_cv_score:.4f}")
        print("Best hyperparameters:")
        for k, v in sorted(best_hyperparams.items()):
            print(f"  {k}: {v}")

        best_hyperparams_data = {
            "best_hyperparams": best_hyperparams,
            "best_cv_score": best_cv_score,
            "n_trials": completed_trials,
            "study_duration_seconds": duration,
        }
        with open(best_hyperparams_path, "w") as f:
            json.dump(best_hyperparams_data, f, indent=4)
        print(f"Saved best hyperparameters to {best_hyperparams_path}")

    # Re-initialize model with best hyperparameters and fit on full training data
    final_model_hyperparams = {
        **best_hyperparams_data["best_hyperparams"],
        "objective": objective_name,
        "eval_metric": eval_metric,
        "verbosity": 0,
        "random_state": args.seed,
        "n_jobs": -1,
    }

    print("\nFitting final XGBoost model with best hyperparameters on full training set...")
    final_model = XGBClassifier(**final_model_hyperparams)
    final_model.fit(X_train, y_train, verbose=False)

    y_probs = final_model.predict_proba(X_test)
    metrics = calc_metrics(y_probs, y_train, y_test)

    output_data = {
        "config": vars(args),
        "metrics": metrics,
        **best_hyperparams_data,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
