import math
import time
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from optuna.samplers import RandomSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from pruning_utils import calculate_roc_auc, memory
from xgboost_utils import calc_metrics, get_xgboost_objective_and_metric, load_xgboost_data



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


# ignore repeat because we use the same optimal hyperparameters for all repeats of a dataset, because we want to save compute.
@memory.cache(ignore=["repeat"])
def find_best_xgboost_hyperparams(dataset, student_n=-1, repeat=0,
                                   n_trials=1000, timeout=600, n_jobs=-1,
                                   cv_folds=5, seed=42):
    print(f"Loading data for dataset: {dataset}")
    X_train, X_test, y_train, y_test = load_xgboost_data(
        dataset, repeat=repeat, student_n=student_n, seed=2
    )
    n_classes = len(np.unique(y_train))
    objective_name, eval_metric = get_xgboost_objective_and_metric(n_classes)

    print(f"Starting Optuna hyperparameter search ({n_trials} trials, {cv_folds}-fold CV)...")
    best_hyperparams, best_cv_score, completed_trials, duration = run_optuna_search(
        X_train=X_train,
        y_train=y_train,
        n_classes=n_classes,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        cv_folds=cv_folds,
        objective_name=objective_name,
        eval_metric=eval_metric,
        seed=seed,
    )
    return {
        "best_hyperparams": best_hyperparams,
        "best_cv_score": best_cv_score,
        "n_trials": completed_trials,
        "study_duration_seconds": duration,
        "objective_name": objective_name,
        "eval_metric": eval_metric,
    }


@memory.cache(ignore=["n_trials", "timeout", "n_jobs", "cv_folds", "seed"])
def train_xgboost_opt(dataset, student_n=-1, repeat=0,
                      n_trials=1000, timeout=600, n_jobs=-1,
                      cv_folds=5, seed=42):
    best_hyperparams_data = find_best_xgboost_hyperparams(
        dataset=dataset,
        student_n=student_n,
        repeat=repeat,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        cv_folds=cv_folds,
        seed=seed,
    )

    X_train, X_test, y_train, y_test = load_xgboost_data(
        dataset, repeat=repeat, student_n=student_n, seed=2
    )

    final_model_hyperparams = {
        **best_hyperparams_data["best_hyperparams"],
        "objective": best_hyperparams_data["objective_name"],
        "eval_metric": best_hyperparams_data["eval_metric"],
        "verbosity": 0,
        "random_state": seed,
        "n_jobs": -1,
    }

    print("\nFitting final XGBoost model with best hyperparameters on full training set...")
    final_model = XGBClassifier(**final_model_hyperparams)
    final_model.fit(X_train, y_train, verbose=False)

    y_probs = final_model.predict_proba(X_test)
    metrics = calc_metrics(y_probs, y_train, y_test)

    return {
        "config": {
            "dataset": dataset,
            "student_n": student_n,
            "repeat": repeat,
            "n_trials": n_trials,
            "timeout": timeout,
            "n_jobs": n_jobs,
            "cv_folds": cv_folds,
            "seed": seed,
        },
        "metrics": metrics,
        "best_hyperparams": best_hyperparams_data["best_hyperparams"],
        "best_cv_score": best_hyperparams_data["best_cv_score"],
        "n_trials": best_hyperparams_data["n_trials"],
        "study_duration_seconds": best_hyperparams_data["study_duration_seconds"],
    }
