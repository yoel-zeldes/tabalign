import numpy as np
from xgboost import XGBClassifier

from data_utils import create_student_training_set
from model_utils import memory
from xgboost_utils import (
    calc_metrics,
    get_xgboost_objective_and_metric,
    load_xgboost_data,
)

# Default hyperparameters from the FT-Transformer paper, as used in the TabSTAR XGBoost baseline:
# https://github.com/alanarazi7/TabSTAR/blob/master/tabstar_paper/baselines/xgboost.py
XGBOOST_N_ESTIMATORS = 2000
XGBOOST_EARLY_STOPPING_ROUNDS = 50
XGBOOST_BOOSTER = "gbtree"
XGBOOST_SEED = 42
VAL_RATIO = 0.1
MAX_VAL_SIZE = 1000


@memory.cache
def train_xgboost(dataset, student_n=-1, repeat=0):
    print(f"Loading data for dataset: {dataset}")
    X_train, X_test, y_train, y_test = load_xgboost_data(
        dataset, repeat=repeat, student_n=student_n, seed=2
    )
    val_size = min(int(len(y_train) * VAL_RATIO), MAX_VAL_SIZE)
    X_val, y_val, X_train, y_train = create_student_training_set(
        X_train, y_train, val_size, return_rest=True
    )

    print(f"Training set size: {len(X_train)}, Val set size: {len(X_val)} (student_n={student_n})")

    n_classes = len(np.unique(y_train))
    objective, eval_metric = get_xgboost_objective_and_metric(n_classes)

    model = XGBClassifier(
        n_estimators=XGBOOST_N_ESTIMATORS,
        early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
        booster=XGBOOST_BOOSTER,
        random_state=XGBOOST_SEED,
        objective=objective,
        eval_metric=eval_metric,
        verbosity=0,
    )

    print("Fitting XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_probs = model.predict_proba(X_test)
    metrics = calc_metrics(y_probs, y_train, y_test)

    return {
        "config": {
            "dataset": dataset,
            "student_n": student_n,
            "repeat": repeat,
        },
        "metrics": metrics,
    }
