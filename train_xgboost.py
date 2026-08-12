import argparse
import json
import os
import numpy as np
from xgboost import XGBClassifier

from pruning_utils import create_filename_from_args, create_student_training_set
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train an XGBoost model on a dataset")
    parser.add_argument("--dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument("--student_n", type=parse_student_n, default=10,
                        help="Number of training examples. Use -1 to train on the full training set.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--force", action="store_true", help="Force training even if output exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = create_filename_from_args(args, extension=".json", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> train_xgboost: Skipping (Output already exists at {output_path})")
        return

    print(f"Loading data for dataset: {args.dataset}")
    X_train, X_test, y_train, y_test = load_xgboost_data(
        args.dataset, repeat=args.repeat, student_n=args.student_n, seed=2
    )
    val_size = min(int(len(y_train) * VAL_RATIO), MAX_VAL_SIZE)
    X_val, y_val, X_train, y_train = create_student_training_set(
        X_train, y_train, val_size, return_rest=True
    )

    print(f"Training set size: {len(X_train)}, Val set size: {len(X_val)} (student_n={args.student_n})")

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
    metrics = calc_metrics(y_probs, y_train, y_test, model_name="XGBoost")

    output_data = {
        "config": vars(args),
        "metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
