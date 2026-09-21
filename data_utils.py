import re
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfn.preprocessing.modality_detection import detect_feature_modalities
from tabpfn.preprocessing.clean import clean_data, fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import FeatureModality
from consts import TABARENA_NAME_TO_TASK_ID


def load_raw_data(dataset_name, repeat=0):
    if not dataset_name.startswith("tabarena/"):
        raise ValueError(f"Unknown dataset name: {dataset_name}")
        
    task = openml.tasks.get_task(TABARENA_NAME_TO_TASK_ID[dataset_name.removeprefix("tabarena/")])
    X, y = task.get_X_and_y(dataset_format="dataframe")
    y = LabelEncoder().fit_transform(y.astype(str))
    train_idx, test_idx = task.get_train_test_split_indices(fold=0, repeat=repeat)
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    n_unique_labels = len(np.unique(y_train))
    if n_unique_labels >= 30:
        raise ValueError(
            f"Dataset '{dataset_name}' has {n_unique_labels} unique labels, so it's probably not a classification dataset."
        )

    # Preprocess categorical/text features into numeric float arrays fit on train
    # and applied to test, ensuring compatibility across all models (TabPFN, TabFM, XGBoost).
    feature_schema = detect_feature_modalities(
        X=X_train.values,
        feature_names=list(X_train.columns),
        min_samples_for_inference=100,
        max_unique_for_category=30,
        min_unique_for_numerical=4,
        min_cardinality_for_text=30,
    )
    X_train_clean, ord_encoder, feature_schema = clean_data(X_train.values, feature_schema)
    inferred_cat_indices = feature_schema.indices_for(FeatureModality.CATEGORICAL)
    X_test_clean = fix_dtypes(pd.DataFrame(X_test.values), cat_indices=inferred_cat_indices)
    X_test_clean = process_text_na_dataframe(X_test_clean, ord_encoder=ord_encoder)
    return X_train_clean, X_test_clean, y_train, y_test, inferred_cat_indices


def load_data(dataset_name, repeat, return_cat_indices=False):
    synthetic_dataset_pattern = r'\[synthetic-n_samples_(\d+)-repeat_(\d+)\]'
    synthetic_match = re.search(synthetic_dataset_pattern, dataset_name)
    is_synthetic = synthetic_match is not None
    dataset_name = re.sub(synthetic_dataset_pattern, '', dataset_name)

    X_train, X_test, y_train, y_test, inferred_cat_indices = load_raw_data(
        dataset_name, repeat=repeat
    )

    if is_synthetic:
        from create_synthetic_dataset import generate_synthetic_dataset
        X_test = generate_synthetic_dataset(
            dataset=dataset_name,
            n_samples=int(synthetic_match.group(1)),
            repeat=int(synthetic_match.group(2)),
        )
        y_test = None

    res = [X_train, X_test, y_train, y_test]
    if return_cat_indices:
        res.append(inferred_cat_indices)
    return tuple(res)


def resolve_student_n(student_n, n_examples):
    if student_n < 0:
        if student_n != -1:
            raise ValueError(f"Invalid student_n value: {student_n}. Negative values must be -1.")
        return int(student_n)
    elif 0 < student_n < 1:
        return max(1, int(round(student_n * n_examples)))
    elif student_n >= 1:
        if student_n > n_examples:
            raise ValueError(
                f"Invalid student_n value: {student_n}. Must be <= number of examples ({n_examples})."
            )
        return int(student_n)
    else:
        raise ValueError(
            f"Invalid student_n value: {student_n}. Must be an integer >= 1, a fraction between 0 and 1, or -1."
        )


def _stratified_subsample(X, y, size, seed):
    """Stratified subsampling to preserve class balance when truncating.
    Without this, datasets with sorted indices (e.g. TabArena) lose minority classes."""
    X_sub, X_rest, y_sub, y_rest = train_test_split(
        X, y, train_size=size, stratify=y, random_state=seed
    )
    return X_sub, y_sub, X_rest, y_rest


def create_student_training_set(X_train, y_train, student_n, seed=1, return_rest=False):
    """Selects student_n examples using stratified sampling.

    If the stratified split leaves any label missing from y_sub or y_rest,
    one example of that label is moved from the other set to fix it.
    """
    student_n = resolve_student_n(student_n, len(X_train))
    unique_labels = np.unique(y_train)
    if student_n < len(unique_labels):
        raise ValueError(
            f"student_n ({student_n}) must be >= number of unique labels ({len(unique_labels)})."
        )

    X_sub, y_sub, X_rest, y_rest = _stratified_subsample(X_train, y_train, student_n, seed=seed)

    # Fix y_sub: for any label missing from y_sub, move one example from y_rest -> y_sub
    for label in unique_labels:
        if label not in y_sub:
            idx = np.where(y_rest == label)[0][0]
            X_sub = np.concatenate([X_sub, X_rest[idx:idx+1]])
            y_sub = np.concatenate([y_sub, y_rest[idx:idx+1]])
            X_rest = np.delete(X_rest, idx, axis=0)
            y_rest = np.delete(y_rest, idx)

    if return_rest:
        # Fix y_rest: for any label missing from y_rest, move one example from y_sub -> y_rest
        for label in unique_labels:
            if label not in y_rest:
                idx = np.where(y_sub == label)[0][0]
                X_rest = np.concatenate([X_rest, X_sub[idx:idx+1]])
                y_rest = np.concatenate([y_rest, y_sub[idx:idx+1]])
                X_sub = np.delete(X_sub, idx, axis=0)
                y_sub = np.delete(y_sub, idx)
        return X_sub, y_sub, X_rest, y_rest

    if not set(np.unique(y_sub)) == set(unique_labels):
        raise ValueError("The resulting subset does not contain all labels found in the original y_train.")

    if return_rest:
        if not set(np.unique(y_rest)) == set(unique_labels):
            raise ValueError("The resulting rest set does not contain all labels found in the original y_train.")

    return X_sub, y_sub


def fill_nans(X, value=0.0):
    if isinstance(X, pd.DataFrame):
        return X.fillna(value)
    elif isinstance(X, np.ndarray):
        return np.nan_to_num(X, nan=value)
    raise ValueError(f"Unknown type: {type(X)}. Supported options are pd.DataFrame and np.ndarray.")

