import argparse
import torch
import copy
import pickle
from tabpfn import TabPFNClassifier
from tabpfn.preprocessing import tag_features_and_sanitize_data
from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.inference_config import InferenceConfig
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import sys
import os
import openml
import re
import hashlib
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import log_loss
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch
from tabpfn.base import create_inference_engine
from tabpfn.validation import ensure_compatible_predict_input_sklearn
from tabpfn.preprocessing.transform import _transform_labels_one
from tabpfn.preprocessing.ensemble import TabPFNPreprocessedEnsembleMember
from cache_utils import OUTPUT_DIR, memory
from create_synthetic_dataset import generate_synthetic_dataset
from data_utils import (
    TABARENA_NAME_TO_TASK_ID,
    stratified_subsample,
    load_raw_data,
)


from tabpfn.settings import settings
settings.tabpfn.allow_cpu_large_dataset = True



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_filename_safe(filename):
    return filename.replace("/", "_").replace(" ", "_").replace('/', '_')

def create_filename_from_args(args, output_dir_arg_name="output_dir", exclude_args=None, extension="", script_name=None, makedirs=False):
    """
    Creates a standardized filename from a script name and a dictionary of arguments.
    Format: {script_name}-{arg1}_{val1}-{arg2}_{val2}...
    """
    if hasattr(args, '__dict__'):
        args = vars(args)
    else:
        args = dict(args)
    if exclude_args is None:
        exclude_args = []
        
    exclude_args.append(output_dir_arg_name)
    exclude_args.append("force")
    
    parts = [
        f"{arg_key}_{str(args[arg_key])}"
        for arg_key in sorted(args.keys()) 
        if arg_key not in exclude_args
    ]        
    filename = "-".join(parts)
    if extension:
        if not extension.startswith('.'):
            extension = f'.{extension}'
        filename += extension

    if script_name is None:
        script_name = os.path.basename(sys.argv[0])     
    script_name = script_name.replace('.py', '')
    filename = make_filename_safe(filename)

    max_filename_len = 255
    if len(filename) > max_filename_len:
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        suffix = f"_{file_hash}{extension}"
        filename = filename[:max_filename_len - len(suffix)] + suffix

    output_dir = args.get(output_dir_arg_name, OUTPUT_DIR)
    res = os.path.join(output_dir, script_name, filename)
    if makedirs:
        os.makedirs(os.path.dirname(res), exist_ok=True)
    return res



def load_data(dataset_name, repeat, return_cat_indices=False, max_num_examples=1000):
    synthetic_dataset_pattern = r'\[synthetic-n_samples_(\d+)-repeat_(\d+)\]'
    synthetic_match = re.search(synthetic_dataset_pattern, dataset_name)
    is_synthetic = synthetic_match is not None
    dataset_name = re.sub(synthetic_dataset_pattern, '', dataset_name)

    X_train, X_test, y_train, y_test, inferred_cat_indices = load_raw_data(
        dataset_name, repeat=repeat, max_num_examples=max_num_examples
    )

    if is_synthetic:
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


def parse_student_n(val_str):
    try:
        val = float(val_str)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid student_n value: '{val_str}'. Must be a number.")

    if val < 0:
        if val != -1:
            raise argparse.ArgumentTypeError(
                f"Invalid student_n value: {val_str}. Negative values must be -1 to indicate full dataset."
            )
        return int(val)
    elif 0 < val < 1:
        return val
    elif val >= 1:
        if not val.is_integer():
            raise argparse.ArgumentTypeError(
                f"Invalid student_n value: {val_str}. Values >= 1 must be integers."
            )
        return int(val)
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid student_n value: {val_str}. Must be an integer >= 1, a fraction between 0 and 1, or -1."
        )


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


def create_model(n_estimators=8, fit_mode="fit_with_cache", model="tabpfn"):
    """
    Creates a TabPFN or TabFM classifier.

    Args:
        n_estimators: Number of estimators.
        fit_mode: TabPFN fit mode. Use 'fit_preprocessors' when a differentiable forward pass is needed.
        model: Model architecture to use ('tabpfn' or 'tabfm'). Default is 'tabpfn'.
    """
    if model == "tabfm":
        if n_estimators > 1:
            raise ValueError("multiple estimators are not supported for now")
        tabfm_model = tabfm_v1_0_0_pytorch.load(model_type="classification")
        return TabFMClassifier(model=tabfm_model, n_estimators=n_estimators)

    if model == "tabpfn":
        return TabPFNClassifier(
            device=get_device(),
            n_estimators=n_estimators,
            fit_mode=fit_mode,
        )

    raise ValueError(f"Unknown model: {model}. Supported options are 'tabpfn' and 'tabfm'.")


def fill_nans(X, value=0.0):
    if isinstance(X, pd.DataFrame):
        return X.fillna(value)
    elif isinstance(X, np.ndarray):
        return np.nan_to_num(X, nan=value)
    raise ValueError(f"Unknown type: {type(X)}. Supported options are pd.DataFrame and np.ndarray.")


def _get_tabfm_preprocessor_state(model):
    """Extract fitted preprocessor state from a fitted TabFM model."""
    # scikit-learn convention: fitted attributes end with an underscore.
    state = {
        k: v
        for k, v in model.__dict__.items()
        if k.endswith("_")
    }
    # Don't save the fitted data and preprocessed data of the ensemble generator.
    ensemble_generator = copy.deepcopy(state["ensemble_generator_"])
    ensemble_generator.X_ = None
    ensemble_generator.y_ = None
    for preprocessor in ensemble_generator.preprocessors_.values():
        preprocessor.X_transformed_ = None
    state["ensemble_generator_"] = ensemble_generator
    return state


def _get_tabpfn_preprocessor_state(model):
    """Extract fitted preprocessor state from a fitted TabPFN model."""
    # scikit-learn convention: fitted attributes end with an underscore.
    # Exclude neural network weights, KV-cache engine, and hardware device settings
    tabpfn_excluded = {"models_", "model_", "executor_", "devices_", "forced_inference_dtype_", "use_autocast_"}
    state = {
        k: v
        for k, v in model.__dict__.items()
        if k.endswith("_") and k not in tabpfn_excluded
    }

    state["executor_ensemble_members"] = [
        TabPFNPreprocessedEnsembleMember(
            config=m.config,
            preprocessor=m.preprocessor,
            cat_ix=m.cat_ix,
            X_train=None,
            y_train=None,
        )
        for m in model.executor_.ensemble_members
    ]
    return state


def _get_model_preprocessor_state(model):
    """Extract lightweight preprocessor state from a fitted model (TabPFN or TabFM), filepath, or dict."""
    if isinstance(model, dict):
        return model

    if isinstance(model, str):
        with open(model, "rb") as f:
            return pickle.load(f)

    if isinstance(model, TabFMClassifier):
        return _get_tabfm_preprocessor_state(model)

    if isinstance(model, TabPFNClassifier):
        return _get_tabpfn_preprocessor_state(model)

    raise ValueError(f"Unknown model type: {type(model)}")


def save_model_preprocessor(model, path):
    """Save the preprocessor state of a fitted model (TabPFN or TabFM)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_get_model_preprocessor_state(model), f)


class _ReusedTabPFNEnsemblePreprocessor:
    """Wraps pre-fitted TabPFN's ensemble member preprocessors."""

    def __init__(self, classifier, executor_ensemble_members):
        self.classifier = classifier
        self.executor_ensemble_members = executor_ensemble_members

    def fit_transform_ensemble_members_iterator(self, X_train, y_train, cat_ix, **kwargs):
        X_clean = ensure_compatible_predict_input_sklearn(X_train, self.classifier)
        X_clean = fix_dtypes(X_clean, cat_indices=self.classifier.inferred_categorical_indices_)
        X_clean = process_text_na_dataframe(X_clean, ord_encoder=self.classifier.preprocessor_)
        y_encoded = self.classifier.label_encoder_.transform(y_train)

        for executor_ensemble_member in self.executor_ensemble_members:
            yield TabPFNPreprocessedEnsembleMember(
                config=executor_ensemble_member.config,
                preprocessor=executor_ensemble_member.preprocessor,
                X_train=executor_ensemble_member.preprocessor.transform(X_clean).X,
                y_train=_transform_labels_one(executor_ensemble_member.config, y_encoded),
                cat_ix=executor_ensemble_member.cat_ix,
            )

    def fit_transform_ensemble_members(self, X_train, y_train, cat_ix):
        return list(self.fit_transform_ensemble_members_iterator(X_train, y_train, cat_ix))


def _fit_tabfm_from_preprocessor(classifier, model_preprocessor, X_train, y_train):
    """Fits a TabFM model in-place using the given preprocessor and training data as ICL context.

    Args:
        classifier: The TabFM model to fit.
        model_preprocessor: a state dict or TabFM model.
        X_train: Training features.
        y_train: Training labels.
    """
    n_estimators = classifier.n_estimators
    classifier.__dict__.update(copy.deepcopy(_get_model_preprocessor_state(model_preprocessor)))
    if n_estimators != classifier.ensemble_generator_.n_estimators:
        raise ValueError("n_estimators of model and model_preprocessor must match.")

    # Encode labels using teacher's fitted encoder
    y_2d = np.array(y_train).reshape(-1, 1)
    y_encoded = classifier.y_encoder_.transform(y_2d).flatten()

    # Transform features using teacher's fitted encoder
    ensemble_generator = classifier.ensemble_generator_
    X_encoded = classifier.X_encoder_.transform(X_train)
    X_encoded = ensemble_generator.unique_filter_.transform(X_encoded)

    # Add cross features using teacher's pre-computed cross_pairs
    if hasattr(ensemble_generator, "cross_pairs_"):
        cross_cols = [X_encoded[:, i] * X_encoded[:, j] for i, j in ensemble_generator.cross_pairs_]
        X_encoded = np.concatenate([X_encoded, np.column_stack(cross_cols)], axis=1)

    # Add SVD features using teacher's fitted SVD pipeline
    if hasattr(ensemble_generator, "svd_pipeline_"):
        svd_feats = ensemble_generator.svd_pipeline_.transform(X_encoded[:, :ensemble_generator.n_original_features_])
        X_encoded = np.concatenate([X_encoded, svd_feats], axis=1)

    # Replace cached training data with student data
    ensemble_generator.X_ = X_encoded
    ensemble_generator.y_ = y_encoded

    # Recompute preprocessed training cache for each preprocessing pipeline
    for norm_method, preprocessor in ensemble_generator.preprocessors_.items():
        preprocessor.X_transformed_ = preprocessor.transform(X_encoded)

    # Reset row subsample patterns to use all student rows
    for norm_method in ensemble_generator.row_subsample_patterns_:
        ensemble_generator.row_subsample_patterns_[norm_method] = [
            None for _ in ensemble_generator.row_subsample_patterns_[norm_method]
        ]


def _fit_tabpfn_from_preprocessor(classifier, model_preprocessor, X_train, y_train):
    """Fits a TabPFN model in-place using the given preprocessor and training data as ICL context.

    Args:
        classifier: The TabPFN model to fit.
        model_preprocessor: a state dict or TabPFN model.
        X_train: Training features.
        y_train: Training labels.
    """
    assert classifier.fit_mode in ("fit_with_cache", "fit_preprocessors"), (
        f"fit_mode '{classifier.fit_mode}' is not supported with reused preprocessors. "
        "Only 'fit_with_cache' and 'fit_preprocessors' are supported."
    )
    assert not getattr(classifier, "differentiable_input", False), (
        "differentiable_input=True is not supported with reused preprocessors."
    )

    n_estimators = classifier.n_estimators
    classifier.__dict__.update(copy.deepcopy(_get_model_preprocessor_state(model_preprocessor)))
    if n_estimators != len(classifier.executor_ensemble_members):
        raise ValueError(
            f"n_estimators ({n_estimators}) does not match preprocessor ({len(classifier.executor_ensemble_members)})."
        )

    byte_size, _ = classifier._initialize_model_variables()
    classifier.executor_ = create_inference_engine(
        fit_mode=classifier.fit_mode,
        X_train=X_train,
        y_train=y_train,
        cat_ix=classifier.inferred_categorical_indices_,
        models=classifier.models_,
        ensemble_preprocessor=_ReusedTabPFNEnsemblePreprocessor(classifier, classifier.executor_ensemble_members),
        devices_=classifier.devices_,
        byte_size=byte_size,
        forced_inference_dtype_=classifier.forced_inference_dtype_,
        memory_saving_mode=classifier.memory_saving_mode,
        use_autocast_=classifier.use_autocast_,
        inference_mode=not classifier.differentiable_input,
    )


def _fit_from_preprocessor(classifier, model_preprocessor, X_train, y_train):
    """Fits a model (TabPFN or TabFM) in-place reusing the teacher preprocessor.

    Args:
        classifier: The TabPFN or TabFM model to fit.
        model_preprocessor: a state dict or fitted teacher model.
        X_train: Training features.
        y_train: Training labels.
    """
    if isinstance(classifier, TabFMClassifier):
        _fit_tabfm_from_preprocessor(classifier, model_preprocessor, X_train, y_train)
    else:
        _fit_tabpfn_from_preprocessor(classifier, model_preprocessor, X_train, y_train)


def fit_model(X_train, y_train, n_estimators=8, fit_mode="fit_with_cache", model="tabpfn", model_preprocessor=None):
    """
    Creates and fits a TabPFN or TabFM model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of estimators.
        fit_mode: See create_model.
        model: See create_model.
        model_preprocessor: Optional. If provided, the student model will reuse
            the preprocessor (encoding, scaling, etc.) from the preprocessor,
            but use X_train/y_train as the ICL context.
    """
    classifier = create_model(
        n_estimators=n_estimators,
        fit_mode=fit_mode,
        model=model,
    )
    if model == "tabfm":
        X_train = fill_nans(X_train)

    if model_preprocessor:
        _fit_from_preprocessor(classifier, model_preprocessor, X_train, y_train)
    else:
        classifier.fit(X_train, y_train)
    return classifier





def calculate_roc_auc(y_true, y_probs):
    """Calculates the appropriate metric based on the number of classes.
      - Binary classification: ROC AUC (higher is better)
      - Multiclass classification: -log_loss (higher is better)
    """
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_probs[:, 1])
    else:
        return -log_loss(y_true, y_probs)

def predict_from_probabilities(classifier, y_probs):
    """Returns class predictions from an array of probabilities using the classifier's classes."""
    return classifier.classes_[np.argmax(y_probs, axis=1)]
