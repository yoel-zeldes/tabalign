import copy
import gc
import os
import pickle
import numpy as np
import torch
from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch
from tabpfn import TabPFNClassifier
from tabpfn.base import create_inference_engine
from tabpfn.preprocessing.clean import fix_dtypes, process_text_na_dataframe
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.ensemble import TabPFNEnsembleMember
from tabpfn.preprocessing.transform import _transform_labels_one
from tabpfn.validation import ensure_compatible_predict_input_sklearn
from cache_utils import memory
from consts import TABFM_DEFAULT_N_ESTIMATORS, TABPFN_DEFAULT_N_ESTIMATORS
from data_utils import fill_nans, load_data
from utils import get_device

from tabpfn.settings import settings
settings.tabpfn.allow_cpu_large_dataset = True


def get_transformer_layer(model, layer_k, model_name="tabpfn"):
    """Returns the k-th transformer block."""
    if model_name == "tabfm":
        return model.model.icl_predictor.tf_icl.blocks[layer_k]
    elif model_name == "tabpfn":
        return model.models_[0].icl_blocks[layer_k]
    raise ValueError(f"Unknown model: {model_name}. Supported options are 'tabpfn' and 'tabfm'.")


def create_model(n_estimators=None, fit_mode="fit_with_cache", model="tabpfn"):
    """
    Creates a TabPFN or TabFM classifier.

    Args:
        n_estimators: Number of estimators.
        fit_mode: TabPFN fit mode. Use 'fit_preprocessors' when a differentiable forward pass is needed.
        model: Model architecture to use ('tabpfn' or 'tabfm'). Default is 'tabpfn'.
    """
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS

    if model == "tabfm":
        tabfm_model = tabfm_v1_0_0_pytorch.load(
            model_type="classification",
            device=str(get_device()),
        )
        return TabFMClassifier(
            model=tabfm_model,
            n_estimators=n_estimators,
            batch_size=1,
        )

    if model == "tabpfn":
        return TabPFNClassifier(
            device=get_device(),
            n_estimators=n_estimators,
            fit_mode=fit_mode,
            keep_cache_on_device=False,
            memory_saving_mode=True,
        )

    raise ValueError(f"Unknown model: {model}. Supported options are 'tabpfn' and 'tabfm'.")


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
        TabPFNEnsembleMember(
            config=m.config,
            cpu_preprocessor=m.cpu_preprocessor,
            gpu_preprocessor=m.gpu_preprocessor,
            feature_schema=m.feature_schema,
            feature_indices=m.feature_indices,
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


@memory.cache
def get_teacher_preprocessor(dataset, repeat=0, model="tabpfn", n_estimators=None):
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS
    X_train, _, y_train, _ = load_data(dataset, repeat=repeat)
    fitted_model = fit_model(
        X_train,
        y_train,
        n_estimators=n_estimators,
        model=model,
        fit_mode="fit_preprocessors",
    )
    state = _get_model_preprocessor_state(fitted_model)
    del fitted_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return state


class _ReusedTabPFNEnsemblePreprocessor:
    """Wraps pre-fitted TabPFN's ensemble member preprocessors."""

    def __init__(self, classifier, executor_ensemble_members):
        self.classifier = classifier
        self.executor_ensemble_members = executor_ensemble_members

    def any_estimator_uses_gpu_svd(self):
        return False

    def fit_transform_ensemble_members(self, X_train, y_train):
        X_clean = ensure_compatible_predict_input_sklearn(X_train, self.classifier)
        X_clean = fix_dtypes(
            X_clean,
            cat_indices=self.classifier.inferred_feature_schema_.indices_for(
                FeatureModality.CATEGORICAL
            ),
        )
        X_clean = process_text_na_dataframe(
            X=X_clean,
            ord_encoder=getattr(self.classifier, "ordinal_encoder_", None),
            passthrough_inf=self.classifier.get_inference_config().PASSTHROUGH_INF,
        )
        y_encoded = self.classifier.label_encoder_._encoder.transform(y_train)

        members = []
        for m in self.executor_ensemble_members:
            X_m = X_clean[:, m.feature_indices] if m.feature_indices is not None else X_clean
            members.append(
                TabPFNEnsembleMember(
                    config=m.config,
                    cpu_preprocessor=m.cpu_preprocessor,
                    gpu_preprocessor=m.gpu_preprocessor,
                    feature_schema=m.feature_schema,
                    feature_indices=m.feature_indices,
                    X_train=m.cpu_preprocessor.transform(X_m).X,
                    y_train=_transform_labels_one(m.config, y_encoded),
                )
            )
        return members


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
    assert classifier.fit_mode == "fit_with_cache", (
        f"fit_mode '{classifier.fit_mode}' is not supported with reused preprocessors. "
        "Only 'fit_with_cache' is supported."
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

    # The teacher preprocessor had keep_fitted_cache=False (due to fit_mode="fit_preprocessors"
    # used by extract_activations.py::get_teacher_preprocessor). Enable it so the student
    # retains preprocessing state for predict_proba.
    for member in classifier.executor_ensemble_members:
        if member.gpu_preprocessor is not None:
            member.gpu_preprocessor.keep_fitted_cache = True

    byte_size = classifier._initialize_model_variables()
    classifier.executor_ = create_inference_engine(
        fit_mode=classifier.fit_mode,
        X_train=X_train,
        y_train=y_train,
        models=classifier.models_,
        ensemble_preprocessor=_ReusedTabPFNEnsemblePreprocessor(
            classifier, classifier.executor_ensemble_members
        ),
        devices_=classifier.devices_,
        byte_size=byte_size,
        forced_inference_dtype_=classifier.forced_inference_dtype_,
        memory_saving_mode=classifier.memory_saving_mode,
        use_autocast_=classifier.use_autocast_,
        task_type="multiclass",
        inference_mode=not classifier.differentiable_input,
        keep_cache_on_device=classifier.keep_cache_on_device,
        kv_cache_precision=classifier.kv_cache_precision,
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


def fit_model(X_train, y_train, n_estimators=None, fit_mode="fit_with_cache", model="tabpfn", model_preprocessor=None):
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
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS

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
