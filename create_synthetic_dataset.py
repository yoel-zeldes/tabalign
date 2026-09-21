import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tqdm import tqdm
from cache_utils import memory
from data_utils import load_raw_data, fill_nans
from utils import get_device


def _sanitize(X):
    """Replace NaN with 0 and clip values to a range that TabPFN can handle."""
    return np.clip(np.nan_to_num(X, nan=0.0), -1e30, 1e30)


class RobustTabPFNRegressor(TabPFNRegressor):
    """TabPFNRegressor that sanitizes inputs to guard against rare infinite/NaN values during synthetic data generation."""
    def predict(self, X, *args, **kwargs):
        return super().predict(_sanitize(X), *args, **kwargs)


class RobustTabPFNClassifier(TabPFNClassifier):
    """TabPFNClassifier that sanitizes inputs to guard against rare infinite/NaN values during synthetic data generation."""
    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(_sanitize(X), *args, **kwargs)


def _generate_synthetic_dataset_tabpfn(X_train, n_samples, batch_size=1024):
    X_train = fill_nans(X_train)
    # Remove constant features before fitting TabPFN (it crashes on them)
    is_constant_mask = np.std(X_train, axis=0) == 0
    n_constant = int(is_constant_mask.sum())

    if n_constant > 0:
        constant_values = X_train[0, is_constant_mask]
        X_train = X_train[:, ~is_constant_mask]

    model = TabPFNUnsupervisedModel(
        tabpfn_clf=RobustTabPFNClassifier(device=get_device()),
        tabpfn_reg=RobustTabPFNRegressor(device=get_device()),
    )
    model.fit(X_train)
    batch_sizes = [batch_size] * (n_samples // batch_size)
    if reminder := n_samples % batch_size:
        batch_sizes.append(reminder)
    synthetic_data = _sanitize(
        np.concatenate([
            model.generate_synthetic_data(n_samples=b, n_permutations=1).cpu().numpy()
            for b in tqdm(batch_sizes, desc="Generating synthetic dataset")
        ], axis=0)
    )

    if n_constant == 0:
        return synthetic_data

    # Re-insert constant features at their original positions
    synthetic_full = np.zeros((n_samples, len(is_constant_mask)), dtype=synthetic_data.dtype)
    synthetic_full[:, ~is_constant_mask] = synthetic_data
    synthetic_full[:, is_constant_mask] = constant_values
    return synthetic_full


@memory.cache
def generate_synthetic_dataset(dataset, n_samples, repeat):
    X_train, _, _, _, _ = load_raw_data(dataset, repeat=repeat)
    return _generate_synthetic_dataset_tabpfn(X_train, n_samples)
