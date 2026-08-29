import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tqdm import tqdm
from cache_utils import memory
from data_utils import load_raw_data


def _generate_synthetic_dataset_tabpfn(X_train, n_samples, batch_size=1024):
    # Remove constant features before fitting TabPFN (it crashes on them)
    is_constant_mask = np.std(X_train, axis=0) == 0
    n_constant = int(is_constant_mask.sum())

    if n_constant > 0:
        constant_values = X_train[0, is_constant_mask]
        X_train = X_train[:, ~is_constant_mask]

    model = TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(),
        tabpfn_reg=TabPFNRegressor()
    )
    model.fit(X_train)
    batch_sizes = [batch_size] * (n_samples // batch_size)
    if reminder := n_samples % batch_size:
        batch_sizes.append(reminder)
    synthetic_data = np.concatenate([
        model.generate_synthetic_data(n_samples=b, n_permutations=1).numpy()
        for b in tqdm(batch_sizes, desc="Generating synthetic dataset")
    ], axis=0)

    if n_constant == 0:
        return synthetic_data

    # Re-insert constant features at their original positions
    synthetic_full = np.zeros((n_samples, len(is_constant_mask)))
    synthetic_full[:, ~is_constant_mask] = synthetic_data
    synthetic_full[:, is_constant_mask] = constant_values
    return synthetic_full


@memory.cache
def generate_synthetic_dataset(dataset, n_samples, repeat):
    X_train, _, _, _, _ = load_raw_data(dataset, repeat=repeat)
    return _generate_synthetic_dataset_tabpfn(X_train, n_samples)
