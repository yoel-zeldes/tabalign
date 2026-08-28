import argparse
import numpy as np
import os
from pruning_utils import load_data, create_filename_from_args
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
from tqdm import tqdm


def generate_synthetic_dataset_tabpfn(X_train, n_samples, batch_size=1024):
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


def generate_synthetic_dataset(dataset_name, n_samples, output_path, repeat=0):
    X_train, _, _, _ = load_data(dataset_name, repeat=repeat)
    X_synthetic = generate_synthetic_dataset_tabpfn(X_train, n_samples)

    n_features = X_train.shape[1]
    with open(output_path, 'w') as f:
        f.write(','.join([f'feature_{i}' for i in range(n_features)]) + '\n')
        for example in X_synthetic:
            f.write(','.join(str(v) for v in example) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset based on marginal feature distributions.')
    parser.add_argument('--dataset', type=str, default='breast_cancer', help='Name of the source dataset')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of synthetic samples to generate')
    parser.add_argument('--output_dir', type=str, default='results/synthetic_data', help='Directory to save the synthetic dataset')
    parser.add_argument('--force', action='store_true', help='Force generation even if output exists')
    parser.add_argument('--repeat', type=int, default=0, help='OpenML repeat index (different repeats use different random splits).')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = create_filename_from_args(args, extension=".csv", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> create_synthetic_dataset: Skipping (Output already exists at {output_path})")
        return
        
    generate_synthetic_dataset(args.dataset, args.n_samples, output_path, repeat=args.repeat)


if __name__ == "__main__":
    main()
