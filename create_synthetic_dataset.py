import argparse
import numpy as np
import os
from pruning_utils import load_data, create_filename_from_args
from tqdm import trange


def generate_synthetic_dataset(dataset_name, n_samples, output_path):
    X_train, _, _, _ = load_data(dataset_name)
    n_features = X_train.shape[1]
    
    with open(output_path, 'w') as f:
        header = ','.join([f'feature_{i}' for i in range(n_features)])
        f.write(header + '\n')
        
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    mins = np.min(X_train, axis=0)
    maxs = np.max(X_train, axis=0)

    for sample_idx in trange(n_samples):
        sample_values = []
        for feature_idx in range(n_features):
            rng = np.random.RandomState((sample_idx + 1) * n_features + (feature_idx + 1))
            value = rng.normal(means[feature_idx], stds[feature_idx])
            value = np.clip(value, mins[feature_idx], maxs[feature_idx])
            sample_values.append(str(value))
        
        with open(output_path, 'a') as f:
            f.write(','.join(sample_values) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset based on marginal feature distributions.')
    parser.add_argument('--dataset', type=str, default='breast_cancer', help='Name of the source dataset')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of synthetic samples to generate')
    parser.add_argument('--output_dir', type=str, default='results/synthetic_data', help='Directory to save the synthetic dataset')
    parser.add_argument('--force', action='store_true', help='Force generation even if output exists')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = create_filename_from_args(args, extension=".csv", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> create_synthetic_dataset: Skipping (Output already exists at {output_path})")
        return
        
    generate_synthetic_dataset(args.dataset, args.n_samples, output_path)


if __name__ == "__main__":
    main()
