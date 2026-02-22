import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pruning_utils import load_data, fit_model, calculate_accuracy
import os

def run_experiment(dataset_name, num_runs, output_dir, n_train=None):
    print(f"Loading dataset: {dataset_name}")
    X_train_full, X_test, y_train_full, y_test = load_data(dataset_name)
    
    if n_train is not None:
        X_train_orig = X_train_full[:n_train]
        y_train_orig = y_train_full[:n_train]
    else:
        X_train_orig = X_train_full
        y_train_orig = y_train_full
        
    n_train_samples = len(X_train_orig)
    print(f"Original training set size: {n_train_samples}")
    
    max_copies = n_train_samples * 10
    
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    
    n_values = np.linspace(0, max_copies, num_runs, dtype=int)
    n_values = np.unique(n_values)
    
    print(f"Running experiments for N values: {n_values}")
    
    results = []
    
    x_first = X_train_orig[0:1] # shape (1, n_features)
    y_first = y_train_orig[0:1] # shape (1,)
    
    for n in tqdm(n_values, desc="Processing N values"):
        if n == 0:
            X_train_curr = X_train_orig
            y_train_curr = y_train_orig
        else:
            X_copies = np.tile(x_first, (n, 1))
            y_copies = np.tile(y_first, (n,))
            
            X_train_curr = np.concatenate([X_train_orig, X_copies], axis=0)
            y_train_curr = np.concatenate([y_train_orig, y_copies], axis=0)
            
        classifier = fit_model(X_train_curr, y_train_curr)
        accuracy = calculate_accuracy(classifier, X_test, y_test)
        
        results.append({
            "n_copies": int(n),
            "total_train_size": len(X_train_curr),
            "test_accuracy": accuracy
        })

    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    output_filename = f"{dataset_name}_duplicate_experiment_results.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(df)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run experiment duplicating the first training example.")
    parser.add_argument("--dataset_name", type=str, default="breast_cancer", help="Name of the dataset")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of experiments (values of N) to run")
    parser.add_argument("--output_dir", type=str, default="results/duplicate_experiment", help="Directory to save results")
    parser.add_argument("--n_train", type=int, default=None, help="Limit initial training set size (optional)")
    
    args = parser.parse_args()
    
    run_experiment(dataset_name=args.dataset_name, 
                   num_runs=args.num_runs, 
                   output_dir=args.output_dir,
                   n_train=args.n_train)

if __name__ == "__main__":
    main()
