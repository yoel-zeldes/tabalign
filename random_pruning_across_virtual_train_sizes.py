from pruning_utils import load_data, fit_model, backup_caches, create_pruning_config, calculate_accuracy
import experiment_utils
from tqdm import tqdm
import pandas as pd


def main(dataset_name, num_runs, max_virtual_train_size, same_across_layers):
    X_train, X_test, y_train, y_test = load_data(dataset_name)
    assert max_virtual_train_size <= len(X_train)
    virtual_train_sizes = list(range(
        2,  # at least 2 samples are required by TabPFNClassifier
        max_virtual_train_size,
        max_virtual_train_size // num_runs
    ))
    virtual_train_sizes[-1] = max_virtual_train_size
    
    classifier = fit_model(X_train, y_train)
    cache_backup = backup_caches(classifier)

    pruning_results = {}
    for virtual_train_size in tqdm(virtual_train_sizes, desc='Pruning'):
        pruning_config = create_pruning_config(
            classifier,
            len(X_train) - virtual_train_size,
            same_across_layers
        )
        pruning_results[virtual_train_size] = calculate_accuracy(
            classifier,
            X_test,
            y_test,
            pruning_config,
            cache_backup
        )

    baseline_results = {}
    for virtual_train_size in tqdm(virtual_train_sizes, desc='Baseline'):
        classifier = fit_model(X_train[:virtual_train_size], y_train[:virtual_train_size])
        baseline_results[virtual_train_size] = calculate_accuracy(classifier, X_test, y_test)

    res_df = pd.DataFrame({
        "virtual_train_size": list(pruning_results.keys()),
        "pruning_accuracy": list(pruning_results.values()),
        "baseline_accuracy": list(baseline_results.values())
    })
    res_df['advantage'] = res_df['pruning_accuracy'] - res_df['baseline_accuracy']
    print(f"--- {dataset_name} (max train size: {len(X_train)}) ---")
    print(res_df)
    print("\n\n")

    return res_df.to_dict()

        
if __name__ == "__main__":
    config = {
        "num_runs": 5,
        "max_virtual_train_size": 32,
        "same_across_layers": False
    }
    
    results = {
        dataset_name: main(dataset_name, **config)
        for dataset_name in tqdm([
            "breast_cancer",
            "wine",
            "iris",
            "digits"
        ], desc="Datasets")
    }
    
    experiment_utils.save_results(config=config, results=results)
