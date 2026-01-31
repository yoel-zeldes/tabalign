from sklearn.model_selection import train_test_split
from pruning_utils import load_data, fit_model, backup_caches, create_pruning_config, calculate_accuracy
from tqdm import trange
import pandas as pd


def main(dataset_name, num_runs, virtual_train_size, same_across_layers, dev_size):
    X_train_full, X_test, y_train_full, y_test = load_data(dataset_name)
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full,
        y_train_full,
        test_size=dev_size,
        random_state=42
    )
    if virtual_train_size > len(X_train):
        raise ValueError("virtual_train_size must be smaller than or equal to len(X_train)")
    
    classifier = fit_model(X_train[:virtual_train_size], y_train[:virtual_train_size])
    baseline_dev_accuracy = calculate_accuracy(classifier, X_dev, y_dev)
    baseline_test_accuracy = calculate_accuracy(classifier, X_test, y_test)

    classifier = fit_model(X_train, y_train)
    cache_backup = backup_caches(classifier)

    best_dev_accuracy = -1.0
    best_seed = None
    
    pruning_dev_accuracies = []
    for seed in trange(1, num_runs + 1, desc='Pruning'):
        pruning_config = create_pruning_config(
            classifier,
            len(X_train) - virtual_train_size,
            same_across_layers,
            seed=seed
        )
        dev_accuracy = calculate_accuracy(classifier, X_dev, y_dev, pruning_config, cache_backup)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_seed = seed
        pruning_dev_accuracies.append(dev_accuracy)

        pruning_dev_accuracies_describe = pd.Series(pruning_dev_accuracies).describe()
        print(f"--- {dataset_name} (train size: {len(X_train)}, dev size: {len(X_dev)}, test size: {len(X_test)}) ---")
        print("Dev set accuracy statistics:")
        print(pruning_dev_accuracies_describe)
        print(f"Baseline dev set accuracy: {baseline_dev_accuracy}\n\n")
    
    pruning_config = create_pruning_config(
        classifier,
        len(X_train) - virtual_train_size,
        same_across_layers,
        seed=best_seed
    )
    best_run_test_accuracy = calculate_accuracy(classifier, X_test, y_test, pruning_config, cache_backup)
    print(f"Best run (seed = {best_seed}) achieved dev accuracy: {best_dev_accuracy}")
    print(f"Test accuracy of best run: {best_run_test_accuracy}")
    print(f"Baseline test accuracy: {baseline_test_accuracy}")


if __name__ == "__main__":
    dataset = 'digits'
    num_runs = 200
    virtual_train_size = 16
    same_across_layers = False
    dev_size = 0.2
    main(dataset, num_runs, virtual_train_size, same_across_layers, dev_size)
