import argparse
import os
import numpy as np
from tqdm import tqdm
from data_utils import load_data
from model_utils import fit_model


def get_args():
    parser = argparse.ArgumentParser(description='Dump teacher logits for a synthetic dataset.')
    parser.add_argument('--dataset', type=str, default='breast_cancer', help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for inference')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to use from the test set (default: all)')
    parser.add_argument('--output_dir', type=str, default='results/teacher', help='Directory to save results')
    return parser.parse_args()


def run_inference(classifier, X_test, batch_size):
    logits_list = []
    for i in tqdm(range(0, len(X_test), batch_size)):
        logits_list.append(classifier.predict_logits(X_test[i : i + batch_size]))
    return np.concatenate(logits_list, axis=0)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    X_train, X_test, y_train, _ = load_data(f"{args.dataset}[synthetic]")
    if args.n_samples is not None:
        X_test = X_test[:args.n_samples]
    n_samples = len(X_test)

    classifier = fit_model(X_train, y_train)
    all_logits = run_inference(classifier, X_test, args.batch_size)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_logits_{n_samples}.npy")
    np.save(output_path, all_logits)


if __name__ == "__main__":
    main()
