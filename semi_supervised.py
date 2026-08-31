"""This script evaluates semi-supervised learning methods on tabular datasets.
It compares baseline student models trained only on small labeled subsets against
teacher models trained on full datasets and semi-supervised models augmented with
pseudo-labeled data.
Unlabeled instances receive predicted labels using the student model, which can
either be assigned using argmax prediction or chosen by evaluating multiple
probabilistically sampled label sets, where each set is scored by how well it
explains the student's labeled data.
"""

import argparse
import json
import os
import numpy as np
import pruning_utils
from cli_utils import parse_student_n
from sklearn.metrics import log_loss
from tqdm import tqdm


def sample_pseudo_labels(classes, pseudo_probs, rng):
    """Sample class labels from predicted probability distributions.

    For each example, samples a class label according to the predicted
    probability distribution (instead of taking argmax).
    """
    n_samples = pseudo_probs.shape[0]
    sampled = np.empty(n_samples, dtype=classes.dtype)
    for i in range(n_samples):
        sampled[i] = rng.choice(classes, p=pseudo_probs[i])
    return sampled


def score_pseudo_labels(pseudo_labels, X_unlabeled, X_student, y_student, n_estimators):
    """Score a set of pseudo-labels by how well they explain the labeled student data.

    Fits a model on (X_unlabeled, pseudo_labels), then computes the negative
    log-loss of predicting y_student from X_student. Higher is better.
    """
    model = pruning_utils.fit_model(X_unlabeled, pseudo_labels, n_estimators=n_estimators)
    probs = model.predict_proba(X_student)
    # Negative log-loss so that higher = better
    return -log_loss(y_student, probs)


def select_best_pseudo_labels(
    classes, pseudo_probs, X_unlabeled, X_student, y_student,
    n_estimators, n_pseudo_samples, seed=42,
):
    """Sample n_pseudo_samples sets of pseudo-labels and return the best one.

    Returns:
        best_pseudo_labels: The pseudo-labels with the highest score.
        score_stats: Dict with mean, std, min, max of all scores.
    """
    rng = np.random.RandomState(seed)
    all_pseudo_labels = []
    scores = []

    # Always include the argmax pseudo-labels as the first candidate
    argmax_pl = classes[np.argmax(pseudo_probs, axis=1)]
    sc = score_pseudo_labels(argmax_pl, X_unlabeled, X_student, y_student, n_estimators)
    scores.append(sc)
    all_pseudo_labels.append(argmax_pl)
    print(f"    Argmax baseline: score={sc:.4f}")

    # Sample additional candidates
    while len(all_pseudo_labels) < n_pseudo_samples + 1:
        pl = sample_pseudo_labels(classes, pseudo_probs, rng)
        # Ensure all classes are represented (TabPFN requires it)
        if len(np.unique(pl)) < len(classes):
            # Skip degenerate samples that drop a class
            continue
        sc = score_pseudo_labels(pl, X_unlabeled, X_student, y_student, n_estimators)
        scores.append(sc)
        all_pseudo_labels.append(pl)
        print(f"    Sample {len(all_pseudo_labels)-1}/{n_pseudo_samples}: score={sc:.4f}")

    if not scores:
        raise ValueError("All sampled pseudo-label sets were degenerate (missing classes).")

    scores = np.array(scores)
    best_idx = np.argmax(scores)
    score_stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "argmax_score": float(scores[0]),
        "n_valid_samples": len(scores),
    }
    return all_pseudo_labels[best_idx], score_stats


def fit_and_evaluate_combined(X_student, y_student, X_unlabeled, pseudo_labels,
                              X_test, y_test, n_estimators, label):
    """Combine student data with pseudo-labeled data, fit a model, and evaluate."""
    X_combined = np.concatenate([X_student, X_unlabeled], axis=0)
    y_combined = np.concatenate([y_student, pseudo_labels], axis=0)
    print(f"  Fitting {label} (N={len(X_combined)})...")
    model = pruning_utils.fit_model(X_combined, y_combined, n_estimators=n_estimators)
    probs = model.predict_proba(X_test)
    return pruning_utils.calculate_roc_auc(y_test, probs)


def run_single(dataset, student_n, repeat, n_estimators, n_pseudo_samples, max_num_examples, output_dir):
    """Run a single semi-supervised experiment and return the result path."""
    result_path = pruning_utils.create_filename_from_args(
        {
            "dataset": dataset,
            "student_n": student_n,
            "repeat": repeat,
            "n_estimators": n_estimators,
            "n_pseudo_samples": n_pseudo_samples,
            "output_dir": output_dir,
            "max_num_examples": max_num_examples,
        },
        script_name="semi_supervised",
        extension=".json",
        makedirs=True,
    )
    if os.path.exists(result_path):
        return result_path

    # Load data
    X_train, X_test, y_train, y_test = pruning_utils.load_data(dataset, repeat=repeat, max_num_examples=max_num_examples)

    # Split into student subset and unlabeled rest
    X_student, y_student, X_unlabeled, y_unlabeled_true = (
        pruning_utils.create_student_training_set(
            X_train, y_train, student_n, return_rest=True
        )
    )

    # 1) Student — trained on student subset only
    print(f"  Fitting Student (N={student_n})...")
    student_model = pruning_utils.fit_model(X_student, y_student, n_estimators=n_estimators)
    student_probs = student_model.predict_proba(X_test)
    student_auc = pruning_utils.calculate_roc_auc(y_test, student_probs)

    # 2) Teacher — trained on full original training set
    print(f"  Fitting Teacher (N={len(X_train)})...")
    teacher_model = pruning_utils.fit_model(X_train, y_train, n_estimators=n_estimators)
    teacher_probs = teacher_model.predict_proba(X_test)
    teacher_auc = pruning_utils.calculate_roc_auc(y_test, teacher_probs)

    # 3) Student + Predicted — student predicts labels for unlabeled, then retrain
    print(f"  Student predicting labels for {len(X_unlabeled)} unlabeled examples...")
    pseudo_probs = student_model.predict_proba(X_unlabeled)

    # Argmax pseudo-labels (used as baseline or as the only approach)
    argmax_labels = pruning_utils.predict_from_probabilities(student_model, pseudo_probs)

    if n_pseudo_samples > 0:
        # Sampling-based pseudo-label selection
        print(f"  Sampling {n_pseudo_samples} pseudo-label sets and scoring...")
        pseudo_labels, score_stats = select_best_pseudo_labels(
            classes=student_model.classes_,
            pseudo_probs=pseudo_probs,
            X_unlabeled=X_unlabeled,
            X_student=X_student,
            y_student=y_student,
            n_estimators=n_estimators,
            n_pseudo_samples=n_pseudo_samples,
            seed=repeat,
        )
        print(f"  Best score: {score_stats['max']:.4f} "
              f"(mean={score_stats['mean']:.4f}, std={score_stats['std']:.4f})")

        # Also evaluate argmax labels for comparison
        argmax_combined_auc = fit_and_evaluate_combined(
            X_student, y_student, X_unlabeled, argmax_labels,
            X_test, y_test, n_estimators, label="Student+Argmax",
        )
    else:
        pseudo_labels = argmax_labels
        score_stats = None
        argmax_combined_auc = None

    combined_auc = fit_and_evaluate_combined(
        X_student, y_student, X_unlabeled, pseudo_labels,
        X_test, y_test, n_estimators, label="Student+Predicted",
    )

    # Sanity check: how accurate were the pseudo-labels?
    pseudo_label_acc = (pseudo_labels == y_unlabeled_true).mean()

    result = {
        "config": {
            "dataset": dataset,
            "student_n": student_n,
            "repeat": repeat,
            "n_estimators": n_estimators,
            "n_pseudo_samples": n_pseudo_samples,
            "train_size": len(X_train),
            "unlabeled_size": len(X_unlabeled),
        },
        "metrics": {
            "student_auc": float(student_auc),
            "teacher_auc": float(teacher_auc),
            "combined_auc": float(combined_auc),
            "pseudo_label_accuracy": float(pseudo_label_acc),
        },
    }
    if argmax_combined_auc is not None:
        result["metrics"]["argmax_combined_auc"] = float(argmax_combined_auc)
    if score_stats is not None:
        result["metrics"]["pseudo_label_score_stats"] = score_stats

    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"  Saved results to {result_path}")

    return result_path


def print_summary_table(all_results):
    """Print the summary table sorted by delta (combined - student)."""
    if not all_results:
        return

    from collections import defaultdict

    grouped = defaultdict(lambda: {"student": [], "teacher": [], "combined": [], "argmax_combined": [], "score_stats": []})
    for r in all_results:
        key = (r["config"]["dataset"], r["config"]["student_n"])
        grouped[key]["student"].append(r["metrics"]["student_auc"])
        grouped[key]["teacher"].append(r["metrics"]["teacher_auc"])
        grouped[key]["combined"].append(r["metrics"]["combined_auc"])
        if "argmax_combined_auc" in r["metrics"]:
            grouped[key]["argmax_combined"].append(r["metrics"]["argmax_combined_auc"])
        if "pseudo_label_score_stats" in r["metrics"]:
            grouped[key]["score_stats"].append(r["metrics"]["pseudo_label_score_stats"])

    has_score_stats = any(vals["score_stats"] for vals in grouped.values())
    has_argmax_combined = any(vals["argmax_combined"] for vals in grouped.values())

    header = (
        f"{'Dataset':<55} {'N':>5}  "
        f"{'Teacher AUC':>17}  {'Student AUC':>17}  {'Student+Pred AUC':>17}  {'Delta':>17}"
    )
    if has_argmax_combined:
        header += f"  {'S+Argmax AUC':>17}"
    if has_score_stats:
        header += f"  {'Score (mean±std)':>20}  {'Score [min, max]':>20}  {'Argmax Score':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    def mean_delta(item):
        vals = item[1]
        return np.mean(vals['combined']) - np.mean(vals['student'])

    def fmt(arr):
        return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

    def fmt_score_stats(stats_list):
        """Format pseudo-label score stats averaged across repeats."""
        if not stats_list:
            return "", "", ""
        means = [s["mean"] for s in stats_list]
        stds = [s["std"] for s in stats_list]
        mins = [s["min"] for s in stats_list]
        maxs = [s["max"] for s in stats_list]
        argmax_scores = [s["argmax_score"] for s in stats_list]
        mean_str = f"{np.mean(means):.4f}±{np.mean(stds):.4f}"
        range_str = f"[{np.mean(mins):.4f}, {np.mean(maxs):.4f}]"
        argmax_str = f"{np.mean(argmax_scores):.4f}"
        return mean_str, range_str, argmax_str

    for (dataset, student_n), vals in sorted(grouped.items(), key=mean_delta, reverse=True):
        delta = [c - s for c, s in zip(vals['combined'], vals['student'])]
        line = (
            f"{dataset:<55} {student_n:>5}  "
            f"{fmt(vals['teacher']):>17}  {fmt(vals['student']):>17}  "
            f"{fmt(vals['combined']):>17}  {fmt(delta):>17}"
        )
        if has_argmax_combined and vals["argmax_combined"]:
            line += f"  {fmt(vals['argmax_combined']):>17}"
        elif has_argmax_combined:
            line += f"  {'':>17}"
        if has_score_stats:
            mean_str, range_str, argmax_str = fmt_score_stats(vals["score_stats"])
            line += f"  {mean_str:>20}  {range_str:>20}  {argmax_str:>14}"
        print(line)

    # Average row
    all_student = [np.mean(v["student"]) for v in grouped.values()]
    all_teacher = [np.mean(v["teacher"]) for v in grouped.values()]
    all_combined = [np.mean(v["combined"]) for v in grouped.values()]
    all_delta = [np.mean(v["combined"]) - np.mean(v["student"]) for v in grouped.values()]
    all_argmax_combined = [np.mean(v["argmax_combined"]) for v in grouped.values() if v["argmax_combined"]]

    def fmt_avg(arr):
        return f"{np.mean(arr):.4f}"

    print("-" * len(header))
    avg_line = (
        f"{'AVERAGE':<55} {'':>5}  "
        f"{fmt_avg(all_teacher):>17}  {fmt_avg(all_student):>17}  "
        f"{fmt_avg(all_combined):>17}  {fmt_avg(all_delta):>17}"
    )
    if has_argmax_combined and all_argmax_combined:
        avg_line += f"  {fmt_avg(all_argmax_combined):>17}"
    print(avg_line)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Semi-supervised evaluation: student, teacher, and student+predicted (pseudo-labels)."
    )
    parser.add_argument("--dataset", type=str, nargs="+", default=["tabarena"])
    parser.add_argument(
        "--student_n", type=parse_student_n, nargs="+", default=[20],
        help="Student training sizes to sweep.",
    )
    parser.add_argument("--n_repeats", type=int, default=3, help="Number of repeats per configuration.")
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument(
        "--n_pseudo_samples", type=int, default=0,
        help="Number of pseudo-label samples to try. 0 = use argmax (original behavior).",
    )
    parser.add_argument("--max_num_examples", type=int, default=10000, help="Max training examples to use.")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # Expand "tabarena" shortcut
    datasets = []
    for d in args.dataset:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in pruning_utils.TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    # Collect all results: list of dicts
    all_results = []

    for dataset in tqdm(datasets, desc="Datasets"):
        for student_n in tqdm(sorted(args.student_n), desc="Student sizes", leave=False):
            for repeat in tqdm(range(args.n_repeats), desc=f"Repeats (N={student_n})", leave=False):
                result_path = run_single(
                    dataset, student_n, repeat, args.n_estimators,
                    args.n_pseudo_samples, args.max_num_examples, args.output_dir,
                )
                if result_path is None:
                    break  # skip remaining repeats for this student_n

                with open(result_path, "r") as f:
                    all_results.append(json.load(f))

                # Print updated table after each iteration
                print_summary_table(all_results)

    # Final summary
    if not all_results:
        print("No results collected.")
        return

    print("\n\n*** FINAL RESULTS ***")
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
