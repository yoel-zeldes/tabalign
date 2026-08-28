import argparse
import json
import os
import time
from typing import Any, Dict, Tuple

import optuna
from optuna.samplers import RandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm
from pruning_utils import get_device, create_filename_from_args, parse_student_n

def suggest_aligner_hyperparams(trial: optuna.Trial) -> Dict[str, Any]:
    hidden_layers_options = {
        "linear": [],
        "mlp_1": [1],
    }
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    hidden_layers = hidden_layers_options[trial.suggest_categorical("arch_type", list(hidden_layers_options.keys()))]
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-6, log=True)

    return {
        "lr": lr,
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "weight_decay": weight_decay,
    }


def validate_metadata(s_meta, t_meta):
    """Ensure student and teacher metadata match."""
    if s_meta["dataset"] != t_meta["dataset"]:
        raise ValueError(f"Dataset mismatch: {s_meta['dataset']} vs {t_meta['dataset']}")
    if s_meta["layer_k"] != t_meta["layer_k"]:
        raise ValueError(f"Layer mismatch: {s_meta['layer_k']} vs {t_meta['layer_k']}")
    if s_meta["n_estimators"] != t_meta["n_estimators"]:
        raise ValueError(f"Estimators mismatch: {s_meta['n_estimators']} vs {t_meta['n_estimators']}")
    print(f"Metadata verified for {s_meta['dataset']} @ Layer {s_meta['layer_k']}")

def prepare_dataloaders(s_activations, t_activations):
    """Squeeze, flatten, and split activations into train/val datasets.
    
    Returns train_ds, val_ds (TensorDatasets) and hidden_dim.
    """
    s_activations = s_activations.squeeze().float()
    t_activations = t_activations.squeeze().float()
    
    if s_activations.shape != t_activations.shape:
        raise ValueError(f"Activation shape mismatch: {s_activations.shape} vs {t_activations.shape}")
        
    hidden_dim = s_activations.shape[-1]
    
    X = s_activations.reshape(-1, hidden_dim)  # [N*n_tokens, hidden_dim]
    Y = t_activations.reshape(-1, hidden_dim) - X  # we're predicting residuals
    
    indices = torch.randperm(X.shape[0])
    split = int(0.8 * X.shape[0])
    
    train_ds = TensorDataset(X[indices[:split]], Y[indices[:split]])
    val_ds = TensorDataset(X[indices[split:]], Y[indices[split:]])
    
    return train_ds, val_ds, hidden_dim


def prepare_concat_datasets(
    s_act_list,
    t_act_list,
) -> Tuple[ConcatDataset, ConcatDataset, int]:
    """Prepare and concatenate train/val datasets across all input activation sets."""
    train_datasets, val_datasets = [], []
    hidden_dim = None
    for s_act, t_act in zip(s_act_list, t_act_list):
        train_ds, val_ds, curr_hidden_dim = prepare_dataloaders(s_act, t_act)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        if hidden_dim is None:
            hidden_dim = curr_hidden_dim
        elif hidden_dim != curr_hidden_dim:
            raise ValueError(f"Hidden dim mismatch across datasets: {hidden_dim} vs {curr_hidden_dim}")

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), hidden_dim


def build_aligner_model(
    input_dim,
    output_dim,
    hyperparams,
):
    """Build an MLP aligner model.
    
    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        hyperparams: Dictionary of hyperparams.
    """
    hidden_layers = [multiplier * input_dim for multiplier in hyperparams["hidden_layers"]]
    layers = []
    for h_dim in hidden_layers:
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.ReLU())
        input_dim = h_dim
    final_layer = nn.Linear(input_dim, output_dim)
    # Near-zero init: scale weights and bias so initial prediction
    # is ~0 (because we predict residuals)
    nn.init.normal_(final_layer.weight, std=1e-3)
    nn.init.normal_(final_layer.bias, std=1e-3)
    layers.append(final_layer)
    return nn.Sequential(*layers)

def train_estimator(
    est_idx,
    train_loader,
    val_loader,
    model,
    lr,
    patience,
    device,
    max_epochs=None,
    weight_decay=0.0,
):
    """Train an aligner for a single estimator.
    
    Trains indefinitely until dev loss does not improve for `patience` consecutive epochs,
    or until max_epochs is reached (if specified).
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    epoch = 0
    
    desc = f"Est {est_idx}"
        
    pbar = tqdm(desc=desc, leave=False)
    while max_epochs is None or epoch < max_epochs:
        epoch += 1
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_x), batch_y).item()
        
        val_loss /= len(val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        pbar.update(1)
        pbar.set_postfix({
            "epoch": epoch,
            "val_loss": f"{val_loss:.6f}",
            "best": f"{best_val_loss:.6f}",
            "no_improve": f"{epochs_without_improvement}/{patience}"
        })
        
        if epochs_without_improvement >= patience:
            break
    
    pbar.close()
    return best_model_state, best_val_loss


def _train_aligner_on_datasets(
    est_idx,
    train_ds,
    val_ds,
    hidden_dim,
    device,
    hyperparams,
    patience,
    max_epochs,
):
    """Build model, construct DataLoaders, and train aligner on train/val datasets."""
    train_loader = DataLoader(train_ds, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hyperparams["batch_size"])

    model = build_aligner_model(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        hyperparams=hyperparams,
    ).to(device)

    return train_estimator(
        est_idx=est_idx,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        lr=hyperparams["lr"],
        patience=patience,
        device=device,
        max_epochs=max_epochs,
        weight_decay=hyperparams["weight_decay"],
    )


def run_aligner_optuna_search(
    s_act_list,
    t_act_list,
    device,
    patience: int = 10,
    n_trials: int = 100,
    timeout: int = 600,
    max_epochs: int | None = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float, int, float, list[Dict[str, Any]]]:
    """Runs Optuna hyperparameter optimization using validation MSE on the training data."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    concat_train_ds, concat_val_ds, hidden_dim = prepare_concat_datasets(
        s_act_list, t_act_list
    )

    def objective(trial: optuna.Trial) -> float:
        hyperparams = suggest_aligner_hyperparams(trial)
        _, best_val_loss = _train_aligner_on_datasets(
            est_idx=0,
            train_ds=concat_train_ds,
            val_ds=concat_val_ds,
            hidden_dim=hidden_dim,
            device=device,
            hyperparams=hyperparams,
            patience=patience,
            max_epochs=max_epochs,
        )
        return best_val_loss

    sampler = RandomSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    pbar = tqdm(total=100, desc="Aligner HP Search", unit="%")
    start_time = time.time()

    def callback(study, trial):
        elapsed = time.time() - start_time
        trials_ratio = len(study.trials) / n_trials
        time_ratio = elapsed / timeout
        target_n = int(max(trials_ratio, time_ratio) * 100)
        pbar.update(target_n - pbar.n)
        time_str = f"{int(elapsed)}s/{timeout}s"
        pbar.set_description(f"Aligner HP Search (trials: {len(study.trials)}/{n_trials}, time: {time_str})")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[callback])
    pbar.close()
    duration = time.time() - start_time

    best_hyperparams = suggest_aligner_hyperparams(study.best_trial)
    best_score = float(study.best_value)
    completed_trials = len(study.trials)

    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                "trial_number": trial.number,
                "params": trial.params,
                "val_loss": float(trial.value),
                "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            })

    return best_hyperparams, best_score, completed_trials, duration, trials_data

def save_aligner(
    output_path,
    students_metadata,
    estimator_idx_to_aligner,
    avg_val_loss,
    hyperparams,
    n_stats=0,
):
    """Save the ensemble of aligner models and metadata."""
    save_obj = {
        "metadata": {
            "students_metadata": students_metadata,
            "avg_mse_loss": avg_val_loss,
            "n_stats": n_stats,
        },
        "hyperparams": hyperparams,
        "estimator_idx_to_aligner": estimator_idx_to_aligner
    }
    
    torch.save(save_obj, output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Train activation aligner")
    parser.add_argument("--dataset", type=str, nargs='+', default=["tabarena/Amazon_employee_access[synthetic]"], help="One or more dataset names. Activations from all datasets are combined (80/20 split per dataset).")
    parser.add_argument("--student_n", type=parse_student_n, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--patience", type=int, default=10, help="Stop training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_layers", type=int, nargs='*', default=[], help="Hidden layer multipliers for MLP aligner. Empty = linear.")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--force", action="store_true", help="Force training even if output exists.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of training epochs. None = unlimited (rely on patience).")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use.")
    parser.add_argument("--opt", action="store_true", help="Perform hyperparameter search.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials to run during hyperparameter search.")
    parser.add_argument("--timeout", type=int, default=600, help="Time budget in seconds for the Optuna study.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna search.")
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    exclude_args = ["n_trials", "timeout", "seed"]
    if not args.opt:
        exclude_args.append("opt")
    output_path = create_filename_from_args(args, extension=".pt", makedirs=True, exclude_args=exclude_args)
    if os.path.exists(output_path) and not args.force:
        print(f">>> train_activation_aligner: Skipping (Output already exists at {output_path})")
        return

    # Load activations for every dataset
    all_student_data = []
    all_teacher_data = []
    for dataset in args.dataset:
        teacher_path = create_filename_from_args({
            "dataset": dataset,
            "student_n": -1,
            "layer_k": args.layer_k,
            "n_estimators": args.n_estimators,
            "repeat": args.repeat,
            "output_dir": args.output_dir,
            "model": args.model,
        }, script_name="extract_activations", extension=".pt")

        student_path = create_filename_from_args({
            "dataset": dataset,
            "student_n": args.student_n,
            "layer_k": args.layer_k,
            "n_estimators": args.n_estimators,
            "repeat": args.repeat,
            "output_dir": args.output_dir,
            "model": args.model,
        }, script_name="extract_activations", extension=".pt")

        print(f"Loading activations for '{dataset}':\n  Teacher: {teacher_path}\n  Student: {student_path}")
        student_data = torch.load(student_path, weights_only=False)
        teacher_data = torch.load(teacher_path, weights_only=False)
        validate_metadata(student_data["metadata"], teacher_data["metadata"])
        all_student_data.append(student_data)
        all_teacher_data.append(teacher_data)

    
    n_estimators, *rest = {student_data["metadata"]["n_estimators"] for student_data in all_student_data}
    if rest:
        raise ValueError(f"All datasets must have the same number of estimators. Found: {n_estimators} and {rest}")
    device = get_device()

    if args.opt:
        best_hyperparams_path = create_filename_from_args(
            args,
            exclude_args=["repeat"],
            extension=".best_hyperparams.json",
            makedirs=True,
        )

        if os.path.exists(best_hyperparams_path) and not args.force:
            print(f"Found existing best hyperparameters at {best_hyperparams_path}. Loading and skipping search...")
            with open(best_hyperparams_path, "r") as f:
                hyperparams = json.load(f)["best_hyperparams"]
        else:
            print(f"Starting Optuna hyperparameter search ({args.n_trials} trials, timeout={args.timeout}s)...")
            # for efficiency, only use the first estimator.
            # All estimators share the same architecture, so the optimal hyperparameters probably transfer
            hyperparams, best_score, completed_trials, duration, trials_data = run_aligner_optuna_search(
                s_act_list=[sd["activations"][0] for sd in all_student_data],
                t_act_list=[td["activations"][0] for td in all_teacher_data],
                device=device,
                patience=args.patience,
                n_trials=args.n_trials,
                timeout=args.timeout,
                max_epochs=args.max_epochs,
                seed=args.seed,
            )

            with open(best_hyperparams_path, "w") as f:
                json.dump({
                    "best_hyperparams": hyperparams,
                    "best_val_loss": best_score,
                    "n_trials": completed_trials,
                    "study_duration_seconds": duration,
                    "trials": trials_data,
                }, f, indent=4)
    else:
        hyperparams = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "hidden_layers": args.hidden_layers,
            "weight_decay": 0.0,
        }

    estimator_idx_to_aligner = {}
    total_val_loss = 0
    num_trained_models = 0

    for est_idx in range(n_estimators):
        s_act_list = [sd["activations"][est_idx] for sd in all_student_data]
        t_act_list = [td["activations"][est_idx] for td in all_teacher_data]

        train_ds, val_ds, hidden_dim = prepare_concat_datasets(
            s_act_list, t_act_list
        )
        state_dict, best_loss = _train_aligner_on_datasets(
            est_idx=est_idx,
            train_ds=train_ds,
            val_ds=val_ds,
            hidden_dim=hidden_dim,
            device=device,
            hyperparams=hyperparams,
            patience=args.patience,
            max_epochs=args.max_epochs,
        )
        estimator_idx_to_aligner[est_idx] = state_dict
        total_val_loss += best_loss
        num_trained_models += 1

    avg_mse = total_val_loss / num_trained_models
    save_aligner(
        output_path,
        [student_data["metadata"] for student_data in all_student_data],
        estimator_idx_to_aligner,
        avg_mse,
        hyperparams=hyperparams,
    )

    print(f"\nTraining complete. Avg MSE: {avg_mse:.6f}")
    print(f"All models saved to {output_path}")

if __name__ == "__main__":
    main()
