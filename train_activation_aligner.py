import time
from typing import Any, Dict, Tuple

import optuna
from optuna.samplers import RandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from consts import (
    TABFM_DEFAULT_N_ESTIMATORS,
    TABPFN_DEFAULT_N_ESTIMATORS,
    TABFM_DEFAULT_LR,
    TABPFN_DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
)
from model_utils import memory
from utils import get_device
from extract_activations import extract_activations

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

def prepare_train_val(s_activations, t_activations, device):
    """Flatten, compute residuals, and split activations into train/val tensors.
    
    Returns train_x, train_y, val_x, val_y.
    """
    if s_activations.shape != t_activations.shape:
        raise ValueError(f"Activation shape mismatch: {s_activations.shape} vs {t_activations.shape}")
        
    hidden_dim = s_activations.shape[-1]
    
    X = s_activations.detach().reshape(-1, hidden_dim).float().to(device, non_blocking=True)
    T = t_activations.detach().reshape(-1, hidden_dim).float().to(device, non_blocking=True)
    Y = T - X  # we're predicting residuals
    
    indices = torch.randperm(X.shape[0], device=device)
    split = int(0.8 * X.shape[0])
    
    train_x = X[indices[:split]]
    train_y = Y[indices[:split]]
    val_x = X[indices[split:]]
    val_y = Y[indices[split:]]
    
    return train_x, train_y, val_x, val_y


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

def _train_aligner_on_tensors(
    est_idx,
    train_x,
    train_y,
    val_x,
    val_y,
    device,
    hyperparams,
    patience,
    max_epochs,
    show_pbar=True,
):
    """Train an aligner for a single estimator.
    
    Trains until dev loss does not improve for `patience` consecutive epochs,
    or until max_epochs is reached (if specified).
    """
    hidden_dim = train_x.shape[-1]
    model = build_aligner_model(
        input_dim=hidden_dim,
        output_dim=hidden_dim,
        hyperparams=hyperparams,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"]
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    epoch = 0
    
    n_train = train_x.shape[0]
    desc = f"Est {est_idx}"
        
    pbar = tqdm(desc=desc, leave=False) if show_pbar else None
    while max_epochs is None or epoch < max_epochs:
        epoch += 1
        model.train()
        perm = torch.randperm(n_train, device=device)  # shuffle train data every epoch
        for i in range(0, n_train, hyperparams["batch_size"]):
            idx = perm[i : i + hyperparams["batch_size"]]
            batch_x, batch_y = train_x[idx], train_y[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Keep state clone on device during training to avoid CPU sync overhead
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "epoch": epoch,
                "val_loss": f"{val_loss:.6f}",
                "best": f"{best_val_loss:.6f}",
                "no_improve": f"{epochs_without_improvement}/{patience}"
            })
        
        if epochs_without_improvement >= patience:
            break
    
    if pbar is not None:
        pbar.close()

    # Move best state dict to CPU once at the end for persistence/caching
    if best_model_state is not None:
        best_model_state = {k: v.cpu() for k, v in best_model_state.items()}
    else:
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    best_epoch = epoch - epochs_without_improvement
    return best_model_state, best_val_loss, best_epoch


def run_aligner_optuna_search(
    s_act,
    t_act,
    device,
    patience: int = 10,
    n_trials: int = 100,
    timeout: int = 600,
    max_epochs: int | None = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], float, int, float, list[Dict[str, Any]]]:
    """Runs Optuna hyperparameter optimization using validation MSE on the training data."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_x, train_y, val_x, val_y = prepare_train_val(s_act, t_act, device=device)

    def objective(trial: optuna.Trial) -> float:
        hyperparams = suggest_aligner_hyperparams(trial)
        _, best_val_loss, _ = _train_aligner_on_tensors(
            est_idx=0,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            device=device,
            hyperparams=hyperparams,
            patience=patience,
            max_epochs=max_epochs,
            show_pbar=False,
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


# ignore repeat because we use the same optimal hyperparameters for all repeats of a dataset, because we want to save compute.
@memory.cache(ignore=["repeat"])
def find_best_aligner_hyperparams(dataset, student_n, layer_k, n_estimators=None,
                                   patience=10, repeat=0, model="tabpfn",
                                   n_trials=100, timeout=600, max_epochs=None, seed=42):
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS
    student_data = extract_activations(
        dataset=dataset, student_n=student_n, layer_k=layer_k,
        n_estimators=n_estimators, repeat=repeat, model=model,
    )
    teacher_data = extract_activations(
        dataset=dataset, student_n=-1, layer_k=layer_k,
        n_estimators=n_estimators, repeat=repeat, model=model,
    )
    device = get_device()
    hyperparams, best_score, completed_trials, duration, trials_data = run_aligner_optuna_search(
        s_act=student_data["activations"][0],
        t_act=teacher_data["activations"][0],
        device=device,
        patience=patience,
        n_trials=n_trials,
        timeout=timeout,
        max_epochs=max_epochs,
        seed=seed,
    )
    return {
        "best_hyperparams": hyperparams,
        "best_val_loss": best_score,
        "n_trials": completed_trials,
        "study_duration_seconds": duration,
        "trials": trials_data,
    }


@memory.cache
def train_aligner(dataset, student_n, layer_k, n_estimators=None, patience=10, lr=None,
                  batch_size=2048, hidden_layers=None, repeat=0, max_epochs=None,
                  model="tabpfn", opt=False, n_trials=100, timeout=600, seed=42):
    if hidden_layers is None:
        hidden_layers = []
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS
    if lr is None:
        lr = TABFM_DEFAULT_LR if model == "tabfm" else TABPFN_DEFAULT_LR


    student_data = extract_activations(
        dataset=dataset, student_n=student_n, layer_k=layer_k,
        n_estimators=n_estimators, repeat=repeat, model=model,
    )
    teacher_data = extract_activations(
        dataset=dataset, student_n=-1, layer_k=layer_k,
        n_estimators=n_estimators, repeat=repeat, model=model,
    )
    validate_metadata(student_data["metadata"], teacher_data["metadata"])

    device = get_device()

    if opt:
        hp_result = find_best_aligner_hyperparams(
            dataset=dataset, student_n=student_n, layer_k=layer_k,
            n_estimators=n_estimators, patience=patience, repeat=repeat,
            model=model, n_trials=n_trials, timeout=timeout,
            max_epochs=max_epochs, seed=seed,
        )
        hyperparams = hp_result["best_hyperparams"]
    else:
        hyperparams = {
            "lr": lr,
            "batch_size": batch_size,
            "hidden_layers": hidden_layers,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
        }

    estimator_idx_to_aligner = {}
    total_val_loss = 0
    total_epochs = 0

    start_time = time.time()
    for est_idx in trange(n_estimators, desc="Training aligners for estimators"):
        s_act = student_data["activations"][est_idx]
        t_act = teacher_data["activations"][est_idx]

        train_x, train_y, val_x, val_y = prepare_train_val(s_act, t_act, device=device)
        state_dict, best_loss, num_epochs = _train_aligner_on_tensors(
            est_idx=est_idx,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            device=device,
            hyperparams=hyperparams,
            patience=patience,
            max_epochs=max_epochs,
            show_pbar=True,
        )
        estimator_idx_to_aligner[est_idx] = state_dict
        total_val_loss += best_loss
        total_epochs += num_epochs

    training_time = time.time() - start_time
    avg_mse = total_val_loss / n_estimators
    avg_num_epochs = total_epochs / n_estimators
    return {
        "metadata": {
            "student_metadata": student_data["metadata"],
            "avg_mse_loss": avg_mse,
            "avg_num_epochs": avg_num_epochs,
            "n_stats": 0,
            "training_time": training_time,
        },
        "hyperparams": hyperparams,
        "estimator_idx_to_aligner": estimator_idx_to_aligner,
    }
