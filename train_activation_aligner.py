import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from pruning_utils import get_device, create_filename_from_args

def validate_metadata(s_meta, t_meta):
    """Ensure student and teacher metadata match."""
    if s_meta["dataset"] != t_meta["dataset"]:
        raise ValueError(f"Dataset mismatch: {s_meta['dataset']} vs {t_meta['dataset']}")
    if s_meta["layer_k"] != t_meta["layer_k"]:
        raise ValueError(f"Layer mismatch: {s_meta['layer_k']} vs {t_meta['layer_k']}")
    if s_meta["n_estimators"] != t_meta["n_estimators"]:
        raise ValueError(f"Estimators mismatch: {s_meta['n_estimators']} vs {t_meta['n_estimators']}")
    print(f"Metadata verified for {s_meta['dataset']} @ Layer {s_meta['layer_k']}")

def prepare_dataloaders(s_activations, t_activations, batch_size, predict_residual=False, token_idx=None):
    """Squeeze, flatten, and split activations into train/val loaders."""
    s_activations = s_activations.squeeze()
    t_activations = t_activations.squeeze()
    
    if s_activations.shape != t_activations.shape:
        raise ValueError(f"Activation shape mismatch: {s_activations.shape} vs {t_activations.shape}")
        
    test_size, n_tokens, hidden_dim = s_activations.shape
    
    if token_idx is not None:
        X = s_activations[:, token_idx, :]
        Y = t_activations[:, token_idx, :]
    else:
        X = s_activations.reshape(-1, hidden_dim)
        Y = t_activations.reshape(-1, hidden_dim)
    if predict_residual:
        Y = Y - X
    
    indices = torch.randperm(X.shape[0])
    split = int(0.8 * X.shape[0])
    
    train_ds = TensorDataset(X[indices[:split]], Y[indices[:split]])
    val_ds = TensorDataset(X[indices[split:]], Y[indices[split:]])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, hidden_dim

def build_aligner_model(hidden_dim, hidden_layers, predict_residual=False):
    """Build an MLP aligner model.
    
    Args:
        hidden_dim: Input and output dimension.
        hidden_layers: List of ints for hidden layer sizes.
        predict_residual: If True, apply near-zero init on the final layer so
            the model starts by outputting near-zero residuals.
    """
    layers = []
    in_dim = hidden_dim
    for h_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.ReLU())
        in_dim = h_dim
    final_layer = nn.Linear(in_dim, hidden_dim)
    if predict_residual:
        # Near-zero init: scale weights and bias so initial residuals ≈ 0
        nn.init.normal_(final_layer.weight, std=1e-3)
        nn.init.normal_(final_layer.bias, std=1e-3)
    layers.append(final_layer)
    return nn.Sequential(*layers)

def train_estimator(est_idx, train_loader, val_loader, model, lr, patience, device, token_idx=None):
    """Train an aligner for a single estimator (and optionally a single token).
    
    Trains indefinitely until dev loss does not improve for `patience` consecutive epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    epoch = 0
    
    desc = f"Est {est_idx}"
    if token_idx is not None:
        desc += f" Token {token_idx}"
        
    pbar = tqdm(desc=desc, leave=False)
    while True:
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
    
    return best_model_state, best_val_loss

def _train_single_aligner(est_idx, s_act, t_act, args, device, token_idx=None):
    """Encapsulate data preparation and training for a single aligner."""
    train_loader, val_loader, hidden_dim = prepare_dataloaders(
        s_act, t_act, args.batch_size, predict_residual=args.predict_residual, token_idx=token_idx
    )
    model = build_aligner_model(hidden_dim, args.hidden_layers, predict_residual=args.predict_residual).to(device)
    return train_estimator(
        est_idx,
        train_loader,
        val_loader,
        model,
        args.lr,
        args.patience,
        device,
        token_idx
    )

def save_aligner(output_path, student_metadata, estimator_idx_to_aligner, avg_val_loss, per_token, hidden_layers, predict_residual):
    """Save the ensemble of aligner models and metadata."""
    save_obj = {
        "metadata": {
            **student_metadata,
            "avg_mse_loss": avg_val_loss,
            "per_token": per_token,
            "hidden_layers": hidden_layers,
            "predict_residual": predict_residual
        },
        "estimator_idx_to_aligner": estimator_idx_to_aligner
    }
    
    torch.save(save_obj, output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Train activation aligner")
    parser.add_argument("--dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument("--student_n", type=int, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--patience", type=int, default=10, help="Stop training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--per_token", action="store_true", help="Train a separate aligner for each token position")
    parser.add_argument("--hidden_layers", type=int, nargs='*', default=[], help="Hidden layer sizes for MLP aligner. Empty = linear.")
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--force", action="store_true", help="Force training even if output exists.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = create_filename_from_args(args, extension=".pt", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> train_activation_aligner: Skipping (Output already exists at {output_path})")
        return

    teacher_path = create_filename_from_args({
        "dataset": args.dataset,
        "student_n": -1,
        "layer_k": args.layer_k,
        "n_estimators": args.n_estimators,
        "output_dir": args.output_dir
    }, script_name="extract_activations", extension=".pt")
    
    student_path = create_filename_from_args({
        "dataset": args.dataset,
        "student_n": args.student_n,
        "layer_k": args.layer_k,
        "n_estimators": args.n_estimators,
        "output_dir": args.output_dir
    }, script_name="extract_activations", extension=".pt")
    
    print(f"Loading activations from:\n  Teacher: {teacher_path}\n  Student: {student_path}")
    student_data = torch.load(student_path)
    teacher_data = torch.load(teacher_path)
    
    validate_metadata(student_data["metadata"], teacher_data["metadata"])
    
    n_estimators = student_data["metadata"]["n_estimators"]
    device = get_device()
    
    estimator_idx_to_aligner = {}
    total_val_loss = 0
    num_trained_models = 0
    
    for est_idx in range(n_estimators):
        s_act = student_data["activations"][est_idx]
        t_act = teacher_data["activations"][est_idx]
        
        if args.per_token:
            token_idx_to_aligner = {}
            n_tokens = s_act.shape[2] # shape is [1, N, Tokens, Hidden]
            est_val_loss = 0
            
            pbar = trange(n_tokens, desc=f"Estimator {est_idx}", leave=False)
            for token_idx in pbar:
                state_dict, best_loss = _train_single_aligner(
                    est_idx, s_act, t_act, args, device, token_idx=token_idx
                )
                token_idx_to_aligner[token_idx] = state_dict
                est_val_loss += best_loss
                num_trained_models += 1
                
            estimator_idx_to_aligner[est_idx] = token_idx_to_aligner
            total_val_loss += est_val_loss
            pbar.set_postfix({"avg_token_mse": f"{est_val_loss / n_tokens:.6f}"})
        else:
            state_dict, best_loss = _train_single_aligner(est_idx, s_act, t_act, args, device)
            estimator_idx_to_aligner[est_idx] = state_dict
            total_val_loss += best_loss
            num_trained_models += 1

    avg_mse = total_val_loss / num_trained_models
    save_aligner(
        output_path,
        student_data["metadata"],
        estimator_idx_to_aligner,
        avg_mse,
        args.per_token,
        args.hidden_layers,
        args.predict_residual
    )
    
    print(f"\nTraining complete. Avg MSE: {avg_mse:.6f}")
    print(f"All models saved to {output_path}")

if __name__ == "__main__":
    main()
