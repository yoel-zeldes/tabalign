import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from pruning_utils import get_device

def validate_metadata(s_meta, t_meta):
    """Ensure student and teacher metadata match."""
    if s_meta["dataset"] != t_meta["dataset"]:
        raise ValueError(f"Dataset mismatch: {s_meta['dataset']} vs {t_meta['dataset']}")
    if s_meta["layer_k"] != t_meta["layer_k"]:
        raise ValueError(f"Layer mismatch: {s_meta['layer_k']} vs {t_meta['layer_k']}")
    if s_meta["n_estimators"] != t_meta["n_estimators"]:
        raise ValueError(f"Estimators mismatch: {s_meta['n_estimators']} vs {t_meta['n_estimators']}")
    print(f"Metadata verified for {s_meta['dataset']} @ Layer {s_meta['layer_k']}")

def prepare_dataloaders(s_activations, t_activations, batch_size, token_idx=None):
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
    
    indices = torch.randperm(X.shape[0])
    split = int(0.8 * X.shape[0])
    
    train_ds = TensorDataset(X[indices[:split]], Y[indices[:split]])
    val_ds = TensorDataset(X[indices[split:]], Y[indices[split:]])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    return train_loader, val_loader, hidden_dim

def train_estimator(est_idx, train_loader, val_loader, hidden_dim, lr, epochs, device, token_idx=None):
    """Train a linear aligner for a single estimator (and optionally a single token)."""
    model = nn.Linear(hidden_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    desc = f"Est {est_idx}"
    if token_idx is not None:
        desc += f" Token {token_idx}"
        
    pbar = trange(epochs, desc=desc, leave=False)
    for _ in pbar:
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
            
        pbar.set_postfix({"val_loss": f"{val_loss:.6f}", "best": f"{best_val_loss:.6f}"})
    
    return best_model_state, best_val_loss

def _train_single_aligner(est_idx, s_act, t_act, args, device, token_idx=None):
    """Encapsulate data preparation and training for a single aligner."""
    train_loader, val_loader, hidden_dim = prepare_dataloaders(
        s_act, t_act, args.batch_size, token_idx=token_idx
    )
    return train_estimator(
        est_idx, train_loader, val_loader, hidden_dim,
        args.lr, args.epochs, device, token_idx=token_idx
    )

def save_aligner(s_meta, estimator_idx_to_aligner, avg_val_loss, n_estimators, output_dir, per_token):
    """Save the ensemble of aligner models and metadata."""
    save_obj = {
        "metadata": {
            **s_meta,
            "avg_mse_loss": avg_val_loss,
            "per_token": per_token
        },
        "estimator_idx_to_aligner": estimator_idx_to_aligner
    }
    
    safe_name = s_meta["dataset"].replace(" ", "_")
    token_label = "_per_token" if per_token else ""
    model_name = f"aligner_{safe_name}_N{s_meta['student_n']}_K{s_meta['layer_k']}_E{n_estimators}{token_label}.pt"
    save_path = os.path.join(output_dir, model_name)
    torch.save(save_obj, save_path)
    return save_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train activation aligner")
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/aligners")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--per_token", action="store_true", help="Train a separate aligner for each token position")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading activations...")
    student_data = torch.load(args.student_path)
    teacher_data = torch.load(args.teacher_path)
    
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
    save_path = save_aligner(
        student_data["metadata"],
        estimator_idx_to_aligner,
        avg_mse,
        n_estimators,
        args.output_dir,
        args.per_token
    )
    
    print(f"\nTraining complete. Avg MSE: {avg_mse:.6f}")
    print(f"All models saved to {save_path}")

if __name__ == "__main__":
    main()
