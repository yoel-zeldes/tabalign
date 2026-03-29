import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm, trange
from pruning_utils import get_device, create_filename_from_args, append_feature_stats

def validate_metadata(s_meta, t_meta):
    """Ensure student and teacher metadata match."""
    if s_meta["dataset"] != t_meta["dataset"]:
        raise ValueError(f"Dataset mismatch: {s_meta['dataset']} vs {t_meta['dataset']}")
    if s_meta["layer_k"] != t_meta["layer_k"]:
        raise ValueError(f"Layer mismatch: {s_meta['layer_k']} vs {t_meta['layer_k']}")
    if s_meta["n_estimators"] != t_meta["n_estimators"]:
        raise ValueError(f"Estimators mismatch: {s_meta['n_estimators']} vs {t_meta['n_estimators']}")
    print(f"Metadata verified for {s_meta['dataset']} @ Layer {s_meta['layer_k']}")

def prepare_dataloaders(s_activations, t_activations, predict_residual=False, token_idx=None, feature_stats=None):
    """Squeeze, flatten, and split activations into train/val datasets.
    
    Returns train_ds, val_ds (TensorDatasets) and hidden_dim.
    
    If feature_stats is provided (shape [n_features, n_stats]), each token's stats
    vector is broadcast and concatenated to the student activation. Only supported
    when token_idx is None.
    """
    s_activations = s_activations.squeeze()
    t_activations = t_activations.squeeze()
    
    if s_activations.shape != t_activations.shape:
        raise ValueError(f"Activation shape mismatch: {s_activations.shape} vs {t_activations.shape}")
        
    _, _, hidden_dim = s_activations.shape
    
    if token_idx is not None:
        X = s_activations[:, token_idx, :]
        Y = t_activations[:, token_idx, :]
        if predict_residual:
            Y = Y - X
    else:
        X = s_activations.reshape(-1, hidden_dim)  # [N*n_tokens, hidden_dim]
        Y = t_activations.reshape(-1, hidden_dim)
        if predict_residual:
            Y = Y - X
        if feature_stats is not None:
            X = append_feature_stats(X, feature_stats)
    
    indices = torch.randperm(X.shape[0])
    split = int(0.8 * X.shape[0])
    
    train_ds = TensorDataset(X[indices[:split]], Y[indices[:split]])
    val_ds = TensorDataset(X[indices[split:]], Y[indices[split:]])
    
    return train_ds, val_ds, hidden_dim

def build_aligner_model(input_dim, output_dim, hidden_layers, predict_residual=False):
    """Build an MLP aligner model.
    
    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        hidden_layers: List of ints for hidden layer sizes.
        predict_residual: If True, apply near-zero init on the final layer so
            the model starts by outputting near-zero residuals.
    """
    layers = []
    for h_dim in hidden_layers:
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.ReLU())
        input_dim = h_dim
    final_layer = nn.Linear(input_dim, output_dim)
    if predict_residual:
        # Near-zero init: scale weights and bias so initial residuals ≈ 0
        nn.init.normal_(final_layer.weight, std=1e-3)
        nn.init.normal_(final_layer.bias, std=1e-3)
    layers.append(final_layer)
    return nn.Sequential(*layers)

def train_estimator(est_idx, train_loader, val_loader, model, lr, patience, device, token_idx=None, max_epochs=None):
    """Train an aligner for a single estimator (and optionally a single token).
    
    Trains indefinitely until dev loss does not improve for `patience` consecutive epochs,
    or until max_epochs is reached (if specified).
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
    
    return best_model_state, best_val_loss

def _train_single_aligner(est_idx, s_act_list, t_act_list, args, device, token_idx=None, feature_stats_list=None):
    """Encapsulate data preparation and training for a single aligner.
    
    s_act_list and t_act_list are lists of activation tensors, one per dataset.
    feature_stats_list (optional): list of [n_features, n_stats] tensors, one per dataset.
    """
    train_datasets, val_datasets = [], []
    hidden_dim = None
    n_stats = 0
    for i, (s_act, t_act) in enumerate(zip(s_act_list, t_act_list)):
        feature_stats = feature_stats_list[i] if feature_stats_list is not None else None
        train_ds, val_ds, curr_hidden_dim = prepare_dataloaders(
            s_act, t_act, predict_residual=args.predict_residual, token_idx=token_idx, feature_stats=feature_stats
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        if hidden_dim is None:
            hidden_dim = curr_hidden_dim
            if feature_stats is not None:
                n_stats = feature_stats.shape[1]
        elif hidden_dim != curr_hidden_dim:
            raise ValueError(f"Hidden dim mismatch across datasets: {hidden_dim} vs {curr_hidden_dim}")

    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size)

    model = build_aligner_model(input_dim=hidden_dim + n_stats, output_dim=hidden_dim, hidden_layers=args.hidden_layers, predict_residual=args.predict_residual).to(device)
    return train_estimator(
        est_idx,
        train_loader,
        val_loader,
        model,
        args.lr,
        args.patience,
        device,
        token_idx,
        max_epochs=args.max_epochs,
    )

def save_aligner(output_path, students_metadata, estimator_idx_to_aligner, avg_val_loss, per_token, hidden_layers, predict_residual, n_stats=0):
    """Save the ensemble of aligner models and metadata."""
    save_obj = {
        "metadata": {
            "students_metadata": students_metadata,
            "avg_mse_loss": avg_val_loss,
            "per_token": per_token,
            "hidden_layers": hidden_layers,
            "predict_residual": predict_residual,
            "n_stats": n_stats,
        },
        "estimator_idx_to_aligner": estimator_idx_to_aligner
    }
    
    torch.save(save_obj, output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Train activation aligner")
    parser.add_argument("--dataset", type=str, nargs='+', default=["tabarena/Amazon_employee_access[synthetic]"], help="One or more dataset names. Activations from all datasets are combined (80/20 split per dataset).")
    parser.add_argument("--student_n", type=int, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--patience", type=int, default=10, help="Stop training after this many consecutive epochs with no improvement in dev loss.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--per_token", action="store_true", help="Train a separate aligner for each token position")
    parser.add_argument("--hidden_layers", type=int, nargs='*', default=[], help="Hidden layer sizes for MLP aligner. Empty = linear.")
    parser.add_argument("--predict_residual", action="store_true", help="Predict residual (teacher - student) instead of teacher activation directly.")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--force", action="store_true", help="Force training even if output exists.")
    parser.add_argument("--use_feature_stats", action="store_true", help="Condition the aligner on per-feature statistics (mean, std, min, max, median) from the teacher's training data.")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of training epochs. None = unlimited (rely on patience).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if len(args.dataset) > 1 and args.per_token:
        raise ValueError("--per_token is not supported when multiple datasets are provided.")
    if args.use_feature_stats and args.per_token:
        raise ValueError("--use_feature_stats is not supported together with --per_token.")

    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = create_filename_from_args(args, extension=".pt", makedirs=True)
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
            "use_feature_stats": args.use_feature_stats,
        }, script_name="extract_activations", extension=".pt")

        student_path = create_filename_from_args({
            "dataset": dataset,
            "student_n": args.student_n,
            "layer_k": args.layer_k,
            "n_estimators": args.n_estimators,
            "repeat": args.repeat,
            "output_dir": args.output_dir,
            "use_feature_stats": args.use_feature_stats,
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

    estimator_idx_to_aligner = {}
    total_val_loss = 0
    num_trained_models = 0

    for est_idx in range(n_estimators):
        s_act_list = [sd["activations"][est_idx] for sd in all_student_data]
        t_act_list = [td["activations"][est_idx] for td in all_teacher_data]
        feature_stats_list = [td["feature_stats"] for td in all_teacher_data] if args.use_feature_stats else None

        if args.per_token:
            token_idx_to_aligner = {}
            s_act, = s_act_list
            t_act, = t_act_list
            n_tokens = s_act.shape[2] # shape is [1, N, Tokens, Hidden]
            est_val_loss = 0

            pbar = trange(n_tokens, desc=f"Estimator {est_idx}", leave=False)
            for token_idx in pbar:
                state_dict, best_loss = _train_single_aligner(
                    est_idx, [s_act], [t_act], args, device, token_idx=token_idx
                )
                token_idx_to_aligner[token_idx] = state_dict
                est_val_loss += best_loss
                num_trained_models += 1

            estimator_idx_to_aligner[est_idx] = token_idx_to_aligner
            total_val_loss += est_val_loss
            pbar.set_postfix({"avg_token_mse": f"{est_val_loss / n_tokens:.6f}"})
        else:
            state_dict, best_loss = _train_single_aligner(
                est_idx, s_act_list, t_act_list, args, device,
                feature_stats_list=feature_stats_list
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
        args.per_token,
        args.hidden_layers,
        args.predict_residual,
        n_stats=feature_stats_list[0].shape[1] if args.use_feature_stats else 0,
    )

    print(f"\nTraining complete. Avg MSE: {avg_mse:.6f}")
    print(f"All models saved to {output_path}")

if __name__ == "__main__":
    main()
