import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pruning_utils import (
    load_data, create_model, create_filename_from_args,
    create_student_training_set, append_feature_stats, get_device,
)
from train_activation_aligner import build_aligner_model, save_aligner
from extract_activations import capture_hook, compute_feature_stats


def _get_underlying_models(student_model):
    """Retrieve the actual nn.Module models from the student's executor,
    one per ensemble member (estimator).

    For InferenceEngineCachePreprocessing, the model_caches list has one entry
    per *checkpoint model* (often just 1), NOT per estimator. Each ensemble
    member's config._model_index tells us which checkpoint model it uses.
    We return models indexed by estimator order.
    """
    executor = student_model.executor_
    device = get_device()

    if hasattr(executor, 'model_caches'):  # MultiDeviceInferenceEngine (e.g. CachePreprocessing)
        return [
            executor.model_caches[em.config._model_index].get(device)
            for em in executor.ensemble_members
        ]
    elif hasattr(executor, 'models'):      # SingleDeviceInferenceEngine (e.g. CacheKV)
        return executor.models
    else:
        raise RuntimeError(f"Cannot extract models from executor type {type(executor)}")


class DispatchingAlignHook:
    """Hook installed once per unique layer that dispatches to the correct aligner.

    When multiple estimators share the same underlying model (and thus the same
    layer object), this single hook is called once per estimator's forward pass.
    It uses a call counter to know which aligner to apply.

    Only TEST activations are aligned; training activations pass through unchanged.
    """
    def __init__(self):
        self.est_sequence = []   # ordered list of (est_idx, aligner, predict_residual, feature_stats, n_test_tokens)
        self.call_idx = 0
        self.aligned_acts = {}   # est_idx -> aligned test activation (with grad)

    def add_estimator(self, est_idx, aligner, predict_residual=False,
                      feature_stats=None, n_test_tokens=None):
        self.est_sequence.append((est_idx, aligner, predict_residual, feature_stats, n_test_tokens))

    def reset(self):
        self.call_idx = 0
        self.aligned_acts = {}

    def __call__(self, module, input, output):
        est_idx, aligner, predict_residual, feature_stats, n_test_tokens = self.est_sequence[self.call_idx]
        self.call_idx += 1

        orig_shape = output.shape
        hidden_dim = orig_shape[-1]
        flat = output.reshape(-1, hidden_dim)

        # Split train / test tokens  (test tokens are at the end)
        train_flat = flat[:-n_test_tokens]
        test_flat = flat[-n_test_tokens:]

        # Align only test portion
        x = test_flat
        if feature_stats is not None:
            x = append_feature_stats(x, feature_stats)
        aligned_test = aligner(x)

        if predict_residual:
            aligned_test = aligned_test + test_flat

        self.aligned_acts[est_idx] = aligned_test  # (n_test_tokens, hidden_dim)

        # Reconstruct full activation: unchanged train + aligned test
        result = torch.cat([train_flat, aligned_test], dim=0)
        return result.reshape(orig_shape)


def _differentiable_predict_proba(student_model, X_test):
    """Run the student model forward with gradients enabled.

    We temporarily set inference_mode=False on the executor AND
    prevent the classifier's forward() from resetting it back to True
    (it calls use_torch_inference_mode inside forward).

    We also disable gradient checkpointing (recompute_layer) on all
    underlying models, since our hooks create tensors that differ between
    the original forward and the recomputation pass.

    Returns a differentiable probability tensor of shape (N, n_classes).
    """
    executor = student_model.executor_

    # Save and disable inference mode
    original_inference_mode = getattr(executor, 'inference_mode', True)
    executor.inference_mode = False

    # Monkey-patch use_torch_inference_mode to prevent forward() from
    # resetting it back to True (line 1324 of classifier.py).
    original_use_torch_inference_mode = executor.use_torch_inference_mode
    executor.use_torch_inference_mode = lambda **kwargs: None

    # Disable gradient checkpointing on all models (incompatible with hooks)
    models = _get_underlying_models(student_model)
    saved_recompute = []
    for m in models:
        saved_recompute.append(m.recompute_layer)
        m.recompute_layer = False

    try:
        raw_logits = student_model.forward(
            X_test, use_inference_mode=True, return_raw_logits=True
        )
        probs = student_model.logits_to_probabilities(raw_logits)
        return probs
    finally:
        for m, orig in zip(models, saved_recompute):
            m.recompute_layer = orig
        executor.use_torch_inference_mode = original_use_torch_inference_mode
        executor.inference_mode = original_inference_mode


def train_aligners(
    student_model,         # single TabPFNClassifier (will be re-fit per dataset)
    per_dataset,           # list of dicts, one per dataset, see main()
    n_estimators,
    layer_k,
    args,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train activation aligner v2 (MSE + KL loss) using concatenated datasets.
    """
    from collections import defaultdict # Added import here for safety

    hidden_dim = next(iter(per_dataset[0]["teacher_acts"].values())).shape[-1]

    # Determine input dim for aligner
    n_stats = per_dataset[0]["feature_stats"].shape[1] if per_dataset[0]["feature_stats"] is not None else 0
    input_dim = hidden_dim + n_stats

    # Build one aligner per estimator
    aligners = {}
    for est_idx in range(n_estimators):
        aligners[est_idx] = build_aligner_model(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_layers=args.hidden_layers,
            predict_residual=args.predict_residual,
        ).to(device)

    # Collect all parameters for a single optimizer
    all_params = []
    for m in aligners.values():
        all_params.extend(m.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr)

    # Prepare teacher targets per dataset (move to device once, KEEP (N, tokens, hidden))
    for ds in per_dataset:
        ds["teacher_probs"] = torch.tensor(
            ds["teacher_probs"], dtype=torch.float32, device=device
        ).clamp(min=1e-10)
        ds["teacher_probs"] = ds["teacher_probs"] / ds["teacher_probs"].sum(dim=-1, keepdim=True)

        ds["teacher_acts"] = {
            k: v.squeeze(0).float().to(device) # Keep unflattened
            for k, v in ds["teacher_acts"].items()
        }

    # Define custom Dataset for indices
    class AlignerDataset(Dataset):
        def __init__(self, per_dataset):
            self.id_to_dataset_sample = []
            for ds_idx, ds in enumerate(per_dataset):
                N = len(ds["teacher_probs"])
                for i in range(N):
                    self.id_to_dataset_sample.append((ds_idx, i))
                    
        def __len__(self):
            return len(self.id_to_dataset_sample)
            
        def __getitem__(self, idx):
            ds_idx, sample_idx = self.id_to_dataset_sample[idx]
            return {"dataset_idx": ds_idx, "sample_idx": sample_idx}

    dataset = AlignerDataset(per_dataset)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_loss = float("inf")
    best_states = None
    epochs_without_improvement = 0
    epoch = 0

    pbar = tqdm(desc="Training", leave=True)
    while args.max_epochs is None or epoch < args.max_epochs:
        epoch += 1
        for m in aligners.values():
            m.train()

        total_mse_val_sum = 0.0
        total_kl_val_sum = 0.0
        total_mse_elements_epoch = 0
        total_kl_samples_epoch = 0
        
        n_batches = len(train_loader)
        last_5_losses = []
        last_fitted_ds_idx = None
        batch_pbar = tqdm(train_loader, desc=f"  Batches (Epoch {epoch})", leave=True)

        for batch_idx, batch in enumerate(batch_pbar):
            optimizer.zero_grad()
            
            dataset_indices = batch["dataset_idx"].tolist()
            sample_indices = batch["sample_idx"].tolist()
            
            grouped = defaultdict(list)
            for i, ds_idx in enumerate(dataset_indices):
                grouped[ds_idx].append(sample_indices[i])
                
            total_sq_error_tensor = torch.tensor(0.0, device=device)
            total_kl_sum_tensor = torch.tensor(0.0, device=device)
            total_mse_elements = 0
            total_kl_samples = 0
            
            for ds_idx, block in grouped.items():
                ds = per_dataset[ds_idx]
                
                if ds_idx != last_fitted_ds_idx:
                    student_model.fit(ds["student_X"], ds["student_y"])
                    last_fitted_ds_idx = ds_idx
                
                X_test_batch = ds["X_test"].iloc[block] if hasattr(ds["X_test"], "iloc") else ds["X_test"][block]
                teacher_probs_batch = ds["teacher_probs"][block] if ds["teacher_probs"] is not None else None
                
                teacher_acts_batch = {}
                for est_idx, acts in ds["teacher_acts"].items():
                    teacher_acts_batch[est_idx] = acts[block].reshape(-1, hidden_dim)
                    
                underlying_models = _get_underlying_models(student_model)
                feature_stats = ds["feature_stats"]

                layer_to_hook = {}
                for est_idx in range(n_estimators):
                    model = underlying_models[est_idx]
                    layer = model.transformer_encoder.layers[layer_k]
                    layer_id = id(layer)
                    if layer_id not in layer_to_hook:
                        layer_to_hook[layer_id] = (layer, DispatchingAlignHook())
                    
                    n_test_tokens = teacher_acts_batch[est_idx].shape[0]
                    
                    layer_to_hook[layer_id][1].add_estimator(
                        est_idx, aligners[est_idx],
                        predict_residual=args.predict_residual,
                        feature_stats=feature_stats,
                        n_test_tokens=n_test_tokens,
                    )

                registered_hooks = []
                all_hook_objs = []
                for layer, hook_obj in layer_to_hook.values():
                    hook_obj.reset()
                    registered_hooks.append(layer.register_forward_hook(hook_obj))
                    all_hook_objs.append(hook_obj)

                aligned_probs = _differentiable_predict_proba(student_model, X_test_batch)

                for h in registered_hooks:
                    h.remove()

                for est_idx in range(n_estimators):
                    aligned_act = None
                    for ho in all_hook_objs:
                        if est_idx in ho.aligned_acts:
                            aligned_act = ho.aligned_acts[est_idx]
                            break
                    
                    if aligned_act is not None:
                        group_mse_sum = F.mse_loss(aligned_act, teacher_acts_batch[est_idx], reduction="sum")
                        total_sq_error_tensor = total_sq_error_tensor + group_mse_sum
                        total_mse_elements += aligned_act.numel()

                if teacher_probs_batch is not None:
                    group_kl_sum = F.kl_div(
                        torch.log(aligned_probs.clamp(min=1e-10)), teacher_probs_batch,
                        reduction="sum", log_target=False,
                    )
                    total_kl_sum_tensor = total_kl_sum_tensor + group_kl_sum
                    total_kl_samples += len(block)

            # Calculation of batch loss as means
            if total_mse_elements > 0:
                batch_mse = total_sq_error_tensor / total_mse_elements
            else:
                batch_mse = torch.tensor(0.0, device=device)
                
            if total_kl_samples > 0:
                batch_kl = total_kl_sum_tensor / total_kl_samples
            else:
                batch_kl = torch.tensor(0.0, device=device)
                
            batch_loss = batch_mse + args.loss_beta * batch_kl
            
            if batch_loss.requires_grad:
                batch_loss.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(all_params, args.clip_grad)
                
            optimizer.step()

            total_mse_val_sum += total_sq_error_tensor.item()
            total_kl_val_sum += total_kl_sum_tensor.item()
            total_mse_elements_epoch += total_mse_elements
            total_kl_samples_epoch += total_kl_samples

            last_5_losses.append(batch_loss.item())
            if len(last_5_losses) > 5:
                last_5_losses.pop(0)
            avg_loss_5 = sum(last_5_losses) / len(last_5_losses)
            batch_pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "mse": f"{batch_mse.item():.4f}",
                "kl": f"{batch_kl.item():.4f}",
            })

        # --- Epoch summary -----------------------------------------------
        if total_mse_elements_epoch > 0:
            mse_loss_val = total_mse_val_sum / total_mse_elements_epoch
        else:
            mse_loss_val = 0.0
            
        if total_kl_samples_epoch > 0:
            kl_loss_val = total_kl_val_sum / total_kl_samples_epoch
        else:
            kl_loss_val = 0.0
            
        current_loss = mse_loss_val + args.loss_beta * kl_loss_val

        # --- Track best model -----------------------------------------------
        if current_loss < best_loss:
            best_loss = current_loss
            best_states = {
                est_idx: {k: v.cpu().clone() for k, v in aligners[est_idx].state_dict().items()}
                for est_idx in range(n_estimators)
            }
            best_mse = mse_loss_val
            best_kl = kl_loss_val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        pbar.update(1)
        pbar.set_postfix({
            "epoch": epoch,
            "loss": f"{current_loss:.6f}",
            "mse": f"{mse_loss_val:.6f}",
            "kl": f"{kl_loss_val:.6f}",
            "best": f"{best_loss:.6f}",
            "no_improv": f"{epochs_without_improvement}/{args.patience}",
        })

        if epochs_without_improvement >= args.patience:
            break

    pbar.close()
    print(f"\nTraining done. Best total={best_loss:.6f}  MSE={best_mse:.6f}  KL={best_kl:.6f}")
    return best_states, best_mse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train activation aligner v2 (MSE + KL loss)"
    )
    parser.add_argument(
        "--dataset", type=str, nargs="+",
        default=["tabarena/Amazon_employee_access[synthetic]"],
        help="One or more dataset names (with [synthetic-...] suffix).",
    )
    parser.add_argument("--student_n", type=int, default=10)
    parser.add_argument("--layer_k", type=int, default=2)
    parser.add_argument("--n_estimators", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Stop after this many epochs with no improvement.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="(Kept for filename compatibility; training uses full-batch.)")
    parser.add_argument(
        "--hidden_layers", type=int, nargs="*", default=[],
        help="Hidden layer sizes for MLP aligner. Empty = linear.",
    )
    parser.add_argument(
        "--predict_residual", action="store_true",
        help="Predict residual (teacher - student) instead of teacher activation.",
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="OpenML repeat index.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Force training even if output exists.")
    parser.add_argument(
        "--use_feature_stats", action="store_true",
        help="Condition aligner on per-feature statistics.",
    )
    parser.add_argument(
        "--clip_grad", type=float, default=None,
        help="Clip gradient norm to this value. None = no clipping.",
    )
    parser.add_argument(
        "--loss_beta", type=float, default=1.0,
        help="Weight for the KL-divergence loss. Total = MSE + loss_beta*KL.",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None,
        help="Maximum number of training epochs. None = unlimited (rely on patience).",
    )
    parser.add_argument(
        "--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn",
        help="Model architecture to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model == "tabfm":
        raise ValueError("tabfm model is currently unsupported in train_activation_aligner_v2")

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = create_filename_from_args(args, extension=".pt", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> train_activation_aligner_v2: Skipping (output exists at {output_path})")
        return

    device = get_device()

    # ------------------------------------------------------------------
    # Create a SINGLE student model (will be re-fit per dataset per epoch)
    # ------------------------------------------------------------------
    print("Creating student model (single instance, will be re-fit per dataset)...")
    student_model = create_model(
        n_estimators=args.n_estimators,
        assure_feature_tokens_are_static=True,
        token_per_feature=args.use_feature_stats,
        fit_mode="fit_preprocessors",
        model=args.model,
    )

    # ------------------------------------------------------------------
    # For each dataset: load teacher data and prepare data
    # ------------------------------------------------------------------
    per_dataset = []
    all_teachers_metadata = []

    for dataset in args.dataset:
        # Load teacher activations + probs from extract_activations output
        teacher_path = create_filename_from_args({
            "dataset": dataset,
            "student_n": -1,
            "layer_k": args.layer_k,
            "n_estimators": args.n_estimators,
            "repeat": args.repeat,
            "output_dir": args.output_dir,
            "use_feature_stats": args.use_feature_stats,
            "model": args.model,
        }, script_name="extract_activations", extension=".pt")

        print(f"Loading teacher data for '{dataset}' from {teacher_path}")
        teacher_data = torch.load(teacher_path, weights_only=False)
        all_teachers_metadata.append(teacher_data["metadata"])

        # Load synthetic X_test + real training data
        print(f"Loading data for '{dataset}'")
        X_train, X_test, y_train, y_test, cat_indices = load_data(
            dataset, repeat=args.repeat, return_cat_indices=True,
        )

        feature_stats = None
        if args.use_feature_stats:
            feature_stats = compute_feature_stats(X_train, y_train, cat_indices=cat_indices)

        # Prepare student training subset (NOT fitting the model here)
        student_X, student_y = create_student_training_set(X_train, y_train, args.student_n)

        per_dataset.append({
            "name": dataset,
            "student_X": student_X,
            "student_y": student_y,
            "X_test": X_test,
            "teacher_acts": teacher_data["activations"],
            "teacher_probs": teacher_data.get("probs"),
            "feature_stats": feature_stats,
        })

    # ------------------------------------------------------------------
    # Train aligners
    # ------------------------------------------------------------------
    best_states, best_mse = train_aligners(
        student_model=student_model,
        per_dataset=per_dataset,
        n_estimators=args.n_estimators,
        layer_k=args.layer_k,
        args=args,
        device=device,
    )

    # ------------------------------------------------------------------
    # Save in V1-compatible format
    # ------------------------------------------------------------------
    students_metadata = []
    for teacher_metadata in all_teachers_metadata:
        metadata = teacher_metadata.copy()
        metadata["student_n"] = args.student_n
        students_metadata.append(metadata)

    n_stats = per_dataset[0]["feature_stats"].shape[1] if args.use_feature_stats else 0
    save_aligner(
        output_path,
        students_metadata=students_metadata,
        estimator_idx_to_aligner=best_states,
        avg_val_loss=best_mse,
        per_token=False,
        hidden_layers=args.hidden_layers,
        predict_residual=args.predict_residual,
        n_stats=n_stats,
    )

    print(f"\nAll models saved to {output_path}")


if __name__ == "__main__":
    main()
