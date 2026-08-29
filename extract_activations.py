import argparse
import os
import torch
import numpy as np
from pruning_utils import load_data, fit_model, create_filename_from_args, create_student_training_set, get_device, save_model_preprocessor, fill_nans, parse_student_n

def capture_hook(module, input, output, captured_storage, model_idx, n_test_tokens):
    test_acts = output[:, -n_test_tokens:, :].detach().clone().float()
    if model_idx not in captured_storage:
        captured_storage[model_idx] = []
    captured_storage[model_idx].append(test_acts)

def main():
    parser = argparse.ArgumentParser(description="Extract TabPFN activations")
    parser.add_argument("--dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument("--student_n", type=parse_student_n, default=10, help="Number of examples for student. Negative for full (teacher).")
    parser.add_argument("--layer_k", type=int, default=2, help="Layer index to extract activations from.")
    parser.add_argument("--n_estimators", type=int, default=8, help="Number of TabPFN estimators.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--repeat", type=int, default=0, help="OpenML repeat index (different repeats use different random splits).")
    parser.add_argument("--force", action="store_true", help="Force extraction even if output exists.")
    parser.add_argument("--model", type=str, choices=["tabpfn", "tabfm"], default="tabpfn", help="Model architecture to use.")
    args = parser.parse_args()

    output_path = create_filename_from_args(args, extension=".pt", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> extract_activations: Skipping (Output already exists at {output_path})")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data: {args.dataset}")
    X_train, X_test, y_train, y_test, cat_indices = load_data(args.dataset, repeat=args.repeat, return_cat_indices=True)
    
    if args.student_n < 0:
        use_X_train = X_train
        use_y_train = y_train
        n_label = "full"
    else:
        use_X_train, use_y_train = create_student_training_set(X_train, y_train, args.student_n)
        n_label = f"N{args.student_n}"
        
    # For student extraction, pass the teacher's saved preprocessor path to reuse its
    # preprocessors (encoding, scaling, filtering). This ensures teacher and student
    # produce activations in the same feature space.
    if args.student_n >= 0:
        teacher_args = dict(vars(args), student_n=-1)
        teacher_act_path = create_filename_from_args(
            teacher_args, script_name="extract_activations", extension=".pt"
        )
        teacher_model_preprocessor_path = teacher_act_path.replace('.pt', '.teacher_model_preprocessor.pkl')
    else:
        teacher_model_preprocessor_path = None

    model = fit_model(
        use_X_train, use_y_train, n_estimators=args.n_estimators,
        model=args.model,
        model_preprocessor=teacher_model_preprocessor_path,
    )

    # For teacher extraction, save the fitted model so student can reuse preprocessors
    if args.student_n < 0:
        teacher_model_preprocessor_path = output_path.replace('.pt', '.teacher_model_preprocessor.pkl')
        save_model_preprocessor(model, teacher_model_preprocessor_path)
    
    captured = {}
    handles = []
    current_batch_n = [0]
    
    if args.model == "tabfm":
        layer = model.model.icl_predictor.tf_icl.blocks[args.layer_k]
        h = layer.register_forward_hook(
            lambda mod, inp, out, stor=captured, idx=0: capture_hook(mod, inp, out, stor, idx, current_batch_n[0])
        )
        handles.append(h)
    else:
        executor = model.executor_
        device = get_device()
        underlying_models = []
        if hasattr(executor, 'model_caches'):
            underlying_models = [executor.model_caches[em.config._model_index].get(device) for em in executor.ensemble_members]
        elif hasattr(executor, 'models'):
            underlying_models = executor.models
        else:
            raise RuntimeError("Cannot extract models from executor type")

        for i, m in enumerate(underlying_models):
            layer = m.transformer_encoder.layers[args.layer_k]
            h = layer.register_forward_hook(
                lambda mod, inp, out, stor=captured, idx=i: capture_hook(mod, inp, out, stor, idx, current_batch_n[0])
            )
            handles.append(h)
    
    if args.model == "tabfm":
        X_test = fill_nans(X_test)
    print(f"Running inference on test set (size {len(X_test)})...")
    current_batch_n[0] = len(X_test)
    with torch.no_grad():
        probs = model.predict_proba(X_test)
        
    for h in handles:
        h.remove()

    concatenated_captured = {}
    for idx, act_list in captured.items():
        concatenated_captured[idx] = torch.cat(act_list, dim=1)

    data_to_save = {
        "metadata": vars(args),
        "activations": concatenated_captured,
        "probs": probs,
    }
    
    torch.save(data_to_save, output_path)
    print(f"Activations saved to {output_path}")

if __name__ == "__main__":
    main()
