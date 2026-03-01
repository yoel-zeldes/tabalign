import argparse
import os
import torch
from pruning_utils import load_data, fit_model, create_filename_from_args, create_student_training_set

def capture_hook(module, input, output, captured_storage, model_idx):
    captured_storage[model_idx] = output.detach().clone()

def main():
    parser = argparse.ArgumentParser(description="Extract TabPFN activations")
    parser.add_argument("--dataset", type=str, default="breast_cancer[synthetic]")
    parser.add_argument("--student_n", type=int, default=10, help="Number of examples for student. Negative for full (teacher).")
    parser.add_argument("--layer_k", type=int, default=2, help="Layer index to extract activations from.")
    parser.add_argument("--n_estimators", type=int, default=8, help="Number of TabPFN estimators.")
    parser.add_argument("--output_dir", type=str, default="results/activations")
    parser.add_argument("--force", action="store_true", help="Force extraction even if output exists.")
    args = parser.parse_args()

    output_path = create_filename_from_args(args, extension=".pt", makedirs=True)
    if os.path.exists(output_path) and not args.force:
        print(f">>> extract_activations: Skipping (Output already exists at {output_path})")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data: {args.dataset}")
    X_train, X_test, y_train, y_test = load_data(args.dataset)
    
    if args.student_n < 0:
        use_X_train = X_train
        use_y_train = y_train
        n_label = "full"
    else:
        use_X_train, use_y_train = create_student_training_set(X_train, y_train, args.student_n)
        n_label = f"N{args.student_n}"
        
    model = fit_model(use_X_train, use_y_train, n_estimators=args.n_estimators, assure_feature_tokens_are_static=True)
    
    captured = {}
    handles = []
    
    for i, m in enumerate(model.executor_.models):
        layer = m.transformer_encoder.layers[args.layer_k]
        h = layer.register_forward_hook(
            lambda mod, inp, out, stor=captured, idx=i: capture_hook(mod, inp, out, stor, idx)
        )
        handles.append(h)
    
    print(f"Running inference on test set (size {len(X_test)})...")
    with torch.no_grad():
        model.predict_proba(X_test)
        
    for h in handles:
        h.remove()
        
    data_to_save = {
        "metadata": vars(args),
        "activations": captured
    }
    
    torch.save(data_to_save, output_path)
    print(f"Activations saved to {output_path}")

if __name__ == "__main__":
    main()
