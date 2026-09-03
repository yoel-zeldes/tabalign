"""Modal app for running tabular experiments in the cloud with persistent caching.

Wraps existing experiment scripts to run on Modal with automatic parallelism
and a persistent Volume for joblib.Memory cache results.
"""
import os
import re
import sys
import threading
import urllib.parse
import modal

APP_DIR = "/app"

if APP_DIR not in sys.path and os.path.exists(APP_DIR):
    # When imported inside a Modal container, modal_app.py is loaded from /root while
    # the project codebase lives at APP_DIR. Add APP_DIR to sys.path so top-level imports below
    # (consts, cli_utils) resolve inside containers. On local machines APP_DIR does not exist,
    # making this a no-op.
    sys.path.insert(0, APP_DIR)

from consts import (
    TABARENA_NAME_TO_TASK_ID,
    MODAL_VOLUME_NAME,
    TABFM_DEFAULT_N_ESTIMATORS,
    TABPFN_DEFAULT_N_ESTIMATORS,
    DEFAULT_N_SAMPLES,
)
from cli_utils import parse_student_n

app = modal.App("tabular-experiments")

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

VOLUME_PATH = "/results"
TIMEOUT_SECONDS = 36_000  # 10 hours

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("matplotlib", "scipy")
    .add_local_dir(
        ".",
        remote_path=APP_DIR,
        copy=True,
        ignore=[
            "*results*", "venv", ".git", "__pycache__", "*.ckpt", ".cache"
        ],
    )
)

def _get_tabpfn_token():
    token = os.environ.get("TABPFN_TOKEN")
    if token:
        return token
    token_file = os.path.expanduser("~/.cache/tabpfn/auth_token")
    if os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()
    return None


def _get_hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        if hf_token := get_token():
            return hf_token
    except Exception:
        pass
    token_file = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_file):
        with open(token_file) as f:
            return f.read().strip()
    return None


APP_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONPATH": APP_DIR,
    "RESULTS_DIR": VOLUME_PATH,
    "OPENML_CACHE_DIRECTORY": f"{VOLUME_PATH}/.openml_cache",
    "XDG_CACHE_HOME": f"{VOLUME_PATH}/.model_cache",
    "HF_HOME": f"{VOLUME_PATH}/.model_cache/huggingface",
    "MPLBACKEND": "Agg",
    "PROFILE_NAME": modal.config._profile,
}
if tabpfn_token := _get_tabpfn_token():
    APP_ENV["TABPFN_TOKEN"] = tabpfn_token
if hf_token := _get_hf_token():
    APP_ENV["HF_TOKEN"] = hf_token


def modal_url_for_file(file_path: str) -> str:
    """Generate the Modal storage web UI URL for a file in the volume."""
    if file_path.startswith(VOLUME_PATH):
        rel_file = file_path[len(VOLUME_PATH):].lstrip("/")
    else:
        rel_file = file_path.lstrip("/")
    rel_dir = os.path.dirname(rel_file)
    quoted_dir = urllib.parse.quote(rel_dir, safe="/:'")
    quoted_file = urllib.parse.quote(rel_file, safe="")
    profile_name = os.environ.get("PROFILE_NAME")
    return f"https://modal.com/storage/{profile_name}/main/volumes/{MODAL_VOLUME_NAME}/{quoted_dir}?file={quoted_file}"


# ---------------------------------------------------------------------------
# Modal functions — each runs one experiment call in its own container
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS, gpu="L4")
def run_extract_activations(kwargs):
    """Run a single extract_activations() call on GPU."""
    volume.reload()
    from extract_activations import extract_activations
    result = extract_activations(**kwargs)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_train_aligner(kwargs):
    """Run a single train_aligner() call on CPU."""
    volume.reload()
    from train_activation_aligner import train_aligner
    result = train_aligner(**kwargs)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS, gpu="L4")
def run_evaluate_aligned(kwargs):
    """Run a single evaluate_aligned_student() call."""
    volume.reload()
    from evaluate_aligned_student import evaluate_aligned_student
    result = evaluate_aligned_student(**kwargs)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_xgboost(kwargs):
    """Run a single train_xgboost() call."""
    volume.reload()
    from train_xgboost import train_xgboost
    result = train_xgboost(**kwargs)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_xgboost_opt(kwargs):
    """Run a single train_xgboost_opt() call."""
    volume.reload()
    from train_xgboost_opt import train_xgboost_opt as _train_xgboost_opt
    result = _train_xgboost_opt(**kwargs)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_sweep_plotting(dataset, args_dict):
    """Run plotting/aggregation for one dataset. All inner calls should be cached."""
    volume.reload()

    import argparse
    import sweep_pipeline_layers

    args = argparse.Namespace(**args_dict)
    sweep_pipeline_layers.run_dataset(args, dataset)
    volume.commit()


# ---------------------------------------------------------------------------
# Cloud sweep function — runs on Modal, fans out all phases
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    env=APP_ENV,
    timeout=TIMEOUT_SECONDS,
)
def sweep(
    dataset: str = "tabarena",
    student_n: str = "20",
    layers: str = "23",
    n_estimators: int = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_repeats: int = 1,
    patience: int = 10,
    lr: float = 1e-3,
    batch_size: int = 2048,
    hidden_layers: str = "",
    max_epochs: int = -1,
    model: str = "tabpfn",
    xgboost_opt: bool = False,
    aligner_opt: bool = False,
):
    """Run sweep_pipeline_layers experiments in parallel on Modal.

    Each (dataset, student_n, repeat, layer_k) combo runs as an independent
    Modal function. Results are cached in a persistent Volume.

    Args:
        dataset: Dataset name(s), comma or space-separated, or "tabarena".
        student_n: Student training size(s), comma or space-separated (e.g. "20", "20 50", "0.1 0.2").
        layers: Layer indices to extract and align, comma or space-separated (default: "23", the last layer).
        n_estimators: Number of estimators (if not specified, uses defaults from consts.py).
        max_epochs: Max training epochs (-1 = unlimited, rely on patience).
    """
    # Parse datasets (can be comma/space-separated, single name, or "tabarena")
    dataset_names = [d.strip() for d in re.split(r"[, ]+", str(dataset)) if d.strip()]
    datasets = []
    for d in dataset_names:
        if d == "tabarena":
            datasets.extend(f"tabarena/{name}" for name in TABARENA_NAME_TO_TASK_ID)
        else:
            datasets.append(d)

    # Parse student_n values (e.g. "20", "20 50", "0.1 0.2")
    student_n = [s.strip() for s in re.split(r"[, ]+", str(student_n)) if s.strip()]
    student_n = [parse_student_n(s) for s in student_n]

    # Parse layers (e.g. "1 2", "1,2", "1 2 5 8 9 10 11")
    layers = [
        int(k.strip())
        for k in re.split(r"[, ]+", str(layers))
        if k.strip()
    ]

    # Parse hidden_layers (e.g. "" or "64 64")
    hidden_layers = [
        int(h.strip())
        for h in re.split(r"[, ]+", str(hidden_layers))
        if h.strip()
    ] if hidden_layers else []

    # Resolve defaults
    if n_estimators is None:
        n_estimators = TABFM_DEFAULT_N_ESTIMATORS if model == "tabfm" else TABPFN_DEFAULT_N_ESTIMATORS

    max_epochs_val = None if max_epochs <= 0 else max_epochs

    # ── Sample efficiency plotter ──
    fractional_student_n = [
        n for n in student_n
        if isinstance(n, float) and 0.0 < n < 1.0
    ]

    plot_lock = threading.Lock()

    def _generate_sample_efficiency_plots():
        if not fractional_student_n:
            return
        with plot_lock:
            import plot_aligned_sample_efficiency
            volume.reload()
            for layer_k in layers:
                table_path = plot_aligned_sample_efficiency.plot(
                    dataset=datasets,
                    student_n=fractional_student_n,
                    layer=layer_k,
                    n_estimators=n_estimators,
                    n_samples=n_samples,
                    patience=patience,
                    lr=lr,
                    batch_size=batch_size,
                    hidden_layers=hidden_layers,
                    n_repeats=n_repeats,
                    max_epochs=max_epochs_val,
                    model=model,
                    xgboost_opt=xgboost_opt,
                    aligner_opt=aligner_opt,
                )
                if table_path:
                    table_url = modal_url_for_file(table_path)
                    print(f"  Table PNG URL (layer={layer_k}):\n  {table_url}")
            volume.commit()

    def _run_phase(name, modal_func, kwargs_list, print_every=10, plot_sample_efficiency=False):
        if not kwargs_list:
            return
        print(f"\n{name} ({len(kwargs_list)} calls)")
        for i, _ in enumerate(modal_func.map(kwargs_list)):
            if (i + 1) % print_every == 0 or i + 1 == len(kwargs_list):
                print(f"  {name}: {i + 1}/{len(kwargs_list)} complete")
            if plot_sample_efficiency:
                _generate_sample_efficiency_plots()

    # ── Phase 1: Extract activations on GPU & run XGBoost on CPU ─────────

    volume.reload()
    from extract_activations import extract_activations
    from train_activation_aligner import train_aligner
    from evaluate_aligned_student import evaluate_aligned_student
    if xgboost_opt:
        from train_xgboost_opt import train_xgboost_opt as xgb_target_func
    else:
        from train_xgboost import train_xgboost as xgb_target_func

    all_extract_kwargs = []
    all_train_aligner_kwargs = []
    all_eval_kwargs = []
    all_xgb_kwargs = []

    for ds in datasets:
        for r in range(n_repeats):
            train_ds_name = f"{ds}[synthetic-n_samples_{n_samples}-repeat_{r}]"

            xgb_kwargs = {
                "dataset": ds,
                "student_n": -1,
                "repeat": r,
            }
            if not xgb_target_func.check_call_in_cache(**xgb_kwargs):
                all_xgb_kwargs.append(xgb_kwargs)

            for layer_k in layers:
                t_extract_kwargs = {
                    "dataset": train_ds_name,
                    "student_n": -1,
                    "layer_k": layer_k,
                    "n_estimators": n_estimators,
                    "repeat": r,
                    "model": model,
                }
                if not extract_activations.check_call_in_cache(**t_extract_kwargs):
                    all_extract_kwargs.append(t_extract_kwargs)

                for n in student_n:
                    s_extract_kwargs = {
                        "dataset": train_ds_name,
                        "student_n": n,
                        "layer_k": layer_k,
                        "n_estimators": n_estimators,
                        "repeat": r,
                        "model": model,
                    }
                    if not extract_activations.check_call_in_cache(**s_extract_kwargs):
                        all_extract_kwargs.append(s_extract_kwargs)

                    aligner_kwargs = {
                        "dataset": train_ds_name,
                        "student_n": n,
                        "layer_k": layer_k,
                        "n_estimators": n_estimators,
                        "patience": patience,
                        "lr": lr,
                        "batch_size": batch_size,
                        "hidden_layers": hidden_layers,
                        "repeat": r,
                        "max_epochs": max_epochs_val,
                        "model": model,
                        "opt": aligner_opt,
                    }
                    if not train_aligner.check_call_in_cache(**aligner_kwargs):
                        all_train_aligner_kwargs.append(aligner_kwargs)

                    eval_kwargs = {
                        "eval_dataset": ds,
                        "train_dataset": train_ds_name,
                        "student_n": n,
                        "layer_k": layer_k,
                        "n_estimators": n_estimators,
                        "patience": patience,
                        "lr": lr,
                        "batch_size": batch_size,
                        "hidden_layers": hidden_layers,
                        "repeat": r,
                        "max_epochs": max_epochs_val,
                        "model": model,
                        "aligner_opt": aligner_opt,
                    }
                    if not evaluate_aligned_student.check_call_in_cache(**eval_kwargs):
                        all_eval_kwargs.append(eval_kwargs)

    xgb_func = run_xgboost_opt if xgboost_opt else run_xgboost
    _run_phase("Phase 1: xgboost (CPU)", xgb_func, all_xgb_kwargs, 5)
    _run_phase("Phase 2: extract_activations (GPU)", run_extract_activations, all_extract_kwargs, 10)
    _run_phase("Phase 3: train_aligner (CPU)", run_train_aligner, all_train_aligner_kwargs, 10)
    _run_phase("Phase 4: evaluate_aligned (GPU)", run_evaluate_aligned, all_eval_kwargs, 10, plot_sample_efficiency=True)


    print(f"\nPhase 5: Generating plots for {len(datasets)} datasets...")
    args_dict = {
        "student_n": student_n,
        "layers": layers,
        "n_estimators": n_estimators,
        "n_samples": n_samples,
        "n_repeats": n_repeats,
        "patience": patience,
        "lr": lr,
        "batch_size": batch_size,
        "hidden_layers": hidden_layers,
        "max_epochs": max_epochs_val,
        "model": model,
        "xgboost_opt": xgboost_opt,
        "aligner_opt": aligner_opt,
    }

    for i, _ in enumerate(
        run_sweep_plotting.starmap([(ds, args_dict) for ds in datasets])
    ):
        print(f"  plot: {i + 1}/{len(datasets)} ({datasets[i]})")

    _generate_sample_efficiency_plots()

    print("\nAll experiments complete!\nRun './venv/bin/python3 download_results.py' to download results.")
