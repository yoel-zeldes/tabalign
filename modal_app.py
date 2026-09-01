"""Modal app for running tabular experiments in the cloud with persistent caching.

Wraps existing experiment scripts to run on Modal with automatic parallelism
and a persistent Volume for joblib.Memory cache results.
"""
from concurrent.futures import ThreadPoolExecutor
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

from consts import TABARENA_NAME_TO_TASK_ID, MODAL_VOLUME_NAME
from cli_utils import parse_student_n

app = modal.App("tabular-experiments")

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

VOLUME_PATH = "/results"
TIMEOUT_SECONDS = 7200
SWEEP_TIMEOUT_SECONDS = 18_000  # 5 hours

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


APP_ENV = {
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

@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_evaluate_aligned(kwargs):
    """Run a single evaluate_aligned_student() call."""
    volume.reload()
    from evaluate_aligned_student import evaluate_aligned_student
    result = evaluate_aligned_student(**kwargs)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_xgboost(dataset, student_n=-1, repeat=0):
    """Run a single train_xgboost() call."""
    volume.reload()
    from train_xgboost import train_xgboost
    result = train_xgboost(dataset=dataset, student_n=student_n, repeat=repeat)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=TIMEOUT_SECONDS)
def run_xgboost_opt(dataset, student_n=-1, repeat=0):
    """Run a single train_xgboost_opt() call."""
    volume.reload()
    from train_xgboost_opt import train_xgboost_opt as _train_xgboost_opt
    result = _train_xgboost_opt(dataset=dataset, student_n=student_n, repeat=repeat)
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_PATH: volume}, env=APP_ENV, timeout=600)
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
    timeout=SWEEP_TIMEOUT_SECONDS,
)
def sweep(
    dataset: str = "tabarena",
    student_n: str = "20",
    layers: str = "23",
    n_estimators: int = None,
    n_samples: int = 10000,
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
        n_estimators: Number of estimators (None = auto: 8 for tabpfn, 32 for tabfm).
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
        n_estimators = 32 if model == "tabfm" else 8
    max_epochs_val = None if max_epochs <= 0 else max_epochs

    # ── Periodic sample efficiency plotter (runs every minute until Phase 2 completes) ──
    fractional_student_n = [
        student_n
        for student_n in student_n
        if isinstance(student_n, float) and 0.0 < student_n < 1.0
    ]

    def _generate_sample_efficiency_plots():
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

    plot_lock = threading.Lock()
    stop_plotting = threading.Event()

    def _periodic_plot_worker():
        while not stop_plotting.is_set():
            if stop_plotting.wait(timeout=60):
                break
            with plot_lock:
                print("\n[Minute update] Updating plot_aligned_sample_efficiency from cached results...")
                _generate_sample_efficiency_plots()

    plotter_thread = None
    if fractional_student_n:
        plotter_thread = threading.Thread(target=_periodic_plot_worker, daemon=True)
        plotter_thread.start()

    # ── Phase 1: Fan out all experiment calls in parallel ────────────────

    eval_args = []
    for ds in datasets:
        for n in student_n:
            for r in range(n_repeats):
                synthetic_dataset = (
                    f"{ds}[synthetic-n_samples_{n_samples}-repeat_{r}]"
                )
                for layer_k in layers:
                    eval_args.append({
                        "eval_dataset": ds,
                        "train_dataset": synthetic_dataset,
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
                    })

    xgb_func = run_xgboost_opt if xgboost_opt else run_xgboost
    xgb_args = [
        (ds, -1, r)
        for ds in datasets
        for r in range(n_repeats)
    ]

    total = len(eval_args) + len(xgb_args)
    print(f"Phase 1: Launching {total} parallel calls")
    print(
        f"  {len(eval_args)} evaluate_aligned "
        f"({len(datasets)} datasets × {len(student_n)} student_ns "
        f"× {n_repeats} repeats × {len(layers)} layers)"
    )
    print(
        f"  {len(xgb_args)} xgboost "
        f"({len(datasets)} datasets × {n_repeats} repeats)"
    )

    # Fan out evaluate_aligned and xgboost calls concurrently
    def _run_eval():
        for i, _ in enumerate(run_evaluate_aligned.map(eval_args)):
            if (i + 1) % 10 == 0 or i + 1 == len(eval_args):
                print(f"  evaluate_aligned: {i + 1}/{len(eval_args)} complete")

    def _run_xgb():
        for i, _ in enumerate(xgb_func.starmap(xgb_args)):
            if (i + 1) % 5 == 0 or i + 1 == len(xgb_args):
                print(f"  xgboost: {i + 1}/{len(xgb_args)} complete")

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_eval = executor.submit(_run_eval)
        fut_xgb = executor.submit(_run_xgb)
        fut_eval.result()
        fut_xgb.result()

    # ── Phase 2: Plotting (all inner calls are cached, so this is fast) ──

    print(f"\nPhase 2: Generating plots for {len(datasets)} datasets...")
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

    # Plot sample efficiency at the end just before thread is stopped
    if plotter_thread is not None:
        with plot_lock:
            print("\nFinalizing plot_aligned_sample_efficiency after Phase 2...")
            _generate_sample_efficiency_plots()
        stop_plotting.set()
        plotter_thread.join(timeout=10)

    print(f"\nAll {total} experiments complete!")
    print("Run './venv/bin/python3 download_results.py' to download results.")
