# Representation Steering & Activation Alignment for Tabular Foundation Models

This repository contains the implementation of **Activation Alignment** for In-Context Learning (ICL) in Tabular Foundation Models (specifically **TabPFN** and **TabFM**). 

The goal of this project is to improve the sample efficiency and performance of tabular foundation models in data-constrained regimes by steering the intermediate transformer activations of a limited-data "student" model toward those of a full-context "teacher" model.

---

## 1. Research Motivation & Problem Formulation

### The Problem
Tabular foundation models (such as TabPFN and TabFM) operate via In-Context Learning (ICL): a set of labeled training examples $(X_{\text{train}}, y_{\text{train}})$ and unlabeled test queries $X_{\text{test}}$ are passed through a transformer architecture. While these models achieve strong performance, their predictive capacity depends heavily on the number of context examples $N$:
1. **Low-Data Degradation**: When few training examples are available ($N_{\text{student}} \ll N_{\text{full}}$), performance drops significantly.
2. **Context Window & Compute Costs**: Providing large context sets increases computational and memory complexity quadratically with sequence length.

### Core Hypothesis
Intermediate transformer representations of a "teacher" model (conditioned on the full dataset) capture rich latent priors, feature interactions, and task-specific representations. If we can learn a lightweight transformation (an **aligner**) that maps the intermediate representations of a data-constrained **student** ($N_{\text{student}}$ samples) to match those of the **teacher**, the student can achieve teacher-level performance at inference time using only its small context.

Furthermore, because these aligners operate directly on query token representations, they can be trained using **unlabeled synthetic data** generated from feature distributions, without requiring additional ground-truth labels.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      TEACHER                           │
                    │   Context: Full Dataset (N_full)                       │
                    │   Input: Synthetic Queries (X_syn)                     │
                    └────────────────────────┬───────────────────────────────┘
                                             │
                                   Extract Teacher Acts
                                        H_teacher^(k)
                                             │
┌──────────────────────────────────────┐     │         ┌─────────────────────┐
│               STUDENT                │     ▼         │       ALIGNER       │
│  Context: Subsample (N_student)      ├─► [Hook] ───► │  f_θ(H_student^(k)) │ ──► Aligned Repr
│  Input: Synthetic Queries (X_syn)    │   Layer k     └─────────────────────┘      to Layer k+1
└──────────────────────────────────────┘                                                 │
                                                                                         ▼
                                                                                Aligned Predictions
                                                                                 (Real Test Data)
```

---

## 2. Algorithmic Workflow & Pipeline Architecture

An experiment is driven by `sweep_pipeline_layers.py` which orchestrates `run_pipeline.py` across datasets, student sample sizes, and transformer layers. The end-to-end pipeline consists of five stages:

```
+------------------------------------------------------------------------------+
|  1. Synthetic Generation      -->  create_synthetic_dataset.py               |
|     (Unsupervised TabPFN Sampling)                                           |
+------------------------------------------------------------------------------+
                                          │
                                          ▼
+------------------------------------------------------------------------------+
|  2. Activation Extraction     -->  extract_activations.py                    |
|     (Capture Layer-k Activations for Teacher & Student on Synthetic Data)    |
+------------------------------------------------------------------------------+
                                          │
                                          ▼
+------------------------------------------------------------------------------+
|  3. Aligner Training          -->  train_activation_aligner.py               |
|     (Fit Linear/MLP Aligner on MSE Loss)                                     |
+------------------------------------------------------------------------------+
                                          │
                                          ▼
+------------------------------------------------------------------------------+
|  4. Downstream Evaluation     -->  evaluate_aligned_student.py               |
|     (Inject Hook at Layer-k on Real Benchmark Test Sets: AUC / Log-Loss)     |
+------------------------------------------------------------------------------+
                                          │
                                          ▼
+------------------------------------------------------------------------------+
|  5. Baseline Comparison       -->  train_xgboost.py                          |
|     (Compare against XGBoost)                                                |
+------------------------------------------------------------------------------+
```

### Stage 1: Synthetic Feature Generation (`create_synthetic_dataset.py`)
Generates unlabeled synthetic query vectors $X_{\text{synthetic}}$ to probe the model's representation space without leaking test labels:
**TabPFN Generative Mode**: Uses `TabPFNUnsupervisedModel` to learn and sample from the joint multivariate distribution of the training features.

### Stage 2: Activation Extraction (`extract_activations.py`)
Attaches forward hooks to layer $k$ of the transformer:
- **Teacher Activations** ($H_{\text{teacher}}^{(k)}$): Extracted using the full training context $(X_{\text{train}}, y_{\text{train}})$.
- **Student Activations** ($H_{\text{student}}^{(k)}$): Extracted using a stratified subset $(X_{\text{student}}, y_{\text{student}})$ of size $N_{\text{student}}$.

### Stage 3: Aligner Training (`train_activation_aligner.py`)
Trains an aligner network $f_\theta$ (Linear or MLP with LayerNorm and ReLU):
**MSE Loss**:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^M \left\| f_\theta\left(H_{\text{student}}^{(k)}\right)_i - \left(H_{\text{teacher}}^{(k)}\right)_i \right\|_2^2$$
**Residual Formulation (`--predict_residual`)**: Predicts the residual $\Delta H = H_{\text{teacher}} - H_{\text{student}}$ with near-zero initialization.

### Stage 4: Evaluation on Real Downstream Data (`evaluate_aligned_student.py`)
Evaluates the aligned student model on real, held-out test data from benchmark datasets (TabArena suite):
- Registers a forward hook at layer $k$ that replaces intermediate student activations with $f_\theta(H_{\text{student}}^{(k)})$.
- Computes metrics: **ROC-AUC** (binary classification), **Negative Log-Loss** (multiclass classification), **Accuracy**, and **Fidelity** to teacher predictions.

### Stage 5: Benchmarking & Baselines (`train_xgboost.py`)
Evaluates competitive tabular baselines:
- **Baseline Student**: TabPFN/TabFM without alignment ($N_{\text{student}}$ examples).
- **Teacher**: Full-data TabPFN/TabFM.
- **XGBoost**: Default configuration (FT-Transformer baseline recipe).

---

## 3. Supported Model Architectures

| Architecture | Model Description | Token Representation | Layer Hook Location |
| :--- | :--- | :--- | :--- |
| **TabPFN** (`tabpfn-v2`) | Prior-Data Fitted Network with multi-estimator ensembling | Feature-level and item-level tokens `[examples, features, hidden]` | `executor_.models[i].transformer_encoder.layers[k]` |
| **TabFM** (`tabfm_v1_0_0`) | Foundation model for tabular data with ICL transformer blocks | Row-level example tokens `[examples, hidden]` | `model.icl_predictor.tf_icl.blocks[k]` |

---

## 4. Repository Structure

```
├── sweep_pipeline_layers.py       # Main entry point: sweeps layers, student sizes & datasets
├── run_pipeline.py                # Single-experiment pipeline orchestrator
├── create_synthetic_dataset.py    # Unsupervised synthetic query generator
├── extract_activations.py         # Forward-hook activation extractor (Teacher & Student)
├── train_activation_aligner.py    # Aligner training
├── evaluate_aligned_student.py    # Downstream test evaluation with hooked aligner
├── train_xgboost.py               # Standard XGBoost baseline
├── pruning_utils.py               # Core utilities: data loading (TabArena/OpenML), model wrappers, metrics
├── xgboost_utils.py               # XGBoost data loading and evaluation helpers
├── activation_patching_experiment.py # Diagnostic activation patching experiments
└── experiments.txt                # Experiment tracker and configuration log
```

---

## 5. Quick Start & Usage

### Environment Setup
Ensure dependencies are installed in your Python environment:
```bash
./venv/bin/python -m pip install -r requirements.txt
```

### Running an Experiment Sweep
The primary way to launch an experiment is via `sweep_pipeline_layers.py`.

#### Example: TabFM Layer Sweep on TabArena Benchmark
```bash
./venv/bin/python sweep_pipeline_layers.py \
    --model tabfm \
    --dataset tabarena \
    --output_dir "results/tabfm" \
    --n_estimators 1 \
    --use_tabpfn \
    --n_samples 1000 \
    --student_n 0.1 0.3 0.5 0.7 0.9 \
    --layers 23 \
    --predict_residual \
    --n_repeats 3 \
    --patience 10 \
    --lr 1e-04
```

### Key Arguments Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Benchmark dataset name(s) or `tabarena` for the full suite | `breast_cancer` |
| `--model` | Tabular foundation model architecture (`tabpfn` or `tabfm`) | `tabpfn` |
| `--student_n` | Training sample size for student (integer count or dataset fraction $0 < N < 1$) | `[20]` |
| `--layers` | Transformer layer indices $k$ to extract and align | `[1, 2, 5, 8, 9, 10, 11]` |
| `--n_samples` | Number of synthetic queries generated for alignment training | `10000` |
| `--use_tabpfn` | Use TabPFN unsupervised model for synthetic data (instead of Gaussians) | `False` |
| `--predict_residual` | Aligner outputs residual update $\Delta H$ rather than direct activation | `False` |
| `--n_repeats` | Number of OpenML random splits/repeats | `1` |
