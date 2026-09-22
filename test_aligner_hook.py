"""Tests for aligner-to-estimator mapping in get_predictions_and_probabilities.

Verifies that the i-th estimator's activations are aligned by the i-th aligner model
for both TabPFN and TabFM, across different ensemble batch sizes.

Only uses ``get_predictions_and_probabilities`` from evaluate_aligned_student.py —
no assumptions about internal hook classes or tracking mechanisms.

Usage:
    ./venv/bin/python -m unittest test_aligner_hook -v
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
from evaluate_aligned_student import get_predictions_and_probabilities
from model_utils import fit_model, get_transformer_layer


# ─── Test Helpers ─────────────────────────────────────────────────────────


class FingerprintAligner(nn.Module):
    """Aligner that records a fingerprint of the activations it receives.

    Returns zeros so the alignment is a no-op (identity residual).
    The fingerprint is the double-precision sum of all elements, which is
    deterministic for identical inputs.
    """

    def __init__(self):
        super().__init__()
        self.fingerprint = None
        # Need a parameter so .to(device) works
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.fingerprint = x.view(-1).double().sum().item()
        return torch.zeros_like(x)


def _capture_reference_fingerprints(model, X_test, layer_k, model_type, n_estimators):
    """Run model.predict_proba directly with a capture hook to get per-estimator
    activation fingerprints.

    The fingerprint is computed in the same way as FingerprintAligner: float32 view,
    then double-precision sum. This gives us the ground-truth fingerprint for each
    estimator in processing order.
    """
    layer = get_transformer_layer(model, layer_k, model_type)
    fingerprints = []

    def capture_hook(module, inp, output):
        x = output[0] if isinstance(output, tuple) else output
        for i in range(x.shape[0]):
            fp = x[i].float().view(-1).double().sum().item()
            fingerprints.append(fp)

    handle = layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model.predict_proba(X_test)
    handle.remove()

    assert len(fingerprints) == n_estimators, (
        f"Expected {n_estimators} fingerprints, got {len(fingerprints)}"
    )
    assert len(set(fingerprints)) == n_estimators, (
        f"Expected {n_estimators} distinct fingerprints, got {len(set(fingerprints))}"
    )
    return fingerprints


# ─── Shared test data ────────────────────────────────────────────────────

np.random.seed(42)
_X_train = np.random.randn(30, 5).astype(np.float32)
_y_train = np.random.randint(0, 2, 30)
_X_test = np.random.randn(5, 5).astype(np.float32)


# ─── Base Test Class ─────────────────────────────────────────────────────


class AlignerMappingTestBase(unittest.TestCase):
    """Base class with the shared aligner mapping verification logic."""

    def _run_aligner_mapping_test(self, model_type, n_estimators, batch_size=None):
        """Verify that aligner i receives estimator i's activations.

        1. Baseline run: call model.predict_proba directly with a capture hook to
           get per-estimator activation fingerprints.
        2. Alignment run: call get_predictions_and_probabilities with FingerprintAligners.
        3. Compare: aligner i's fingerprint must match the baseline fingerprint for
           estimator i.
        """
        model = fit_model(_X_train, _y_train, n_estimators=n_estimators, model=model_type)

        if batch_size is not None:
            model.batch_size = batch_size

        layer_k = 1

        # Step 1: capture reference fingerprints (one per estimator, in processing order)
        reference_fingerprints = _capture_reference_fingerprints(
            model, _X_test, layer_k=layer_k, model_type=model_type, n_estimators=n_estimators,
        )

        # Step 2: run with FingerprintAligners via the public API (i.e. get_predictions_and_probabilities)
        aligners = {i: FingerprintAligner() for i in range(n_estimators)}

        preds, probs = get_predictions_and_probabilities(
            model, _X_test, aligner_models=aligners, layer_k=layer_k, model_type=model_type,
        )

        # Step 3: verify that aligner i received estimator i's activations
        for i in range(n_estimators):
            self.assertIsNotNone(aligners[i].fingerprint, f"Aligner {i} was never called")
            self.assertAlmostEqual(
                aligners[i].fingerprint, reference_fingerprints[i], places=4,
                msg=(
                    f"Aligner {i} fingerprint mismatch: "
                    f"got {aligners[i].fingerprint}, expected {reference_fingerprints[i]} "
                    f"(diff={abs(aligners[i].fingerprint - reference_fingerprints[i]):.2e})"
                ),
            )


# ─── TabFM Tests ─────────────────────────────────────────────────────────


class TestTabFM(AlignerMappingTestBase):

    def test_batch_size_1(self):
        """TabFM with batch_size=1: estimators forwarded one at a time."""
        self._run_aligner_mapping_test("tabfm", n_estimators=4, batch_size=1)

    def test_batch_size_2(self):
        """TabFM with batch_size=2: estimators forwarded in pairs."""
        self._run_aligner_mapping_test("tabfm", n_estimators=4, batch_size=2)

    def test_batch_size_all(self):
        """TabFM with batch_size=0 (all at once): all estimators in one forward pass."""
        self._run_aligner_mapping_test("tabfm", n_estimators=4, batch_size=0)

    def test_single_estimator(self):
        """TabFM with a single estimator."""
        self._run_aligner_mapping_test("tabfm", n_estimators=1, batch_size=1)


# ─── TabPFN Tests ────────────────────────────────────────────────────────


class TestTabPFN(AlignerMappingTestBase):

    def test_2_estimators(self):
        """TabPFN with 2 estimators."""
        self._run_aligner_mapping_test("tabpfn", n_estimators=2)

    def test_4_estimators(self):
        """TabPFN with 4 estimators."""
        self._run_aligner_mapping_test("tabpfn", n_estimators=4)

    def test_single_estimator(self):
        """TabPFN with a single estimator."""
        self._run_aligner_mapping_test("tabpfn", n_estimators=1)


if __name__ == "__main__":
    unittest.main()
