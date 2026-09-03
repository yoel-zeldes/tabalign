"""Unit tests for train_activation_aligner.

Verifies that the closed-form analytical ridge regression solver (_fit_linear_aligner_closed_form)
and the iterative AdamW optimizer (_train_aligner_on_tensors) yield almost the same solution
for a linear aligner.

Usage:
    ./venv/bin/python3 -m unittest test_train_activation_aligner -v
"""

import unittest
import torch
from train_activation_aligner import (
    _fit_linear_aligner_closed_form,
    _train_aligner_on_tensors,
    prepare_train_val,
)


class TestLinearAlignerSolvers(unittest.TestCase):
    """Test equivalence between closed-form and iterative linear aligner solvers."""

    def _assert_closed_form_matches_iterative(
        self,
        train_x,
        train_y,
        val_x,
        val_y,
        patience=30,
        max_epochs=400,
        loss_delta=5e-3,
        weight_atol=2e-2,
        weight_rtol=0.1,
        bias_atol=0.03,
        bias_rtol=0.2,
    ):
        hyperparams = {
            "hidden_layers": [],
            "lr": 0.01,
            "weight_decay": 1e-4,
            "batch_size": 2048,
        }

        cf_state, cf_loss = _fit_linear_aligner_closed_form(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            weight_decay=hyperparams["weight_decay"],
        )

        it_state, it_loss, _ = _train_aligner_on_tensors(
            est_idx=0,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            device="cpu",
            hyperparams=hyperparams,
            patience=patience,
            max_epochs=max_epochs,
            min_improvement_delta=1e-6,
            show_pbar=False,
        )

        self.assertAlmostEqual(cf_loss, it_loss, delta=loss_delta)
        torch.testing.assert_close(
            cf_state["0.weight"],
            it_state["0.weight"],
            atol=weight_atol,
            rtol=weight_rtol,
        )
        torch.testing.assert_close(
            cf_state["0.bias"],
            it_state["0.bias"],
            atol=bias_atol,
            rtol=bias_rtol,
        )

    def test_closed_form_matches_iterative_for_linear_aligner(self):
        n_samples = 1000
        dim = 128

        torch.manual_seed(42)
        s_act = torch.randn(n_samples, dim)
        true_w = torch.randn(dim, dim) * 0.1
        true_b = torch.randn(dim) * 0.05
        # Teacher activations with linear relationship + noise
        t_act = s_act + s_act @ true_w + true_b + 0.01 * torch.randn(n_samples, dim)

        train_x, train_y, val_x, val_y = prepare_train_val(s_act, t_act, device="cpu")
        self._assert_closed_form_matches_iterative(train_x, train_y, val_x, val_y)

    def test_closed_form_matches_iterative_underdetermined_n_less_than_d(self):
        """Verify closed-form dual solver matches iterative solver when N < D."""
        torch.manual_seed(42)
        n_samples = 200
        dim = 256

        s_act = torch.randn(n_samples, dim)
        true_w = torch.randn(dim, dim) / (dim ** 0.5) * 0.1
        true_b = torch.randn(dim) * 0.01
        t_act = s_act + s_act @ true_w + true_b + 0.001 * torch.randn(n_samples, dim)

        train_x, train_y, val_x, val_y = prepare_train_val(s_act, t_act, device="cpu")
        self.assertLess(train_x.shape[0], dim)  # Ensure underdetermined (N < D)
        self._assert_closed_form_matches_iterative(train_x, train_y, val_x, val_y)


if __name__ == "__main__":
    unittest.main()
