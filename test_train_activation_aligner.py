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
            patience=20,
            max_epochs=300,
            min_improvement_delta=1e-6,
            show_pbar=False,
        )

        self.assertAlmostEqual(cf_loss, it_loss, delta=1e-4)

        torch.testing.assert_close(
            cf_state["0.weight"],
            it_state["0.weight"],
            atol=5e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            cf_state["0.bias"],
            it_state["0.bias"],
            atol=5e-3,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
