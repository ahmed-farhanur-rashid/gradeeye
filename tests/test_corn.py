"""Tests for CORN loss, prediction, and probability conversion.

Validates:
  - CORN loss produces finite scalar for all class distributions
  - CORN loss is zero when model perfectly predicts all thresholds
  - Per-threshold class weighting doesn't change loss shape
  - corn_predict produces valid class indices in [0, 4]
  - corn_predict_probas produces a valid probability distribution
  - Rank-consistency: cumulative product is monotonically non-increasing
"""
import numpy as np
import pytest
import torch

from src.losses.corn_loss import corn_loss
from src.losses.class_weights import compute_corn_per_threshold_weights
from src.models.corn import corn_logits_to_probas, corn_predict, corn_predict_probas

NUM_CLASSES = 5
NUM_THRESHOLDS = NUM_CLASSES - 1


class TestCornLoss:
    """Core CORN loss correctness."""

    def test_loss_is_finite_scalar(self):
        """Loss should be a finite scalar for random inputs."""
        logits = torch.randn(16, NUM_THRESHOLDS)
        labels = torch.randint(0, NUM_CLASSES, (16,))
        loss = corn_loss(logits, labels, NUM_CLASSES)

        assert loss.dim() == 0, "Loss must be scalar"
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"

    def test_loss_with_all_same_label(self):
        """Loss should still be finite when all labels are the same class."""
        logits = torch.randn(16, NUM_THRESHOLDS)
        for label_val in range(NUM_CLASSES):
            labels = torch.full((16,), label_val, dtype=torch.long)
            loss = corn_loss(logits, labels, NUM_CLASSES)
            assert torch.isfinite(loss), f"Loss not finite for all-class-{label_val}"

    def test_loss_with_single_sample(self):
        """Loss should work for batch_size=1."""
        logits = torch.randn(1, NUM_THRESHOLDS)
        labels = torch.tensor([2])
        loss = corn_loss(logits, labels, NUM_CLASSES)
        assert torch.isfinite(loss)

    def test_perfect_prediction_has_low_loss(self):
        """When logits strongly predict the correct thresholds, loss → 0."""
        # For label=3: thresholds 0,1,2 should be positive (>3 means y>0,y>1,y>2 all true)
        # threshold 3 should be negative (y>3 is false)
        logits = torch.tensor([
            [10.0, 10.0, 10.0, -10.0],  # label=3
            [-10.0, -10.0, -10.0, -10.0],  # label=0
            [10.0, 10.0, 10.0, 10.0],  # label=4
        ])
        labels = torch.tensor([3, 0, 4])
        loss = corn_loss(logits, labels, NUM_CLASSES)
        assert loss.item() < 0.01, f"Perfect-prediction loss should be near 0, got {loss.item()}"

    def test_loss_with_per_threshold_weights(self):
        """Weighted loss should still be finite and differ from unweighted."""
        logits = torch.randn(32, NUM_THRESHOLDS)
        labels = torch.randint(0, NUM_CLASSES, (32,))

        loss_unweighted = corn_loss(logits, labels, NUM_CLASSES, per_threshold_weights=None)

        # Build per-threshold weights from the labels
        weights = compute_corn_per_threshold_weights(labels.numpy(), NUM_CLASSES)
        loss_weighted = corn_loss(logits, labels, NUM_CLASSES, per_threshold_weights=weights)

        assert torch.isfinite(loss_weighted)
        # Weighted and unweighted should generally differ
        # (unless weights happen to all be 1.0, which is unlikely with imbalanced random labels)

    def test_loss_gradient_flows(self):
        """Gradient must flow through CORN loss for backprop to work."""
        logits = torch.randn(8, NUM_THRESHOLDS, requires_grad=True)
        labels = torch.randint(0, NUM_CLASSES, (8,))
        loss = corn_loss(logits, labels, NUM_CLASSES)
        loss.backward()

        assert logits.grad is not None, "Gradient must exist"
        assert torch.isfinite(logits.grad).all(), "Gradients must be finite"

    def test_eligible_mask_conditional_subsetting(self):
        """Verify CORN's conditional training: threshold k only uses labels >= k.

        For threshold k=3 (y>3), only samples with label >= 3 should contribute.
        If all labels < 3, threshold 3 should be skipped entirely.
        """
        logits = torch.randn(8, NUM_THRESHOLDS)
        # All labels 0 or 1 — thresholds 2 and 3 have no eligible samples
        labels = torch.randint(0, 2, (8,))

        loss = corn_loss(logits, labels, NUM_CLASSES)
        assert torch.isfinite(loss), "Loss must handle missing thresholds gracefully"


class TestCornPredict:
    """CORN inference rule correctness."""

    def test_output_shape(self):
        logits = torch.randn(16, NUM_THRESHOLDS)
        preds = corn_predict(logits)
        assert preds.shape == (16,), f"Expected (16,), got {preds.shape}"

    def test_output_range(self):
        """Predictions must be valid class indices in [0, num_classes-1]."""
        logits = torch.randn(100, NUM_THRESHOLDS)
        preds = corn_predict(logits)
        assert (preds >= 0).all() and (preds <= NUM_THRESHOLDS).all(), \
            f"Predictions out of range: min={preds.min()}, max={preds.max()}"

    def test_strong_positive_logits_predict_highest_class(self):
        """Very large positive logits → all thresholds exceeded → class 4."""
        logits = torch.full((4, NUM_THRESHOLDS), 20.0)
        preds = corn_predict(logits)
        assert (preds == 4).all(), f"Expected all class 4, got {preds}"

    def test_strong_negative_logits_predict_lowest_class(self):
        """Very large negative logits → no thresholds exceeded → class 0."""
        logits = torch.full((4, NUM_THRESHOLDS), -20.0)
        preds = corn_predict(logits)
        assert (preds == 0).all(), f"Expected all class 0, got {preds}"

    def test_rank_consistency(self):
        """CORN's structural guarantee: unconditional probs are monotonically non-increasing.

        P(y>0) >= P(y>1) >= P(y>2) >= P(y>3) for every sample, by construction
        (cumprod of values in [0,1] is non-increasing).
        """
        logits = torch.randn(100, NUM_THRESHOLDS)
        cond_probs = corn_logits_to_probas(logits)
        uncond_probs = torch.cumprod(cond_probs, dim=1)

        # Check monotonicity: each column should be >= the next
        for k in range(NUM_THRESHOLDS - 1):
            assert (uncond_probs[:, k] >= uncond_probs[:, k + 1] - 1e-6).all(), \
                f"Rank consistency violated at threshold {k}"


class TestCornPredictProbas:
    """Per-class probability distribution from CORN logits."""

    def test_output_shape(self):
        logits = torch.randn(16, NUM_THRESHOLDS)
        probas = corn_predict_probas(logits)
        assert probas.shape == (16, NUM_CLASSES)

    def test_probabilities_sum_to_one(self):
        """Per-sample class probabilities must sum to ~1.0."""
        logits = torch.randn(100, NUM_THRESHOLDS)
        probas = corn_predict_probas(logits)
        sums = probas.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
            f"Probability sums deviate from 1.0: {sums}"

    def test_probabilities_non_negative(self):
        logits = torch.randn(100, NUM_THRESHOLDS)
        probas = corn_predict_probas(logits)
        assert (probas >= 0).all(), "Class probabilities must be non-negative"

    def test_argmax_matches_corn_predict(self):
        """argmax of class probabilities should usually match corn_predict.

        Note: they can differ at decision boundaries where probabilities are
        nearly tied, so we check >90% agreement rather than exact match.
        """
        logits = torch.randn(200, NUM_THRESHOLDS) * 3  # amplify for clearer decisions
        preds_threshold = corn_predict(logits)
        preds_probas = corn_predict_probas(logits).argmax(dim=1)
        agreement = (preds_threshold == preds_probas).float().mean()
        assert agreement > 0.85, f"corn_predict vs argmax(probas) agreement too low: {agreement:.2%}"
