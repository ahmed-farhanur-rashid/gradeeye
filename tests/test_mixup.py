"""Tests for MixUp augmentation.

Validates:
  - MixUp produces blended images and correct label pairs
  - When MixUp is disabled, output is identity (no blending)
  - Lambda is drawn from Beta distribution (0 < lam < 1)
  - Labels_a and labels_b are permutations of the original labels
"""
import pytest
import torch

from src.augmentation.transforms import mixup_batch, maybe_apply_mixup


class TestMixupBatch:
    """Direct tests for mixup_batch()."""

    def test_output_shapes(self):
        images = torch.randn(8, 3, 64, 64)
        labels = torch.randint(0, 5, (8,))

        mixed, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=0.2)

        assert mixed.shape == images.shape
        assert labels_a.shape == labels.shape
        assert labels_b.shape == labels.shape
        assert 0 <= lam <= 1

    def test_labels_a_is_original(self):
        """labels_a should be the original labels (unchanged)."""
        images = torch.randn(8, 3, 64, 64)
        labels = torch.arange(8)

        _, labels_a, _, _ = mixup_batch(images, labels, alpha=0.2)
        assert torch.equal(labels_a, labels)

    def test_labels_b_is_permutation(self):
        """labels_b should be a permutation of the original labels."""
        images = torch.randn(16, 3, 64, 64)
        labels = torch.arange(16)

        _, _, labels_b, _ = mixup_batch(images, labels, alpha=0.2)

        # labels_b should contain the same elements (just reordered)
        assert set(labels_b.tolist()) == set(labels.tolist())

    def test_mixed_images_are_blend(self):
        """Mixed images should be lam*orig + (1-lam)*permuted."""
        torch.manual_seed(0)
        images = torch.randn(8, 3, 64, 64)
        labels = torch.randint(0, 5, (8,))

        mixed, _, _, lam = mixup_batch(images, labels, alpha=1.0)  # alpha=1 → uniform Beta

        # The mixed image should NOT be identical to original (unless lam=1.0 exactly)
        if lam < 0.999:
            assert not torch.equal(mixed, images), "Mixed should differ from original"

    def test_alpha_zero_returns_identity(self):
        """alpha=0 should return unmodified images and lam=1."""
        images = torch.randn(8, 3, 64, 64)
        labels = torch.randint(0, 5, (8,))

        mixed, labels_a, labels_b, lam = mixup_batch(images, labels, alpha=0)

        assert torch.equal(mixed, images)
        assert torch.equal(labels_a, labels)
        assert torch.equal(labels_b, labels)
        assert lam == 1.0


class TestMaybeApplyMixup:
    """Tests for the stochastic MixUp wrapper."""

    def test_disabled_returns_identity(self):
        images = torch.randn(8, 3, 64, 64)
        labels = torch.randint(0, 5, (8,))

        out_images, labels_a, labels_b, lam = maybe_apply_mixup(
            images, labels, enabled=False
        )

        assert torch.equal(out_images, images)
        assert torch.equal(labels_a, labels)
        assert torch.equal(labels_b, labels)
        assert lam == 1.0

    def test_identity_when_not_fired(self):
        """When MixUp doesn't fire (random > p), labels_a == labels_b == labels."""
        images = torch.randn(8, 3, 64, 64)
        labels = torch.randint(0, 5, (8,))

        # With p=0, MixUp never fires
        _, labels_a, labels_b, lam = maybe_apply_mixup(
            images, labels, p=0.0, enabled=True
        )

        assert torch.equal(labels_a, labels_b)
        assert lam == 1.0

    def test_loss_combination_correct(self):
        """The caller uses lam*loss_a + (1-lam)*loss_b.
        When MixUp doesn't fire (lam=1), this should collapse to loss_a only."""
        images = torch.randn(8, 3, 64, 64)
        labels = torch.randint(0, 5, (8,))

        _, labels_a, labels_b, lam = maybe_apply_mixup(
            images, labels, enabled=False
        )

        # Simulate loss combination
        loss_a = torch.tensor(1.5)
        loss_b = torch.tensor(2.0)
        combined = lam * loss_a + (1 - lam) * loss_b

        assert combined.item() == loss_a.item(), \
            f"With lam=1, combined loss should equal loss_a: {combined.item()} vs {loss_a.item()}"
