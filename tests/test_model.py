"""Tests for the full model architecture (backbone + CBAM + head).

Validates:
  - Model forward pass produces correct output shape for CORN and CE modes
  - freeze_backbone / unfreeze_backbone toggle requires_grad correctly
  - build_model reads config correctly
  - CBAM attention preserves spatial dimensions
"""
import pytest
import torch

from src.models.dr_model import DRGradingModel, build_model
from src.models.cbam import CBAM


class TestDRGradingModel:
    """Model architecture and forward pass."""

    def test_corn_output_shape(self):
        """CORN model outputs (B, 4) raw logits."""
        model = DRGradingModel(pretrained=False, use_cbam=True, num_thresholds=4,
                                head_hidden_dim=64, dropout=0.1, output_mode="corn")
        x = torch.randn(2, 3, 384, 384)
        out = model(x)
        assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    def test_softmax_output_shape(self):
        """CE model outputs (B, 5) raw logits."""
        model = DRGradingModel(pretrained=False, use_cbam=False, num_thresholds=4,
                                head_hidden_dim=64, dropout=0.1, output_mode="softmax")
        x = torch.randn(2, 3, 384, 384)
        out = model(x)
        assert out.shape == (2, 5), f"Expected (2, 5), got {out.shape}"


class TestFreezeUnfreeze:
    """Backbone freeze/unfreeze for phased training."""

    def test_freeze_disables_backbone_grad(self):
        model = DRGradingModel(pretrained=False, use_cbam=True, num_thresholds=4,
                                head_hidden_dim=64, dropout=0.1)
        model.freeze_backbone()

        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone params should be frozen"
        for p in model.cbam_modules.parameters():
            assert not p.requires_grad, "CBAM params should be frozen"
        for p in model.head.parameters():
            assert p.requires_grad, "Head params should remain trainable"

    def test_unfreeze_enables_all_grad(self):
        model = DRGradingModel(pretrained=False, use_cbam=True, num_thresholds=4,
                                head_hidden_dim=64, dropout=0.1)
        model.freeze_backbone()
        model.unfreeze_backbone()

        for p in model.backbone.parameters():
            assert p.requires_grad, "Backbone should be unfrozen"
        for p in model.cbam_modules.parameters():
            assert p.requires_grad, "CBAM should be unfrozen"


class TestBuildModel:
    """build_model() reads config correctly."""

    def test_corn_config(self):
        config = {
            "loss_type": "corn",
            "model": {
                "pretrained": False,
                "use_cbam": True,
                "cbam_num_stages": 2,
                "num_thresholds": 4,
                "head_hidden_dim": 64,
                "dropout": 0.1,
            },
        }
        model = build_model(config)
        model.eval()  # BN1d requires batch_size>1 in training mode
        x = torch.randn(1, 3, 384, 384)
        out = model(x)
        assert out.shape == (1, 4)

    def test_ce_config(self):
        config = {
            "loss_type": "ce",
            "model": {
                "pretrained": False,
                "use_cbam": False,
                "num_thresholds": 4,
                "head_hidden_dim": 64,
                "dropout": 0.1,
            },
        }
        model = build_model(config)
        model.eval()  # BN1d requires batch_size>1 in training mode
        x = torch.randn(1, 3, 384, 384)
        out = model(x)
        assert out.shape == (1, 5)


class TestCBAM:
    """CBAM attention module."""

    def test_preserves_spatial_dims(self):
        cbam = CBAM(channels=64)
        x = torch.randn(2, 64, 12, 12)
        out = cbam(x)
        assert out.shape == x.shape, f"CBAM should preserve shape: {x.shape} -> {out.shape}"

    def test_output_differs_from_input(self):
        """Attention should modify the features (not identity)."""
        cbam = CBAM(channels=64)
        x = torch.randn(2, 64, 12, 12)
        out = cbam(x)
        assert not torch.equal(out, x), "CBAM output should differ from input"
