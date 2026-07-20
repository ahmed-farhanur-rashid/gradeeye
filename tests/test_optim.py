"""Tests for optimizer parameter groups and weight-decay exemption.

Validates:
  - Bias, BatchNorm, and LayerNorm parameters get weight_decay=0.0
  - Convolution and Linear weights get the configured weight_decay
  - Phase 1 (frozen backbone) only optimizes head params
  - Phase 2/3 has correct per-module LR groups (head > cbam > backbone)
  - All model parameters with requires_grad=True are covered by some group
"""
import pytest
import torch

from src.models.dr_model import DRGradingModel
from src.training.optim import build_optimizer, _NO_DECAY_KEYWORDS


def _build_test_model(use_cbam=True, output_mode="corn"):
    return DRGradingModel(
        pretrained=False,
        use_cbam=use_cbam,
        cbam_num_stages=2 if use_cbam else 0,
        num_thresholds=4,
        head_hidden_dim=64,
        dropout=0.1,
        output_mode=output_mode,
    )


def _all_optimized_params(optimizer):
    """Collect all parameter ids from all groups."""
    return {id(p) for g in optimizer.param_groups for p in g["params"]}


class TestWeightDecayExemption:
    """Plan Section 6: WD only on Conv/Linear weights, not on bias/BN/LN."""

    def test_bias_params_have_zero_weight_decay(self):
        model = _build_test_model()
        model.unfreeze_backbone()
        optimizer = build_optimizer(model, phase="phase2_full_training",
                                    head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4)

        bias_param_ids = {id(p) for n, p in model.named_parameters()
                         if "bias" in n and p.requires_grad}

        for group in optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            overlap = bias_param_ids & group_ids
            if overlap:
                assert group["weight_decay"] == 0.0, \
                    f"Bias params in group with weight_decay={group['weight_decay']} != 0.0"

    def test_batchnorm_params_have_zero_weight_decay(self):
        model = _build_test_model()
        model.unfreeze_backbone()
        optimizer = build_optimizer(model, phase="phase2_full_training",
                                    head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4)

        bn_param_ids = set()
        for n, p in model.named_parameters():
            if any(nd in n.lower() for nd in ("bn", "norm")) and p.requires_grad:
                bn_param_ids.add(id(p))

        for group in optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            overlap = bn_param_ids & group_ids
            if overlap:
                assert group["weight_decay"] == 0.0, \
                    f"BN params in group with weight_decay={group['weight_decay']} != 0.0"

    def test_conv_linear_weights_have_nonzero_weight_decay(self):
        model = _build_test_model()
        model.unfreeze_backbone()
        optimizer = build_optimizer(model, phase="phase2_full_training",
                                    head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4)

        # Find a Conv/Linear weight (not bias, not norm)
        conv_linear_ids = set()
        for n, p in model.named_parameters():
            if p.requires_grad and not any(nd in n.lower() for nd in _NO_DECAY_KEYWORDS):
                conv_linear_ids.add(id(p))

        found_nonzero_wd = False
        for group in optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            overlap = conv_linear_ids & group_ids
            if overlap and group["weight_decay"] > 0:
                found_nonzero_wd = True

        assert found_nonzero_wd, "Conv/Linear weight params should have nonzero weight_decay"


class TestOptimizerPhaseConfig:
    """Correct parameter grouping across training phases."""

    def test_phase1_only_optimizes_head(self):
        model = _build_test_model()
        model.freeze_backbone()

        optimizer = build_optimizer(model, phase="phase1_frozen",
                                    head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4)

        optimized_ids = _all_optimized_params(optimizer)
        head_ids = {id(p) for p in model.head.parameters() if p.requires_grad}

        assert optimized_ids == head_ids, \
            "Phase 1 should only optimize head params"

    def test_phase2_includes_all_modules(self):
        model = _build_test_model(use_cbam=True)
        model.unfreeze_backbone()

        optimizer = build_optimizer(model, phase="phase2_full_training",
                                    head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4)

        optimized_ids = _all_optimized_params(optimizer)
        all_trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

        assert optimized_ids == all_trainable_ids, \
            "Phase 2 should include all trainable params"

    def test_head_lr_higher_than_backbone_lr(self):
        model = _build_test_model(use_cbam=True)
        model.unfreeze_backbone()

        head_lr, backbone_lr = 1e-3, 1e-5
        optimizer = build_optimizer(model, phase="phase2_full_training",
                                    head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=1e-4)

        head_param_ids = {id(p) for p in model.head.parameters() if p.requires_grad}
        backbone_param_ids = {id(p) for p in model.backbone.parameters() if p.requires_grad}

        for group in optimizer.param_groups:
            group_ids = {id(p) for p in group["params"]}
            if group_ids & head_param_ids:
                assert group["lr"] == head_lr, f"Head LR should be {head_lr}"
            if group_ids & backbone_param_ids:
                assert group["lr"] == backbone_lr, f"Backbone LR should be {backbone_lr}"

    def test_no_cbam_model_has_no_empty_groups(self):
        """Baseline (no CBAM) should still produce valid optimizer."""
        model = _build_test_model(use_cbam=False)
        model.unfreeze_backbone()

        optimizer = build_optimizer(model, phase="phase2_full_training",
                                    head_lr=1e-3, backbone_lr=1e-5, weight_decay=1e-4)

        for group in optimizer.param_groups:
            assert len(group["params"]) > 0, "Optimizer should have no empty param groups"
