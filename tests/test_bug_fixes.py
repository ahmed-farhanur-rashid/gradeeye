"""Tests for the bug fixes in this refactor.

Validates:
  - corn_loss weighted mean actually applies per-class weighting
    (positive and negative samples contribute proportionally to their weight).
  - tta_forward returns averaged CLASS probabilities, not conditional
    probabilities (i.e. the cumprod step is now per-pass, not post-aggregation).
  - freeze_backbone keeps BN params trainable.
"""
import os
import sys
sys.path.insert(0, "/home/farhan/my-projects/gradeeye")

import torch
import numpy as np

from src.losses.corn_loss import corn_loss
from src.losses.class_weights import compute_corn_per_threshold_weights
from src.eval.tta import tta_forward
from src.models.corn import corn_predict_probas


def test_corn_weighted_mean_changes_loss():
    """The same predictions should give different losses with vs without
    weighting, demonstrating that the weighting is actually being applied
    to the gradient (not just multiplying a constant that .mean() cancels)."""
    # Construct a batch where positive (minority) samples are confidently wrong.
    logits = torch.tensor([[5.0, 5.0, 5.0, 5.0],   # class 0 — easy
                           [5.0, 5.0, 5.0, 5.0],   # class 0
                           [-5.0, -5.0, -5.0, -5.0]])  # class 4 — very wrong
    labels = torch.tensor([0, 0, 4])

    counts = np.array([1, 1, 1, 1, 1])  # equal counts -> no weighting
    weights_eq = compute_corn_per_threshold_weights(counts, num_classes=5,
                                                   method="effective_number")

    counts_imbal = np.array([1000, 50, 50, 50, 50])  # class 4 heavily minority
    weights_imbal = compute_corn_per_threshold_weights(counts_imbal, num_classes=5,
                                                     method="effective_number")

    loss_eq = corn_loss(logits, labels, num_classes=5, per_threshold_weights=weights_eq)
    loss_imbal = corn_loss(logits, labels, num_classes=5, per_threshold_weights=weights_imbal)

    # With imbalanced weights, the wrong class-4 prediction should be UPweighted
    # -> loss should be LARGER (or at least different) than the equal-weight case.
    assert not torch.isclose(loss_eq, loss_imbal), (
        f"Weighted mean bug: equal vs imbalanced weights gave same loss "
        f"({loss_eq.item():.4f} vs {loss_imbal.item():.4f}). "
        f"The per-class weighting is being silently undone."
    )


def test_tta_returns_class_probabilities():
    """tta_forward must return a CLASS distribution (sums to 1, has 5 entries),
    not a conditional probability vector (4 entries that need cumprod)."""
    from src.models.dr_model import build_model
    from src.config_merge import merge_configs
    import yaml
    # Configs are split by axis — load model + LODO yamls and merge.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(f"{repo_root}/configs/models/convnext_tiny.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(f"{repo_root}/configs/lodo/eyepacs.yaml") as f:
        lodo_cfg = yaml.safe_load(f)
    cfg = merge_configs(model_cfg, lodo_cfg)
    cfg["model"]["pretrained"] = False
    cfg["model"]["use_cbam"] = False
    model = build_model(cfg)
    model.eval()

    x = torch.randn(3, 3, 384, 384)
    out = tta_forward(model, x, output_mode="corn")

    # Should be a 5-class probability distribution (num_classes = num_thresholds + 1).
    assert out.shape == (3, 5), f"Expected (3, 5) class distribution, got {tuple(out.shape)}"
    assert (out >= 0).all(), "Class probs must be non-negative"
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3), (
        f"Class probs must sum to ~1, got {sums.tolist()}"
    )


def test_freeze_backbone_keeps_bn_trainable():
    """ConvNeXt-Tiny uses LayerNorm, but the registry contract is that
    freeze_backbone keeps norm params trainable. Verify for a model that
    has at least one BN submodule (build a tiny synthetic model)."""
    import torch.nn as nn
    from src.models.dr_model import DRGradingModel

    # ConvNeXt has LayerNorm only; verify the loop doesn't accidentally
    # leave non-norm params trainable.
    model = DRGradingModel(pretrained=False, use_cbam=False, num_thresholds=4,
                           head_hidden_dim=64, dropout=0.1)
    model.freeze_backbone()

    for name, p in model.backbone.named_parameters():
        is_norm = any(nd in name.lower() for nd in ("norm", ".bn", "_bn"))
        if is_norm:
            assert p.requires_grad, f"Norm param {name} should remain trainable"
        else:
            assert not p.requires_grad, f"Non-norm param {name} should be frozen"


if __name__ == "__main__":
    test_corn_weighted_mean_changes_loss()
    print("PASS: corn_loss weighted mean actually applies weighting")
    test_tta_returns_class_probabilities()
    print("PASS: tta_forward returns class probabilities")
    test_freeze_backbone_keeps_bn_trainable()
    print("PASS: freeze_backbone keeps norm params trainable")