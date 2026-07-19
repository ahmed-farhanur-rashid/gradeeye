"""
Plain cross-entropy baseline, no ordinal structure.

Used for:
  - Baseline run (plain CE, no ordinal structure, no class weighting, no CBAM)
  - Ablation run (plain CE + inverse-freq weighting, WITH CBAM) — isolates
    CORN's specific contribution by holding attention constant and only
    swapping the loss/output head.

Since the model architecture always outputs `num_thresholds` = num_classes-1
raw values for CORN compatibility, the CE baseline path instead expects a
model configured with a standard Linear(num_classes) head. See
scripts/train.py for how the baseline run swaps head output width.
"""
import torch
import torch.nn.functional as F


def ce_loss(logits: torch.Tensor, labels: torch.Tensor,
            class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    logits: (B, num_classes) raw logits (standard softmax classification head).
    labels: (B,) long tensor, ordinal class indices in [0, num_classes-1]
            (treated as plain nominal classes here — that's the point of
            the baseline, it does NOT use ordinal structure).
    class_weights: optional (num_classes,) tensor, inverse-frequency weights
                   per class (5-class, not the CORN per-threshold scheme).
    """
    return F.cross_entropy(logits, labels, weight=class_weights)


def compute_5class_inverse_sqrt_weights(class_counts) -> torch.Tensor:
    """
    Standard 5-class inverse-sqrt-frequency weights, used only by the
    ablation run (plain CE + weighting). This is deliberately separate from
    the CORN per-threshold weighting scheme in class_weights.py — plan
    Section 5 explicitly warns against reusing raw 5-class weights on the
    CORN binary decomposition, so the two weighting schemes are kept apart.
    """
    import numpy as np
    counts = np.clip(np.asarray(class_counts), a_min=1, a_max=None)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
