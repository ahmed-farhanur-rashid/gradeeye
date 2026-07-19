"""
CORN loss (conditional training scheme).

Chosen over CORAL (plan Section 5) because CORN guarantees rank-consistency
structurally rather than as a soft training penalty. Each of the 4 binary
sub-problems is trained only on the subset of samples that "survived" past
the previous threshold (that's the conditional part -- it's what makes
CORN's rank-consistency structural rather than a soft constraint).
"""
import torch
import torch.nn.functional as F

from src.losses.class_weights import build_sample_weights_for_threshold


def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int,
              per_threshold_weights=None) -> torch.Tensor:
    """
    logits: (B, num_thresholds) raw CORN logits, num_thresholds = num_classes - 1.
    labels: (B,) long tensor, ordinal class indices in [0, num_classes-1].
    per_threshold_weights: optional list of length num_thresholds, each a
                     [w_negative, w_positive] np.ndarray for that threshold's
                     binary sub-problem (from class_weights.py
                     compute_corn_per_threshold_weights). Pass None to
                     disable weighting entirely (used by the baseline run
                     in the plan's run matrix, which has no class weighting).

    Returns scalar loss.
    """
    num_thresholds = num_classes - 1
    device = logits.device
    total_loss = torch.tensor(0.0, device=device)
    total_terms = 0

    for k in range(num_thresholds):
        # Task k: predict whether label > k, but ONLY among samples with
        # label >= k (the conditional subset -- samples that already passed
        # threshold k-1 are the only ones eligible to be evaluated on
        # threshold k). This conditioning is what makes CORN's guarantee
        # structural instead of a soft penalty like CORAL's.
        eligible_mask = labels >= k
        if eligible_mask.sum() == 0:
            continue

        eligible_labels = labels[eligible_mask]
        target_k = (eligible_labels > k).float()
        logits_k = logits[eligible_mask, k]

        loss_k = F.binary_cross_entropy_with_logits(logits_k, target_k, reduction="none")

        if per_threshold_weights is not None:
            weights_k = build_sample_weights_for_threshold(eligible_labels, k, per_threshold_weights[k])
            loss_k = loss_k * weights_k

        total_loss = total_loss + loss_k.sum()
        total_terms += eligible_mask.sum().item()

    if total_terms == 0:
        return total_loss  # degenerate empty-batch edge case

    return total_loss / total_terms
