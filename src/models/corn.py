"""
CORN (Conditional Ordinal Regression for Neural networks) output layer.

Per plan Section 4 / handoff note 2 — this changes the final layer SHAPE,
not just the loss:

  - NOT a standard Linear(num_classes=5) + softmax.
  - Reformulates the 5-class ordinal problem (0-4 ICDR) as 4 binary
    sub-problems: P(severity > 0), P(severity > 1), P(severity > 2),
    P(severity > 3), each a sigmoid output.
  - CORN's conditional training scheme structurally guarantees
    rank-consistency at inference (fixes CORAL's soft-constraint-only
    guarantee — CORAL can produce non-monotonic threshold probabilities,
    CORN cannot by construction).
  - Inference: sum of exceeded thresholds -> predicted ordinal class.

Reference: Shi, Cao & Raschka (2021), "Deep Neural Networks for Rank
Consistent Ordinal Regression Based on Conditional Probabilities".
"""
import torch


def corn_logits_to_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw CORN logits (B, num_thresholds) to conditional probabilities
    P(y > k | y > k-1) via sigmoid. These are CONDITIONAL, not marginal,
    probabilities — that's what CORN's training scheme is built around.
    """
    return torch.sigmoid(logits)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Inference rule: convert conditional probabilities to unconditional
    P(y > k) by cumulative product, threshold at 0.5, and sum the exceeded
    thresholds to get the predicted ordinal class in [0, num_thresholds].

    logits: (B, num_thresholds) raw CORN logits.
    Returns: (B,) long tensor of predicted classes.
    """
    conditional_probas = corn_logits_to_probas(logits)  # P(y > k | y > k-1)

    # Unconditional P(y > k) = product of all conditional probs up to k
    # (rank-consistency is structural here: cumprod is monotonically
    # non-increasing by construction, so thresholds can't "uncross").
    unconditional_probas = torch.cumprod(conditional_probas, dim=1)

    predicted_class = (unconditional_probas > 0.5).sum(dim=1)
    return predicted_class


def corn_predict_probas(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns per-class probability distribution (B, num_classes) derived
    from the unconditional P(y > k) values, useful for AUC-ROC computation
    which needs per-class scores rather than a single predicted label.
    """
    conditional_probas = corn_logits_to_probas(logits)
    unconditional_probas = torch.cumprod(conditional_probas, dim=1)  # (B, K) P(y>0)...P(y>K-1)

    batch_size = logits.size(0)
    num_thresholds = logits.size(1)
    num_classes = num_thresholds + 1

    # P(y=0) = 1 - P(y>0)
    # P(y=k) = P(y>k-1) - P(y>k) for 0 < k < num_classes-1
    # P(y=num_classes-1) = P(y>K-1)
    class_probas = torch.zeros(batch_size, num_classes, device=logits.device)
    class_probas[:, 0] = 1 - unconditional_probas[:, 0]
    for k in range(1, num_thresholds):
        class_probas[:, k] = unconditional_probas[:, k - 1] - unconditional_probas[:, k]
    class_probas[:, num_thresholds] = unconditional_probas[:, num_thresholds - 1]

    return class_probas.clamp(min=0.0)  # guard against tiny negative floating-point noise
