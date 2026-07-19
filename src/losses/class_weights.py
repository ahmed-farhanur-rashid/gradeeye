"""
Class imbalance handling: inverse-frequency weighting applied PER CORN
binary sub-problem, not as raw 5-class weights (plan Section 5).

Each of the 4 rank-threshold splits (y>0, y>1, y>2, y>3) has its own
imbalance profile, since the "eligible" subset shrinks as k increases
(fewer samples have label >= k for larger k). Uses inverse SQUARE-ROOT
frequency or clipped inverse frequency, not pure 1/count — pure inverse
frequency risks destabilizing training given how small the
Severe/Proliferative classes are (2.02-2.49% of EyePACS per plan Section 5).
"""
import numpy as np
import torch


def compute_class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        counts[c] = (labels == c).sum()
    return counts


def inverse_sqrt_freq_weights(class_counts: np.ndarray) -> np.ndarray:
    """w_c = 1 / sqrt(count_c), normalized to mean 1."""
    counts = np.clip(class_counts, a_min=1, a_max=None)  # guard against zero-count classes
    weights = 1.0 / np.sqrt(counts)
    return weights / weights.mean()


def clipped_inverse_freq_weights(class_counts: np.ndarray, clip_min: float = 0.5,
                                  clip_max: float = 5.0) -> np.ndarray:
    """w_c = 1 / count_c, normalized to mean 1, then clipped to [clip_min, clip_max]."""
    counts = np.clip(class_counts, a_min=1, a_max=None)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return np.clip(weights, clip_min, clip_max)


def compute_corn_per_threshold_weights(labels: np.ndarray, num_classes: int,
                                        method: str = "inverse_sqrt") -> list[np.ndarray]:
    """
    For each CORN threshold k in [0, num_classes-2], compute a 2-element
    weight array [w_negative, w_positive] over the ELIGIBLE subset
    (label >= k) for the binary task "label > k".

    This is the "per CORN binary sub-problem" weighting the plan calls for
    — each threshold's binary split gets its own imbalance-aware weights,
    not a single 5-class weight vector reused across all 4 sub-problems.
    """
    if method not in ("inverse_sqrt", "clipped_inverse"):
        raise ValueError(f"Unknown method: {method!r}")

    num_thresholds = num_classes - 1
    per_threshold_weights = []

    for k in range(num_thresholds):
        eligible = labels[labels >= k]
        if len(eligible) == 0:
            per_threshold_weights.append(np.array([1.0, 1.0]))
            continue

        n_negative = (eligible <= k).sum()  # label == k, i.e. NOT > k
        n_positive = (eligible > k).sum()
        counts = np.array([n_negative, n_positive])

        if method == "inverse_sqrt":
            w = inverse_sqrt_freq_weights(counts)
        else:
            w = clipped_inverse_freq_weights(counts)

        per_threshold_weights.append(w)

    return per_threshold_weights


def build_sample_weights_for_threshold(labels: torch.Tensor, k: int,
                                        threshold_weights: np.ndarray) -> torch.Tensor:
    """
    Given the eligible-subset labels for threshold k (labels >= k already
    filtered by the caller in corn_loss.py) and the [w_neg, w_pos] pair for
    that threshold, build a per-sample weight tensor.
    """
    is_positive = (labels > k).long()
    weights_lookup = torch.tensor(threshold_weights, dtype=torch.float32, device=labels.device)
    return weights_lookup[is_positive]
