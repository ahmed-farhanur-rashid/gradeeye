"""
Probability calibration for ordinal classification.

This module is the *confidence* calibration counterpart to
`src/eval/threshold_calibration.py`:

  - threshold_calibration.py learns per-class decision-rule bonuses to
    maximize QWK on a calibration set. It does NOT change the model's
    probability estimates — only the argmax-with-threshold step.

  - probability_calibration.py (this file) learns a temperature T on the
    logits so that the post-temperature-sigmoid / softmax probabilities
    better match observed frequencies. This is the standard
    "temperature scaling" technique (Guo et al., 2017) applied to CORN's
    4-conditional-probability output.

The two are complementary, NOT competing:
  - Apply temperature scaling first to get well-calibrated probabilities.
  - Then apply threshold calibration on those calibrated probabilities to
    maximize QWK under the cross-domain prior shift.

Implementation notes:
  - We support two flavors of temperature scaling:
      * per_thresh: 4 learned temperatures, one per CORN sub-problem.
      * global:     1 shared temperature across all 4 sub-problems.
    The per_thresh vs global ablation is a paper result.
  - Binning for ECE uses equal-mass (quantile) binning because per-fold
    test sets are tiny (Messidor-2 full = 227, matched = 227, APTOS test
    ~ 367). Equal-width bins on 227 samples leave most bins empty.
  - All math is pure numpy; no torch dependency (operates on already-
    collected logits + labels arrays, as called by Phase 3 step 3).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar


# Number of CORN sub-problems = num_classes - 1 = 5 - 1 = 4
NUM_THRESHOLDS = 4
NUM_CLASSES = 5


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def per_threshold_ece(corn_probs: np.ndarray, y_true: np.ndarray,
                        k: int, n_bins: int = 10) -> float:
    """
    Expected Calibration Error for the k-th CORN sub-problem.

    Args:
        corn_probs: (N, 4) array of conditional probabilities P(y > threshold_k | y > threshold_{k-1}).
                    This is the SIGMOID output of the logits, NOT the cumulative
                    class distribution. Callers pass `sigmoid(corn_logits)`.
        y_true:     (N,) integer labels in [0, 4].
        k:          which sub-problem to evaluate (0..3).
        n_bins:     number of equal-mass bins.

    Returns:
        ECE: float in [0, 1]. Lower = more calibrated.
    """
    if corn_probs.shape[1] != NUM_THRESHOLDS:
        raise ValueError(
            f"corn_probs must have shape (N, {NUM_THRESHOLDS}); got {corn_probs.shape}"
        )
    conf = corn_probs[:, k]
    labels_k = (y_true > k).astype(np.float64)
    return _equal_mass_ece(conf, labels_k, n_bins)


def _equal_mass_ece(conf: np.ndarray, outcomes: np.ndarray,
                     n_bins: int) -> float:
    """Equal-mass (quantile) binning ECE, robust on small N."""
    n = len(conf)
    if n == 0:
        return 0.0
    # Quantile edges: split conf into n_bins groups of equal size.
    # ties = 'drop' because every-conf-is-equal (e.g. sigmoid outputs all 0.5)
    # would otherwise produce percentile edges that all collapse to one value.
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(conf, quantiles, method="linear")
    # Ensure edges are strictly increasing (tie-collapse can introduce dupes)
    edges = np.unique(edges)
    if len(edges) < 2:
        return float(np.abs(conf.mean() - outcomes.mean()))

    # np.digitize: bin index is 1..len(edges)-1 for in-range, 0 for below-first
    bin_idx = np.digitize(conf, edges[1:-1], right=False)
    ece = 0.0
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_conf = conf[mask].mean()
        bin_acc = outcomes[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_diagram_data(conf: np.ndarray, outcomes: np.ndarray,
                              n_bins: int = 10) -> list[tuple[float, float, float]]:
    """Bin-level (bin_conf, bin_acc, bin_count) tuples for plotting.

    Returns list of (lower_bound, upper_bound, accuracy, count) per populated bin.
    Same equal-mass binning as `_equal_mass_ece`.
    """
    n = len(conf)
    if n == 0:
        return []
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(conf, quantiles, method="linear")
    edges = np.unique(edges)
    if len(edges) < 2:
        return []
    bin_idx = np.digitize(conf, edges[1:-1], right=False)
    bins = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        bins.append((float(edges[b]), float(edges[b + 1]),
                     float(outcomes[mask].mean()),
                     int(mask.sum())))
    return bins


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _log_loss(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Binary cross-entropy / negative log-likelihood."""
    p = np.clip(probs, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fit_temperature_per_threshold(logits: np.ndarray, y_true: np.ndarray,
                                    n_bins: int = 10) -> np.ndarray:
    """Fit one temperature per CORN sub-problem (k=0..3).

    Args:
        logits:  (N, 4) raw CORN logits (BEFORE sigmoid).
        y_true:  (N,) integer labels in [0, 4].

    Returns:
        temperatures: (4,) array, one T per sub-problem. T > 0.
    """
    if logits.shape[1] != NUM_THRESHOLDS:
        raise ValueError(
            f"logits must have shape (N, {NUM_THRESHOLDS}); got {logits.shape}"
        )
    temperatures = np.ones(NUM_THRESHOLDS, dtype=np.float64)
    for k in range(NUM_THRESHOLDS):
        labels_k = (y_true > k).astype(np.float64)

        def neg_log_lik(T: float) -> float:
            if T <= 0:
                return 1e9
            probs = _sigmoid(logits[:, k] / T)
            return _log_loss(probs, labels_k)

        # Bracketed scalar search: T=1 (baseline) is always a defensible
        # boundary; allow [0.05, 5.0] to cover aggressive sharpening and
        # softening. Most calibrated models end up in [0.5, 2.0].
        result = minimize_scalar(neg_log_lik, bounds=(0.05, 5.0), method="bounded",
                                   options={"xatol": 1e-4})
        temperatures[k] = float(result.x)
    return temperatures


def fit_temperature_global(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Fit one shared temperature across all 4 CORN sub-problems.

    Returns:
        T: float, T > 0.
    """
    if logits.shape[1] != NUM_THRESHOLDS:
        raise ValueError(
            f"logits must have shape (N, {NUM_THRESHOLDS}); got {logits.shape}"
        )

    def neg_log_lik(T: float) -> float:
        if T <= 0:
            return 1e9
        # Concat all 4 sub-problems into one joint binary task.
        all_probs = _sigmoid(logits.flatten() / T)
        all_labels = np.concatenate([
            (y_true > k).astype(np.float64) for k in range(NUM_THRESHOLDS)
        ])
        return _log_loss(all_probs, all_labels)

    result = minimize_scalar(neg_log_lik, bounds=(0.05, 5.0), method="bounded",
                               options={"xatol": 1e-4})
    return float(result.x)


def apply_temperature(logits: np.ndarray, temperatures: np.ndarray | float) -> np.ndarray:
    """Divide logits by temperature(s) and apply sigmoid.

    Args:
        logits:        (N, 4) raw CORN logits.
        temperatures:  (4,) array OR scalar float.
    Returns:
        calibrated probs: (N, 4) post-sigmoid.
    """
    if isinstance(temperatures, (int, float)):
        return _sigmoid(logits / float(temperatures))
    if temperatures.shape != (NUM_THRESHOLDS,):
        raise ValueError(
            f"temperatures must be scalar or shape ({NUM_THRESHOLDS},); got {temperatures.shape}"
        )
    return _sigmoid(logits / temperatures[None, :])
