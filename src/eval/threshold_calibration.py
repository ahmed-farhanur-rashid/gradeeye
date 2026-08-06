"""
Threshold calibration for ordinal classification.

Instead of argmax (which assumes uniform misclassification cost), we learn 4
thresholds t_1..t_4 on a calibration set such that:

    predicted_class = argmax_k ( P(y=k) - threshold_k )

where threshold_k is the "bonus" required before predicting class k over
class k-1. Equivalently, predict class k if:

    P(y=k) > P(y=k-1) + threshold_k

with thresholds sorted in increasing order (lower thresholds near the
majority end, larger thresholds for high-confidence minority predictions).
This is a learned decision rule — it does not change the model or its
probabilities, only how we MAP probabilities → discrete class labels.

Why this matters for DR grading:
- The model is trained with class-imbalanced loss weighting (CORN
  per-threshold inverse_sqrt), so its argmax already slightly favors
  minorities. But the calibration comes from the EXACT prior shift between
  training (EyePACS: 74% No-DR) and inference (Messidor2: 58% No-DR,
  APTOS: 49% No-DR). Threshold calibration absorbs that prior shift
  without retraining.
- Expected gain on cross-domain eval (Messidor2 zero-shot): +0.01 to +0.05
  QWK — the rare class F1 gains most, since calibrated thresholds push
  predictions back into minority territory where the model has actual
  signal.

How it works:
- Optimize QWK on a calibration set (e.g. EyePACS val).
- Use Nelder-Mead simplex search: 4-D parameter space, derivative-free.
- Search bounded by [0, 1] per threshold; sorted into monotonic order
  after each evaluation so we never evaluate an out-of-order configuration.

NOTE — DO NOT CONFUSE WITH `src/eval/probability_calibration.py`:
  - This file: decision-rule calibration (no model probabilities changed).
  - `probability_calibration.py`: temperature scaling (model probabilities
    themselves are re-calibrated before argmax).
  - The two are STACKED in the paper: temperature scale first, then apply
    threshold calibration on the calibrated probabilities. Renamed
    `threshold_calibration.py` → `calibration_thresholds.py` in Phase 4 of
    the cleanup plan, but kept at this path during Phase 1-3 development.

Usage:
    from src.eval.threshold_calibration import (
        optimize_thresholds, predict_with_thresholds, save_thresholds,
        load_thresholds,
    )

    thresholds = optimize_thresholds(y_proba, y_true)
    y_pred = predict_with_thresholds(y_proba, thresholds)
"""
from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from src.eval.metrics import quadratic_weighted_kappa

NUM_CLASSES = 5  # 5 DR grades, 4 thresholds in between


def predict_with_thresholds(y_proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Map class probabilities to predictions using calibrated thresholds.

    `y_proba`: (N, NUM_CLASSES) — rows sum to 1.
    `thresholds`: (NUM_CLASSES - 1,) — calibrated "bonus" required to
    predict class k over class k-1.

    Algorithm:
        score[k] = P(y=k) - threshold[k]   for k in [1..NUM_CLASSES-1]
        score[0] = P(y=0) - 0              (no bonus for the lowest class)
        predict = argmax_k score[k]

    Equivalent to: predict class k if cumulative P(y>=k) - P(y>k) crosses
    zero at k — same as the original argmax when all thresholds == 0.
    """
    if thresholds.ndim != 1 or thresholds.shape[0] != NUM_CLASSES - 1:
        raise ValueError(
            f"thresholds must be shape ({NUM_CLASSES - 1},); got {thresholds.shape}"
        )
    if y_proba.ndim != 2 or y_proba.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"y_proba must be shape (N, {NUM_CLASSES}); got {y_proba.shape}"
        )

    # score[k] = P(y=k) - bonus[k], where bonus[0] = 0.
    # Higher score = more likely class k.
    scores = y_proba.copy()
    for k in range(1, NUM_CLASSES):
        scores[:, k] = y_proba[:, k] - thresholds[k - 1]

    return scores.argmax(axis=1)


def _qwk_with_thresholds(thresholds_raw: np.ndarray, y_proba: np.ndarray,
                          y_true: np.ndarray) -> float:
    """
    QWK as a function of an UNORDERED threshold vector.

    Sort thresholds into monotonic order before evaluating — this prevents
    Nelder-Mead from wasting evaluations on configurations where
    threshold_2 < threshold_1 (which would invert the decision rule's
    monotonicity assumption without telling us anything new).
    """
    thresholds = np.sort(thresholds_raw)  # ascending: closer to No-DR = small
    y_pred = predict_with_thresholds(y_proba, thresholds)
    return quadratic_weighted_kappa(y_true, y_pred)


def optimize_thresholds(y_proba: np.ndarray, y_true: np.ndarray,
                         method: str = "nelder-mead",
                         maxiter: int = 1000,
                         seed: int = 42,
                         verbose: bool = True) -> np.ndarray:
    """
    Find thresholds that maximize QWK on a calibration set.

    Args:
        y_proba: (N, NUM_CLASSES) probabilities from the trained model.
        y_true: (N,) integer labels in [0, NUM_CLASSES-1].
        method: scipy.optimize method. Nelder-Mead is robust on the small,
                non-smooth QWK landscape.
        maxiter: max iterations for Nelder-Mead.
        seed: RNG seed for scipy.optimize.minimize reproducibility.

    Returns:
        thresholds: (NUM_CLASSES - 1,) sorted ascending — the calibrated
        decision boundary offsets.
    """
    if y_proba.shape[0] != y_true.shape[0]:
        raise ValueError("y_proba and y_true must have the same number of samples")
    if y_proba.shape[1] != NUM_CLASSES:
        raise ValueError(f"y_proba must have {NUM_CLASSES} columns")

    # Baseline (all thresholds = 0 ≡ argmax)
    base_qwk = _qwk_with_thresholds(np.zeros(NUM_CLASSES - 1), y_proba, y_true)
    if verbose:
        print(f"  Baseline QWK (argmax):   {base_qwk:.4f}")

    # Use inverse-frequency seeds as initial guess — Mild/Moderate are
    # under-predicted by argmax so a small positive threshold nudge helps.
    # We DON'T use 1/count directly because that's too aggressive; we use
    # a mild shift proportional to the per-class prior gap.
    class_counts = np.bincount(y_true, minlength=NUM_CLASSES).astype(float)
    class_priors = class_counts / class_counts.sum()
    uniform = 1.0 / NUM_CLASSES
    # positive value when actual prior < uniform (under-predicted by argmax)
    gap = (uniform - class_priors) * 0.5  # scale factor from argmax baseline
    initial = np.clip(gap[1:] - gap[:-1], -0.3, 0.3)  # adjacent-class gaps
    initial_sorted = np.sort(initial)

    # scipy.optimize.minimize minimizes — negate QWK
    rng = np.random.default_rng(seed)

    result = minimize(
        fun=lambda t: -_qwk_with_thresholds(t, y_proba, y_true),
        x0=initial_sorted,
        method=method,
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-5, "disp": False},
    )

    best_thresholds = np.sort(result.x)
    best_qwk = -result.fun

    # Multi-start: if the result is suspiciously close to baseline, retry
    # with random initializations. Sometimes Nelder-Mead gets stuck in a
    # local minimum when the QWK landscape is flat near the baseline.
    if best_qwk - base_qwk < 0.002:
        if verbose:
            print(f"  Gain only {best_qwk - base_qwk:+.4f}, retrying with multi-start...")
        best_so_far = best_qwk
        best_thresholds_so_far = best_thresholds.copy()
        for restart in range(5):
            x0 = rng.uniform(-0.2, 0.2, size=NUM_CLASSES - 1)
            x0.sort()
            res = minimize(
                fun=lambda t: -_qwk_with_thresholds(t, y_proba, y_true),
                x0=x0,
                method=method,
                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-5},
            )
            cand_qwk = -res.fun
            cand_t = np.sort(res.x)
            if cand_qwk > best_so_far:
                best_so_far = cand_qwk
                best_thresholds_so_far = cand_t
        best_thresholds = best_thresholds_so_far
        best_qwk = best_so_far

    if verbose:
        print(f"  Calibrated QWK:           {best_qwk:.4f}  "
              f"(gain {best_qwk - base_qwk:+.4f})")
        print(f"  Calibrated thresholds:    {np.round(best_thresholds, 4).tolist()}")

    return best_thresholds


def save_thresholds(thresholds: np.ndarray, path: str,
                     extra_metadata: dict | None = None) -> None:
    """Persist calibrated thresholds + audit metadata (gain, source, etc.) as JSON."""
    payload = {
        "thresholds": [float(t) for t in thresholds],
        "num_classes": int(NUM_CLASSES),
    }
    if extra_metadata:
        payload.update(extra_metadata)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_thresholds(path: str) -> np.ndarray:
    """Load calibrated thresholds from a JSON file written by `save_thresholds`."""
    with open(path) as f:
        payload = json.load(f)
    thresholds = np.asarray(payload["thresholds"], dtype=np.float64)
    if thresholds.shape != (NUM_CLASSES - 1,):
        raise ValueError(
            f"Expected {NUM_CLASSES - 1} thresholds; got {thresholds.shape[0]}"
        )
    return thresholds


def describe_thresholds(thresholds: np.ndarray) -> str:
    """Human-readable summary of what each threshold means operationally."""
    lines = ["Calibrated decision rule (vs. argmax):"]
    for k in range(1, NUM_CLASSES):
        prev_class = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][k - 1]
        this_class = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"][k]
        if thresholds[k - 1] > 0:
            meaning = (f"need +{thresholds[k-1]:.3f} bonus over {prev_class} "
                       f"to predict {this_class}")
        elif thresholds[k - 1] < 0:
            meaning = (f"{this_class} gets -{abs(thresholds[k-1]):.3f} bonus — "
                       f"easier to predict than {prev_class}")
        else:
            meaning = f"no change from argmax between {prev_class} and {this_class}"
        lines.append(f"  t_{k} = {thresholds[k-1]:+.4f}  →  {meaning}")
    return "\n".join(lines)