"""
src/ensemble: ensemble evaluation of multiple DR grading models.

This package holds all ensemble-related code (moved from the legacy
`scripts/ensemble_evaluate.py`). The single-model variant lives in
`src/eval/evaluate.py` (calibrated thresholds, single checkpoint);
this package handles the multi-model case.

Library entry point: `src.ensemble.ensemble.evaluate_ensemble(...)`
CLI entry point:    `python -m src.ensemble.evaluate ...`

Key design constraint: averaging happens at the CLASS PROBABILITY level
(after each model's own CORN decode), NOT at the conditional-threshold
level. Averaging conditional probabilities P(y>k | y>k-1) across models
and then decoding breaks CORN's rank-consistency guarantee.
"""
from src.ensemble.ensemble import (
    NUM_CLASSES,
    default_seg_dir_for_manifest,
    ensemble_predict_probas,
    evaluate_ensemble,
)

__all__ = [
    "NUM_CLASSES",
    "default_seg_dir_for_manifest",
    "ensemble_predict_probas",
    "evaluate_ensemble",
]