"""src/eval: evaluation, calibration, and metrics library + CLI entry points."""
from src.eval.metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    format_confusion_matrix_str,
    quadratic_weighted_kappa,
)
from src.eval.threshold_calibration import NUM_CLASSES
from src.eval.threshold_calibration import (
    describe_thresholds,
    load_thresholds,
    optimize_thresholds,
    predict_with_thresholds,
    save_thresholds,
)
from src.eval.probability_calibration import (
    fit_temperature_global,
    fit_temperature_per_threshold,
    apply_temperature,
    per_threshold_ece,
    reliability_diagram_data,
)

__all__ = [
    "NUM_CLASSES",
    "compute_all_metrics",
    "compute_confusion_matrix",
    "format_confusion_matrix_str",
    "quadratic_weighted_kappa",
    # Decision-rule threshold calibration
    "optimize_thresholds",
    "predict_with_thresholds",
    "save_thresholds",
    "load_thresholds",
    "describe_thresholds",
    # Probability calibration (temperature scaling)
    "fit_temperature_per_threshold",
    "fit_temperature_global",
    "apply_temperature",
    "per_threshold_ece",
    "reliability_diagram_data",
]