"""
Evaluation metrics per plan Section 7.

Quadratic Weighted Kappa (QWK) is the PRIMARY metric — this is the
standard metric for the Kaggle DR competitions and the one comparable to
published benchmarks. Plain accuracy is reported as a secondary metric
only, since accuracy doesn't penalize a Severe-graded-as-No-DR error the
same as a Severe-graded-as-Moderate error (which matters clinically).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Primary metric. Comparable directly to published Kaggle DR leaderboard scores."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray | None = None) -> dict:
    """
    y_true, y_pred: (N,) int arrays, classes 0-4.
    y_proba: optional (N, 5) per-class probability array, needed for
             macro-averaged one-vs-rest AUC-ROC.
    """
    metrics = {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)

    for i, name in enumerate(CLASS_NAMES):
        metrics[f"f1_{name.replace(' ', '_')}"] = per_class_f1[i]
        metrics[f"recall_{name.replace(' ', '_')}"] = per_class_recall[i]
        metrics[f"precision_{name.replace(' ', '_')}"] = per_class_precision[i]

    if y_proba is not None:
        try:
            metrics["macro_auc_roc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=list(range(5))
            )
        except ValueError:
            # Can happen if a class is entirely absent from y_true in a small eval split.
            metrics["macro_auc_roc"] = float("nan")

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(5)))


def format_confusion_matrix_str(cm: np.ndarray) -> str:
    """Human-readable confusion matrix with class name headers."""
    header = "        " + "".join(f"{n[:6]:>8}" for n in CLASS_NAMES)
    lines = [header]
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>8}" for v in row)
        lines.append(f"{CLASS_NAMES[i][:6]:>8}{row_str}")
    return "\n".join(lines)
