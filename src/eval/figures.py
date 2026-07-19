"""
Publication-ready figures for a conference paper, saved as both PNG
(for drafts/slides) and PDF (vector, for camera-ready LaTeX submission).

Covers:
  - Training curves (loss + QWK vs. epoch, spanning all 3 phases)
  - Confusion matrix heatmap
  - Per-class ROC curves (one-vs-rest)
  - Class distribution bar chart (per source, for the dataset section)
  - Multi-run comparison bar chart (baseline vs ablation vs full_method)

All functions save to the given output directory and return the saved
file paths. Uses a consistent, minimal style suitable for print (no
default matplotlib gridlines/colors).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Consistent style: serif fonts read better in most conference LaTeX templates.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save_both(fig, out_dir: str, name: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_training_curves(epoch_log_csv: str, out_dir: str, run_name: str) -> dict:
    """
    Reads the per-epoch CSV log written by scripts/train.py
    (saved/logs/{run_name}_epoch_log.csv) and plots loss + QWK curves
    spanning all 3 training phases, with vertical lines marking phase
    boundaries.
    """
    df = pd.read_csv(epoch_log_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Loss curve
    axes[0].plot(df["global_epoch_idx"], df["train_loss"], label="Train", linewidth=1.5)
    axes[0].plot(df["global_epoch_idx"], df["val_loss"], label="Val", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend(frameon=False)

    # QWK curve (primary metric)
    axes[1].plot(df["global_epoch_idx"], df["val_qwk"], color="darkgreen", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Quadratic Weighted Kappa")
    axes[1].set_title("Validation QWK")
    axes[1].set_ylim(0, 1)

    # Mark phase boundaries with vertical dashed lines.
    phase_changes = df[df["phase"] != df["phase"].shift(1)]
    for _, row in phase_changes.iloc[1:].iterrows():
        for ax in axes:
            ax.axvline(row["global_epoch_idx"], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.suptitle(f"Training curves — {run_name}")
    fig.tight_layout()
    return _save_both(fig, out_dir, f"{run_name}_training_curves")


def plot_confusion_matrix(confusion_matrix: np.ndarray, out_dir: str, run_name: str,
                           normalize: bool = True) -> dict:
    """
    confusion_matrix: (5, 5) array from src/eval/metrics.py:compute_confusion_matrix.
    normalize: if True, show row-normalized (recall) percentages instead of raw counts.
    """
    import seaborn as sns

    cm = confusion_matrix.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={"label": "Proportion" if normalize else "Count"}, ax=ax)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(f"Confusion matrix — {run_name}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _save_both(fig, out_dir, f"{run_name}_confusion_matrix")


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, out_dir: str, run_name: str) -> dict:
    """
    One-vs-rest ROC curve per class plus macro-average.

    y_true: (N,) int labels 0-4.
    y_proba: (N, 5) per-class probabilities (e.g. from corn_predict_probas).
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))

    fpr_grid = np.linspace(0, 1, 200)
    tprs_interp = []

    for c in range(5):
        y_true_binary = (y_true == c).astype(int)
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            continue  # class absent or all-present in this split, ROC undefined
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.3, label=f"{CLASS_NAMES[c]} (AUC={roc_auc:.3f})")
        tprs_interp.append(np.interp(fpr_grid, fpr, tpr))

    if tprs_interp:
        macro_tpr = np.mean(tprs_interp, axis=0)
        macro_auc = auc(fpr_grid, macro_tpr)
        ax.plot(fpr_grid, macro_tpr, color="black", linestyle="--", linewidth=1.8,
                label=f"Macro-average (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves (one-vs-rest) — {run_name}")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return _save_both(fig, out_dir, f"{run_name}_roc_curves")


def plot_class_distribution(manifest_csvs: dict, out_dir: str) -> dict:
    """
    manifest_csvs: dict of {source_name: manifest_csv_path}, e.g.
        {"EyePACS": "data/processed/eyepacs_manifest.csv",
         "APTOS": "data/processed/aptos_manifest.csv",
         "Messidor-2": "data/processed/messidor2_manifest.csv"}

    Grouped bar chart of class distribution per source — standard figure
    for the dataset section of a DR grading paper.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sources = list(manifest_csvs.keys())
    x = np.arange(5)
    width = 0.8 / len(sources)

    for i, source in enumerate(sources):
        df = pd.read_csv(manifest_csvs[source])
        counts = df["label"].value_counts().reindex(range(5), fill_value=0)
        proportions = counts / counts.sum()
        ax.bar(x + i * width, proportions, width=width, label=source)

    ax.set_xticks(x + width * (len(sources) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Proportion of dataset")
    ax.set_title("Class distribution by source")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_both(fig, out_dir, "class_distribution")


def plot_run_comparison(results: dict, out_dir: str, metric: str = "qwk") -> dict:
    """
    results: dict of {run_name: metrics_dict}, where metrics_dict is the
    output of src/eval/metrics.py:compute_all_metrics for that run's test
    set evaluation. E.g.:
        {"baseline": {...}, "ablation_ce_weighted_cbam": {...}, "full_method": {...}}

    Bar chart comparing the run matrix on a single metric (default QWK,
    the primary metric) — the standard "ablation table as a figure" for
    a results section.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    run_names = list(results.keys())
    values = [results[r][metric] for r in run_names]

    bars = ax.bar(run_names, values, color=["#888888", "#4C72B0", "#55A868"][:len(run_names)])
    ax.set_ylabel(metric.upper() if metric == "qwk" else metric.replace("_", " ").title())
    ax.set_title(f"Run comparison — {metric.upper() if metric == 'qwk' else metric}")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return _save_both(fig, out_dir, f"run_comparison_{metric}")
