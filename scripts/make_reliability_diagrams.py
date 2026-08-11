#!/usr/bin/env python3
"""Generate reliability diagrams from the calibration JSON outputs.

Reads saved/logs/calibration/lodo_<fold>_convnext_tiny_probability_calibration.json
and produces one reliability-diagram PNG per (fold, threshold) on the held-out
test set, plus one 4-threshold montage per fold.

Each bin shows the predicted-probability midpoint (x) vs the empirical
frequency of the binary "grade >= t" outcome (y). A perfectly calibrated
model would lie on the diagonal y = x.

Outputs to docs/analysis/figures/calibration/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIB_DIR = REPO_ROOT / "saved/logs/calibration"
OUT_DIR = REPO_ROOT / "docs/analysis/figures/calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = ["eyepacs", "aptos", "messidor2", "ddr"]
FOLD_NAMES = {"eyepacs": "EyePACS", "aptos": "APTOS",
              "messidor2": "Messidor-2", "ddr": "DDR"}
THRESHOLD_LABELS = ["t=1 (Mild or worse)", "t=2 (Moderate or worse)",
                    "t=3 (Severe or worse)", "t=4 (Proliferative)"]


def load_fold(fold: str) -> dict:
    p = CALIB_DIR / f"lodo_{fold}_convnext_tiny_probability_calibration.json"
    return json.loads(p.read_text())["result"]


def plot_single_threshold(ax, bin_data, ece, title):
    """Plot one reliability diagram on existing axes."""
    bins = bin_data["reliability_bins"]
    midpoints = np.array([(b["lower"] + b["upper"]) / 2 for b in bins])
    counts = np.array([b["count"] for b in bins])
    accs = np.array([b["accuracy"] for b in bins])

    # Bar widths proportional to count (resize so they tile)
    lower = np.array([b["lower"] for b in bins])
    upper = np.array([b["upper"] for b in bins])
    centers = (lower + upper) / 2
    widths = (upper - lower) * 0.95

    # Confidence band via normal approximation (count weighted)
    se = np.sqrt(accs * (1 - accs) / np.maximum(counts, 1))
    ax.bar(centers, accs, width=widths, color="#1f77b4", alpha=0.7,
           edgecolor="white", linewidth=0.5, yerr=se, error_kw={"linewidth": 0.5})
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(grade ≥ t)")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def plot_single_fold(fold_data, fold_name):
    """4-threshold montage for one fold."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    raw_per = fold_data["raw_ece"]["full"]["per_threshold"]
    for i, (ax, label, thresh_data) in enumerate(zip(axes, THRESHOLD_LABELS, raw_per)):
        plot_single_threshold(ax, thresh_data, thresh_data["ece"],
                              f"{fold_name} — {label}")
    fig.suptitle(f"{fold_name}: Reliability diagrams (no calibration, held-out test)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / f"reliability_{fold_name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def plot_single_threshold_only(fold_data, fold_name, threshold_idx):
    """Single-panel reliability diagram for one (fold, threshold)."""
    raw_per = fold_data["raw_ece"]["full"]["per_threshold"]
    thresh_data = raw_per[threshold_idx]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    plot_single_threshold(ax, thresh_data, thresh_data["ece"],
                          f"{fold_name} — {THRESHOLD_LABELS[threshold_idx]}")
    fig.tight_layout()
    out = OUT_DIR / f"reliability_{fold_name}_t{threshold_idx+1}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    for fold in FOLDS:
        d = load_fold(fold)
        name = FOLD_NAMES[fold]
        plot_single_fold(d, name)
        for i in range(4):
            plot_single_threshold_only(d, name, i)
    print(f"\nAll reliability diagrams written to {OUT_DIR}")


if __name__ == "__main__":
    main()