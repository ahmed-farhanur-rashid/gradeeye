#!/usr/bin/env python3
"""Generate decision-boundary sweep plots from saved threshold-tuning records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
THR_PATH = REPO_ROOT / "saved/logs/threshold_tuning.json"
OUT_DIR = REPO_ROOT / "docs/analysis/figures/decision_boundary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = ["eyepacs", "aptos", "messidor2", "ddr"]
VARIANTS = ["3ch_balanced", "3ch_unbalanced", "4ch_soft", "4ch_tversky", "4ch_morph"]
FOLD_NAMES = {"eyepacs": "EyePACS", "aptos": "APTOS", "messidor2": "Messidor-2", "ddr": "DDR"}
VARIANT_NAMES = {
    "3ch_balanced":   "3ch balanced",
    "3ch_unbalanced": "3ch unbalanced",
    "4ch_soft":       "4ch soft",
    "4ch_tversky":    "4ch Tversky",
    "4ch_morph":      "4ch morph",
}


def plot_one_variant(rows_for_variant: list, variant: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows_for_variant:
        trace = r["bias_trace"]
        bs = [t["bias"] for t in trace]
        qs = [t["qwk"] for t in trace]
        ax.plot(bs, qs, label=f"{FOLD_NAMES[r['fold']]} (baseline {r['baseline_qwk']:.3f})", lw=1.5)
        ax.scatter([r["best_bias"]], [r["best_qwk"]], s=70, marker="*", zorder=5)
    ax.axvline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Decision-boundary bias b (additive to all 4 sigmoid thresholds)")
    ax.set_ylabel("QWK")
    ax.set_title(f"Decision-boundary sweep — {VARIANT_NAMES[variant]} (star = OOB-CV best)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"decision_boundary_{variant}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_summary_grid(all_rows: list):
    """Grid of 5 variants × 4 folds of bias-sweep curves."""
    fig, axes = plt.subplots(len(VARIANTS), len(FOLDS), figsize=(16, 14), sharex=True)
    for i, v in enumerate(VARIANTS):
        for j, f in enumerate(FOLDS):
            ax = axes[i, j]
            r = next((x for x in all_rows if x["variant"] == v and x["fold"] == f), None)
            if r is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            trace = r["bias_trace"]
            bs = [t["bias"] for t in trace]
            qs = [t["qwk"] for t in trace]
            ax.plot(bs, qs, color="#3b82f6", lw=1.0)
            ax.scatter([r["best_bias"]], [r["best_qwk"]], s=30, marker="*", color="red", zorder=5)
            ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.4)
            ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
            ax.grid(alpha=0.2)
            if i == 0:
                ax.set_title(FOLD_NAMES[f], fontsize=10)
            if j == 0:
                ax.set_ylabel(VARIANT_NAMES[v], fontsize=10)
            if i == len(VARIANTS) - 1:
                ax.set_xlabel("bias b")
    fig.suptitle("Decision-boundary sweep (all variants × all folds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "decision_boundary_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    rows = json.loads(THR_PATH.read_text())
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        if rs:
            plot_one_variant(rs, v)
    plot_summary_grid(rows)


if __name__ == "__main__":
    main()