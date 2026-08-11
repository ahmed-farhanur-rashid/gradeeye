#!/usr/bin/env python3
"""Generate all analysis figures from the eval_results.jsonl log.

Outputs are PNGs at docs/analysis/figures/, suitable for embedding in
docs/analysis/analysis.tex.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "saved/logs/eval_results.jsonl"
OUT = REPO_ROOT / "docs/analysis/figures"
OUT.mkdir(parents=True, exist_ok=True)

# Source identification by test-set size
SOURCE_BY_N = {88702: "EyePACS", 3662: "APTOS", 1744: "Messidor-2", 12522: "DDR"}


def load_results():
    """Load eval_results.jsonl, dedupe by (variant, fold), keep highest QWK."""
    raw = []
    with open(LOG) as f:
        for line in f:
            d = json.loads(line)
            raw.append(d)
    seen = {}
    for d in raw:
        cp = d["checkpoint"]
        n = d["n_samples"]
        fold = SOURCE_BY_N.get(n, f"N={n}")
        parts = cp.split("/")
        is_unbalanced = "baseline_3ch_unbalanced" in cp
        variant = parts[-2]
        if variant in (
            "four_ch_soft", "four_ch_tversky", "four_ch_morph",
            "four_ch_sobel", "five_ch_soft_sobel", "five_ch_tversky_sobel",
        ):
            canonical = variant
            arch = "convnext_tiny"
        elif is_unbalanced:
            canonical = f"baseline_3ch_unbalanced_{variant}"
            arch = variant
        else:
            canonical = f"baseline_3ch_{variant}"
            arch = variant
        key = (canonical, fold)
        prev = seen.get(key)
        if prev is None or d["metrics"]["qwk"] > prev["metrics"]["qwk"]:
            seen[key] = d
    rows = []
    for (canonical, fold), d in seen.items():
        rows.append({
            "canonical": canonical,
            "fold": fold,
            "arch": arch,
            "n": d["n_samples"],
            "qwk": d["metrics"]["qwk"],
            "acc": d["metrics"]["accuracy"],
            "f1": d["metrics"]["macro_f1"],
            "auc": d["metrics"].get("macro_auc_roc", None),
        })
    return rows


def fig_baseline_per_fold(rows):
    """Figure: QWK by architecture × fold for the 3-channel balanced baseline."""
    folds = ["EyePACS", "APTOS", "Messidor-2", "DDR"]
    archs = ["baseline_3ch_convnext_tiny", "baseline_3ch_deit3_small_patch16_384", "baseline_3ch_maxvit_tiny_tf_384"]
    arch_labels = ["ConvNeXt-Tiny", "DeiT-3 Small/384", "MaxViT-Tiny/384"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(folds))
    width = 0.27

    for i, (arch, label) in enumerate(zip(archs, arch_labels)):
        qwks = []
        for f in folds:
            row = next((r for r in rows if r["canonical"] == arch and r["fold"] == f), None)
            qwks.append(row["qwk"] if row else 0.0)
        bars = ax.bar(x + (i - 1) * width, qwks, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("QWK (held-out test)")
    ax.set_title("3-Channel Baseline: QWK by Architecture × LODO Fold")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5, label="QWK=0.5 (moderate)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_baseline_qwk_per_arch.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_baseline_per_metric(rows):
    """Figure: per-fold QWK + Acc + F1 + AUC for ConvNeXt-Tiny baseline (the primary arch)."""
    folds = ["EyePACS", "APTOS", "Messidor-2", "DDR"]
    arch = "baseline_3ch_convnext_tiny"
    metrics = {"QWK": [], "Accuracy": [], "F1 (macro)": [], "AUC (macro)": []}
    for f in folds:
        row = next((r for r in rows if r["canonical"] == arch and r["fold"] == f), None)
        if row:
            metrics["QWK"].append(row["qwk"])
            metrics["Accuracy"].append(row["acc"])
            metrics["F1 (macro)"].append(row["f1"])
            metrics["AUC (macro)"].append(row["auc"] if row["auc"] is not None else 0)
        else:
            for k in metrics:
                metrics[k].append(0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(folds))
    width = 0.18
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for i, (label, vals) in enumerate(metrics.items()):
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Metric value")
    ax.set_title("ConvNeXt-Tiny (3-Channel Baseline): Per-Metric Breakdown by LODO Fold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_convnext_metrics_per_fold.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_seg_ablation(rows):
    """Figure: Seg ablation. Per-fold QWK of 3-channel vs 4-channel variants."""
    folds = ["EyePACS", "APTOS", "Messidor-2", "DDR"]
    variants = ["baseline_3ch_convnext_tiny", "four_ch_soft", "four_ch_tversky", "four_ch_morph"]
    labels = ["3-ch baseline", "4-ch soft", "4-ch Tversky", "4-ch morph"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(folds))
    width = 0.2
    for i, (v, lbl) in enumerate(zip(variants, labels)):
        qwks = []
        for f in folds:
            row = next((r for r in rows if r["canonical"] == v and r["fold"] == f), None)
            qwks.append(row["qwk"] if row else np.nan)
        ax.bar(x + (i - 1.5) * width, qwks, width, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("QWK (held-out test)")
    ax.set_title("Segmentation Auxiliary Ablation: 3-ch vs 4-ch (ConvNeXt-Tiny)")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_seg_ablation_qwk.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_delta_seg_ablation(rows):
    """Figure: Delta QWK vs baseline (4-ch minus 3-ch)."""
    folds = ["EyePACS", "APTOS", "Messidor-2", "DDR"]
    variants = ["four_ch_soft", "four_ch_tversky", "four_ch_morph"]
    labels = ["4-ch soft", "4-ch Tversky", "4-ch morph"]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(folds))
    width = 0.25

    base_qwk = {}
    for f in folds:
        row = next((r for r in rows if r["canonical"] == "baseline_3ch_convnext_tiny" and r["fold"] == f), None)
        base_qwk[f] = row["qwk"] if row else None

    for i, (v, lbl) in enumerate(zip(variants, labels)):
        deltas = []
        for f in folds:
            row = next((r for r in rows if r["canonical"] == v and r["fold"] == f), None)
            if row and base_qwk[f] is not None:
                deltas.append(row["qwk"] - base_qwk[f])
            else:
                deltas.append(np.nan)
        ax.bar(x + (i - 1) * width, deltas, width, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel(r"$\Delta$QWK vs 3-channel baseline")
    ax.set_title("Seg Ablation: Improvement Threshold Check (≥2%)")
    ax.axhline(0.02, color="green", linestyle="--", alpha=0.5, label="+2% threshold")
    ax.axhline(-0.02, color="red", linestyle="--", alpha=0.5, label="−2% threshold")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_seg_ablation_delta.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_qwk_distribution(rows):
    """Figure: Distribution of per-fold QWK across all variants (boxplot)."""
    folds = ["EyePACS", "APTOS", "Messidor-2", "DDR"]
    data_per_fold = []
    for f in folds:
        vals = [r["qwk"] for r in rows if r["fold"] == f]
        data_per_fold.append(vals)

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data_per_fold, tick_labels=folds, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#1f77b4")
        patch.set_alpha(0.5)
    ax.set_ylabel("QWK (held-out test)")
    ax.set_title("Per-Fold QWK Distribution Across All Variants")
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT / "fig_qwk_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    rows = load_results()
    print(f"Loaded {len(rows)} unique (variant, fold) results.")
    fig_baseline_per_fold(rows)
    fig_baseline_per_metric(rows)
    fig_seg_ablation(rows)
    fig_delta_seg_ablation(rows)
    fig_qwk_distribution(rows)
    print("\nAll figures written to", OUT)


if __name__ == "__main__":
    main()
