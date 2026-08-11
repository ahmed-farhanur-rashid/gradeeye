#!/usr/bin/env python3
"""Summarize all evaluation results into a single CSV at docs/analysis/results_summary.csv.

Deduplicates eval_results.jsonl by (variant, fold), keeping the highest-QWK entry
and producing one row per (variant, fold) with all metrics.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "saved/logs/eval_results.jsonl"
OUT = REPO_ROOT / "docs/analysis/results_summary.csv"

SOURCE_BY_N = {88702: "EyePACS", 3662: "APTOS", 1744: "Messidor-2", 12522: "DDR"}


def main():
    raw = []
    with open(LOG) as f:
        for line in f:
            raw.append(json.loads(line))

    # Dedupe by (variant, fold); keep highest QWK
    seen = {}
    for d in raw:
        cp = d["checkpoint"]
        n = d["n_samples"]
        fold = SOURCE_BY_N.get(n, f"N={n}")
        parts = cp.split("/")
        # Detect unbalanced baseline via parent dir name.
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
        if prev is None or d["metrics"]["qwk"] > prev[4]["metrics"]["qwk"]:
            seen[key] = (canonical, arch, fold, n, d)

    rows = []
    for key, (canonical, arch, fold, n, d) in seen.items():
        m = d["metrics"]
        rows.append({
            "variant": canonical,
            "arch": arch,
            "fold": fold,
            "n_test": n,
            "qwk": round(m["qwk"], 4),
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "macro_auc_roc": round(m.get("macro_auc_roc", 0), 4),
            "checkpoint": d["checkpoint"],
            "timestamp": d["timestamp"],
        })

    rows.sort(key=lambda r: (r["variant"], r["fold"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "arch", "fold", "n_test", "qwk", "accuracy",
                        "macro_f1", "macro_auc_roc", "checkpoint", "timestamp"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")
    for r in rows:
        print(f"  {r['variant']:30s} × {r['fold']:12s}  QWK={r['qwk']:.4f}  Acc={r['accuracy']:.4f}  F1={r['macro_f1']:.4f}  AUC={r['macro_auc_roc']:.4f}")


if __name__ == "__main__":
    main()
