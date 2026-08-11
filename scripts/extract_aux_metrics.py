#!/usr/bin/env python3
"""Extract per-variant secondary metrics and per-class F1 for the aux-channel ablation.

Reads saved/logs/eval_results.jsonl and emits:

  --secondary  : per-variant secondary metrics (Acc, F1, AUC) per fold for
                 baseline + 4-ch soft + 4-ch Tversky + 4-ch morph.
  --f1-per-class: per-class F1 across all aux variants × 4 folds.
  --write      : patch docs/results.md §4.2 and §4.3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = REPO_ROOT / "saved/logs/eval_results.jsonl"
RESULTS_MD = REPO_ROOT / "docs/results.md"

FOLDS = ["EyePACS", "APTOS", "Messidor-2", "DDR"]
FOLD_BY_N = {88702: "EyePACS", 3662: "APTOS", 1744: "Messidor-2", 12522: "DDR"}
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
GRADE_KEYS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
VARIANT_ORDER = [
    ("baseline_3ch_balanced_convnext_tiny", "3ch baseline"),
    ("four_ch_soft", "4ch soft mask"),
    ("four_ch_tversky", "4ch Tversky"),
    ("four_ch_morph", "4ch morph"),
]


def load_deduped():
    records = []
    with open(LOG) as f:
        for line in f:
            records.append(json.loads(line))
    seen = {}
    for d in records:
        cp = d["checkpoint"]
        parts = cp.split("/")
        is_unbalanced = "baseline_3ch_unbalanced" in cp
        variant = parts[-2]
        n = d["n_samples"]
        fold = FOLD_BY_N.get(n, f"N={n}")
        if variant in (
            "four_ch_soft", "four_ch_tversky", "four_ch_morph",
            "four_ch_sobel", "five_ch_soft_sobel", "five_ch_tversky_sobel",
        ):
            canonical = variant
        elif is_unbalanced:
            canonical = f"baseline_3ch_unbalanced_{variant}"
        else:
            canonical = f"baseline_3ch_balanced_{variant}"
        key = (canonical, fold)
        prev = seen.get(key)
        if prev is None or d["metrics"]["qwk"] > prev["metrics"]["qwk"]:
            seen[key] = d
    return seen


def secondary_metrics_table(seen):
    """Per-variant per-fold QWK/Acc/F1/AUC, plus mean across folds."""
    lines = []
    # Header
    lines.append("| Variant | EyePACS | APTOS | Messidor-2 | DDR | Mean ± SD |")
    lines.append("|:--------|--------:|------:|-----------:|----:|----------:|")
    # Three metric groups; one row per (variant, metric)
    for metric_key, metric_label in [
        ("qwk", "QWK"),
        ("accuracy", "Accuracy"),
        ("macro_f1", "F1 (macro)"),
        ("macro_auc_roc", "AUC (macro)"),
    ]:
        lines.append(f"| **{metric_label}** |  |  |  |  |  |")
        for canonical, label in VARIANT_ORDER:
            vals = []
            for fold in FOLDS:
                d = seen.get((canonical, fold))
                vals.append(d["metrics"][metric_key] if d else float("nan"))
            valid = [v for v in vals if v == v]  # not NaN
            if valid:
                mean = sum(valid) / len(valid)
                if len(valid) > 1:
                    sd = (sum((v - mean) ** 2 for v in valid) / (len(valid) - 1)) ** 0.5
                    mean_sd = f"{mean:.3f} ± {sd:.3f}"
                else:
                    mean_sd = f"{mean:.3f}"
            else:
                mean_sd = "—"
            cells = [f"{v:.3f}" if v == v else "—" for v in vals]
            lines.append(f"| &nbsp;&nbsp;&nbsp;{label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {mean_sd} |")
    return "\n".join(lines)


def f1_per_class_table(seen):
    """Per-variant per-fold per-class F1 (mean across folds at bottom)."""
    lines = []
    lines.append("| Variant | Class | EyePACS | APTOS | Messidor-2 | DDR | Mean |")
    lines.append("|:--------|:------|--------:|------:|-----------:|----:|-----:|")
    for canonical, label in VARIANT_ORDER:
        for grade_name, grade_key in zip(GRADE_NAMES, GRADE_KEYS):
            key = f"f1_{grade_key}"
            vals = []
            for fold in FOLDS:
                d = seen.get((canonical, fold))
                vals.append(d["metrics"][key] if d else float("nan"))
            valid = [v for v in vals if v == v]
            mean = sum(valid) / len(valid) if valid else float("nan")
            cells = [f"{v:.3f}" if v == v else "—" for v in vals]
            mean_cell = f"{mean:.3f}" if mean == mean else "—"
            lines.append(f"| {label} | {grade_name} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {mean_cell} |")
        # Macro F1 row
        macro_vals = []
        for fold in FOLDS:
            d = seen.get((canonical, fold))
            macro_vals.append(d["metrics"]["macro_f1"] if d else float("nan"))
        valid = [v for v in macro_vals if v == v]
        mean = sum(valid) / len(valid) if valid else float("nan")
        cells = [f"{v:.3f}" if v == v else "—" for v in macro_vals]
        mean_cell = f"{mean:.3f}" if mean == mean else "—"
        lines.append(f"| **{label}** | **macro F1** | **{cells[0]}** | **{cells[1]}** | **{cells[2]}** | **{cells[3]}** | **{mean_cell}** |")
        lines.append("|  |  |  |  |  |  |  |")  # spacer
    return "\n".join(lines)


def patch_results_md(seen):
    text = RESULTS_MD.read_text()

    # §4.2 secondary metrics
    sec_block = secondary_metrics_table(seen)
    sec_marker = "| 3ch baseline |  |  |  |  |  |"
    # The existing §4.2 has a flat empty table; we need to find a header
    sec_header = "| Variant | QWK | Acc | F1 (macro) | AUC (macro) |"
    if sec_header in text:
        idx = text.find(sec_header)
        lines = text[idx:].split("\n")
        end = idx
        n_table = 0
        for i, line in enumerate(lines):
            if line.startswith("|"):
                n_table += 1
            else:
                if n_table > 0:
                    end = idx + sum(len(l) + 1 for l in lines[:i])
                    break
        text = text[:idx] + sec_block + text[end:]
        print(f"  patched §4.2 secondary metrics in {RESULTS_MD}")

    # §4.3 per-class F1
    f1_block = f1_per_class_table(seen)
    f1_header = "| Variant | F1 (Grade 0) | F1 (Grade 1) | F1 (Grade 2) | F1 (Grade 3) | F1 (Grade 4) |"
    if f1_header in text:
        idx = text.find(f1_header)
        lines = text[idx:].split("\n")
        end = idx
        n_table = 0
        for i, line in enumerate(lines):
            if line.startswith("|"):
                n_table += 1
            else:
                if n_table > 0:
                    end = idx + sum(len(l) + 1 for l in lines[:i])
                    break
        text = text[:idx] + f1_block + text[end:]
        print(f"  patched §4.3 per-class F1 in {RESULTS_MD}")

    RESULTS_MD.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secondary", action="store_true")
    ap.add_argument("--f1-per-class", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    seen = load_deduped()
    print(f"Loaded {len(seen)} unique (variant, fold) results.\n")

    if args.secondary:
        print("=== Per-variant secondary metrics ===")
        print(secondary_metrics_table(seen))

    if args.f1_per_class:
        print("\n=== Per-variant per-class F1 ===")
        print(f1_per_class_table(seen))

    if args.write:
        patch_results_md(seen)


if __name__ == "__main__":
    main()