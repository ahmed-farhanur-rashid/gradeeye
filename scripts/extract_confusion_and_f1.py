#!/usr/bin/env python3
"""Extract per-fold confusion matrices and per-class F1 from eval_results.jsonl.

Reads saved/logs/eval_results.jsonl, dedupes by (variant, fold), and emits:

  --confusion   : per-fold 5x5 confusion matrix for the 3-channel ConvNeXt-Tiny
                  balanced baseline, row-normalized (%).
  --f1-balancing: per-fold per-class F1 for balanced vs unbalanced ConvNeXt-Tiny.
  --write       : patch docs/results.md §2 (confusion) and §6 (F1).

Usage:
    python scripts/extract_confusion_and_f1.py --confusion
    python scripts/extract_confusion_and_f1.py --f1-balancing
    python scripts/extract_confusion_and_f1.py --confusion --f1-balancing --write
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


def confusion_table(seen, variant="baseline_3ch_balanced_convnext_tiny"):
    """Row-normalized 5x5 confusion matrices for one variant, all folds."""
    lines = []
    for fold in FOLDS:
        d = seen.get((variant, fold))
        if d is None or "confusion_matrix" not in d:
            lines.append(f"### {fold}\n_missing\n")
            continue
        cm = d["confusion_matrix"]
        rows = []
        # Normalize by row (true class)
        row_sums = [sum(r) for r in cm]
        for i, name in enumerate(GRADE_NAMES):
            cells = []
            for j in range(5):
                pct = 100.0 * cm[i][j] / row_sums[i] if row_sums[i] else 0.0
                cells.append(f"{pct:.1f}")
            rows.append(f"| {name} (true) | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |")
        header = "| True \\ Pred | " + " | ".join(GRADE_NAMES) + " |\n"
        header += "|:------------:|" + ":-:|" * 5 + "\n"
        lines.append(f"### {fold} (n={row_sums[0]+row_sums[1]+row_sums[2]+row_sums[3]+row_sums[4]})\n")
        lines.append(header)
        lines.extend(rows)
    return "\n".join(lines)


def f1_balancing_table(seen):
    """Per-class F1 for balanced vs unbalanced ConvNeXt-Tiny, per fold."""
    lines = []
    # Header
    lines.append(
        "| Fold | Class | F1 (balanced) | F1 (unbalanced) | Δ |"
    )
    lines.append(
        "|:-----|:------|:--------------:|:----------------:|:-:|"
    )
    for fold in FOLDS:
        bal = seen.get(("baseline_3ch_balanced_convnext_tiny", fold))
        unb = seen.get(("baseline_3ch_unbalanced_convnext_tiny", fold))
        if bal is None or unb is None:
            continue
        bm = bal["metrics"]
        um = unb["metrics"]
        # Per-class F1 keys: f1_No_DR, f1_Mild, f1_Moderate, f1_Severe, f1_Proliferative_DR
        class_keys = [
            ("No DR (0)", "f1_No_DR"),
            ("Mild (1)", "f1_Mild"),
            ("Moderate (2)", "f1_Moderate"),
            ("Severe (3)", "f1_Severe"),
            ("Proliferative (4)", "f1_Proliferative_DR"),
        ]
        for name, key in class_keys:
            b = bm.get(key)
            u = um.get(key)
            if b is None or u is None:
                continue
            d = u - b
            sign = "+" if d >= 0 else ""
            lines.append(f"| {fold} | {name} | {b:.4f} | {u:.4f} | {sign}{d:.4f} |")
        # Mean row
        bm_f1 = [bm[k] for _, k in class_keys if k in bm]
        um_f1 = [um[k] for _, k in class_keys if k in um]
        mb = sum(bm_f1) / len(bm_f1)
        mu = sum(um_f1) / len(um_f1)
        dmu = mu - mb
        sign = "+" if dmu >= 0 else ""
        lines.append(f"| **{fold} (macro)** | — | **{mb:.4f}** | **{mu:.4f}** | **{sign}{dmu:.4f}** |")
    return "\n".join(lines)


def patch_results_md(seen):
    text = RESULTS_MD.read_text()

    # Patch §2: confusion matrix table for ConvNeXt-Tiny
    cm_block = confusion_table(seen, "baseline_3ch_balanced_convnext_tiny")
    cm_marker = "**Confusion matrix (summed across folds, normalized by row):**"
    if cm_marker in text:
        # Find the line with the marker and replace the table block after it
        idx = text.find(cm_marker)
        # Find the end of the section (next ## heading or EOF)
        rest = text[idx:]
        # Find next occurrence of '\n## ' or '\nPer-fold'
        # Strategy: find the line after the empty header, then find the next "Per-fold" or "## "
        # The existing markdown has a single aggregated table after the marker, then "Per-fold raw"
        # Replace the aggregated empty table with per-fold breakdowns.
        end_marker = "\nPer-fold raw confusion matrices:"
        end_idx = text.find(end_marker, idx)
        if end_idx >= 0:
            # Replace from after marker to end_marker with our cm_block
            after_marker = idx + len(cm_marker)
            text = text[:after_marker] + "\n\n" + cm_block + "\n\n" + text[end_idx:]
            print(f"  patched confusion matrices in {RESULTS_MD}")
    # Patch §6: per-class F1
    f1_block = f1_balancing_table(seen)
    # Find the §6 per-class F1 table by header
    f1_marker = "| Class | F1 (balanced) | F1 (unbalanced) | Δ |"
    if f1_marker in text:
        idx = text.find(f1_marker)
        # Find end of contiguous table
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
        print(f"  patched per-class F1 in {RESULTS_MD}")

    RESULTS_MD.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", action="store_true", help="Print confusion matrices")
    ap.add_argument("--f1-balancing", action="store_true", help="Print per-class F1 table")
    ap.add_argument("--write", action="store_true", help="Patch docs/results.md")
    args = ap.parse_args()

    seen = load_deduped()
    print(f"Loaded {len(seen)} unique (variant, fold) results.\n")

    if args.confusion:
        print("=== Per-fold confusion matrices (3-ch ConvNeXt-Tiny baseline, row-normalized %) ===")
        print(confusion_table(seen))

    if args.f1_balancing:
        print("\n=== Per-class F1: balanced vs unbalanced ConvNeXt-Tiny ===")
        print(f1_balancing_table(seen))

    if args.write:
        patch_results_md(seen)


if __name__ == "__main__":
    main()