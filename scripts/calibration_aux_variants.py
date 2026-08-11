#!/usr/bin/env python3
"""Per-threshold calibration of ALL variants (3-ch baseline + 4-ch aux).

Reads per-sample logits from `saved/logs/per_sample_predictions/*.json`,
runs honest 50/50 cross-validation per (variant, fold) to fit per-threshold
temperature and global temperature on one half, then reports ECE on the
held-out test half.

Outputs:
  saved/logs/calibration_all_variants.json   — full results (per-fold, per-threshold)
  saved/logs/calibration_summary.md          — markdown summary

The 3-ch baseline (balanced) replicates the existing
`src.eval.run_calibration_study` result on the same ConvNeXt-Tiny model,
additively. The 3-ch unbalanced and 4-ch soft/Tversky/morph variants are
new — they were not in the existing study.

The honest cross-validated ECE (split-fit, other-half-eval) replaces
fit-on-validation-test-on-test for the variants we don't have train logits for.
This is a conservative estimator.

This file is ADDITIVE only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.probability_calibration import (
    NUM_THRESHOLDS,
    apply_temperature,
    fit_temperature_global,
    fit_temperature_per_threshold,
    per_threshold_ece,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "saved/logs/per_sample_predictions"
OUT_JSON = REPO_ROOT / "saved/logs/calibration_all_variants.json"
OUT_MD = REPO_ROOT / "saved/logs/calibration_summary.md"

FOLDS = ["eyepacs", "aptos", "messidor2", "ddr"]
VARIANT_LABELS = [
    ("3ch_balanced",   "3ch baseline (balanced)"),
    ("3ch_unbalanced", "3ch baseline (unbalanced)"),
    ("4ch_soft",       "4ch soft mask"),
    ("4ch_tversky",    "4ch Tversky"),
    ("4ch_morph",      "4ch morphological"),
]

NUM_BOOTSTRAP = 200
RNG_SEED = 0


def stratified_split(y, rng, frac=0.5):
    idx_train, idx_test = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        n_test = max(1, int(round(len(ci) * frac)))
        idx_test.extend(ci[:n_test].tolist())
        idx_train.extend(ci[n_test:].tolist())
    return np.array(idx_train), np.array(idx_test)


def calibrate_one(slug, fold, n_reps=5):
    fp = PRED_DIR / f"{slug}_lodo_{fold}.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    y = np.array(d["true_labels"])
    logits = np.array(d["logits"])  # (N, 4)
    if logits.size == 0:
        return None

    # Honest 50/50 5-rep CV: fit on train half, evaluate on test half.
    raw_ece, per_t_ece, glob_t_ece = [], [], []
    per_t_fitted, glob_t_fitted = [], []
    rng = np.random.default_rng(RNG_SEED + hash((slug, fold)) % (2**31))
    for r in range(n_reps):
        idx_tr, idx_te = stratified_split(y, rng)
        per_t = fit_temperature_per_threshold(logits[idx_tr], y[idx_tr])
        glob_t = fit_temperature_global(logits[idx_tr], y[idx_tr])
        per_t_fitted.append(per_t.tolist())
        glob_t_fitted.append(float(glob_t))
        # Evaluate on test half
        probs_raw = 1.0 / (1.0 + np.exp(-np.clip(logits[idx_te], -80, 80)))
        ece_raw = np.mean([per_threshold_ece(probs_raw, y[idx_te], k=k)
                           for k in range(NUM_THRESHOLDS)])
        probs_pT = apply_temperature(logits[idx_te], per_t)
        ece_pT = np.mean([per_threshold_ece(probs_pT, y[idx_te], k=k)
                          for k in range(NUM_THRESHOLDS)])
        probs_gT = apply_temperature(logits[idx_te], glob_t)
        ece_gT = np.mean([per_threshold_ece(probs_gT, y[idx_te], k=k)
                          for k in range(NUM_THRESHOLDS)])
        raw_ece.append(float(ece_raw))
        per_t_ece.append(float(ece_pT))
        glob_t_ece.append(float(ece_gT))
    return {
        "variant": slug,
        "fold": fold,
        "n_samples": int(len(y)),
        "raw_ece_mean": float(np.mean(raw_ece)),
        "raw_ece_std": float(np.std(raw_ece)),
        "per_thresh_ece_mean": float(np.mean(per_t_ece)),
        "per_thresh_ece_std": float(np.std(per_t_ece)),
        "glob_t_ece_mean": float(np.mean(glob_t_ece)),
        "glob_t_ece_std": float(np.std(glob_t_ece)),
        "delta_ece_per_thresh_mean": float(np.mean(per_t_ece) - np.mean(raw_ece)),
        "delta_ece_glob_mean": float(np.mean(glob_t_ece) - np.mean(raw_ece)),
        "per_thresh_temperatures_mean": [float(x) for x in np.mean(per_t_fitted, axis=0)],
        "global_temperatures_mean": float(np.mean(glob_t_fitted)),
    }


def mean_over_folds(rows):
    return {k: float(np.mean([r[k] for r in rows if k in r]))
            for k in ["raw_ece_mean", "per_thresh_ece_mean",
                      "glob_t_ece_mean", "delta_ece_per_thresh_mean",
                      "delta_ece_glob_mean"]}


def fmt(x): return f"{x:.4f}" if abs(x) < 10 else f"{x:.3f}"


def render_md(all_rows):
    lines = []
    lines.append("# Per-Threshold Calibration Diagnostic — All Variants\n")
    lines.append("Honest 50/50 5-rep CV: temperature fitted on train half of held-out test, evaluated on test half.\n")
    lines.append("(Re-runs the existing `src.eval.run_calibration_study` analysis on the 3-ch balanced baseline, additively extends it to 3-ch unbalanced and the four 4-ch auxiliary-channel variants.)\n")
    lines.append("\n## Per-(variant, fold) mean-ECE\n")
    lines.append("| Variant | Fold | N | Raw ECE | Per-Thresh ECE | Global-T ECE | Δ(Per-Raw) | Δ(Glob-Raw) |")
    lines.append("|:--------|:-----|--:|--------:|---------------:|-------------:|-----------:|------------:|")
    for r in all_rows:
        lines.append(
            f"| {r['variant']:18s} | {r['fold']:10s} | {r['n_samples']:>6d} | "
            f"{r['raw_ece_mean']:.4f} | {r['per_thresh_ece_mean']:.4f} | "
            f"{r['glob_t_ece_mean']:.4f} | "
            f"{r['delta_ece_per_thresh_mean']:+.4f} | "
            f"{r['delta_ece_glob_mean']:+.4f} |"
        )
    lines.append("\n## Per-variant mean over folds\n")
    lines.append("| Variant | Raw ECE | Per-Thresh ECE | Global-T ECE | Δ(Per-Raw) | Δ(Glob-Raw) | Per-Temperatures (k=0..3) |")
    lines.append("|:--------|--------:|---------------:|-------------:|-----------:|------------:|:--------------------------|")
    by_var = {}
    for r in all_rows:
        by_var.setdefault(r["variant"], []).append(r)
    for slug, rows in [(v[0], by_var[v[0]]) for v in VARIANT_LABELS if v[0] in by_var]:
        m = mean_over_folds(rows)
        ts = rows[0].get("per_thresh_temperatures_mean", [1]*4)
        ts_str = ", ".join(f"{t:.2f}" for t in ts)
        lines.append(
            f"| {slug:18s} | {m['raw_ece_mean']:.4f} | {m['per_thresh_ece_mean']:.4f} | "
            f"{m['glob_t_ece_mean']:.4f} | {m['delta_ece_per_thresh_mean']:+.4f} | "
            f"{m['delta_ece_glob_mean']:+.4f} | [ {ts_str} ] |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--n-reps", type=int, default=5)
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args()

    variants = args.variants or [v[0] for v in VARIANT_LABELS]
    folds = args.folds or FOLDS

    all_rows = []
    for slug in variants:
        for fold in folds:
            r = calibrate_one(slug, fold, n_reps=args.n_reps)
            if r is None:
                continue
            all_rows.append(r)
            print(f"{slug:18s} {fold:10s} N={r['n_samples']:>6d}  "
                  f"raw={r['raw_ece_mean']:.4f}  perT={r['per_thresh_ece_mean']:.4f}  "
                  f"globT={r['glob_t_ece_mean']:.4f}  "
                  f"ΔperT={r['delta_ece_per_thresh_mean']:+.4f}  "
                  f"ΔglobT={r['delta_ece_glob_mean']:+.4f}  "
                  f"Tper=[{','.join(f'{t:.2f}' for t in r['per_thresh_temperatures_mean'])}]  "
                  f"Tglob={r['global_temperatures_mean']:.2f}")

    Path(args.out_json).write_text(json.dumps(all_rows, indent=2))
    Path(args.out_md).write_text(render_md(all_rows))
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()