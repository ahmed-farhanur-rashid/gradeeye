"""Render LaTeX tables from saved/logs JSON artifacts."""
import json
from pathlib import Path

REPO = Path("/home/farhan/my-projects/gradeeye")

# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Per-(variant, fold) QWK with bootstrap 95% CIs (ConvNeXt-Tiny)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Table: per-(variant, fold) QWK with bootstrap 95% CIs ===\n")
with open(REPO / "saved/logs/bootstrap_cis.json") as f:
    cis = json.load(f)
ci_map = {(r["variant"], r["fold"]): r["qwk_ci"] for r in cis}
variants = ["3ch_balanced", "3ch_unbalanced", "4ch_soft", "4ch_tversky", "4ch_morph"]
folds = ["eyepacs", "aptos", "messidor2", "ddr"]
print("\\begin{table}[ht]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Variant & EyePACS & APTOS & Messidor-2 & DDR \\\\")
print("\\midrule")
for v in variants:
    row = [v.replace("_", " ")]
    for f in folds:
        ci = ci_map.get((v, f))
        if ci is None:
            row.append("—")
        else:
            row.append(f"{ci['point']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")
    print(" & ".join(row) + " \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Per-fold QWK with paired-bootstrap 95\\% confidence intervals (1000 resamples) on the held-out test set. ConvNeXt-Tiny backbone.}\n\\label{tab:bootstrap-cis}}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Decision-boundary threshold tuning (ConvNeXt-Tiny)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Table: decision-boundary threshold tuning ===\n")
with open(REPO / "saved/logs/threshold_tuning.json") as f:
    thr = json.load(f)
thr_map = {(r["variant"], r["fold"]): r for r in thr}
print("\\begin{table}[ht]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{llccccc}")
print("\\toprule")
print("Variant & Held-out & Baseline QWK & Tuned QWK & $\\Delta$QWK & Best bias $b$ & $n$ \\\\")
print("\\midrule")
for v in variants:
    for fold in folds:
        r = thr_map.get((v, fold))
        if r is None:
            continue
        print(f"{v.replace('_', ' ')} & {fold} & "
              f"{r['baseline_qwk']:.4f} & {r['best_qwk']:.4f} & "
              f"{r['delta_qwk']:+.4f} & {r['best_bias']:+.2f} & {r['n_samples']:,} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Decision-boundary post-processing. Tuned QWK is the mean of 5 honest 50/50 stratified splits: bias is fitted on one half of the held-out test set, evaluated on the other half. No logits or weights are modified.}\n\\label{tab:decision-boundary}}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Per-(variant, fold) ECE — all 5 variants (calibration)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Table: per-(variant, fold) ECE on all 5 variants ===\n")
with open(REPO / "saved/logs/calibration_all_variants.json") as f:
    cal = json.load(f)
print("\\begin{table}[ht]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{llcccccc}")
print("\\toprule")
print("Variant & Fold & $N$ & Raw ECE & Per-Thresh ECE & Global-$T$ ECE & $\\Delta$(Per-Raw) & $\\Delta$(Glob-Raw) \\\\")
print("\\midrule")
for v in variants:
    rows = [r for r in cal if r["variant"] == v]
    for r in rows:
        print(f"{v.replace('_', ' ')} & {r['fold']} & {r['n_samples']:,} & "
              f"{r['raw_ece_mean']:.4f} & {r['per_thresh_ece_mean']:.4f} & "
              f"{r['glob_t_ece_mean']:.4f} & "
              f"{r['delta_ece_per_thresh_mean']:+.4f} & "
              f"{r['delta_ece_glob_mean']:+.4f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Per-(variant, fold) ECE (mean across the four CORN thresholds) under three regimes: no calibration (raw), per-threshold temperature scaling, and single-global temperature scaling. Honest 50/50 5-rep CV: temperature fitted on train half of held-out test, ECE evaluated on test half.}\n\\label{tab:cal-all-variants}}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Per-fold QWK × 4-class breakdown + ΔQWK vs 3ch baseline
# (Uses saved per-sample predictions as the single source of truth — these
#  are the same arrays bootstrap/calibration/decision-boundary consume.)
# ─────────────────────────────────────────────────────────────────────────────
print("=== Table: ΔQWK (4ch vs 3ch balanced) ===\n")
import numpy as np
from sklearn.metrics import cohen_kappa_score
seen = {}
for fp in sorted((REPO / "saved/logs/per_sample_predictions").glob("*.json")):
    with open(fp) as f:
        d = json.load(f)
    name = fp.stem  # e.g. "3ch_balanced_lodo_eyepacs"
    parts = name.split("_lodo_")
    if len(parts) != 2:
        continue
    var, fold = parts
    labels = np.array(d["true_labels"])
    preds = np.array(d["pred_labels"])
    q = float(cohen_kappa_score(labels, preds, weights="quadratic"))
    seen[(var, fold)] = {"metrics": {"qwk": q}}

import statistics
print("\\begin{table}[ht]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("Variant & EyePACS & APTOS & Messidor-2 & DDR & Mean $\\pm$ SD \\\\")
print("\\midrule")
for v in variants:
    qs = []
    cells = []
    for f in folds:
        d = seen.get((v, f))
        if d is None:
            cells.append("---")
        else:
            q = d["metrics"]["qwk"]
            qs.append(q)
            cells.append(f"{q:.3f}")
    if qs:
        mu = statistics.mean(qs)
        sd = statistics.stdev(qs) if len(qs) > 1 else 0
        mcell = f"{mu:.3f} $\\pm$ {sd:.3f}"
    else:
        mcell = "---"
    print(f"{v.replace('_', ' ')} & {' & '.join(cells)} & {mcell} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Per-fold held-out test QWK (ConvNeXt-Tiny). Mean $\\pm$ SD is across the four folds.}\n\\label{tab:qwk-per-variant}}")
print()

print("=== Table: ΔQWK (variant minus 3ch balanced) ===\n")
print("\\begin{table}[ht]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("Variant & EyePACS & APTOS & Messidor-2 & DDR & Mean $\\Delta$QWK \\\\")
print("\\midrule")
base_qs = {f: seen[("3ch_balanced", f)]["metrics"]["qwk"] for f in folds if ("3ch_balanced", f) in seen}
for v in ["3ch_unbalanced", "4ch_soft", "4ch_tversky", "4ch_morph"]:
    cells = []
    deltas = []
    for f in folds:
        d = seen.get((v, f))
        if d is None or f not in base_qs:
            cells.append("---")
        else:
            q = d["metrics"]["qwk"]
            dq = q - base_qs[f]
            deltas.append(dq)
            sign = "+" if dq >= 0 else ""
            cells.append(f"{sign}{dq*100:.2f}\\%")
    if deltas:
        mcell = f"{'+' if statistics.mean(deltas) >= 0 else ''}{statistics.mean(deltas)*100:.2f}\\%"
    else:
        mcell = "---"
    print(f"{v.replace('_', ' ')} & {' & '.join(cells)} & {mcell} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{Per-fold $\\Delta$QWK against the 3-channel balanced baseline. The pre-registered improvement threshold was $\\pm 2$ absolute percentage points.}\n\\label{tab:delta-qwk}}")