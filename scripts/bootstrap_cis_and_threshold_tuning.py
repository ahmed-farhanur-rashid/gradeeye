#!/usr/bin/env python3
"""Bootstrap 95% CIs + Decision-boundary threshold tuning.

Reads per-sample predictions from `saved/logs/per_sample_predictions/*.json`
and writes:

  saved/logs/bootstrap_cis.json     — per (variant, fold, metric) ±95% CI
  saved/logs/threshold_tuning.json  — optimal decision-boundary thresholds

Bootstrap CIs: resample per-sample arrays with replacement 1000 times,
recompute QWK / accuracy / macro-F1 each time, take 2.5 / 97.5 percentile.

Decision-boundary tuning: for each K-1 thresholds {t1..t4} in the per-threshold
probs, sweep an additive bias b in [-0.4, 0.4] step 0.01 applied to all
thresholds; convert biased sigmoid probs to ordinal class predictions via
argmax of the joint class probability (CORN); recompute QWK on the same
held-out set and report the optimal (b, qwk) per (variant, fold).

The baseline argmax decision rule (current eval pipeline) is included as
bias=0 for comparison. Decision-boundary tuning is purely post-hoc — no
logits or weights change.

This file is ADDITIVE only — no existing log/results file is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "saved/logs/per_sample_predictions"
OUT_CI = REPO_ROOT / "saved/logs/bootstrap_cis.json"
OUT_THR = REPO_ROOT / "saved/logs/threshold_tuning.json"

FOLDS = ["eyepacs", "aptos", "messidor2", "ddr"]
VARIANTS = ["3ch_balanced", "3ch_unbalanced", "4ch_soft", "4ch_tversky", "4ch_morph"]
K_CLASSES = 5
K_THR = K_CLASSES - 1  # 4

NUM_BOOTSTRAP = 1000
RNG_SEED = 0

# CORN joint class probability from K-1 sigmoid threshold probs.
# class c (0..K-1) is predicted when c-1 thresholds > 0.5 and c thresholds ≤ 0.5.
# Equivalently: class c = sum_{k=c}^{K-1} (1 - p_k) * prod_{j=0}^{c-1} p_j
# where p_k is sigmoid(threshold_k).
def corn_class_probs(per_thresh_probs: np.ndarray) -> np.ndarray:
    """per_thresh_probs: (N, K-1) sigmoid outputs. Returns (N, K) class probs."""
    p = per_thresh_probs  # (N, K-1)
    # class_k = prod_{j<k} p_j * (1 - p_k) for k in 0..K-2
    # class_{K-1} = prod_{j<K-1} p_j
    N = p.shape[0]
    cls = np.zeros((N, K_CLASSES), dtype=np.float64)
    for k in range(K_CLASSES - 1):
        left = np.ones(N)
        for j in range(k):
            left = left * p[:, j]
        cls[:, k] = left * (1.0 - p[:, k])
    # last class
    left = np.ones(N)
    for j in range(K_THR):
        left = left * p[:, j]
    cls[:, K_CLASSES - 1] = left
    return cls


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def metrics_for_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "qwk": qwk(y_true, y_pred),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def stratified_split(y: np.ndarray, rng: np.random.Generator, test_frac: float = 0.5):
    """Per-class 50/50 split. Returns (idx_train, idx_test)."""
    idx_train, idx_test = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        rng.shuffle(ci)
        n_test = max(1, int(round(len(ci) * test_frac)))
        idx_test.extend(ci[:n_test].tolist())
        idx_train.extend(ci[n_test:].tolist())
    return np.array(idx_train, dtype=int), np.array(idx_test, dtype=int)


def bootstrap_ci(metric_fn, y_true: np.ndarray, y_pred: np.ndarray,
                 n: int = NUM_BOOTSTRAP, rng: np.random.Generator = None) -> dict:
    rng = rng or np.random.default_rng(RNG_SEED)
    n_samples = len(y_true)
    point = metric_fn(y_true, y_pred)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])
    return {
        "point": float(point),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "std": float(samples.std()),
        "n_bootstrap": n,
    }


def decision_rule(probs: np.ndarray, bias: float) -> np.ndarray:
    """Apply additive bias to per-threshold sigmoid probs, return predicted class."""
    biased = np.clip(probs + bias, 1e-6, 1 - 1e-6)
    cls = corn_class_probs(biased)
    return cls.argmax(axis=1)


def sweep_bias(y_true: np.ndarray, probs: np.ndarray,
               biases: np.ndarray = None) -> tuple[float, float, list]:
    """Sweep bias to maximize QWK on (y_true, probs). Return (best_bias, best_qwk, trace)."""
    if biases is None:
        biases = np.arange(-0.4, 0.41, 0.01)
    best_b, best_q = 0.0, -1.0
    trace = []
    for b in biases:
        pred = decision_rule(probs, b)
        q = qwk(y_true, pred)
        trace.append({"bias": float(b), "qwk": float(q)})
        if q > best_q:
            best_q = q
            best_b = float(b)
    return best_b, best_q, trace


def process_one(slug: str, fold: str) -> dict | None:
    fp = PRED_DIR / f"{slug}_lodo_{fold}.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    y_true = np.array(d["true_labels"])
    y_pred = np.array(d["pred_labels"])
    probs = np.array(d["per_threshold_probs"])  # (N, 4)

    rng = np.random.default_rng(RNG_SEED + hash((slug, fold)) % (2**31))
    cis = {k: bootstrap_ci(metric_fn, y_true, y_pred, rng=rng)
           for k, metric_fn in [("qwk", qwk), ("accuracy", lambda t, p: float((p == t).mean())),
                                ("macro_f1", macro_f1)]}

    # Decision-boundary threshold tuning — HONEST cross-validated:
    # Stratified 50/50 split (per class) of the held-out test set. Tune bias
    # on one half, evaluate on the other half. The reported best_qwk is the
    # *out-of-bias-sample* QWK on the held-out half; this removes the
    # train/test leak that selecting bias on the same eval set would cause.
    # We repeat the split with 5 different random seeds and average the
    # out-of-bias-sample QWKs for a more stable estimate.
    baseline_q = qwk(y_true, y_pred)
    oob_qwks, chosen_biases, traces = [], [], []
    for split_seed in range(5):
        rng2 = np.random.default_rng(RNG_SEED + 1000 + split_seed)
        idx_train, idx_test = stratified_split(y_true, rng2)
        b, _q, trace = sweep_bias(y_true[idx_train], probs[idx_train])
        oob_pred = decision_rule(probs[idx_test], b)
        oob_q = qwk(y_true[idx_test], oob_pred)
        oob_qwks.append(oob_q)
        chosen_biases.append(b)
        if split_seed == 0:
            traces.append(trace)
    honest_best_q = float(np.mean(oob_qwks))
    honest_best_b = float(np.mean(chosen_biases))
    return {
        "variant": slug,
        "fold": fold,
        "n_samples": int(len(y_true)),
        "baseline_qwk": baseline_q,
        "best_bias": honest_best_b,
        "best_qwk": honest_best_q,
        "delta_qwk": honest_best_q - baseline_q,
        "bias_trace": traces[0],  # full sweep for plotting (first seed)
        "cv_qwks": oob_qwks,
        "cv_biases": chosen_biases,
        "bootstrap_cis": cis,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-only", action="store_true")
    ap.add_argument("--threshold-only", action="store_true")
    ap.add_argument("--num-bootstrap", type=int, default=NUM_BOOTSTRAP)
    ap.add_argument("--out-ci", default=str(OUT_CI))
    ap.add_argument("--out-thr", default=str(OUT_THR))
    args = ap.parse_args()
    if not args.bootstrap_only and not args.threshold_only:
        args.bootstrap_only = args.threshold_only = True

    ci_results = {}
    thr_results = {}
    for slug in VARIANTS:
        for fold in FOLDS:
            r = process_one(slug, fold)
            if r is None:
                print(f"  SKIP {slug}/{fold}: predictions file missing")
                continue
            ci_results[(slug, fold)] = r["bootstrap_cis"]
            thr_results[(slug, fold)] = {
                "n_samples": r["n_samples"],
                "baseline_qwk": r["baseline_qwk"],
                "best_bias": r["best_bias"],
                "best_qwk": r["best_qwk"],
                "delta_qwk": r["delta_qwk"],
                "bias_trace": r["bias_trace"],
            }
            print(f"{slug:18s} {fold:10s}  qwk={r['baseline_qwk']:.4f}  "
                  f"CI=[{r['bootstrap_cis']['qwk']['ci_low']:.4f}, "
                  f"{r['bootstrap_cis']['qwk']['ci_high']:.4f}]  "
                  f"tuned_qwk={r['best_qwk']:.4f} @ b={r['best_bias']:+.2f}  "
                  f"Δqwk={r['delta_qwk']:+.4f}")

    if args.bootstrap_only:
        # write as list of {variant, fold, ...} records (JSON-safe)
        out = []
        for (slug, fold), cis in ci_results.items():
            out.append({"variant": slug, "fold": fold, **{f"{m}_ci": c for m, c in cis.items()}})
        Path(args.out_ci).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {len(out)} bootstrap CI records to {args.out_ci}")
    if args.threshold_only:
        out = []
        for (slug, fold), t in thr_results.items():
            out.append({"variant": slug, "fold": fold, **t})
        Path(args.out_thr).write_text(json.dumps(out, indent=2))
        print(f"Wrote {len(out)} threshold-tuning records to {args.out_thr}")


if __name__ == "__main__":
    main()