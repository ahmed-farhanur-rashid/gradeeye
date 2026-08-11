#!/usr/bin/env python3
"""Collect per-sample predictions from existing LODO checkpoints.

For each (variant, fold) pair seen in eval_results.jsonl, loads the *best.pt
checkpoint, runs inference on the held-out test manifest, and writes a JSON
file containing per-sample {true_label, pred_label, per_threshold_prob}.

This file is ADDITIVE only — it does not modify the existing
eval_results.jsonl or calibration JSON files. It exists so downstream
analyses (bootstrap CIs, decision-boundary threshold tuning, per-class
metrics) can compute statistics that need the full per-sample array.

Usage:
    python scripts/collect_per_sample_predictions.py
    python scripts/collect_per_sample_predictions.py --variants 3ch_balanced
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.models.corn import corn_predict_probas
from src.training.checkpoint_io import load_model_from_checkpoint
from torch.utils.data import DataLoader

OUT_DIR = REPO_ROOT / "saved/logs/per_sample_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_LOG = REPO_ROOT / "saved/logs/eval_results.jsonl"

FOLD_BY_N = {88702: "eyepacs", 3662: "aptos", 1744: "messidor2", 12522: "ddr"}
FOLD_FROM_HOLDOUT = {"EyePACS": "eyepacs", "APTOS": "aptos",
                     "Messidor-2": "messidor2", "DDR": "ddr"}

# Where seg aux channel lives (only used for 4-ch variants)
SEG_DIR_BY_VARIANT = {
    "four_ch_soft":    "data/segmentation/preds_soft",
    "four_ch_tversky": "data/segmentation/preds_soft",  # Tversky uses soft-mask predictions
    "four_ch_morph":   "data/segmentation/preds_soft",  # Morph uses soft-mask predictions
    "four_ch_sobel":   "data/segmentation/preds_sobel",
    "five_ch_soft_sobel":    "data/segmentation/preds_soft",
    "five_ch_tversky_sobel": "data/segmentation/preds_soft",
}

# Canonical variant name → short slug for output filename
VARIANT_SLUG = {
    "baseline_3ch":                  "3ch_balanced",
    "baseline_3ch_unbalanced":       "3ch_unbalanced",
    "four_ch_soft":                  "4ch_soft",
    "four_ch_tversky":               "4ch_tversky",
    "four_ch_morph":                 "4ch_morph",
    "four_ch_sobel":                 "4ch_sobel",
    "five_ch_soft_sobel":            "5ch_soft_sobel",
    "five_ch_tversky_sobel":         "5ch_tversky_sobel",
}


def load_target_pairs():
    """Return list of (variant_canonical, fold_slug, checkpoint_path, manifest_path, seg_dir_or_none).

    Scans eval_results.jsonl to find the highest-QWK checkpoint per
    (variant, fold). variant_canonical is the parent dir of the checkpoint
    (e.g. "four_ch_soft") with the `_best` removed. fold_slug is the
    lowercase LODO holdout (eyepacs, aptos, messidor2, ddr).
    """
    # checkpoint_parent / manifest_parent determine the variant+fold axes
    # uniquely; we don't have to parse run names.
    seen: dict[tuple, dict] = {}
    with open(EVAL_LOG) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            cp = d.get("checkpoint")
            if not cp or not cp.endswith("_best.pt"):
                continue
            # skip 5-arch ablation etc — only keep convnext_tiny for now
            if d.get("model_arch") and d["model_arch"] != "convnext_tiny":
                continue
            # Determine variant from checkpoint path. Layouts:
            #   baseline_3ch/{fold}/{arch}/x.pt          → variant="baseline_3ch"
            #   baseline_3ch_unbalanced/{fold}/{arch}/x.pt → variant="baseline_3ch_unbalanced"
            #   four_ch_soft/x.pt                         → variant="four_ch_soft"
            pp = Path(cp).parent
            if pp.name in ("convnext_tiny", "deit3_small_patch16_384",
                           "maxvit_tiny_tf_384", "swinv2_base_window12to24_192to384"):
                # 3-arch layout: pp=arch dir, grandparent=fold, great-grandparent=variant
                variant = pp.parent.parent.name
            else:
                # flat layout: pp is the variant dir
                variant = pp.name
            n = d.get("n_samples", 0)
            fold = FOLD_BY_N.get(n)
            if not fold:
                continue
            key = (variant, fold)
            prev = seen.get(key)
            if prev is None or d.get("metrics", {}).get("qwk", 0) > prev.get("metrics", {}).get("qwk", 0):
                seen[key] = d

    out = []
    for (variant, fold), d in sorted(seen.items()):
        cp = d["checkpoint"]
        manifest = d["manifest"]
        seg_dir = SEG_DIR_BY_VARIANT.get(variant)
        slug = VARIANT_SLUG.get(variant, variant)
        out.append((variant, slug, fold, cp, manifest, seg_dir))
    return out


def collect_one(variant: str, fold: str, cp: str, manifest: str, seg_dir: str | None,
                device: str, batch_size: int = 64):
    """Collect per-sample predictions for one (variant, fold)."""
    cp_p = Path(cp)
    if not cp_p.exists():
        print(f"  SKIP {variant}/{fold}: checkpoint not found at {cp_p}")
        return None

    print(f"\n=== {variant} / {fold} ===")
    print(f"  checkpoint: {cp_p}")
    print(f"  manifest:   {manifest}")

    model, config = load_model_from_checkpoint(str(cp_p), device, use_ema=True)
    img_size = config.get("model", {}).get("img_size")

    dataset = DRDataset(
        manifest,
        transform=build_eval_transforms(img_size=img_size),
        seg_dir=seg_dir,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=device.startswith("cuda"),
    )

    logits_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.numpy())

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)

    # Convert raw logits -> probabilities per threshold via sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    # Get categorical class probabilities (K=5) via CORN joint
    class_probs = corn_predict_probas(torch.from_numpy(logits)).numpy()
    pred_labels = class_probs.argmax(axis=1)

    print(f"  n={len(labels)}, K={probs.shape[1]} thresholds, pred_acc={(pred_labels == labels).mean():.4f}")

    return {
        "variant": variant,
        "fold": fold,
        "n_samples": int(len(labels)),
        "true_labels": labels.astype(int).tolist(),
        "pred_labels": pred_labels.astype(int).tolist(),
        "per_threshold_probs": probs.tolist(),  # [N, 4]
        "class_probs": class_probs.tolist(),     # [N, 5]
        "logits": logits.tolist(),              # [N, 4] raw CORN logits (for calibration)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variants", nargs="+", default=None,
        help="Filter by variant slug (e.g. 3ch_balanced, 4ch_soft). Default: all in eval_results.jsonl",
    )
    ap.add_argument("--folds", nargs="+", default=None,
                    help="Filter by fold slug (eyepacs, aptos, messidor2, ddr). Default: all")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run even if output JSON exists")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pairs = load_target_pairs()
    print(f"Found {len(pairs)} (variant, fold) pairs in eval_results.jsonl\n")

    for variant, slug, fold, cp, manifest, seg_dir in pairs:
        if args.variants and slug not in args.variants:
            continue
        if args.folds and fold not in args.folds:
            continue
        out_path = OUT_DIR / f"{slug}_lodo_{fold}.json"
        if out_path.exists() and not args.overwrite:
            print(f"  EXISTS {out_path} — skipping (use --overwrite to re-run)")
            continue
        data = collect_one(variant, fold, cp, manifest, seg_dir, device, args.batch_size)
        if data is None:
            continue
        out_path.write_text(json.dumps(data))
        print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    print(f"\nAll predictions written to {OUT_DIR}")


if __name__ == "__main__":
    main()