"""Run probability calibration studies for completed LODO checkpoints.

For each fold, this command collects raw CORN logits on the held-out
validation, full-test, and matched-test manifests; reports per-threshold ECE;
fits per-threshold and global temperature scaling on validation; evaluates
those temperatures on both held-out test variants; and writes JSON artifacts
containing ECE tables, temperatures, and reliability-diagram bins.

Usage:
    python -m src.eval.run_calibration_study

The command intentionally uses the existing checkpoint config and evaluation
loader. It does not alter model weights or decision thresholds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.eval.metrics import compute_all_metrics
from src.eval.probability_calibration import (
    NUM_THRESHOLDS,
    apply_temperature,
    fit_temperature_global,
    fit_temperature_per_threshold,
    per_threshold_ece,
    reliability_diagram_data,
)
from src.models.corn import corn_predict_probas
from src.training.checkpoint_io import load_model_from_checkpoint


FOLDS = ("eyepacs", "aptos", "messidor2", "ddr")


def collect_logits(
    model: torch.nn.Module,
    manifest: str,
    batch_size: int,
    device: str,
    img_size: int | None = None,
    seg_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw CORN logits and integer labels from one manifest."""
    dataset = DRDataset(
        manifest,
        transform=build_eval_transforms(img_size=img_size),
        seg_dir=seg_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.numpy())
    if not logits_all:
        raise ValueError(f"Manifest contains no samples: {manifest}")
    return np.concatenate(logits_all), np.concatenate(labels_all)


def _ece_table(logits: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Return ECE and reliability bins for all CORN thresholds."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    rows = []
    for k in range(NUM_THRESHOLDS):
        outcomes = (labels > k).astype(np.float64)
        rows.append({
            "threshold": k,
            "ece": per_threshold_ece(probs, labels, k=k, n_bins=n_bins),
            "reliability_bins": [
                {
                    "lower": lower,
                    "upper": upper,
                    "accuracy": accuracy,
                    "count": count,
                }
                for lower, upper, accuracy, count in reliability_diagram_data(
                    probs[:, k], outcomes, n_bins=n_bins
                )
            ],
        })
    return {
        "per_threshold": rows,
        "mean_ece": float(np.mean([row["ece"] for row in rows])),
    }


def _evaluate(logits: np.ndarray, labels: np.ndarray, temperatures: np.ndarray | float) -> dict[str, float]:
    """Evaluate CORN class predictions after temperature scaling."""
    scaled = apply_temperature(logits, temperatures)
    class_probs = corn_predict_probas(torch.from_numpy(np.asarray(
        np.log(np.clip(scaled, 1e-12, 1 - 1e-12) / np.clip(1 - scaled, 1e-12, 1))
    ))).numpy()
    predictions = class_probs.argmax(axis=1)
    metrics = compute_all_metrics(labels, predictions, class_probs)
    return {key: float(value) for key, value in metrics.items() if np.isscalar(value)}


def run_fold(fold: str, args: argparse.Namespace, device: str) -> dict[str, Any]:
    """Run one fold and return a serializable result dictionary."""
    checkpoint = Path(args.checkpoint_root) / fold / "convnext_tiny" / f"lodo_{fold}_convnext_tiny_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    print(f"\n=== {fold}: {checkpoint} ===")
    model, config = load_model_from_checkpoint(str(checkpoint), device, use_ema=not args.no_ema)
    img_size = args.img_size or config.get("model", {}).get("img_size")
    batch_size = args.batch_size

    # Read manifest paths from the saved checkpoint config (top-level manifest_val,
    # manifest_test) so we don't have to re-derive the LODO manifest layout. If not present,
    # fall back to args.split_root / lodo_<fold>_<split>.csv.
    val_manifest = Path(config.get("manifest_val") or (Path(args.split_root) / f"lodo_{fold}_val.csv"))
    full_manifest = Path(config.get("manifest_test") or (Path(args.split_root) / f"lodo_{fold}_test_full.csv"))
    matched_manifest = Path(args.split_root) / f"lodo_{fold}_test_matched.csv"
    for path in (val_manifest, full_manifest, matched_manifest):
        if not path.exists():
            raise FileNotFoundError(path)

    print(f"Collecting validation logits ({val_manifest})")
    val_logits, val_labels = collect_logits(model, str(val_manifest), batch_size, device, img_size)
    print(f"Collecting full-test logits ({full_manifest})")
    full_logits, full_labels = collect_logits(model, str(full_manifest), batch_size, device, img_size)
    print(f"Collecting matched-test logits ({matched_manifest})")
    matched_logits, matched_labels = collect_logits(model, str(matched_manifest), batch_size, device, img_size)

    raw_ece = {
        "validation": _ece_table(val_logits, val_labels, args.n_bins),
        "full": _ece_table(full_logits, full_labels, args.n_bins),
        "matched": _ece_table(matched_logits, matched_labels, args.n_bins),
    }
    per_t = fit_temperature_per_threshold(val_logits, val_labels)
    global_t = fit_temperature_global(val_logits, val_labels)

    calibrated = {
        "validation": {
            "raw": _evaluate(val_logits, val_labels, 1.0),
            "per_threshold": _evaluate(val_logits, val_labels, per_t),
            "global": _evaluate(val_logits, val_labels, global_t),
        },
        "full": {
            "raw": _evaluate(full_logits, full_labels, 1.0),
            "per_threshold": _evaluate(full_logits, full_labels, per_t),
            "global": _evaluate(full_logits, full_labels, global_t),
        },
        "matched": {
            "raw": _evaluate(matched_logits, matched_labels, 1.0),
            "per_threshold": _evaluate(matched_logits, matched_labels, per_t),
            "global": _evaluate(matched_logits, matched_labels, global_t),
        },
    }

    # Also recompute ECE per threshold after temperature scaling — the raw_ece above
    # is pre-temperature only, but the comparison §3.2/§3.3/§3.4 needs post-T ECE.
    def _post_t_ece(logits, labels, temperatures):
        scaled = apply_temperature(logits, temperatures)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -80.0, 80.0)))
        return _ece_table(probs, labels, args.n_bins)

    calibrated_ece = {
        "validation": {
            "raw": raw_ece["validation"],
            "per_threshold": _post_t_ece(val_logits, val_labels, per_t),
            "global": _post_t_ece(val_logits, val_labels, global_t),
        },
        "full": {
            "raw": raw_ece["full"],
            "per_threshold": _post_t_ece(full_logits, full_labels, per_t),
            "global": _post_t_ece(full_logits, full_labels, global_t),
        },
        "matched": {
            "raw": raw_ece["matched"],
            "per_threshold": _post_t_ece(matched_logits, matched_labels, per_t),
            "global": _post_t_ece(matched_logits, matched_labels, global_t),
        },
    }
    return {
        "fold": fold,
        "checkpoint": str(checkpoint),
        "validation_manifest": str(val_manifest),
        "full_manifest": str(full_manifest),
        "matched_manifest": str(matched_manifest),
        "n_samples": {
            "validation": int(len(val_labels)),
            "full": int(len(full_labels)),
            "matched": int(len(matched_labels)),
        },
        "temperatures": {
            "per_threshold": [float(value) for value in per_t],
            "global": float(global_t),
        },
        "raw_ece": raw_ece,
        "calibrated_metrics": calibrated,
        "calibrated_ece": calibrated_ece,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", choices=FOLDS, default=list(FOLDS))
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--checkpoint-root", default="saved/checkpoints")
    parser.add_argument("--output-dir", default="saved/logs/calibration")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in args.folds:
        result = run_fold(fold, args, device)
        results.append(result)
        out_path = output_dir / f"lodo_{fold}_convnext_tiny_probability_calibration.json"
        out_path.write_text(json.dumps({
            "created_at": dt.datetime.now().isoformat(),
            "device": device,
            "n_bins": args.n_bins,
            "result": result,
        }, indent=2) + "\n")
        print(f"Saved {out_path}")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "created_at": dt.datetime.now().isoformat(),
        "device": device,
        "results": results,
    }, indent=2) + "\n")
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
