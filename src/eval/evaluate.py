"""
Single-model evaluation entry point — supports calibrated thresholds and TTA.

Usage:
    python -m src.eval.evaluate \
        --checkpoint saved/checkpoints/lodo_eyepacs_convnext_tiny_best.pt \
        --manifest data/splits/lodo_aptos_test_full.csv \
        --thresholds saved/checkpoints/calibrated_thresholds.json \
        --tta

This is the canonical single-checkpoint evaluator. For ensemble evaluation
(multiple checkpoints → averaged class probabilities), see `src/ensemble`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.eval.metrics import compute_all_metrics, compute_confusion_matrix, format_confusion_matrix_str
from src.eval.threshold_calibration import (
    load_thresholds,
    predict_with_thresholds,
)
from src.training.checkpoint_io import load_model_from_checkpoint


def evaluate_checkpoint(
    checkpoint_path: str,
    manifest_csv: str,
    batch_size: int = 64,
    use_ema: bool = True,
    use_tta: bool = False,
    thresholds_path: str | None = None,
    img_size: int | None = None,
    seg_dir: str | None = None,
    device: str | None = None,
) -> dict:
    """Run a single model on a manifest, returning metrics + report dict.

    Uses argmax if `thresholds_path` is None; otherwise applies the
    calibrated thresholds from that JSON file.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model_from_checkpoint(checkpoint_path, device, use_ema=use_ema)
    arch = config.get("model", {}).get("arch", "unknown")
    img_size = img_size or config.get("model", {}).get("img_size")

    dataset = DRDataset(
        manifest_csv,
        transform=build_eval_transforms(img_size=img_size),
        seg_dir=seg_dir,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    all_probas, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            if use_tta:
                from src.eval.tta import tta_forward
                probas = tta_forward(model, images, "corn")
            else:
                logits = model(images)
                from src.models.corn import corn_predict_probas
                probas = corn_predict_probas(logits)
            all_probas.append(probas.cpu().numpy())
            all_labels.append(labels.numpy())

    y_proba = np.concatenate(all_probas)
    y_true = np.concatenate(all_labels)

    if thresholds_path:
        thresholds = load_thresholds(thresholds_path)
        y_pred = predict_with_thresholds(y_proba, thresholds)
        mode = "calibrated"
    else:
        y_pred = y_proba.argmax(axis=1)
        mode = "argmax"

    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "confusion_matrix_str": format_confusion_matrix_str(cm),
        "n_samples": int(len(y_true)),
        "model_arch": arch,
        "mode": mode,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-model evaluation with optional calibrated thresholds.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--thresholds", default=None,
        help="Path to calibrated_thresholds.json. If omitted, uses argmax.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--seg-dir", default=None)
    parser.add_argument("--log-dir", default="saved/logs")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating: {os.path.basename(args.checkpoint)} on {args.manifest}")
    print(f"TTA: {args.tta}, Thresholds: {args.thresholds or 'argmax (none)'}")

    result = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        batch_size=args.batch_size,
        use_ema=not args.no_ema,
        use_tta=args.tta,
        thresholds_path=args.thresholds,
        img_size=args.img_size,
        seg_dir=args.seg_dir,
        device=device,
    )
    arch = result["model_arch"]
    mode = result["mode"]
    metrics = result["metrics"]

    print(f"\nEvaluated {result['n_samples']} samples ({arch}, {mode})")
    print(f"QWK (primary): {metrics['qwk']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    if "macro_auc_roc" in metrics:
        print(f"Macro AUC-ROC: {metrics['macro_auc_roc']:.4f}")
    print("\nConfusion matrix:")
    print(result["confusion_matrix_str"])

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "eval_results.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "single_model_calibrated" if args.thresholds else "single_model_argmax",
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_arch": arch,
        "manifest": os.path.abspath(args.manifest),
        "use_ema": not args.no_ema,
        "use_tta": args.tta,
        "thresholds_path": os.path.abspath(args.thresholds) if args.thresholds else None,
        "n_samples": result["n_samples"],
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\n→ Results saved to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())