"""
Single-model evaluation with calibrated thresholds.

Same as scripts/evaluate.py, but uses pre-fitted per-class thresholds
instead of argmax. If --thresholds isn't given, falls back to argmax.

Usage:
    python scripts/evaluate_calibrated.py \
        --checkpoint saved/checkpoints/full_method_convnext_best.pt \
        --manifest data/splits/messidor2_test.csv \
        --thresholds saved/checkpoints/calibrated_thresholds.json \
        --tta
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import build_eval_transforms
from src.data.datasets import DRDataset
from src.eval.metrics import compute_all_metrics, compute_confusion_matrix, format_confusion_matrix_str
from src.eval.threshold_calibration import (
    load_thresholds,
    predict_with_thresholds,
)
from scripts.ensemble_evaluate import load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate with calibrated thresholds.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--thresholds", default=None,
                        help="Path to calibrated_thresholds.json. If omitted, uses argmax.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--seg-dir", default=None)
    parser.add_argument("--log-dir", default="saved/logs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating: {os.path.basename(args.checkpoint)} on {args.manifest}")
    print(f"TTA: {args.tta}, Thresholds: {args.thresholds or 'argmax (none)'}")

    model, config = load_model_from_checkpoint(args.checkpoint, device,
                                                 use_ema=not args.no_ema)
    arch = config.get("model", {}).get("arch", "unknown")
    img_size = args.img_size or config.get("model", {}).get("img_size")

    dataset = DRDataset(args.manifest, transform=build_eval_transforms(img_size=img_size),
                        seg_dir=args.seg_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    all_probas, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            if args.tta:
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

    if args.thresholds:
        thresholds = load_thresholds(args.thresholds)
        y_pred = predict_with_thresholds(y_proba, thresholds)
        mode = "calibrated"
    else:
        y_pred = y_proba.argmax(axis=1)
        mode = "argmax"

    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)

    print(f"\nEvaluated {len(y_true)} samples ({arch}, {mode})")
    print(f"QWK (primary): {metrics['qwk']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    if "macro_auc_roc" in metrics:
        print(f"Macro AUC-ROC: {metrics['macro_auc_roc']:.4f}")
    print("\nConfusion matrix:")
    print(format_confusion_matrix_str(cm))

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
        "n_samples": int(len(y_true)),
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\n→ Results saved to {log_path}")


if __name__ == "__main__":
    main()