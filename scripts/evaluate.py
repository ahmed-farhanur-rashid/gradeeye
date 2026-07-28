"""
Single-model evaluation on a held-out manifest (e.g. Messidor2 zero-shot).

Usage:
    python scripts/evaluate.py \
        --checkpoint saved/checkpoints/full_method_convnext_best.pt \
        --manifest data/splits/messidor2_test.csv \
        --tta

Wraps the ensemble machinery: with one checkpoint, it produces the same
metrics (QWK, accuracy, macro-F1, AUC-ROC) as a 1-model ensemble.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ensemble_evaluate import evaluate_ensemble
from src.eval.metrics import compute_all_metrics, compute_confusion_matrix, format_confusion_matrix_str


def main():
    parser = argparse.ArgumentParser(description="Single-model evaluation on a held-out manifest.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--manifest", required=True, help="Path to evaluation manifest CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta", action="store_true",
                        help="4-way TTA (orig, hflip, vflip, rot180) per forward pass")
    parser.add_argument("--seg-dir", default=None,
                        help="Segmentation mask dir for Option A (4-ch) models. "
                             "Auto-detected from manifest basename if not given.")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Override input size (e.g. 256 for Swin@256)")
    parser.add_argument("--log-dir", default="saved/logs")
    args = parser.parse_args()

    print(f"Evaluating: {os.path.basename(args.checkpoint)}")
    print(f"On: {args.manifest}")
    print(f"TTA: {args.tta}")

    result = evaluate_ensemble(
        [args.checkpoint], args.manifest, None,
        batch_size=args.batch_size, use_ema=not args.no_ema, use_tta=args.tta,
        seg_dir=args.seg_dir, img_size=args.img_size,
    )

    metrics = result["metrics"]
    print(f"\nEvaluated {result['n_samples']} samples")
    print(f"Model: {result['model_archs'][0]}")
    print(f"\nQWK (primary): {metrics['qwk']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    if "macro_auc_roc" in metrics:
        print(f"Macro AUC-ROC: {metrics['macro_auc_roc']:.4f}")
    print("\nConfusion matrix:")
    print(format_confusion_matrix_str(result["confusion_matrix"]))

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "eval_results.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "single_model",
        "checkpoint": os.path.abspath(args.checkpoint),
        "model_arch": result["model_archs"][0],
        "manifest": os.path.abspath(args.manifest),
        "use_ema": not args.no_ema,
        "use_tta": args.tta,
        "n_samples": result["n_samples"],
        "metrics": {k: round(v, 6) if isinstance(v, float) else v
                    for k, v in metrics.items()},
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\n→ Results saved to {log_path}")


if __name__ == "__main__":
    main()
