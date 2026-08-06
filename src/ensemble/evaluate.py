"""
CLI entry point: ensemble evaluation of multiple DR models.

Usage:
    python -m src.ensemble.evaluate \
        --checkpoints saved/checkpoints/lodo_eyepacs_convnext_tiny_best.pt \
                      saved/checkpoints/lodo_aptos_convnext_tiny_best.pt \
        --manifest data/splits/lodo_messidor2_test_full.csv \
        --tta
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

from src.ensemble.ensemble import evaluate_ensemble


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation of multiple DR grading models.",
    )
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="Paths to checkpoint files to ensemble.",
    )
    parser.add_argument("--manifest", required=True, help="Manifest CSV to evaluate on.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--log-dir", default="saved/logs")
    parser.add_argument(
        "--img-size", type=int, default=None,
        help="Override input size. Per-checkpoint img_size is the default.",
    )
    parser.add_argument(
        "--seg-dir", default=None,
        help="Segmentation mask dir for Option A (4-ch) models.",
    )
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(f"\nEnsemble of {len(args.checkpoints)} models:")
    result = evaluate_ensemble(
        args.checkpoints,
        args.manifest,
        batch_size=args.batch_size,
        use_ema=not args.no_ema,
        use_tta=args.tta,
        seg_dir=args.seg_dir,
        img_size=args.img_size,
        device=args.device,
    )

    print(f"\nEvaluated {result['n_samples']} samples from {args.manifest}")
    print(f"Models: {', '.join(result['model_archs'])}\n")
    print(f"QWK (primary metric): {result['metrics']['qwk']:.4f}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {result['metrics']['macro_f1']:.4f}")
    if "macro_auc_roc" in result["metrics"]:
        print(f"Macro AUC-ROC: {result['metrics']['macro_auc_roc']:.4f}")
    print("\nConfusion matrix:")
    print(result["confusion_matrix_str"])

    os.makedirs(args.log_dir, exist_ok=True)
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "ensemble",
        "checkpoints": [os.path.abspath(p) for p in args.checkpoints],
        "model_archs": result["model_archs"],
        "manifest": os.path.abspath(args.manifest),
        "use_ema": not args.no_ema,
        "use_tta": args.tta,
        "n_samples": result["n_samples"],
        "metrics": {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in result["metrics"].items()},
        "confusion_matrix": result["confusion_matrix"].tolist(),
    }
    log_path = os.path.join(args.log_dir, "eval_results.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\n→ Results saved to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())