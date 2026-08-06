"""
Threshold calibration entry point — learn per-class decision thresholds on a
calibration set, then evaluate the gain on test sets.

This is post-training, no-fine-tune threshold calibration:
  1. Run trained model on calibration manifest (default: a LODO val set)
     → collect (y_proba, y_true) pairs.
  2. Optimize 4 thresholds (between 5 classes) to maximize QWK.
  3. Save calibrated thresholds to JSON.
  4. Report QWK with argmax vs. calibrated thresholds on the calibration set
     and on the held-out test sets.

Usage:
    python -m src.eval.calibrate_thresholds \
        --checkpoint saved/checkpoints/lodo_eyepacs_convnext_tiny_best.pt \
        --calib-manifest data/splits/lodo_eyepacs_val.csv \
        --thresholds-out saved/checkpoints/calibrated_thresholds.json

Threshold calibration changes only the decision rule (argmax vs.
threshold-aware argmax); it does NOT modify the model or its weights.
It is COMPOSABLE with `src.eval.probability_calibration` (temperature
scaling on the model's probabilities). Apply temperature scaling first
for well-calibrated probabilities, then threshold calibration to absorb
cross-domain prior shift.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.eval.metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    format_confusion_matrix_str,
)
from src.eval.threshold_calibration import (
    NUM_CLASSES,
    describe_thresholds,
    optimize_thresholds,
    predict_with_thresholds,
    save_thresholds,
)
from src.models.corn import corn_predict_probas
from src.training.checkpoint_io import load_model_from_checkpoint


def collect_probas(
    model, manifest_csv: str, batch_size: int, device: str,
    use_tta: bool = False, img_size: int | None = None,
    seg_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on manifest, return (y_proba [N,5], y_true [N])."""
    ds = DRDataset(
        manifest_csv,
        transform=build_eval_transforms(img_size=img_size),
        seg_dir=seg_dir,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_probas, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in dl:
            images = images.to(device, non_blocking=True)
            if use_tta:
                from src.eval.tta import tta_forward
                probas = tta_forward(model, images, "corn")
            else:
                logits = model(images)
                probas = corn_predict_probas(logits)
            all_probas.append(probas.cpu().numpy())
            all_labels.append(labels.numpy())

    y_proba = np.concatenate(all_probas)
    y_true = np.concatenate(all_labels)
    return y_proba, y_true


def evaluate_with_thresholds(
    y_proba: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray | None,
) -> dict:
    """Return full metrics dict using argmax (None) or calibrated thresholds."""
    if thresholds is None:
        y_pred = y_proba.argmax(axis=1)
    else:
        y_pred = predict_with_thresholds(y_proba, thresholds)
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)
    return {"metrics": metrics, "confusion_matrix": cm}


def fmt_section(title: str) -> str:
    return f"\n{'─' * 60}\n  {title}\n{'─' * 60}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate decision thresholds on a held-out set.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--calib-manifest", default="data/splits/lodo_eyepacs_val.csv",
        help="Held-out set used to fit thresholds.",
    )
    parser.add_argument(
        "--thresholds-out",
        default="saved/checkpoints/calibrated_thresholds.json",
    )
    parser.add_argument(
        "--test-manifests", nargs="+",
        default=[
            "data/splits/lodo_aptos_test_full.csv",
            "data/splits/lodo_messidor2_test_full.csv",
        ],
        help="Manifests to evaluate gain on.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--seg-dir", default=None)
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--log-dir", default="saved/logs")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(fmt_section(f"Loading checkpoint: {os.path.basename(args.checkpoint)}"))
    t0 = time.time()
    model, config = load_model_from_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Arch: {config.get('model', {}).get('arch', 'unknown')}")

    ckpt_in_chans = config.get("model", {}).get("in_chans", 3)
    if img_size := (args.img_size or config.get("model", {}).get("img_size")):
        print(f"  img_size: {img_size}, in_chans: {ckpt_in_chans}")

    print(fmt_section(f"Step 1: collecting probabilities on calibration set ({args.calib_manifest})"))
    t0 = time.time()
    y_proba_cal, y_true_cal = collect_probas(
        model, args.calib_manifest, args.batch_size, device,
        use_tta=args.tta, img_size=args.img_size, seg_dir=args.seg_dir,
    )
    print(f"  Collected {len(y_true_cal)} samples in {time.time() - t0:.1f}s")
    print(f"  Class distribution: {np.bincount(y_true_cal, minlength=NUM_CLASSES).tolist()}")

    print(fmt_section("Step 2: optimizing thresholds (Nelder-Mead on QWK)"))
    t0 = time.time()
    thresholds = optimize_thresholds(y_proba_cal, y_true_cal, maxiter=args.maxiter)
    print(f"  Optimized in {time.time() - t0:.1f}s")
    print(describe_thresholds(thresholds))

    save_thresholds(thresholds, args.thresholds_out, extra_metadata={
        "checkpoint": os.path.abspath(args.checkpoint),
        "calib_manifest": os.path.abspath(args.calib_manifest),
        "n_calib_samples": int(len(y_true_cal)),
        "use_ema": not args.no_ema,
        "use_tta": args.tta,
        "maxiter": args.maxiter,
        "calibrated_at": datetime.datetime.now().isoformat(),
    })
    print(f"\n  → Saved to {args.thresholds_out}")

    print(fmt_section("Step 3a: in-domain evaluation (calibration set — sanity check)"))
    res_argmax = evaluate_with_thresholds(y_proba_cal, y_true_cal, None)
    res_calib = evaluate_with_thresholds(y_proba_cal, y_true_cal, thresholds)
    print(f"  Argmax  QWK: {res_argmax['metrics']['qwk']:.4f}  |  "
          f"Macro F1: {res_argmax['metrics']['macro_f1']:.4f}")
    print(f"  Thresh  QWK: {res_calib['metrics']['qwk']:.4f}  |  "
          f"Macro F1: {res_calib['metrics']['macro_f1']:.4f}")
    print(f"  In-domain gain (calibration set): "
          f"{res_calib['metrics']['qwk'] - res_argmax['metrics']['qwk']:+.4f}")
    print("\n  Argmax confusion matrix:")
    print(format_confusion_matrix_str(res_argmax["confusion_matrix"]))

    print(fmt_section("Step 3b: test-set evaluation (measuring transfer)"))
    summary_rows = []
    for manifest in args.test_manifests:
        if not os.path.exists(manifest):
            print(f"\n  Skipping missing manifest: {manifest}")
            continue
        print(f"\n  ── {os.path.basename(manifest)} ──")
        y_proba_test, y_true_test = collect_probas(
            model, manifest, args.batch_size, device,
            use_tta=args.tta, img_size=args.img_size, seg_dir=args.seg_dir,
        )
        ra = evaluate_with_thresholds(y_proba_test, y_true_test, None)
        rc = evaluate_with_thresholds(y_proba_test, y_true_test, thresholds)

        qwk_arg = ra["metrics"]["qwk"]
        qwk_cal = rc["metrics"]["qwk"]
        gain = qwk_cal - qwk_arg

        print(f"    Argmax  QWK: {qwk_arg:.4f}  |  Macro F1: {ra['metrics']['macro_f1']:.4f}")
        print(f"    Thresh  QWK: {qwk_cal:.4f}  |  Macro F1: {rc['metrics']['macro_f1']:.4f}")
        print(f"    >>> GAIN: {gain:+.4f} QWK")
        if gain > 0:
            print("    ✓ Calibration helps this set.")
        elif gain > -0.002:
            print("    ~ No meaningful change.")
        else:
            print("    ✗ Calibration HURT this set — consider not applying it here.")

        per_class_recall_argmax = ra["metrics"].get("recall_Mild", float("nan"))
        per_class_recall_thresh = rc["metrics"].get("recall_Mild", float("nan"))
        if not np.isnan(per_class_recall_argmax):
            print(f"    Mild recall: argmax={per_class_recall_argmax:.3f} → "
                  f"calibrated={per_class_recall_thresh:.3f}")

        summary_rows.append({
            "manifest": os.path.abspath(manifest),
            "n_samples": int(len(y_true_test)),
            "argmax_qwk": round(qwk_arg, 6),
            "thresh_qwk": round(qwk_cal, 6),
            "gain_qwk": round(gain, 6),
            "argmax_macro_f1": round(ra["metrics"]["macro_f1"], 6),
            "thresh_macro_f1": round(rc["metrics"]["macro_f1"], 6),
        })

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "calibration_runs.jsonl")
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": os.path.abspath(args.checkpoint),
        "calib_manifest": os.path.abspath(args.calib_manifest),
        "thresholds_path": os.path.abspath(args.thresholds_out),
        "thresholds": [float(t) for t in thresholds],
        "use_ema": not args.no_ema,
        "use_tta": args.tta,
        "in_domain_qwk_argmax": round(res_argmax["metrics"]["qwk"], 6),
        "in_domain_qwk_thresh": round(res_calib["metrics"]["qwk"], 6),
        "test_results": summary_rows,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\n  → Run log saved to {log_path}")

    print(fmt_section("DONE"))
    print("  Calibrated thresholds applied to:")
    print("    python -m src.eval.evaluate ... --thresholds <json>")
    print("    python -m src.ensemble.evaluate ... --thresholds <json>")
    print("  Or call predict_with_thresholds(y_proba, thresholds) in your own eval.")
    return 0


if __name__ == "__main__":
    sys.exit(main())