"""
Ensemble evaluation: load multiple CORN-headed models trained via the
same 3-phase pipeline, average their per-threshold sigmoid probabilities,
and decode once on the averaged probabilities.

Key insight: averaging at the CORN *probability* level (before rank-
consistent decode) preserves ordinal confidence information that majority
voting on final class labels would throw away. This is the standard way
to ensemble ordinal-regression models.

Usage:
    python scripts/ensemble_evaluate.py \
        --checkpoints saved/checkpoints/full_method_best.pt \
                      saved/checkpoints/ensemble_effnetv2_best.pt \
        --manifest data/splits/aptos_test.csv \
        --norm-stats data/processed/aptos_norm_stats.json \
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
from src.models.corn import corn_predict_probas
from src.models.dr_model import DRGradingModel
from src.training.checkpoint import load_checkpoint

NUM_CLASSES = 5


def _strip_compile_prefix(sd: dict) -> dict:
    """Strip torch.compile's _orig_mod. prefix from state_dict keys."""
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def load_model_from_checkpoint(checkpoint_path: str, device: str, use_ema: bool = True):
    """Load a single model from a self-contained checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_cfg = config.get("model", {})
    loss_type = config.get("loss_type", "corn")
    output_mode = "corn" if loss_type == "corn" else "softmax"

    model = DRGradingModel(
        pretrained=False,
        use_cbam=model_cfg.get("use_cbam", True),
        cbam_num_stages=model_cfg.get("cbam_num_stages", 2),
        num_thresholds=model_cfg.get("num_thresholds", 4),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.3),
        output_mode=output_mode,
        arch=model_cfg.get("arch", "convnext_tiny"),
    )

    if use_ema and checkpoint.get("ema_state_dict") is not None:
        model.load_state_dict(_strip_compile_prefix(checkpoint["ema_state_dict"]))
    else:
        model.load_state_dict(_strip_compile_prefix(checkpoint["model_state_dict"]))

    model.to(device)
    model.eval()
    return model, config


def ensemble_predict_probas(models, images, use_tta: bool = False):
    """Average CORN sigmoid probabilities across all models, then decode.

    Each model produces per-threshold logits → sigmoid → P(y > k).
    We average these probabilities across models, then convert to class
    probabilities via the standard CORN decode:
        P(y = k) = P(y > k-1) - P(y > k)
    """
    all_probas = []

    for model in models:
        if use_tta:
            from src.eval.tta import tta_forward
            # tta_forward returns averaged logits across augmented views
            avg_logits = tta_forward(model, images, "corn")
            # Convert logits to per-threshold probabilities
            threshold_probs = torch.sigmoid(avg_logits)
        else:
            logits = model(images)
            threshold_probs = torch.sigmoid(logits)

        all_probas.append(threshold_probs)

    # Average per-threshold probabilities across models
    avg_threshold_probs = torch.stack(all_probas).mean(dim=0)  # (B, num_thresholds)

    # Decode averaged thresholds to class probabilities (CORN rank-consistent)
    # P(y=0) = 1 - P(y>0)
    # P(y=k) = P(y>k-1) - P(y>k)  for 0 < k < K
    # P(y=K) = P(y>K-1)
    num_thresholds = avg_threshold_probs.shape[1]
    num_classes = num_thresholds + 1

    class_probs = torch.zeros(avg_threshold_probs.shape[0], num_classes,
                              device=avg_threshold_probs.device)
    class_probs[:, 0] = 1.0 - avg_threshold_probs[:, 0]
    for k in range(1, num_thresholds):
        class_probs[:, k] = avg_threshold_probs[:, k - 1] - avg_threshold_probs[:, k]
    class_probs[:, num_classes - 1] = avg_threshold_probs[:, num_thresholds - 1]

    # Clamp to valid range (floating point can cause tiny negatives)
    class_probs = class_probs.clamp(min=0.0)
    # Re-normalize
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)

    preds = class_probs.argmax(dim=1)
    return preds, class_probs


def evaluate_ensemble(checkpoint_paths: list[str], manifest_csv: str,
                      norm_stats_path: str, batch_size: int = 32,
                      device: str | None = None, use_ema: bool = True,
                      use_tta: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load all models
    models = []
    configs = []
    for path in checkpoint_paths:
        model, config = load_model_from_checkpoint(path, device, use_ema=use_ema)
        models.append(model)
        configs.append(config)
        arch = config.get("model", {}).get("arch", "convnext_tiny")
        print(f"  Loaded: {os.path.basename(path)} ({arch})")

    # Build dataloader
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    dataset = DRDataset(manifest_csv, norm_stats, transform=build_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            preds, probas = ensemble_predict_probas(models, images, use_tta=use_tta)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probas.append(probas.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probas)

    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    cm = compute_confusion_matrix(y_true, y_pred)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "confusion_matrix_str": format_confusion_matrix_str(cm),
        "n_samples": len(y_true),
        "model_archs": [c.get("model", {}).get("arch", "convnext_tiny") for c in configs],
        "checkpoint_paths": checkpoint_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation of multiple DR models.")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to checkpoint files to ensemble")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--log-dir", default="saved/logs")
    args = parser.parse_args()

    print(f"\nEnsemble of {len(args.checkpoints)} models:")
    result = evaluate_ensemble(
        args.checkpoints, args.manifest, args.norm_stats,
        batch_size=args.batch_size, use_ema=not args.no_ema, use_tta=args.tta,
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

    # Save to log
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


if __name__ == "__main__":
    main()
