"""
Standalone evaluation script per plan Section 7.

Loads a self-contained checkpoint (config + class_names embedded, per
checkpoint.py) and runs it against a given manifest CSV (val or test split
of any source), reporting QWK + full metric suite + confusion matrix.

Usage:
    python -m src.eval.evaluate --checkpoint saved/checkpoints/run1_best.pt \
        --manifest data/splits/aptos_test.csv --norm-stats data/processed/aptos_norm_stats.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import build_eval_transforms
from src.data.datasets import DRDataset
from src.eval.metrics import compute_all_metrics, compute_confusion_matrix, format_confusion_matrix_str
from src.models.corn import corn_predict, corn_predict_probas
from src.models.dr_model import DRGradingModel
from src.training.checkpoint import load_checkpoint


def evaluate_checkpoint(checkpoint_path: str, manifest_csv: str, norm_stats_path: str,
                         batch_size: int = 32, device: str | None = None,
                         use_ema: bool = True, use_tta: bool = False) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    class_names = checkpoint.get("class_names", None)

    model_cfg = config.get("model", {}) if config else {}
    # CRITICAL: output_mode must match how the checkpoint was trained.
    # loss_type="corn" -> CORN-shaped head (num_thresholds=4 raw values).
    # loss_type="ce"   -> softmax-shaped head (num_thresholds+1=5 raw values).
    # Getting this wrong either crashes on state_dict shape mismatch or,
    # worse, loads "successfully" with garbage predictions if shapes
    # happen to coincide. baseline.yaml and ablation_ce_weighted_cbam.yaml
    # both use loss_type="ce", so they MUST NOT go through corn_predict.
    loss_type = config.get("loss_type", "corn") if config else "corn"
    output_mode = "corn" if loss_type == "corn" else "softmax"

    model = DRGradingModel(
        pretrained=False,  # weights come from checkpoint, not ImageNet re-download
        use_cbam=model_cfg.get("use_cbam", True),
        cbam_num_stages=model_cfg.get("cbam_num_stages", 2),
        num_thresholds=model_cfg.get("num_thresholds", 4),
        head_hidden_dim=model_cfg.get("head_hidden_dim", 512),
        dropout=model_cfg.get("dropout", 0.3),
        output_mode=output_mode,
    )

    # torch.compile wraps param names with '_orig_mod.' prefix during training.
    # Strip it so weights load into the non-compiled eval model.
    def _strip(sd):
        return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    if use_ema and checkpoint.get("ema_state_dict") is not None:
        model.load_state_dict(_strip(checkpoint["ema_state_dict"]))
    else:
        model.load_state_dict(_strip(checkpoint["model_state_dict"]))

    model.to(device)
    model.eval()

    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    dataset = DRDataset(manifest_csv, norm_stats, transform=build_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            if use_tta:
                from src.eval.tta import tta_forward, tta_predict, tta_predict_probas
                avg_probas = tta_forward(model, images, output_mode)
                preds = tta_predict(avg_probas, output_mode)
                probas = tta_predict_probas(avg_probas, output_mode)
            else:
                logits = model(images)
                if output_mode == "corn":
                    preds = corn_predict(logits)
                    probas = corn_predict_probas(logits)
                else:
                    preds = logits.argmax(dim=1)
                    probas = torch.softmax(logits, dim=1)

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
        "class_names": class_names,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DR grading checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA shadow.")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint, args.manifest, args.norm_stats,
        batch_size=args.batch_size, use_ema=not args.no_ema, use_tta=args.tta,
    )

    print(f"\nEvaluated {result['n_samples']} samples from {args.manifest}\n")
    print(f"QWK (primary metric): {result['metrics']['qwk']:.4f}")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Macro F1: {result['metrics']['macro_f1']:.4f}")
    if "macro_auc_roc" in result["metrics"]:
        print(f"Macro AUC-ROC: {result['metrics']['macro_auc_roc']:.4f}")
    print("\nConfusion matrix:")
    print(result["confusion_matrix_str"])


if __name__ == "__main__":
    main()
