"""
Ensemble library: load multiple trained models, average their class
probability distributions, and decode once on the averaged probabilities.

Used by `src.ensemble.evaluate` (the CLI entry point). Also re-exported
via `src.ensemble.__init__` for callers who want to invoke the pipeline
from Python without going through argparse.

The `load_model_from_checkpoint` and `default_seg_dir_for_manifest`
helpers are shared with single-model eval via `src.training.checkpoint_io`
(the legacy `scripts/ensemble_evaluate.py` used to hold its own copies;
they were extracted to the shared module and re-imported here).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.eval.metrics import compute_all_metrics, compute_confusion_matrix, format_confusion_matrix_str
from src.models.corn import corn_predict_probas
from src.training.checkpoint_io import (
    default_seg_dir_for_manifest,
    load_model_from_checkpoint,
)

NUM_CLASSES = 5


def ensemble_predict_probas(models, images, use_tta: bool = False):
    """Average CLASS PROBABILITY distributions across all models, then predict.

    Each model independently produces its own full class distribution:
      CORN: logits → sigmoid → cumprod → P(y=k) via corn_predict_probas()
      CE:   logits → softmax

    Averaging complete class distributions across models preserves each
    model's internal rank-consistency; averaging the conditional
    probabilities P(y>k | y>k-1) directly would break CORN's monotonicity
    guarantee, so we decode per-model first.
    """
    all_class_probas = []
    for model in models:
        if use_tta:
            from src.eval.tta import tta_forward
            avg_probas = tta_forward(model, images, "corn")
            class_probs = avg_probas
        else:
            logits = model(images)
            class_probs = corn_predict_probas(logits)
        all_class_probas.append(class_probs)

    avg_class_probs = torch.stack(all_class_probas).mean(dim=0)
    avg_class_probs = avg_class_probs.clamp(min=0.0)
    avg_class_probs = avg_class_probs / avg_class_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    preds = avg_class_probs.argmax(dim=1)
    return preds, avg_class_probs


def evaluate_ensemble(
    checkpoint_paths: list[str],
    manifest_csv: str,
    batch_size: int = 32,
    device: str | None = None,
    use_ema: bool = True,
    use_tta: bool = False,
    seg_dir: str | None = None,
    img_size: int | None = None,
) -> dict[str, Any]:
    """Load N checkpoints, ensemble-predict on a manifest, return metrics + report.

    Returns:
        dict with keys: metrics, confusion_matrix, confusion_matrix_str,
        n_samples, model_archs, checkpoint_paths.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    models, configs = [], []
    for path in checkpoint_paths:
        model, config = load_model_from_checkpoint(path, device, use_ema=use_ema)
        models.append(model)
        configs.append(config)
        arch = config.get("model", {}).get("arch", "convnext_tiny")
        print(f"  Loaded: {os.path.basename(path)} ({arch})")

    # If any model has in_chans=4 (Option A segmentation), require seg_dir.
    needs_seg = any(c.get("model", {}).get("in_chans", 3) > 3 for c in configs)
    if needs_seg and seg_dir is None:
        seg_dir = default_seg_dir_for_manifest(manifest_csv)
        if seg_dir is None:
            raise ValueError(
                "Model was trained with in_chans=4 (segmentation) but no seg_dir "
                "was provided AND no default candidate exists. Pass --seg-dir."
            )
        print(f"  Using default seg_dir: {seg_dir}")

    if img_size is None and configs:
        sizes = [c.get("model", {}).get("img_size") for c in configs
                 if c.get("model", {}).get("img_size") is not None]
        if sizes:
            img_size = max(sizes) if len(set(sizes)) > 1 else sizes[0]
            print(f"  Auto-detected img_size={img_size} from checkpoint(s)")

    dataset = DRDataset(
        manifest_csv,
        transform=build_eval_transforms(img_size=img_size),
        seg_dir=seg_dir,
    )
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


# `os` is imported lazily by callers (we keep the top of the file clean).
import os  # noqa: E402