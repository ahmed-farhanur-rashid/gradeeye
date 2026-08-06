"""
checkpoint_io: load a trained DRGradingModel from a self-contained checkpoint.

Used by both `src.eval.evaluate` and `src.ensemble.ensemble`. Centralizing
here avoids the original duplication where both had their own copy of
the same loader (and got out of sync).

The self-contained checkpoint format (`src/training/checkpoint.py:save_checkpoint`)
stores:
  - model_state_dict
  - config  (full YAML that produced the model)
  - ema_state_dict (optional)
  - class_names, best_metric, phase_name
so we don't need the original YAML to load a model.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from src.models.dr_model import DRGradingModel
from src.training.checkpoint import load_checkpoint


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Strip torch.compile's `_orig_mod.` prefix from state_dict keys."""
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    use_ema: bool = True,
) -> tuple[DRGradingModel, dict[str, Any]]:
    """Load a single model + the config dict that produced it.

    Returns:
        (model_on_device, config_dict)
    """
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
        in_chans=model_cfg.get("in_chans", 3),
        img_size=model_cfg.get("img_size", 384),
    )

    if use_ema and checkpoint.get("ema_state_dict") is not None:
        model.load_state_dict(_strip_compile_prefix(checkpoint["ema_state_dict"]))
    else:
        model.load_state_dict(_strip_compile_prefix(checkpoint["model_state_dict"]))

    model.to(device)
    model.eval()
    return model, config


def default_seg_dir_for_manifest(manifest_csv: str) -> str | None:
    """Best-effort guess at the segmentation mask directory for a given test manifest.

    Looks for `data/processed/segmentation/<source>/` where <source> is derived
    from the manifest filename prefix (e.g. `aptos_test.csv` -> `data/processed/segmentation/aptos/`).

    Returns None if no candidate exists.
    """
    base = os.path.basename(manifest_csv)
    source = base.split("_")[0]
    candidate = f"data/processed/segmentation/{source}"
    return candidate if os.path.isdir(candidate) else None