"""
Tests for the codebase restructure (Phase 0-3 cleanup):
  - `src/data/_00_manifests.py`
  - `src/data/_01_preprocess.py`
  - `src/data/_02_splits.py`
  - `src/data/_03_segmentation.py`
  - `src/data/_04_preprocessing_ops.py`
  - `src/data/_05_dataset.py`
  - `src/training/checkpoint_io.py`
  - `src/training/orchestrate.py`
  - `src/data/lodo_configs.py` (5×3 matrix generator)

These tests cover:
  - The LODO module produces all 4 expected CSVs per fold (3 folds total).
  - Source separation: train pool does NOT include held-out images.
  - Matched test is downsampled (eyepacs/eyepacs case) to Messidor-2 size.
  - 5-class validation raises on out-of-range labels.
  - All 15 LODO configs load and have the required keys.
  - Each config's arch is registered in `src.models.backbone.ARCH_REGISTRY`.
  - The `_00_` re-export pattern works.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Ensure repo root on sys.path (helpful when running pytest from anywhere)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# src/data/_00_manifests.py
# ─────────────────────────────────────────────────────────────────────────────

def test_manifests_module_has_builders():
    from src.data._00_manifests import BUILDERS
    assert set(BUILDERS.keys()) == {"eyepacs", "aptos", "messidor2"}


def test_manifests_validates_5class(tmp_path):
    """_validate_5class raises on any label outside [0, 4]."""
    from src.data._00_manifests import _validate_5class
    df = pd.DataFrame({"image_path": ["a.png", "b.png"], "label": [0, 5]})  # 5 = invalid
    with pytest.raises(ValueError, match="outside 5-class range"):
        _validate_5class(df, "test_source")


def test_aptos_manifest_from_synthetic_csv(tmp_path):
    """build_aptos_manifest works on a synthetic train.csv + train_images dir."""
    from src.data._00_manifests import build_aptos_manifest

    # Build a synthetic APTOS layout under tmp_path.
    raw_dir = tmp_path / "raw" / "aptos"
    images_dir = raw_dir / "train_images"
    images_dir.mkdir(parents=True)
    (raw_dir / "train.csv").write_text(
        "id_code,diagnosis\n"
        "img001,0\n"
        "img002,1\n"
        "img003,4\n"
    )
    # Create dummy image files (we don't actually read them in build_aptos_manifest).
    for stem in ("img001", "img002", "img003"):
        (images_dir / f"{stem}.png").write_bytes(b"")

    out_csv = tmp_path / "aptos_manifest.csv"
    result = build_aptos_manifest(
        raw_dir=str(raw_dir),
        labels_csv=str(raw_dir / "train.csv"),
        out_path=str(out_csv),
    )
    assert result == str(out_csv)
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["image_path", "label"]
    assert set(df["label"].unique()).issubset({0, 1, 2, 3, 4})


# ─────────────────────────────────────────────────────────────────────────────
# src/data/_02_splits.py (LODO)
# ─────────────────────────────────────────────────────────────────────────────

def test_lodo_split_source_separation(tmp_path, monkeypatch):
    """LODO fold's train pool must NOT include any images from the held-out source."""
    from src.data import lodo_split
    from src.data._02_splits import build_lodo_split

    # lodo_split._manifest_path is hardcoded to data/processed/{source}_manifest.csv.
    # Patch it to point at our tmp_path layout.
    def fake_manifest_path(source: str) -> str:
        return str(tmp_path / "data" / "processed" / f"{source}_manifest.csv")

    monkeypatch.setattr(lodo_split, "_manifest_path", fake_manifest_path)

    # Build three synthetic manifests large enough for stratified splits
    # (each class needs ≥1 sample in both train and val).
    (tmp_path / "data" / "processed").mkdir(parents=True)
    n_per_source = 200  # >= 5 classes * 2 (one each for train/val) comfortably
    for src in ("eyepacs", "aptos", "messidor2"):
        rows = []
        for cls in range(5):
            for i in range(n_per_source // 5):
                rows.append({"image_path": f"/fake/{src}/img_{cls}_{i}.png", "label": cls})
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / "data" / "processed" / f"{src}_manifest.csv", index=False)

    paths = build_lodo_split(
        holdout_source="eyepacs",
        out_dir=str(tmp_path / "splits"),
    )

    # Files written
    for key in ("train", "val", "test_full", "test_matched"):
        assert os.path.exists(paths[key]), f"missing {key} CSV"

    # Train pool should only contain APTOS + Messidor-2 rows.
    train_df = pd.read_csv(paths["train"])
    train_sources = set(train_df["source"].unique())
    assert "eyepacs" not in train_sources, "EyepACS leaked into eyepacs-holdout train pool"
    assert train_sources.issubset({"aptos", "messidor2"})

    # Test_full should be 100% eyepacs
    test_full_df = pd.read_csv(paths["test_full"])
    assert (test_full_df["source"] == "eyepacs").all()
    assert len(test_full_df) == n_per_source

    # Test_matched is downsampled (≤ 227)
    test_matched_df = pd.read_csv(paths["test_matched"])
    assert (test_matched_df["source"] == "eyepacs").all()
    assert len(test_matched_df) <= n_per_source


def test_lodo_split_three_folds(tmp_path, monkeypatch):
    """build_all_lodo_folds produces 3 folds × 4 CSVs = 12 files."""
    from src.data import lodo_split
    from src.data._02_splits import build_all_lodo_folds

    monkeypatch.setattr(
        lodo_split, "_manifest_path",
        lambda source: str(tmp_path / "data" / "processed" / f"{source}_manifest.csv"),
    )

    (tmp_path / "data" / "processed").mkdir(parents=True)
    n_per_source = 200
    for src in ("eyepacs", "aptos", "messidor2"):
        rows = []
        for cls in range(5):
            for i in range(n_per_source // 5):
                rows.append({"image_path": f"/fake/{src}/img_{cls}_{i}.png", "label": cls})
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / "data" / "processed" / f"{src}_manifest.csv", index=False)

    folds = build_all_lodo_folds(out_dir=str(tmp_path / "splits"))
    assert set(folds.keys()) == {"eyepacs", "aptos", "messidor2"}
    for holdout, paths in folds.items():
        for key in ("train", "val", "test_full", "test_matched"):
            assert os.path.exists(paths[key]), f"{holdout}/{key} missing"


def test_lodo_split_rejects_unknown_source(tmp_path):
    from src.data._02_splits import build_lodo_split
    with pytest.raises(ValueError, match="not in sources"):
        build_lodo_split(holdout_source="kaggle", out_dir=str(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
# src/data/_04_preprocessing_ops.py + _01_preprocess.py
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocessing_ops_re_exports():
    from src.data._04_preprocessing_ops import (
        crop_pad_resize,
        color_correction_pipeline,
        apply_anisotropic_filter,
        normalize_image,
    )
    assert callable(crop_pad_resize)
    assert callable(color_correction_pipeline)
    assert callable(apply_anisotropic_filter)
    assert callable(normalize_image)


# ─────────────────────────────────────────────────────────────────────────────
# src/data/_05_dataset.py + dataset re-export
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_re_exports():
    from src.data._05_dataset import (
        DRDataset,
        NUM_CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )
    assert NUM_CLASSES == 5
    assert len(IMAGENET_MEAN) == 3
    assert len(IMAGENET_STD) == 3
    # DRDataset should be the same class as the legacy import path
    from src.data.datasets import DRDataset as LegacyDRDataset
    assert DRDataset is LegacyDRDataset


# ─────────────────────────────────────────────────────────────────────────────
# Configs: split by axis — configs/models/<arch>.yaml + configs/lodo/<holdout>.yaml
# ─────────────────────────────────────────────────────────────────────────────

ARCHS = [
    "convnext_tiny",
    "convnext_small",
    "deit3_small_patch16_384",
    "maxvit_tiny_tf_384",
    "swinv2_base_window12to24_192to384",
]
HOLDOUTS = ["eyepacs", "aptos", "messidor2"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("arch", ARCHS)
def test_model_config_loads_and_has_required_keys(arch):
    """Each configs/models/<arch>.yaml must have all the model-axis fields."""
    import yaml
    from src.models.backbone import ARCH_REGISTRY

    cfg_path = os.path.join(REPO_ROOT, "configs", "models", f"{arch}.yaml")
    assert os.path.exists(cfg_path), f"missing model config {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Required top-level keys (the model axis)
    for k in ("arch", "model", "phases", "ema_decay", "loss_type",
              "use_class_weighting", "use_mixup"):
        assert k in cfg, f"missing top-level key {k!r} in {cfg_path}"

    # Arch registry consistency
    assert cfg["arch"] == arch
    assert arch in ARCH_REGISTRY, f"arch {arch!r} not in ARCH_REGISTRY"
    assert cfg["model"]["arch"] == arch
    assert cfg["model"]["channels_last"] == ARCH_REGISTRY[arch].channels_last

    # Phases must have phase1 (frozen warmup) and phase2 (full fine-tune)
    assert "phase1_frozen" in cfg["phases"]
    assert "phase2_full_training" in cfg["phases"]
    # Phase 1 freezes the backbone
    assert cfg["phases"]["phase1_frozen"]["freeze_backbone"] is True
    # Phase 2 unfreezes it
    assert cfg["phases"]["phase2_full_training"]["freeze_backbone"] is False
    # Phase 2 must have batch_size (per-arch hyperparams live here)
    assert "batch_size" in cfg["phases"]["phase2_full_training"]


@pytest.mark.parametrize("holdout", HOLDOUTS)
def test_lodo_config_loads_and_has_required_keys(holdout):
    """Each configs/lodo/<holdout>.yaml must have only the data-axis fields."""
    import yaml

    cfg_path = os.path.join(REPO_ROOT, "configs", "lodo", f"{holdout}.yaml")
    assert os.path.exists(cfg_path), f"missing LODO config {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Required LODO-axis keys
    for k in ("holdout_source", "manifest_train", "manifest_val"):
        assert k in cfg, f"missing LODO key {k!r} in {cfg_path}"

    # Manifest paths are correctly derived from holdout name
    assert cfg["manifest_train"] == f"data/splits/lodo_{holdout}_train.csv"
    assert cfg["manifest_val"] == f"data/splits/lodo_{holdout}_val.csv"

    # The LODO file must NOT contain model/phases — those belong on the model axis.
    # This guards against future regressions where someone adds `arch:` to the LODO file.
    for forbidden in ("arch", "model", "phases", "run_name"):
        assert forbidden not in cfg, \
            f"LODO file {cfg_path} should not contain {forbidden!r} (model-axis field)"


@pytest.mark.parametrize("holdout", HOLDOUTS)
@pytest.mark.parametrize("arch", ARCHS)
def test_model_x_lodo_merge_produces_full_config(arch, holdout):
    """Merging configs/models/<arch>.yaml with configs/lodo/<holdout>.yaml
    produces the same shape the legacy monolithic lodo_<holdout>_<arch>.yaml
    had: run_name, holdout_source, arch, model, phases with manifest paths
    injected into both phases."""
    import yaml
    from src.config_merge import merge_configs

    model_cfg = yaml.safe_load(
        open(os.path.join(REPO_ROOT, "configs", "models", f"{arch}.yaml"))
    )
    lodo_cfg = yaml.safe_load(
        open(os.path.join(REPO_ROOT, "configs", "lodo", f"{holdout}.yaml"))
    )
    merged = merge_configs(model_cfg, lodo_cfg)

    # All keys the trainer needs
    for k in ("run_name", "holdout_source", "arch", "model", "phases",
              "ema_decay", "loss_type", "use_class_weighting", "use_mixup"):
        assert k in merged, f"missing key {k!r} in merged config"

    # Naming is derived deterministically
    assert merged["run_name"] == f"lodo_{holdout}_{arch}"
    assert merged["holdout_source"] == holdout
    assert merged["arch"] == arch

    # Both phases must point at the correct LODO manifests
    expected_train = f"data/splits/lodo_{holdout}_train.csv"
    expected_val = f"data/splits/lodo_{holdout}_val.csv"
    assert merged["phases"]["phase1_frozen"]["manifest_train"] == expected_train
    assert merged["phases"]["phase1_frozen"]["manifest_val"] == expected_val
    assert merged["phases"]["phase2_full_training"]["manifest_train"] == expected_train
    assert merged["phases"]["phase2_full_training"]["manifest_val"] == expected_val


def test_generate_lodo_configs_count():
    """There must be exactly 5 model yamls and 3 LODO yamls (8 total) on disk."""
    import glob
    model_cfgs = glob.glob(os.path.join(REPO_ROOT, "configs", "models", "*.yaml"))
    lodo_cfgs = glob.glob(os.path.join(REPO_ROOT, "configs", "lodo", "*.yaml"))
    assert len(model_cfgs) == 5, f"expected 5 model configs, got {len(model_cfgs)}"
    assert len(lodo_cfgs) == 3, f"expected 3 LODO configs, got {len(lodo_cfgs)}"
    # Sanity: no monolithic lodo_<holdout>_<arch>.yaml files leak back in
    leaked = glob.glob(os.path.join(REPO_ROOT, "configs", "lodo_*_*.yaml"))
    assert leaked == [], f"legacy monolithic yamls leaked back in: {leaked}"


# ─────────────────────────────────────────────────────────────────────────────
# src/training/checkpoint_io.py + orchestrate.py
# ─────────────────────────────────────────────────────────────────────────────

def test_checkpoint_io_strip_compile_prefix():
    from src.training.checkpoint_io import _strip_compile_prefix
    sd = {"_orig_mod.layer.weight": 1.0, "layer.bias": 2.0}
    cleaned = _strip_compile_prefix(sd)
    assert "_orig_mod.layer.weight" not in cleaned
    assert "layer.weight" in cleaned
    assert "layer.bias" in cleaned


def test_orchestrate_discover_phase_names():
    from src.training.orchestrate import _discover_phase_names
    cfg = {
        "phases": {
            "phase1_frozen": {"num_epochs": 5},
            "phase2_full_training": {"num_epochs": 35},
            # phase3 deliberately omitted
        }
    }
    names = _discover_phase_names(cfg)
    assert names == ["phase1_frozen", "phase2_full_training"]


def test_orchestrate_set_seed_is_deterministic():
    import numpy as np
    import torch
    from src.training.orchestrate import set_seed

    set_seed(42)
    a = (torch.rand(3).tolist(), np.random.rand(3).tolist())
    set_seed(42)
    b = (torch.rand(3).tolist(), np.random.rand(3).tolist())
    assert a == b


# ─────────────────────────────────────────────────────────────────────────────
# src/data/_03_segmentation.py — light smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_segmentation_module_imports():
    """Just verify the segmentation library functions exist (no GPU required)."""
    from src.data._03_segmentation import (
        finetune_unet,
        load_unet,
        infer_masks_for_source,
        infer_masks_for_sources,
        DEVICE,
    )
    assert callable(finetune_unet)
    assert callable(load_unet)
    assert callable(infer_masks_for_source)
    assert callable(infer_masks_for_sources)
    assert DEVICE in ("cuda", "cpu")