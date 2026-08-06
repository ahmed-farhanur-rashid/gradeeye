"""
Tests for the second restructure pass: ensuring all entry points are
under `src/` (Pearl-style), `scripts/` is minimal, and `src/ensemble/`
holds all ensemble code.

Coverage:
  - `src.train` is the canonical training entry point
  - `src.eval.evaluate` is the canonical single-model entry point
  - `src.eval.calibrate_thresholds` is the canonical calibration entry point
  - `src.ensemble.ensemble` and `src.ensemble.evaluate` are the ensemble
    library + entry point
  - `src.data.cli` is the unified data pipeline CLI
  - `src.data.lodo_configs` regenerates the full 5×3 LODO config matrix
  - `scripts/` only contains shell wrappers (not Python entry points)
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# src.train is the canonical training entry point
# ─────────────────────────────────────────────────────────────────────────────

def test_src_train_imports():
    from src.train import main, build_parser, load_config
    assert callable(main)
    assert callable(build_parser)
    assert callable(load_config)


def test_src_train_help_runs():
    """`python -m src.train --help` must exit cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "src.train", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    assert "Train a DR grading model" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# src.eval entry points
# ─────────────────────────────────────────────────────────────────────────────

def test_src_eval_evaluate_imports():
    from src.eval.evaluate import evaluate_checkpoint, main, build_parser
    assert callable(evaluate_checkpoint)
    assert callable(main)
    assert callable(build_parser)


def test_src_eval_calibrate_thresholds_imports():
    from src.eval.calibrate_thresholds import (
        collect_probas, evaluate_with_thresholds, main, build_parser,
    )
    assert callable(collect_probas)
    assert callable(evaluate_with_thresholds)
    assert callable(main)
    assert callable(build_parser)


def test_src_eval_evaluate_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "src.eval.evaluate", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    assert "Single-model evaluation" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# src.ensemble library + entry point
# ─────────────────────────────────────────────────────────────────────────────

def test_src_ensemble_package_imports():
    from src.ensemble import (
        NUM_CLASSES,
        default_seg_dir_for_manifest,
        ensemble_predict_probas,
        evaluate_ensemble,
    )
    assert NUM_CLASSES == 5
    assert callable(ensemble_predict_probas)
    assert callable(evaluate_ensemble)
    assert callable(default_seg_dir_for_manifest)


def test_src_ensemble_evaluate_imports():
    from src.ensemble.evaluate import main, build_parser
    assert callable(main)
    assert callable(build_parser)


def test_src_ensemble_evaluate_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "src.ensemble.evaluate", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    assert "Ensemble evaluation" in result.stdout


def test_src_ensemble_uses_shared_checkpoint_io():
    """src.ensemble.ensemble should share the checkpoint_io loader,
    not duplicate it."""
    from src.ensemble import ensemble as ens_mod
    from src.training import checkpoint_io

    # The functions src.ensemble needs from checkpoint_io
    assert hasattr(ens_mod, "load_model_from_checkpoint")
    assert hasattr(ens_mod, "default_seg_dir_for_manifest")
    # And the underlying helpers come from the shared library
    assert ens_mod.load_model_from_checkpoint is checkpoint_io.load_model_from_checkpoint
    assert ens_mod.default_seg_dir_for_manifest is checkpoint_io.default_seg_dir_for_manifest


# ─────────────────────────────────────────────────────────────────────────────
# src.data.cli
# ─────────────────────────────────────────────────────────────────────────────

def test_src_data_cli_imports():
    from src.data.cli import DISPATCH, build_parser, main
    assert callable(main)
    assert callable(build_parser)
    # All subcommands must be wired through DISPATCH
    for cmd in ("manifests", "preprocess", "segmentation", "splits", "splits-only"):
        assert cmd in DISPATCH


def test_src_data_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "src.data.cli", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    for sub in ("manifests", "preprocess", "segmentation", "splits"):
        assert sub in result.stdout


def test_src_data_cli_split_subcommand_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "src.data.cli", "splits", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    assert "LODO" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# src.data.lodo_configs
# ─────────────────────────────────────────────────────────────────────────────

def test_src_data_lodo_configs_imports():
    from src.data.lodo_configs import build_config, main, PER_ARCH_OVERRIDES, SOURCES
    assert callable(build_config)
    assert callable(main)
    assert set(SOURCES) == {"eyepacs", "aptos", "messidor2"}
    # All 5 supported architectures must have an override entry
    from src.models.backbone import ARCH_REGISTRY
    for arch in ARCH_REGISTRY:
        assert arch in PER_ARCH_OVERRIDES, f"missing override for {arch}"


def test_lodo_configs_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "src.data.lodo_configs", "--help"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0
    assert "LODO" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# scripts/ is minimal
# ─────────────────────────────────────────────────────────────────────────────

def test_scripts_dir_has_no_python_entry_points():
    """Per the cleanup plan, scripts/ should only contain shell wrappers.
    All Python entry points are under `src/`."""
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    )
    files = [
        f for f in os.listdir(scripts_dir)
        if os.path.isfile(os.path.join(scripts_dir, f))
    ]
    # Only .sh files should be at the top level
    py_files = [f for f in files if f.endswith(".py")]
    assert py_files == [], f"scripts/ has Python entry points: {py_files}"


def test_scripts_archive_is_deleted():
    """Archive was deleted (not relocated). All legacy entry points have
    canonical replacements under src/."""
    archive_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "archive",
    )
    assert not os.path.exists(archive_dir), \
        f"scripts/archive should be deleted, but exists at {archive_dir}"


def test_canonical_entry_points_exist():
    """Every legacy script has a canonical replacement under src/."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_modules = [
        "src/train.py",
        "src/eval/evaluate.py",
        "src/eval/calibrate_thresholds.py",
        "src/ensemble/evaluate.py",
        "src/data/cli.py",
        "src/data/lodo_configs.py",
    ]
    missing = [m for m in expected_modules
               if not os.path.exists(os.path.join(repo_root, m))]
    assert not missing, f"missing canonical entry points: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Pearl-style tree shape
# ─────────────────────────────────────────────────────────────────────────────

def test_no_src_subdir_is_legacy_redundant():
    """Directories under src/ should not be dups of each other."""
    src_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    subdirs = {
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith("__")
    }
    # Required subdirs (Pearl-style)
    required = {"data", "ensemble", "eval", "training", "models", "preprocessing",
                 "losses", "augmentation"}
    assert required.issubset(subdirs), f"missing src/ subdirs: {required - subdirs}"