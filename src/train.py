"""
Unified GradeEye training entry point — runs the LODO pipeline for any
(architecture, variant, holdout) triple.

Usage (preferred three-axis split form):
    python -m src.train \\
        --model-config   configs/models/convnext_tiny.yaml \\
        --variant-config configs/variants/convnext_tiny_4ch_sobel.yaml \\
        --lodo-config    configs/lodo/eyepacs.yaml

Usage (two-axis split form, 3-channel baseline):
    python -m src.train \\
        --model-config configs/models/convnext_tiny.yaml \\
        --lodo-config   configs/lodo/eyepacs.yaml

Usage (legacy monolithic form, back-compat shim):
    python -m src.train --config configs/lodo_eyepacs_convnext_tiny.yaml

There is exactly one training pipeline (`src.training.orchestrate.run_training`).
The 5 backbones × 7 variants (3ch baseline + 6 channel variants) × 4 LODO holdouts
= 140 supported training runs are constructed at invocation time by
deep-merging:

  1. A model-axis yaml (`configs/models/<arch>.yaml`) — what the backbone is.
  2. An optional variant-axis yaml (`configs/variants/<variant>.yaml`) — what
     auxiliary channels to apply (4ch / 5ch variants). Variants overlay only
     `model.in_chans`, `phases.*.seg_dir`, `log_dir`, and `checkpoint_dir`;
     everything else (arch, batch_size, lr, etc.) is inherited from the model
     config. Omit `--variant-config` for the 3-channel baseline.
  3. A LODO-axis yaml (`configs/lodo/<experiment>/<holdout>.yaml`) — which
     source is held out.

Run-name is derived deterministically: `run_name = f"lodo_{holdout}_{arch}"` —
never stored in any yaml.

Architectures (configs/models/):
    convnext_tiny, convnext_small, deit3_small_patch16_384,
    maxvit_tiny_tf_384, swinv2_base_window12to24_192to384

Variants (configs/variants/):
    four_ch_soft, four_ch_sobel, four_ch_tversky, four_ch_morph,
    five_ch_soft_sobel, five_ch_tversky_sobel.
    (No variant file is needed for the 3-channel baseline.)

Holdout sources (configs/lodo_balanced/):
    eyepacs, aptos, messidor2, ddr

Output isolation (so different experiments never overwrite each other):

- A variant config sets its own `log_dir` / `checkpoint_dir` (e.g.
  `saved/logs/four_ch_sobel/`). All four LODO folds of that variant share
  the directory but are distinguished by `run_name` in the filename.
- If a LODO config is under `configs/<experiment_name>/<holdout>.yaml`,
  the trainer scopes logs and checkpoints under that experiment_name:
  `saved/logs/<experiment_name>/<holdout>/<arch>/`.
- Pass `--experiment-name` to override the auto-detected parent dir.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import yaml

# CRITICAL FIX for DataLoader deadlock on Linux: prevent OpenCV from spawning
# its own threads inside the PyTorch multiprocessing workers.
cv2.setNumThreads(0)

from src.config_merge import merge_configs
from src.training.orchestrate import run_training


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _infer_experiment_name(lodo_cfg_path: str) -> str | None:
    """If the LODO config is at configs/<experiment_name>/<file>.yaml, return
    `<experiment_name>`. Returns None for configs/lodo/ or configs at the repo
    root (no parent dir), preserving the legacy unscoped layout."""
    p = Path(lodo_cfg_path).resolve()
    parent = p.parent.name
    grandparent = p.parent.parent.name
    # Only treat the parent as an experiment_name when the grandparent is
    # `configs/` — that filters out e.g. scripts/eval/foo.yaml.
    if grandparent == "configs" and parent not in ("lodo", "models", "variants", ""):
        return parent
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DR grading model on one LODO fold "
                    "(three-axis split: model + variant + lodo, or two-axis, "
                    "or legacy monolithic config).",
    )

    split = parser.add_argument_group("three-axis split form (preferred)")
    split.add_argument("--model-config", default=None,
                       help="Path to configs/models/<arch>.yaml "
                            "(what the model is).")
    split.add_argument("--variant-config", default=None,
                       help="Path to configs/variants/<variant>.yaml "
                            "(auxiliary channels: 4ch / 5ch variant). "
                            "Omit for the 3-channel baseline. A variant may "
                            "only set model.in_chans, phases.*.seg_dir, "
                            "log_dir, and checkpoint_dir.")
    split.add_argument("--lodo-config", default=None,
                       help="Path to configs/<experiment>/<holdout>.yaml "
                             "(which source is held out). The <experiment> "
                             "parent dir scopes logs/checkpoints so different "
                             "experiments never overwrite each other.")

    legacy = parser.add_argument_group("legacy monolithic form (back-compat)")
    legacy.add_argument("--config", default=None,
                        help="[DEPRECATED] Path to a monolithic "
                             "lodo_<holdout>_<arch>.yaml. If set, "
                             "--model-config, --variant-config, and "
                             "--lodo-config are ignored.")

    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. 'cuda', 'cuda:0', 'cpu'). "
                             "Default: auto-detect.")
    parser.add_argument("--experiment-name", default=None,
                        help="Override the experiment_name used to scope "
                             "logs/checkpoints (default: parent dir of "
                             "--lodo-config). Use this when configs are not "
                             "under configs/<experiment_name>/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.config:
        # Legacy monolithic form: load as-is, no merge.
        config = load_yaml(args.config)
    else:
        if not (args.model_config and args.lodo_config):
            print("ERROR: either --config (legacy) OR both "
                  "--model-config and --lodo-config are required.",
                  file=sys.stderr)
            return 2
        model_cfg = load_yaml(args.model_config)
        lodo_cfg = load_yaml(args.lodo_config)
        variant_cfg = load_yaml(args.variant_config) if args.variant_config else None

        experiment_name = args.experiment_name
        if experiment_name is None:
            experiment_name = _infer_experiment_name(args.lodo_config)
        config = merge_configs(model_cfg, lodo_cfg,
                                variant_cfg=variant_cfg,
                                experiment_name=experiment_name)

    if args.device is not None:
        config["device"] = args.device
    run_training(config, resume_from=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
