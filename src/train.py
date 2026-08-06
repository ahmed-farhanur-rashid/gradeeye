"""
Unified GradeEye training entry point — runs the LODO pipeline for any
(architecture, holdout_source) pair.

Usage (preferred split form):
    python -m src.train \\
        --model-config configs/models/convnext_tiny.yaml \\
        --lodo-config   configs/lodo/eyepacs.yaml

Usage (legacy monolithic form, back-compat shim):
    python -m src.train --config configs/lodo_eyepacs_convnext_tiny.yaml
    # OR any single-file yaml matching the legacy shape — it's loaded
    # as-is and forwarded to run_training without modification.

There is exactly one training pipeline (`src.training.orchestrate.run_training`).
The 5 architectures × 3 LODO folds = 15 supported training runs are
constructed at invocation time by deep-merging one model-axis yaml
(`configs/models/<arch>.yaml`) with one LODO-axis yaml
(`configs/lodo/<holdout>.yaml`). Run-name is derived deterministically:
`run_name = f"lodo_{holdout}_{arch}"` — never stored in either yaml.

Architectures (see src.models.backbone.ARCH_REGISTRY):
    convnext_tiny, convnext_small, deit3_small_patch16_384,
    maxvit_tiny_tf_384, swinv2_base_window12to24_192to384

Holdout sources: eyepacs, aptos, messidor2
"""
from __future__ import annotations

import argparse
import os
import sys

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DR grading model on one LODO fold "
                    "(split model+data axes, or legacy monolithic config).",
    )

    split = parser.add_argument_group("split form (preferred)")
    split.add_argument("--model-config", default=None,
                       help="Path to configs/models/<arch>.yaml "
                            "(what the model is).")
    split.add_argument("--lodo-config", default=None,
                       help="Path to configs/lodo/<holdout>.yaml "
                             "(which source is held out).")

    legacy = parser.add_argument_group("legacy monolithic form (back-compat)")
    legacy.add_argument("--config", default=None,
                        help="[DEPRECATED] Path to a monolithic "
                             "lodo_<holdout>_<arch>.yaml. If set, --model-config "
                             "and --lodo-config are ignored.")

    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume training from.")
    parser.add_argument("--device", default=None,
                        help="Override device (e.g. 'cuda', 'cuda:0', 'cpu'). "
                             "Default: auto-detect.")
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
        config = merge_configs(model_cfg, lodo_cfg)

    if args.device is not None:
        config["device"] = args.device
    run_training(config, resume_from=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
