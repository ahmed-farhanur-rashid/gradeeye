"""Convert GradeEye checkpoints to Hugging Face .safetensors format.

For each classifier checkpoint:
  - Write <stem>_ema.safetensors (PRIMARY, reproduces paper QWK metrics)
  - Write <stem>.safetensors (SECONDARY, raw model_state_dict)
  - Write <stem>.meta.json (training metadata, no optimizer/scheduler)

For each segmentation checkpoint:
  - Write best.safetensors
  - Write best.meta.json

Source layout (verified):
  baseline_3ch/<fold>/<arch>/lodo_<fold>_<arch>_best.pt
  baseline_3ch_unbalanced/<fold>/<arch>/lodo_<fold>_<arch>_best.pt
  four_ch_<variants>/lodo_<fold>_convnext_tiny_best.pt
  seg_unet_<variants>/best.pt

Target layout (preserved per repo requirements):
  baseline-3ch:                <fold>/<arch>/lodo_<fold>_<arch>_best_ema.safetensors
  baseline-3ch-unbalanced:     <fold>/<arch>/lodo_<fold>_<arch>_best_ema.safetensors
  four-ch-soft/-morph/-tversky: lodo_<fold>_convnext_tiny_best_ema.safetensors (flat)
  seg-unet-bcedice/-tversky:   best.safetensors

calibration_layout/ is excluded (symlinks to baseline_3ch).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / "saved/checkpoints"

REPO_TARGETS: dict[str, str] = {
    "baseline_3ch": "baseline-3ch",
    "baseline_3ch_unbalanced": "baseline-3ch-unbalanced",
    "four_ch_morph": "four-ch-morph",
    "four_ch_soft": "four-ch-soft",
    "four_ch_tversky": "four-ch-tversky",
    "seg_unet_bcedice": "seg-unet-bcedice",
    "seg_unet_tversky": "seg-unet-tversky",
}

EXCLUDE_DIRS = {"calibration_layout"}


def strip_prefix(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def save_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def convert_classifier(
    src: Path,
    repo_dir: Path,
    rel_target: str,
    inspection_row: dict[str, Any],
) -> list[Path]:
    """Convert one classifier checkpoint to its primary + secondary safetensors.

    Returns the list of files written under repo_dir (relative paths).
    """
    obj = torch.load(src, map_location="cpu", weights_only=False)
    sd_raw = strip_prefix(obj["model_state_dict"])
    ema_raw = strip_prefix(obj["ema_state_dict"])
    assert isinstance(ema_raw, dict) and len(ema_raw) == len(sd_raw), (
        f"EMA missing or key mismatch for {src}"
    )

    stem = src.stem
    ema_path = repo_dir / rel_target / f"{stem}_ema.safetensors"
    raw_path = repo_dir / rel_target / f"{stem}.safetensors"

    ema_contig = {k: v.contiguous() for k, v in ema_raw.items()}
    raw_contig = {k: v.contiguous() for k, v in sd_raw.items()}

    save_file(ema_contig, ema_path)
    save_file(raw_contig, raw_path)

    cfg = obj.get("config", {})
    model_cfg = cfg.get("model", {})
    phases = cfg.get("phases", {})

    # Prefer paths from the inspection report (single source of truth).
    meta = {
        "checkpoint_source": str(src.relative_to(REPO)),
        "primary_weights": f"{stem}_ema.safetensors",
        "secondary_weights": f"{stem}.safetensors",
        "primary_weights_kind": "ema",
        "secondary_weights_kind": "raw_model_state_dict",
        "best_metric": inspection_row.get("best_metric"),
        "epoch": inspection_row.get("epoch"),
        "global_step": inspection_row.get("global_step"),
        "phase_name": inspection_row.get("phase_name"),
        "class_names": inspection_row.get("class_names"),
        "ema_state_dict_present": True,
        "training_config": {
            "arch": model_cfg.get("arch"),
            "in_chans": model_cfg.get("in_chans"),
            "img_size": model_cfg.get("img_size"),
            "head_hidden_dim": model_cfg.get("head_hidden_dim"),
            "dropout": model_cfg.get("dropout"),
            "use_cbam": model_cfg.get("use_cbam"),
            "cbam_num_stages": model_cfg.get("cbam_num_stages"),
            "num_thresholds": model_cfg.get("num_thresholds"),
            "channels_last": model_cfg.get("channels_last"),
            "loss_type": cfg.get("loss_type"),
            "use_class_weighting": cfg.get("use_class_weighting"),
            "use_mixup": cfg.get("use_mixup"),
            "ema_decay": cfg.get("ema_decay"),
            "phase_batches": {
                p: phases[p].get("batch_size") for p in phases if isinstance(phases[p], dict)
            },
        },
    }
    meta_path = repo_dir / rel_target / f"{stem}.meta.json"
    save_meta(meta_path, meta)

    return [ema_path, raw_path, meta_path]


def convert_seg(
    src: Path,
    repo_dir: Path,
    inspection_row: dict[str, Any],
) -> list[Path]:
    obj = torch.load(src, map_location="cpu", weights_only=False)
    sd = obj["model_state"]
    sd_contig = {k: v.contiguous() for k, v in sd.items()}
    out_path = repo_dir / "best.safetensors"
    save_file(sd_contig, out_path)

    meta = {
        "checkpoint_source": str(src.relative_to(REPO)),
        "primary_weights": "best.safetensors",
        "primary_weights_kind": "raw_model_state",
        "loss": inspection_row.get("loss"),
        "epochs": inspection_row.get("epochs"),
        "best_val_dice": inspection_row.get("best_val_dice"),
        "best_epoch": inspection_row.get("best_epoch"),
        "encoder": inspection_row.get("encoder"),
        "img_size": inspection_row.get("img_size"),
        "training_source": inspection_row.get("training_source"),
    }
    meta_path = repo_dir / "best.meta.json"
    save_meta(meta_path, meta)
    return [out_path, meta_path]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="staging/hf",
                    help="Root staging directory (will be wiped & recreated).")
    ap.add_argument("--inspection-report",
                    default="scripts/hf_upload/inspection_report.json")
    args = ap.parse_args()

    staging = REPO / args.staging
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    report = json.loads((REPO / args.inspection_report).read_text())
    by_path = {r["path"]: r for r in report}

    # Discover source files and dispatch to the right repo directory.
    written: list[Path] = []
    repos_files: dict[str, list[Path]] = {v: [] for v in REPO_TARGETS.values()}

    for src in sorted(CKPT_ROOT.rglob("*_best.pt")):
        parts = src.relative_to(CKPT_ROOT).parts
        if parts[0] in EXCLUDE_DIRS:
            continue
        if parts[0] not in REPO_TARGETS:
            print(f"WARN: skipping {src} (not in REPO_TARGETS)", file=sys.stderr)
            continue
        repo = REPO_TARGETS[parts[0]]
        repo_dir = staging / repo
        repo_dir.mkdir(parents=True, exist_ok=True)

        # Preserve source subpath inside the repo.
        if parts[0] == "baseline_3ch" or parts[0] == "baseline_3ch_unbalanced":
            # baseline_3ch/<fold>/<arch>/<file>.pt  ->  <fold>/<arch>/<files>.safetensors
            rel_target = "/".join(parts[1:-1])
        elif parts[0].startswith("four_ch_"):
            # four_ch_<v>/<file>.pt  -> flat (no subdir)
            rel_target = ""
        else:
            rel_target = ""

        if rel_target:
            (repo_dir / rel_target).mkdir(parents=True, exist_ok=True)

        # Use the inspection report record for metadata.
        rel_src = str(src.relative_to(REPO))
        row = by_path.get(rel_src)
        if row is None:
            print(f"ERROR: no inspection row for {rel_src}", file=sys.stderr)
            return 1
        try:
            new_files = convert_classifier(src, repo_dir, rel_target, row)
        except Exception as e:
            print(f"ERROR converting {src}: {e}", file=sys.stderr)
            return 1
        for f in new_files:
            written.append(f)
            repos_files[repo].append(f)

    for src in [CKPT_ROOT / "seg_unet_bcedice" / "best.pt",
                CKPT_ROOT / "seg_unet_tversky" / "best.pt"]:
        if not src.is_file():
            print(f"WARN: missing {src}", file=sys.stderr)
            continue
        repo = REPO_TARGETS["seg_unet_" + src.parent.name.split("seg_unet_")[-1]]
        repo_dir = staging / repo
        repo_dir.mkdir(parents=True, exist_ok=True)
        row = by_path.get(str(src.relative_to(REPO)))
        if row is None:
            print(f"ERROR: no inspection row for {src}", file=sys.stderr)
            return 1
        new_files = convert_seg(src, repo_dir, row)
        for f in new_files:
            written.append(f)
            repos_files[repo].append(f)

    # Print summary.
    print("\n" + "=" * 70)
    print(f"Staging directory: {staging.relative_to(REPO)}")
    print(f"Total files written: {len(written)}")
    for repo, files in sorted(repos_files.items()):
        print(f"  {repo}: {len(files)} files")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())