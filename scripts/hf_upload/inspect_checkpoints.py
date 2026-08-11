"""Inspect all source checkpoints to be uploaded to Hugging Face.

Walks the seven intended source folders, excludes calibration_layout/
(symlinks to baseline_3ch/), and validates every .pt file's wrapper
schema, parameter counts, state-dict keys, and EMA presence (for
classifier checkpoints).

Output: prints a per-file table and writes
`scripts/hf_upload/inspection_report.json` with the same data.

This script does NOT modify any files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO / "saved/checkpoints"

CLASSIFIER_FOLDERS = [
    "baseline_3ch",
    "baseline_3ch_unbalanced",
    "four_ch_morph",
    "four_ch_soft",
    "four_ch_tversky",
]
SEG_FOLDERS = ["seg_unet_bcedice", "seg_unet_tversky"]


def strip_prefix(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def inspect_classifier_ckpt(p: Path) -> dict[str, Any]:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    top_keys = sorted(obj.keys())
    sd = obj["model_state_dict"]
    ema = obj.get("ema_state_dict")
    sd_clean = strip_prefix(sd)
    ema_clean = strip_prefix(ema) if isinstance(ema, dict) else None
    nparams = sum(v.numel() for v in sd.values())
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    ema_keys_match = (
        ema_clean is not None
        and set(sd_clean.keys()) == set(ema_clean.keys())
    )
    return {
        "path": str(p.relative_to(REPO)),
        "type": "classifier",
        "wrapper_keys": top_keys,
        "n_params": int(nparams),
        "dtypes": dtypes,
        "state_dict_keys_have_orig_mod_prefix": any(
            k.startswith("_orig_mod.") for k in sd.keys()
        ),
        "ema_state_dict_present": ema is not None and isinstance(ema, dict),
        "ema_keys_match_model": bool(ema_keys_match),
        "best_metric": obj.get("best_metric"),
        "phase_name": obj.get("phase_name"),
        "epoch": obj.get("epoch"),
        "global_step": obj.get("global_step"),
        "class_names": obj.get("class_names"),
        "arch": obj.get("config", {}).get("model", {}).get("arch"),
        "in_chans": obj.get("config", {}).get("model", {}).get("in_chans"),
        "img_size": obj.get("config", {}).get("model", {}).get("img_size"),
        "head_hidden_dim": obj.get("config", {}).get("model", {}).get("head_hidden_dim"),
        "dropout": obj.get("config", {}).get("model", {}).get("dropout"),
        "use_cbam": obj.get("config", {}).get("model", {}).get("use_cbam"),
        "loss_type": obj.get("config", {}).get("loss_type"),
        "ema_decay": obj.get("config", {}).get("ema_decay"),
    }


def inspect_seg_ckpt(p: Path) -> dict[str, Any]:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    sd = obj["model_state"]
    nparams = sum(v.numel() for v in sd.values())
    dtypes = sorted({str(v.dtype) for v in sd.values()})
    return {
        "path": str(p.relative_to(REPO)),
        "type": "segmentation",
        "wrapper_keys": sorted(obj.keys()),
        "n_params": int(nparams),
        "dtypes": dtypes,
        "loss": obj.get("loss"),
        "epochs": obj.get("epochs"),
        "best_val_dice": obj.get("best_val_dice"),
        "best_epoch": obj.get("best_epoch"),
        "encoder": obj.get("encoder"),
        "img_size": obj.get("img_size"),
        "training_source": obj.get("training_source"),
    }


def collect_classifier_files() -> list[Path]:
    out = []
    for folder in CLASSIFIER_FOLDERS:
        root = CKPT_ROOT / folder
        if not root.is_dir():
            print(f"WARN: missing {root}", file=sys.stderr)
            continue
        out.extend(sorted(root.rglob("*_best.pt")))
    return out


def collect_seg_files() -> list[Path]:
    out = []
    for folder in SEG_FOLDERS:
        p = CKPT_ROOT / folder / "best.pt"
        if not p.is_file():
            print(f"WARN: missing {p}", file=sys.stderr)
            continue
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="scripts/hf_upload/inspection_report.json")
    args = ap.parse_args()

    classifier_paths = collect_classifier_files()
    seg_paths = collect_seg_files()

    print(f"\nClassifiers to inspect: {len(classifier_paths)}")
    for p in classifier_paths:
        print(f"  {p.relative_to(REPO)}")
    print(f"\nSeg models to inspect: {len(seg_paths)}")
    for p in seg_paths:
        print(f"  {p.relative_to(REPO)}")

    report: list[dict[str, Any]] = []
    failures: list[str] = []
    for p in classifier_paths:
        try:
            r = inspect_classifier_ckpt(p)
            report.append(r)
            assert r["ema_state_dict_present"], f"missing EMA in {p}"
            assert r["ema_keys_match_model"], f"EMA keys diverge from model in {p}"
            assert r["state_dict_keys_have_orig_mod_prefix"], (
                f"raw state_dict unexpectedly lacks _orig_mod. prefix in {p}"
            )
            assert r["arch"] in (
                "convnext_tiny",
                "deit3_small_patch16_384",
                "maxvit_tiny_tf_384",
            ), f"unexpected arch {r['arch']} in {p}"
            assert r["img_size"] == 384, f"img_size != 384 in {p}"
        except Exception as e:
            failures.append(f"{p}: {e}")

    for p in seg_paths:
        try:
            r = inspect_seg_ckpt(p)
            report.append(r)
            assert r["encoder"] == "efficientnet_b4", f"unexpected encoder in {p}"
            assert r["img_size"] == 384, f"img_size != 384 in {p}"
        except Exception as e:
            failures.append(f"{p}: {e}")

    out_path = REPO / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    # Pretty-print summary table.
    print("\n" + "=" * 100)
    print(f"{'TYPE':<14} {'PARAMS':>12} {'ARCH':<32} {'FOLD':<14} {'IMG':>4} {'EMA?':<6} {'BEST':>8}")
    print("=" * 100)
    for r in report:
        if r["type"] == "classifier":
            parts = r["path"].split("/")
            # baseline_3ch/<fold>/<arch>/<file>.pt -> fold = parts[1]
            # four_ch_morph/<file>.pt -> fold = parts[0].split('_')[-1]
            if parts[0].startswith("four_ch_"):
                stem = parts[1] if len(parts) > 1 else ""
                # lodo_<fold>_convnext_tiny_best.pt
                fold = stem.split("_")[1] if "_" in stem else "?"
            else:
                fold = parts[1] if len(parts) > 1 else "?"
            print(
                f"{'classifier':<14} {r['n_params']:>12,} "
                f"{r['arch']:<32} {fold:<14} {r['img_size']:>4} "
                f"{'yes' if r['ema_state_dict_present'] else 'no':<6} "
                f"{r['best_metric']:>8.4f}"
            )
        else:
            fold = r["path"].split("/")[0]
            print(
                f"{'segmentation':<14} {r['n_params']:>12,} "
                f"{r['encoder']:<32} {fold:<14} {r['img_size']:>4} "
                f"{'n/a':<6} {r['best_val_dice']:>8.4f}"
            )
    print("=" * 100)
    n_class = sum(1 for r in report if r["type"] == "classifier")
    n_seg = sum(1 for r in report if r["type"] == "segmentation")
    print(f"\nTotal checkpoints inspected: {len(report)} ({n_class} classifier, {n_seg} segmentation)")
    print(f"Total failures: {len(failures)}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
    print(f"\nWrote {out_path.relative_to(REPO)} ({out_path.stat().st_size / 1024:.1f} KB)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())