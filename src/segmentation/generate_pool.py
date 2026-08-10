"""Generate one aux pool across all 4 sources using a trained U-Net.

Usage:
    PYTHONPATH=. python -m src.segmentation.generate_pool \\
        --checkpoint saved/checkpoints/seg_unet_bcedice/best.pt \\
        --pool-name soft \\
        [--mode soft|binary]

This wraps `src.segmentation.generate_soft_masks` (default, 16-bit soft
probs) or `src.segmentation.generate_masks` (binary 8-bit, --mode binary)
to write per-source aux pools under
`data/processed/segmentation_<pool_name>/<source>/`. The "soft" mode is
the default because downstream tooling (e.g. `morph_soft_masks.py`)
expects uint16 inputs.

We loop over sources ourselves because the upstream scripts either accept
a single flat output dir or `--sources` writing into a single shared
`segmentation/<source>/` dir, neither of which supports per-pool
top-level naming.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SOURCES = ("eyepacs", "aptos", "messidor2", "ddr")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pool-name", required=True,
                   help="e.g. 'soft' -> writes to "
                        "data/processed/segmentation_<pool_name>/<source>/")
    p.add_argument("--sources", nargs="*", default=list(SOURCES))
    p.add_argument("--root-data", default="data/processed")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Only used when --mode binary.")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mode", choices=["soft", "binary"], default="soft",
                   help="soft: 16-bit probability PNGs (uint16). "
                        "binary: 8-bit thresholded PNGs (uint8).")
    ns = p.parse_args()

    checkpoint = Path(ns.checkpoint)
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1

    module = ("src.segmentation.generate_soft_masks"
              if ns.mode == "soft"
              else "src.segmentation.generate_masks")
    for source in ns.sources:
        in_dir = Path(ns.root_data) / source
        out_dir = Path(ns.root_data) / f"segmentation_{ns.pool_name}" / source
        if not in_dir.is_dir():
            print(f"SKIP {source}: {in_dir} not found")
            continue
        if out_dir.is_dir() and any(out_dir.iterdir()):
            existing = sum(1 for _ in out_dir.glob("*.png"))
            print(f"SKIP {source}: {out_dir} already has {existing} masks")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", "-m", module,
            "--checkpoint", str(checkpoint),
            "--sources", source,
            "--root-data", str(Path(ns.root_data)),
            "--output-root", str(Path(ns.root_data).parent),
            "--output-dirname", str(Path(ns.root_data) / f"segmentation_{ns.pool_name}"),
            "--img-size", str(ns.img_size),
            "--batch-size", str(ns.batch_size),
        ]
        if ns.mode == "binary":
            # generate_masks supports --input-dir/--output-dir directly.
            cmd = [
                sys.executable, "-u", "-m", module,
                "--checkpoint", str(checkpoint),
                "--input-dir", str(in_dir),
                "--output-dir", str(out_dir),
                "--threshold", str(ns.threshold),
                "--img-size", str(ns.img_size),
                "--batch-size", str(ns.batch_size),
            ]
        print(f"\n=== {source} -> {out_dir} (mode={ns.mode}) ===")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"ERROR: {source} failed with rc={rc}")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
