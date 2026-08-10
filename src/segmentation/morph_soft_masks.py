"""
Apply morphology cleanup to soft masks (Option 3b).

Reads 16-bit PNGs from segmentation_pooled_soft/, applies:
  - Morphological opening (3x3) to remove speckle noise
  - Morphological closing (3x3) to fill small holes
  - Slight Gaussian blur (sigma 0.5) to soften edges

Saves to segmentation_pooled_morph/. Maintains 16-bit format so the
4-channel loader handles them as soft probabilities.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def morph_one(path: Path, out_dir: Path):
    out_path = out_dir / path.name
    if out_path.exists():
        return
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return
    # m is uint16, scale to uint8 for morphology ops
    m8 = (m.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    # Use cv2 morphologyEx with 3x3 kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (3, 3), 0.5)
    # Back to uint16
    out16 = (blurred.astype(np.float32) * 257.0).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_path), out16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/processed/segmentation_pooled_soft")
    parser.add_argument("--output-dir", default="data/processed/segmentation_pooled_morph")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(in_dir.glob("*.png"))
    print(f"input={in_dir} ({len(paths)} files) -> {out_dir}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(lambda p: morph_one(p, out_dir), paths),
                  total=len(paths), desc="morph"))
    print("done")


if __name__ == "__main__":
    main()