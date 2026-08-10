"""
Generate Sobel edge maps for any image directory.

For each image, computes a normalized Sobel gradient magnitude (uint8 0-255)
and saves it as a PNG with the same basename. Used as an alternative
structural prior to the U-Net lesion masks.

Pure CPU work — fast and OOD-invariant (no learned biases).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sobel_mask(img_path: Path) -> np.ndarray:
    """Compute Sobel magnitude map. Returns uint8 (H, W) in [0,255]."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    # Per-image normalization so each map uses the full 0-255 range.
    # For 4ch classifier use, the loader expects uint8 / 255.
    if mag.max() > 0:
        mag = (mag / mag.max() * 255.0)
    return mag.astype(np.uint8)


def process_one(args):
    in_path, out_dir = args
    out_path = out_dir / (in_path.stem + ".png")
    if out_path.exists():
        return  # skip already-done
    mask = sobel_mask(in_path)
    cv2.imwrite(str(out_path), mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*",
                        default=["eyepacs", "aptos", "messidor2"])
    parser.add_argument("--root-data", default="data/processed")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    for source in args.sources:
        in_dir = Path(args.root_data) / source
        out_dir = Path(args.output_root) / "segmentation_pooled_sobel"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in in_dir.glob("*.*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        print(f"source={source}: {len(paths)} images -> {out_dir}")
        tasks = [(p, out_dir) for p in paths]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(tqdm(ex.map(process_one, tasks), total=len(tasks),
                      desc=f"sobel/{source}"))
    print("done")


if __name__ == "__main__":
    main()