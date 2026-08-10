"""
Generate soft (float32) lesion probability maps for any image directory.

Unlike generate_masks.py which binarizes at a threshold, this saves the raw
sigmoid probabilities as 16-bit PNGs (range 0-65535 = 0.0-1.0). The
classifier gets continuous "lesion-ness" scores instead of binary masks.

Output: data/processed/segmentation_pooled_soft/<basename>.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.segmentation.generate_masks import (
    InferenceDataset, load_segmenter,
)


@torch.no_grad()
def predict_soft_masks(model, paths, out_dir, img_size=384, batch_size=16,
                        device="cuda"):
    """Save raw sigmoid probabilities as 16-bit PNGs (0-65535 = 0.0-1.0)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ds = InferenceDataset(paths, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    written = []
    for batch in loader:
        images, names = batch
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        probs = torch.sigmoid(logits).float().cpu().numpy()  # (B,1,H,W)
        for i, name in enumerate(names):
            soft = probs[i, 0].astype(np.float32)
            soft_16bit = (soft * 65535.0).clip(0, 65535).astype(np.uint16)
            stem = Path(name).stem
            out_path = os.path.join(out_dir, stem + ".png")
            cv2.imwrite(out_path, soft_16bit)
            written.append((name, out_path))
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sources", nargs="*",
                        default=["eyepacs", "aptos", "messidor2"])
    parser.add_argument("--root-data", default="data/processed")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--output-dirname", default="segmentation_pooled_soft",
                        help="Subdirectory under output-root for soft masks. "
                             "Use a unique name per checkpoint to avoid "
                             "overwriting other mask variants.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_segmenter(args.checkpoint, device)
    print(f"loaded {args.checkpoint} val_iou={ckpt.get('val_iou'):.4f}")

    for source in args.sources:
        in_dir = Path(args.root_data) / source
        out_dir = Path(args.output_root) / args.output_dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted(str(p) for p in in_dir.glob("*.*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        print(f"source={source}: {len(paths)} images -> {out_dir}")
        written = predict_soft_masks(model, paths, str(out_dir),
                                      img_size=args.img_size,
                                      batch_size=args.batch_size,
                                      device=device)
        print(f"  wrote {len(written)} soft masks")
    print("done")


if __name__ == "__main__":
    main()