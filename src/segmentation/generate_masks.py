"""
Generate lesion masks for any image directory using a trained DDR U-Net.

Two modes:
  1. Single-directory inference (--input-dir ... --output-dir ...) for the
     OOD qualitative check: drop masks into a temp folder, eyeball them.
  2. Multi-source mass generation via --sources (aptos, eyepacs, messidor2):
     read manifests from data/{train,val,test}/<source>/manifest.csv and
     write masks to data/processed/segmentation/<source>/<basename>.png.

Mask format: uint8 grayscale {0,255}, single channel, 384x384 (matches
preprocessed image size). The 4-channel dataset loader reads these as-is.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.normalize import normalize_image


class InferenceDataset(Dataset):
    """Loads images from disk, normalizes ImageNet-style, no augmentation."""

    def __init__(self, paths: list[str], img_size: int = 384):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # LongestMaxSize + PadIfNeeded (matches training resize)
        scale = self.img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        img = normalize_image(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return tensor, os.path.basename(path)


def load_segmenter(ckpt_path: str, device: str = "cuda"):
    from src.models.segmentation import UNetVessel
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder = ckpt.get("encoder", "efficientnet_b4")
    model = UNetVessel(encoder=encoder, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_masks(model, paths: list[str], out_dir: str, img_size: int = 384,
                   batch_size: int = 16, device: str = "cuda",
                   threshold: float = 0.5) -> list[tuple[str, str]]:
    """Run model over `paths`, write uint8 PNGs to `out_dir`.

    Output filenames are always .png regardless of input extension, so the
    4-channel dataset loader (which expects <basename>.png) finds them.
    """
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
            mask = (probs[i, 0] > threshold).astype(np.uint8) * 255
            stem = Path(name).stem
            out_path = os.path.join(out_dir, stem + ".png")
            cv2.imwrite(out_path, mask)
            written.append((name, out_path))
    return written


def collect_from_manifests(manifest_csvs: Iterable[str]) -> list[str]:
    paths = []
    for csv in manifest_csvs:
        if not Path(csv).exists():
            print(f"warn: missing {csv}, skipping")
            continue
        df = pd.read_csv(csv)
        paths.extend(df["image_path"].tolist())
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    # Mode A: single directory
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    # Mode B: multi-source from manifest CSVs
    parser.add_argument("--sources", nargs="*", default=None,
                        help="e.g. eyepacs aptos messidor2 — uses data/processed/<source>/")
    parser.add_argument("--root-data", default="data/processed")
    parser.add_argument("--limit", type=int, default=None,
                        help="For qualitative check: only generate first N masks")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_segmenter(args.checkpoint, device)
    val_iou = ckpt.get("val_iou")
    val_dice = ckpt.get("best_val_dice", ckpt.get("val_dice"))
    print(f"loaded {args.checkpoint}, encoder={ckpt.get('encoder')} "
          f"val_iou={val_iou if val_iou is not None else float('nan'):.4f} "
          f"val_dice={val_dice if val_dice is not None else float('nan'):.4f}")

    if args.sources:
        for source in args.sources:
            in_dir = Path(args.root_data) / source
            out_dir = Path(args.root_data) / "segmentation" / source
            paths = sorted(str(p) for p in in_dir.glob("*.*")
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            if args.limit:
                paths = paths[:args.limit]
            print(f"source={source} input={in_dir} n={len(paths)} -> {out_dir}")
            written = predict_masks(
                model, paths, str(out_dir),
                img_size=args.img_size, batch_size=args.batch_size,
                device=device, threshold=args.threshold,
            )
            print(f"  wrote {len(written)} masks to {out_dir}")
    else:
        assert args.input_dir and args.output_dir, \
            "need --input-dir and --output-dir, or --sources"
        paths = sorted(str(p) for p in Path(args.input_dir).glob("*.*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if args.limit:
            paths = paths[:args.limit]
        print(f"input={args.input_dir} n={len(paths)} -> {args.output_dir}")
        written = predict_masks(
            model, paths, args.output_dir,
            img_size=args.img_size, batch_size=args.batch_size,
            device=device, threshold=args.threshold,
        )
        print(f"  wrote {len(written)} masks")


if __name__ == "__main__":
    main()