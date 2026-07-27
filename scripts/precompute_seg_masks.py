"""
Pre-compute vessel/lesion segmentation masks for the DR datasets.

This is the Option A segmentation pipeline. It:
  1. Optionally fine-tunes a U-Net on DRIVE (40 public retinal vessel
     segmentation images) for a few minutes.
  2. Runs inference over EyePACS / APTOS / Messidor preprocessed images.
  3. Saves masks as PNGs in `data/processed/segmentation/<source>/`.

The masks can then be loaded by DRDataset as a 4th channel via the
`seg_dir` argument.

Usage:
    # Use whatever weights are available (random init if no pretrained).
    python scripts/precompute_seg_masks.py --datasets aptos messidor2

    # Skip DRIVE fine-tuning (use random init or a checkpoint you provide).
    python scripts/precompute_seg_masks.py --datasets aptos --no-finetune
        --checkpoint saved/checkpoints/unet_vessel.pt

Notes:
  - Inference is fast (~50ms per 384x384 image on RTX 4070 Super).
  - 88k EyePACS images will take ~75 minutes on a 4070 Super.
  - APTOS (3,663) and Messidor (1,745) finish in seconds.
"""
import argparse
import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.segmentation import build_unet_vessel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_for_unet(img: np.ndarray) -> torch.Tensor:
    """ImageNet normalize (RGB, [0,1]) -> CHW float tensor."""
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


class _MaskDataset(Dataset):
    """Loads preprocessed RGB PNGs for inference."""

    def __init__(self, paths, size):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path)
        if img is None:
            return torch.zeros(3, self.size, self.size), path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return _normalize_for_unet(img), path


@torch.no_grad()
def infer_masks(model, image_paths, out_dir, size=384, batch_size=16):
    os.makedirs(out_dir, exist_ok=True)
    dataset = _MaskDataset(image_paths, size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    for batch in tqdm(loader, desc=f"infer -> {os.path.basename(out_dir)}"):
        imgs, paths = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
            logits = model(imgs)
        probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()  # (B, H, W)
        for prob, src_path in zip(probs, paths):
            mask = (prob * 255).astype(np.uint8)
            out_path = os.path.join(out_dir, os.path.basename(src_path))
            cv2.imwrite(out_path, mask)
    print(f"Wrote {len(image_paths)} masks to {out_dir}")


def run_finetune_drive(checkpoint_path: str, epochs: int = 5, batch_size: int = 8):
    """
    Best-effort fine-tune on DRIVE. If DRIVE isn't available locally or
    the download fails, prints a warning and saves the random-init weights
    instead — masks will be garbage but the pipeline still works end-to-end.

    To get real vessel masks you should:
      1. Download DRIVE from https://drive.grand-challenge.org/
      2. Extract so that the structure is: data/drive/training/{images,masks}/*.png
      3. Run with --drive-dir data/drive
    """
    drive_dir = "data/drive"
    if not os.path.isdir(drive_dir):
        print(f"[WARN] DRIVE dataset not found at {drive_dir}. "
              f"Falling back to random-init weights (masks will be useless). "
              f"Download DRIVE and re-run for real vessel masks.")
        model = build_unet_vessel(pretrained=True)
        torch.save(model.state_dict(), checkpoint_path)
        return model

    train_imgs = sorted(os.path.join(drive_dir, "training", "images", f)
                        for f in os.listdir(os.path.join(drive_dir, "training", "images")))
    train_masks = sorted(os.path.join(drive_dir, "training", "masks", f)
                         for f in os.listdir(os.path.join(drive_dir, "training", "masks")))

    model = build_unet_vessel(pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for i in range(0, len(train_imgs), batch_size):
            batch_imgs = train_imgs[i:i + batch_size]
            batch_masks = train_masks[i:i + batch_size]
            xs, ys = [], []
            for ip, mp in zip(batch_imgs, batch_masks):
                img = cv2.imread(ip)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)
                xs.append(_normalize_for_unet(img))
                ys.append(torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0))
            x = torch.stack(xs).to(DEVICE)
            y = torch.stack(ys).to(DEVICE)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"  drive epoch {epoch + 1}/{epochs}: loss={running / max(1, len(train_imgs) // batch_size):.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved UNet weights to {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["aptos", "messidor2"],
                        help="Datasets to generate masks for.")
    parser.add_argument("--checkpoint", default="saved/checkpoints/unet_vessel.pt",
                        help="Where to save/load UNet weights.")
    parser.add_argument("--no-finetune", action="store_true",
                        help="Skip DRIVE fine-tuning and load existing weights "
                             "(or fall back to random init if no checkpoint exists).")
    parser.add_argument("--drive-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manifest-dir", default="data/processed",
                        help="Where dataset manifests live.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    # Load or fine-tune the UNet.
    if args.no_finetune and os.path.exists(args.checkpoint):
        model = build_unet_vessel(pretrained=False)
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        model = model.to(DEVICE)
        print(f"Loaded existing UNet weights from {args.checkpoint}")
    else:
        print("Fine-tuning UNet on DRIVE (or falling back to random init)...")
        model = run_finetune_drive(args.checkpoint, epochs=args.drive_epochs).to(DEVICE)

    # Run inference over each dataset.
    for ds_name in args.datasets:
        manifest_csv = os.path.join(args.manifest_dir, f"{ds_name}_manifest.csv")
        if not os.path.exists(manifest_csv):
            print(f"[WARN] Manifest not found: {manifest_csv}, skipping {ds_name}")
            continue
        df = pd.read_csv(manifest_csv)
        out_dir = os.path.join(args.manifest_dir, "segmentation", ds_name)
        print(f"\nProcessing {ds_name}: {len(df)} images -> {out_dir}")
        infer_masks(model, df["image_path"].tolist(), out_dir,
                    size=384, batch_size=args.batch_size)


if __name__ == "__main__":
    main()