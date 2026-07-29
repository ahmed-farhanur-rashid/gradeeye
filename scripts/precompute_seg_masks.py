"""
Pre-compute vessel/lesion segmentation masks for the DR datasets.

This is the Option A segmentation pipeline. It:
  1. Fine-tunes a U-Net on DRIVE (40 retinal vessel images) + IDRiD
     (54 training images with microaneurysm / haemorrhage / hard exudate /
     soft exudate lesion annotations) — combined "retinal structures" target.
  2. Runs inference over EyePACS / APTOS / Messidor preprocessed images.
  3. Saves masks as PNGs in `data/processed/segmentation/<source>/`.

The masks can then be loaded by DRDataset as a 4th channel via the
`seg_dir` argument.

Usage:
    # Fine-tune U-Net on DRIVE+IDRiD, then generate masks for the listed datasets.
    python scripts/precompute_seg_masks.py --datasets aptos messidor2 eyepacs

    # Skip fine-tuning and just generate masks with whatever weights exist.
    python scripts/precompute_seg_masks.py --datasets aptos --no-finetune

Notes:
  - Inference is fast (~50ms per 384x384 image on RTX 4070 Super).
  - 88k EyePACS images will take ~75 minutes on a 4070 Super.
  - APTOS (3,663) and Messidor (1,745) finish in seconds.
  - DRIVE (20 train) + IDRiD (54 train) = 74 images, ~10 fine-tune epochs.
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
    out_size = size  # save at native training resolution
    for batch in tqdm(loader, desc=f"infer -> {os.path.basename(out_dir)}"):
        imgs, paths = batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.bfloat16):
            logits = model(imgs)
        # U-Net output is downsampled by 32 (192x192 from 384x384 input).
        # Bilinearly upsample to full res for downstream consumption.
        if logits.shape[-1] != out_size:
            logits = F.interpolate(logits, size=(out_size, out_size),
                                   mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()  # (B, H, W)
        for prob, src_path in zip(probs, paths):
            mask = (prob * 255).astype(np.uint8)
            out_path = os.path.join(out_dir, os.path.basename(src_path))
            cv2.imwrite(out_path, mask)
    print(f"Wrote {len(image_paths)} masks to {out_dir}")


def run_finetune_drive(checkpoint_path: str, epochs: int = 5, batch_size: int = 8):
    """
    Fine-tune U-Net on DRIVE vessels + IDRiD lesions (union of all retinal
    structures) so the model learns "where there is anything notable in
    the retina" — useful as a 4th-channel prior for the grader.

    DRIVE: 20 train images, *_training.tif + *_manual1.gif vessels + FOV mask
    IDRiD: 54 train images, IDRiD_NN.jpg + per-lesion TIFFs
           (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates).
    """
    train_imgs, train_masks = [], []

    # ----- DRIVE (vessels) -----
    drive_dir = "data/drive"
    if os.path.isdir(drive_dir):
        img_dir = os.path.join(drive_dir, "training", "images")
        lab_dir = os.path.join(drive_dir, "training", "1st_manual")
        fov_dir = os.path.join(drive_dir, "training", "mask")
        if os.path.isdir(img_dir) and os.path.isdir(lab_dir):
            img_files = sorted(os.listdir(img_dir))
            for img_name in img_files:
                ip = os.path.join(img_dir, img_name)
                # DRIVE labels are in 1st_manual/, named *_manual1.gif
                # Image basename e.g. 21_training.tif -> 21_manual1.gif
                lab_name = img_name.replace("_training.tif", "_manual1.gif")
                mp = os.path.join(lab_dir, lab_name)
                fov_name = img_name.replace("_training.tif", "_training_mask.gif")
                fp = os.path.join(fov_dir, fov_name)
                if os.path.exists(mp):
                    train_imgs.append((ip, mp, fp))  # (img, mask, fov)
        print(f"  DRIVE: {len(train_imgs)} train images")

    # ----- IDRiD (lesions) -----
    idrid_root = "data/A. Segmentation/1. Original Images/a. Training Set"
    idrid_lab_root = "data/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set"
    idrid_train_pairs = []
    if os.path.isdir(idrid_root) and os.path.isdir(idrid_lab_root):
        # Map lesion subfolder -> suffix
        lesion_folders = {
            "1. Microaneurysms": "_MA.tif",
            "2. Haemorrhages":   "_HE.tif",
            "3. Hard Exudates":  "_EX.tif",
            "4. Soft Exudates":  "_SE.tif",
        }
        for img_name in sorted(os.listdir(idrid_root)):
            ip = os.path.join(idrid_root, img_name)
            stem = img_name.replace(".jpg", "")
            # Build a single combined lesion mask by unioning all lesions.
            combined = None
            for folder, suffix in lesion_folders.items():
                lab_path = os.path.join(idrid_lab_root, folder, stem + suffix)
                if os.path.exists(lab_path):
                    m = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        combined = m if combined is None else np.maximum(combined, m)
            if combined is not None:
                # Save combined mask to a temp path in lab_dir/0_all/
                all_dir = os.path.join(idrid_lab_root, "0_all")
                os.makedirs(all_dir, exist_ok=True)
                all_path = os.path.join(all_dir, stem + "_ALL.png")
                if not os.path.exists(all_path):
                    cv2.imwrite(all_path, combined)
                idrid_train_pairs.append((ip, all_path, None))  # no FOV for IDRiD
        train_imgs.extend(idrid_train_pairs)
        print(f"  IDRiD: {len(idrid_train_pairs)} train images")

    if not train_imgs:
        print(f"[WARN] No segmentation datasets found. Falling back to random-init "
              f"weights (masks will be useless).")
        model = build_unet_vessel(pretrained=True)
        torch.save(model.state_dict(), checkpoint_path)
        return model

    # Dice loss is standard for binary vessel segmentation (BCE alone
    # converges to all-zero on these extremely imbalanced masks).
    bce = nn.BCEWithLogitsLoss()

    def dice_loss(logits, target, eps=1e-6):
        p = torch.sigmoid(logits)
        num = 2 * (p * target).sum(dim=(1, 2, 3))
        den = (p + target).sum(dim=(1, 2, 3)) + eps
        return (1 - (num / den)).mean()

    model = build_unet_vessel(pretrained=True).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_steps = 0
        # Shuffle every epoch
        rng = np.random.default_rng(epoch)
        perm = rng.permutation(len(train_imgs))
        train_ep = [train_imgs[i] for i in perm]
        for i in range(0, len(train_ep), batch_size):
            batch = train_ep[i:i + batch_size]
            xs, ys = [], []
            for ip, mp, fp in batch:
                img = cv2.imread(ip)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                # U-Net downsamples by 32 (encoder strides). Resize to model
                # output resolution so loss matches.
                mask = cv2.resize(mask, (192, 192), interpolation=cv2.INTER_NEAREST)
                # Restrict loss to inside-FOV (DRIVE only)
                if fp is not None and os.path.exists(fp):
                    fov = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if fov is not None:
                        fov = cv2.resize(fov, (192, 192), interpolation=cv2.INTER_NEAREST)
                        mask = (mask * (fov > 127)).astype(np.uint8)
                xs.append(_normalize_for_unet(img))
                ys.append(torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0))
            if not xs:
                continue
            x = torch.stack(xs).to(DEVICE)
            y = torch.stack(ys).to(DEVICE)
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            n_steps += 1
        scheduler.step()
        print(f"  drive+idrid epoch {epoch + 1}/{epochs}: loss={running / max(1, n_steps):.4f}")

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