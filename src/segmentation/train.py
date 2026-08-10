"""
Train a U-Net lesion segmenter on the DDR lesion segmentation subset.

Architecture: UNetVessel (src/models/segmentation.py) with a timm encoder
swapped to EfficientNet-B4 (B3 also viable if VRAM tight). Single-channel
sigmoid output trained against the fused lesion mask (OR over EX/HE/MA/SE).

Loss: BCE + Dice (weighted 0.5/0.5 by default).
Schedule: AdamW + cosine LR with 3-epoch warmup, 30 epochs.
Image size: 384 (matches classifier input).

Usage:
    PYTHONPATH=. python -m src.segmentation.train \\
        --encoder efficientnet_b4 \\
        --img-size 384 \\
        --epochs 30 \\
        --batch-size 8 \\
        --lr 1e-4 \\
        --output-dir saved/checkpoints/ddr_unet_effb4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.segmentation import UNetVessel
from src.segmentation.ddr_dataset import DDRLesionDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """BCE + (1 - Dice). Stable for sparse-mask lesion segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: float = 25.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Strong pos_weight compensates for ~0.005% positive pixel fraction.
        # 25x is a middle-ground that prevents gradient starvation without
        # flooding false positives — adjust by val precision/recall if needed.
        self.register_buffer(
            "pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: (B, 1, H, W)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device)
        )
        probs = torch.sigmoid(logits)
        # Dice on flattened per-sample then averaged (avoids one big empty
        # image dominating the loss).
        b, _, h, w = logits.shape
        probs_flat = probs.view(b, -1)
        targets_flat = targets.view(b, -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + 1.0) / (denom + 1.0)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    """Tversky loss — generalizes Dice with FP/FN weights.

    alpha > beta penalizes false positives more (more conservative).
    alpha < beta penalizes false negatives more (more sensitive).
    For sparse lesion segmentation we want more recall, so beta > alpha.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight
        self.gamma = gamma  # focal exponent

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        b, _, h, w = logits.shape
        p = probs.view(b, -1)
        t = targets.view(b, -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        loss = (1.0 - tversky) ** self.gamma
        return loss.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Mean IoU over a batch. Operates on logits (will sigmoid)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(iou.mean())


def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return float(dice.mean())


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n_batches = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach())
        total_iou += iou_score(logits.detach(), masks)
        n_batches += 1
    return total_loss / max(n_batches, 1), total_iou / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n_batches = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += float(loss.detach())
        total_iou += iou_score(logits, masks)
        total_dice += dice_score(logits, masks)
        n_batches += 1
    n = max(n_batches, 1)
    return total_loss / n, total_iou / n, total_dice / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="efficientnet_b4")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=25.0)
    parser.add_argument("--loss", choices=["bce_dice", "tversky"],
                        default="bce_dice")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="saved/checkpoints/ddr_unet_effb4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from <output-dir>/best.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} encoder={args.encoder} img_size={args.img_size}")

    train_ds = DDRLesionDataset("train", img_size=args.img_size, augment=True)
    valid_ds = DDRLesionDataset("valid", img_size=args.img_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    print(f"train={len(train_ds)} valid={len(valid_ds)} "
          f"batch={args.batch_size} epochs={args.epochs}")

    model = UNetVessel(encoder=args.encoder, pretrained=True).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    out_dir = Path(args.output_dir)
    if args.resume:
        ckpt_path = out_dir / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            print(f"resumed from {ckpt_path}, epoch={ckpt.get('epoch')} "
                  f"val_iou={ckpt.get('val_iou'):.4f}")
        else:
            print(f"--resume set but no {ckpt_path} found, training from scratch")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = BCEDiceLoss(args.bce_weight, args.dice_weight, args.pos_weight) \
        if args.loss == "bce_dice" else TverskyLoss(alpha=0.3, beta=0.7)
    scaler = torch.amp.GradScaler("cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.json"
    log = []

    best_iou = -1.0
    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_t0 = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
        )
        val_loss, val_iou, val_dice = validate(model, valid_loader, criterion, device)
        scheduler.step()

        ep_time = time.time() - ep_t0
        log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
            "lr": optimizer.param_groups[0]["lr"],
            "ep_time_s": ep_time,
        })
        print(
            f"epoch {epoch+1:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
            f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} val_dice={val_dice:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={ep_time:.1f}s"
        )
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt = {
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "img_size": args.img_size,
                "epoch": epoch + 1,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  -> new best, val_iou={val_iou:.4f}, saved {out_dir / 'best.pt'}")

    total_time = time.time() - t0
    print(f"done in {total_time/60:.1f} min, best_val_iou={best_iou:.4f}")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "best_val_iou": best_iou,
            "epochs": args.epochs,
            "encoder": args.encoder,
            "img_size": args.img_size,
            "total_time_min": total_time / 60,
        }, f, indent=2)


if __name__ == "__main__":
    main()