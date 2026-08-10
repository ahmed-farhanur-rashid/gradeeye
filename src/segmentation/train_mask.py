"""
Train the U-Net lesion segmenter on **DDR lesion segmentation** (the primary
source) with **IDRiD** as a small auxiliary source and IDRiD's test split as
the held-out evaluation.

Why this script exists:
- The pre-existing 5 segmentation pools (segmentation_pooled_soft, _tversky,
  _morph, _sobel, plus the original _pooled) were wiped before this runbook
  was written (Rule 4 of docs/todo.md).
- The pre-existing U-Net checkpoints were also wiped.
- The pre-existing `src/segmentation/train.py` tried to load DDR lesion
  segmentation data from `data/ddr/DDR-dataset/lesion_segmentation/`, but
  that path only contains README files (the per-image TIFs are not in this
  checkout). It is unusable as-is.
- The DDR lesion-segmentation data IS on disk at
  `data/raw/DDR-dataset/lesion_segmentation/` (383 training + 149 valid + 225
  test images, four lesion types per image: MA, HE, EX, SE).
- The IDRiD lesion-segmentation data is also on disk at
  `data/raw/A. Segmentation/` (54 training + 27 test images, four lesion
  types per image).
- The existing `src/segmentation/ablation_3arm.py` provides the
  `collect_idrid_samples` helper and the BCE+Dice baseline loss, but no
  equivalent for DDR. This script adds a DDR loader and uses IDRiD only as
  auxiliary training data plus the held-out evaluation set.

Training source composition:
- **Primary**: DDR lesion_segmentation (757 images with 4 lesion types each:
  MA = microaneurysms, HE = hemorrhages, EX = hard exudates, SE = soft
  exudates). The four per-lesion masks are unioned into a single binary
  "any-lesion" mask for training.
- **Auxiliary (optional)**: IDRiD lesion-segmentation training set (54
  images). Adds 54 more diverse-annotation samples.
- **Evaluation (held-out)**: IDRiD lesion-segmentation test set (27 images).
  We use IDRiD-test as the test set rather than DDR-test because IDRiD has
  the gold-standard per-lesion-type GT that the methodology references, and
  because we want the test set to be drawn from a different domain than
  training (true generalization test, not just a random split).

What this script does:
- Trains a single U-Net (EfficientNet-B4 encoder), fixed 4-fold split
  (no CV), with a configurable loss family (BCE+Dice or Tversky).
- Saves the best checkpoint to `saved/checkpoints/seg_unet_<loss>/best.pt`.
- Writes a small JSON log to `saved/logs/seg_unet_<loss>/train_log.json`.
- Reports Dice / IoU / Tversky on the IDRiD test set.

Usage:
    PYTHONPATH=. python -m src.segmentation.train_mask \\
        --loss bce_dice \\
        --epochs 30 \\
        --output-dir saved/checkpoints/seg_unet_bcedice

Then `src/segmentation/generate_masks.py` is run with --checkpoint pointing
at the saved file to populate `data/processed/segmentation_pooled_<loss>/`
for all four sources (eyepacs, aptos, messidor2, ddr).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.models.segmentation import build_unet_vessel
from src.segmentation.ablation_3arm import (
    IDRID_IMG_DIR, IDRID_LAB_DIR,
    IDRiDSample, _load_pair, _safe_div, dice_iou, predict_mask,
    baseline_loss, _idrid_lesion_folder, LESION_TYPES,
)
from src.segmentation.train import TverskyLoss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# DDR lesion-segmentation paths.
DDR_SEG_ROOT = Path("data/raw/DDR-dataset/lesion_segmentation")
DDR_TRAIN_IMG = DDR_SEG_ROOT / "train/image"
DDR_TRAIN_LAB = DDR_SEG_ROOT / "train/label"
DDR_VALID_IMG = DDR_SEG_ROOT / "valid/image"
DDR_VALID_LAB = DDR_SEG_ROOT / "valid/segmentation label"
DDR_TEST_IMG = DDR_SEG_ROOT / "test/image"
DDR_TEST_LAB = DDR_SEG_ROOT / "test/label"


@dataclass
class DDRSample:
    """Same shape as IDRiDSample (image_path, stem, union_mask_path,
    per_lesion_mask_paths) so the rest of the pipeline can be source-agnostic."""
    image_path: str
    stem: str
    union_mask_path: str
    per_lesion_mask_paths: dict[str, str]


def collect_ddr_samples(split: str = "train") -> list[DDRSample]:
    """Collect DDR lesion-segmentation samples for a given split.

    Args:
        split: one of "train" (383 images), "valid" (149), "test" (225).
               The four per-lesion masks are unioned into a single binary
               "any-lesion" mask; the union PNG is written to a sibling
               directory the first time it is requested and reused thereafter.

    Returns:
        list of DDRSample. The union_mask_path is set to a cached PNG in
        the same split directory under "0_all/" so subsequent calls are fast.
    """
    if split == "train":
        img_dir, lab_root = DDR_TRAIN_IMG, DDR_TRAIN_LAB
    elif split == "valid":
        img_dir, lab_root = DDR_VALID_IMG, DDR_VALID_LAB
    elif split == "test":
        img_dir, lab_root = DDR_TEST_IMG, DDR_TEST_LAB
    else:
        raise ValueError(f"Unknown split: {split}")

    all_dir = lab_root.parent / "0_all"
    all_dir.mkdir(exist_ok=True)
    samples: list[DDRSample] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        per_lesion: dict[str, str] = {}
        union = None
        for lesion in LESION_TYPES:
            lab_path = lab_root / lesion / f"{stem}.tif"
            if not lab_path.exists():
                continue
            per_lesion[lesion] = str(lab_path)
            m = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            union = m if union is None else np.maximum(union, m)
        if union is None:
            continue
        all_path = all_dir / f"{stem}_ALL.png"
        if not all_path.exists():
            cv2.imwrite(str(all_path), union)
        samples.append(DDRSample(
            image_path=str(img_path),
            stem=stem,
            union_mask_path=str(all_path),
            per_lesion_mask_paths=per_lesion,
        ))
    return samples


def tversky_loss_fn(logits: torch.Tensor, target: torch.Tensor,
                    bce: nn.Module) -> torch.Tensor:
    """Tversky loss wrapped in the (logits, target, bce) signature the train
    loop uses. The bce arg is unused — Tversky is its own loss — but keeping
    the signature lets the train loop stay arm-agnostic."""
    tl = TverskyLoss(alpha=0.3, beta=0.7, gamma=1.0).to(logits.device)
    return tl(logits, target)


def split_ddr(samples: list[DDRSample], n_splits: int = 4,
              seed: int = 42) -> tuple[list[DDRSample], list[DDRSample]]:
    """Fixed K-fold split (stratified by presence of any lesion). Returns
    (train, val) for the first fold. The val set is taken from the DDR
    training set itself (a held-out subset of training) so the test set
    (DDR test split) is reserved for end-of-run reporting and IDRiD-test
    remains the cross-domain evaluation set."""
    rng = np.random.default_rng(seed)
    has_lesion = []
    for s in samples:
        m = cv2.imread(s.union_mask_path, cv2.IMREAD_GRAYSCALE)
        has_lesion.append((m > 0).any() if m is not None else False)
    pos_idx = [i for i, h in enumerate(has_lesion) if h]
    neg_idx = [i for i, h in enumerate(has_lesion) if not h]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    val_size = max(1, len(samples) // n_splits)
    val_idx = pos_idx[: val_size // 2] + neg_idx[: val_size - val_size // 2]
    val_idx_set = set(val_idx)
    train = [s for i, s in enumerate(samples) if i not in val_idx_set]
    val = [s for i, s in enumerate(samples) if i in val_idx_set]
    return train, val


def evaluate_segmentation(model: nn.Module, samples) -> dict:
    """Compute Dice / IoU / Tversky on a list of samples (IDRiDSample or
    DDRSample — the only required attribute is image_path + union_mask_path)."""
    dices, ious, tverskies = [], [], []
    for s in samples:
        pred = predict_mask(model, s.image_path)
        gt = cv2.imread(s.union_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 127).astype(np.uint8)
        d, i = dice_iou(pred, gt_bin)
        dices.append(d)
        ious.append(i)
        p = pred.astype(bool)
        t = gt_bin.astype(bool)
        tp = float(np.logical_and(p, t).sum())
        fp = float(np.logical_and(p, ~t).sum())
        fn = float(np.logical_and(~p, t).sum())
        tverskies.append(_safe_div(tp, tp + 0.5 * fp + 0.5 * fn))
    return {
        "n": len(dices),
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "tversky": float(np.mean(tverskies)) if tverskies else 0.0,
    }


def train_one(loss_kind: str, ddr_train: list, ddr_val: list,
              idrid_train: list | None,
              epochs: int = 30, batch_size: int = 8, lr: float = 1e-4,
              seed: int = 42) -> tuple[nn.Module, dict]:
    """Train a single U-Net for the specified loss kind. The training pool
    is the union of `ddr_train` and (optionally) `idrid_train`. Returns
    (best_model, log). The validation set is `ddr_val` only — IDRiD is held
    out for end-of-run evaluation, not for checkpoint selection, because
    the goal is to select a checkpoint that generalizes across domains."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_unet_vessel(pretrained=True, encoder="efficientnet_b4").to(DEVICE)
    bce = nn.BCEWithLogitsLoss()

    if loss_kind == "bce_dice":
        loss_fn = lambda l, t, b: baseline_loss(l, t, b)
    elif loss_kind == "tversky":
        loss_fn = tversky_loss_fn
    else:
        raise ValueError(f"Unknown loss: {loss_kind}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_pairs: list = [(s.image_path, s.union_mask_path, None)
                          for s in ddr_train]
    if idrid_train is not None:
        train_pairs += [(s.image_path, s.union_mask_path, None)
                         for s in idrid_train]
    val_pairs = [(s.image_path, s.union_mask_path, None) for s in ddr_val]

    log = {"loss": loss_kind, "epochs": [], "best_val_dice": 0.0,
           "best_epoch": -1, "n_train_ddr": len(ddr_train),
           "n_train_idrid": len(idrid_train) if idrid_train else 0,
           "n_val_ddr": len(ddr_val)}
    best_state = None
    best_dice = -1.0

    for epoch in range(epochs):
        model.train()
        running, n_steps = 0.0, 0
        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(len(train_pairs))
        shuffled = [train_pairs[i] for i in perm]
        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            xs, ys = [], []
            for ip, mp, fp in batch:
                pair = _load_pair(ip, mp, fp)
                if pair is None:
                    continue
                xs.append(pair[0])
                ys.append(pair[1])
            if not xs:
                continue
            x = torch.stack(xs).to(DEVICE)
            y = torch.stack(ys).to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y, bce)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            n_steps += 1
        scheduler.step()

        val_metrics = evaluate_segmentation(model, ddr_val)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, n_steps),
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_tversky": val_metrics["tversky"],
        }
        log["epochs"].append(epoch_log)
        print(f"  [{loss_kind}] epoch {epoch + 1}/{epochs}: "
              f"train_loss={epoch_log['train_loss']:.4f} "
              f"val_dice={val_metrics['dice']:.4f} "
              f"val_iou={val_metrics['iou']:.4f} "
              f"val_tversky={val_metrics['tversky']:.4f}")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log["best_val_dice"] = best_dice
            log["best_epoch"] = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["bce_dice", "tversky"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-idrid-train", action="store_true",
                        help="EXCLUDE IDRiD's 54 training images from the "
                             "training pool. Default: IDRiD is INCLUDED. "
                             "IDRiD shares DDR's lesion-segmentation label "
                             "space (4 lesion types: MA/HE/EX/SE) and adds "
                             "14% more diverse-annotation training data "
                             "(54/383). IDRiD test (27 images) is always "
                             "held out for cross-domain evaluation "
                             "regardless of this flag.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write best.pt. Also: ../logs/<name>/ "
                             "is auto-created for the JSON log.")
    parser.add_argument("--name", default=None,
                        help="Optional run name (default: derived from output-dir).")
    args = parser.parse_args()

    if not DDR_SEG_ROOT.is_dir():
        print(f"ERROR: DDR lesion segmentation not found at {DDR_SEG_ROOT}.")
        return 1
    if not (IDRID_IMG_DIR.is_dir() and IDRID_LAB_DIR.is_dir()):
        print(f"ERROR: IDRiD not found at {IDRID_IMG_DIR} / {IDRID_LAB_DIR}.")
        return 1

    print(f"Collecting DDR lesion-segmentation samples (train)...")
    ddr_train_full = collect_ddr_samples("train")
    if len(ddr_train_full) == 0:
        print("ERROR: no DDR train samples. Stop.")
        return 1
    ddr_train, ddr_val = split_ddr(ddr_train_full, n_splits=4, seed=args.seed)
    print(f"  DDR: {len(ddr_train)} train / {len(ddr_val)} val")

    idrid_train = None
    if not args.no_idrid_train:
        from src.segmentation.ablation_3arm import collect_idrid_samples
        idrid_train = collect_idrid_samples()  # 54 samples
        print(f"  IDRiD train (auxiliary): {len(idrid_train)} samples")
    else:
        print(f"  IDRiD train (auxiliary): excluded via --no-idrid-train")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or out_dir.name
    log_dir = Path("saved/logs") / name
    log_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, log = train_one(args.loss, ddr_train, ddr_val,
                           idrid_train=idrid_train,
                           epochs=args.epochs, batch_size=args.batch_size,
                           lr=args.lr, seed=args.seed)
    elapsed = time.time() - t0

    # Save checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "loss": args.loss,
        "epochs": args.epochs,
        "best_val_dice": log["best_val_dice"],
        "best_epoch": log["best_epoch"],
        "encoder": "efficientnet_b4",
        "img_size": 384,
        "training_source": "ddr" + ("+idrid" if idrid_train else ""),
    }
    ckpt_path = out_dir / "best.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Final eval on IDRiD test (cross-domain) and DDR test (in-domain).
    idrid_test = _collect_idrid_test_samples()
    ddr_test = collect_ddr_samples("test")
    idrid_test_metrics = evaluate_segmentation(model, idrid_test) if idrid_test else {}
    ddr_test_metrics = evaluate_segmentation(model, ddr_test) if ddr_test else {}
    log["idrid_test_metrics"] = idrid_test_metrics
    log["ddr_test_metrics"] = ddr_test_metrics
    log["elapsed_sec"] = elapsed

    log_path = log_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")
    print(f"Final IDRiD test metrics: {idrid_test_metrics}")
    print(f"Final DDR test metrics:    {ddr_test_metrics}")
    return 0


def _collect_idrid_test_samples() -> list[IDRiDSample]:
    """Build IDRiDSample list from the IDRiD testing-set directory only.
    The IDRiD directory layout is::

        1. Original Images/
            a. Training Set/    <- IDRID_IMG_DIR points here
            b. Testing Set/     <- this is what we want for evaluation
        2. All Segmentation Groundtruths/
            a. Training Set/    <- IDRID_LAB_DIR points here
            b. Testing Set/     <- this is what we want for evaluation

    So the test set is a *sibling* of the train set, not a child.
    """
    test_img_dir = IDRID_IMG_DIR.parent / "b. Testing Set"
    test_lab_dir = IDRID_LAB_DIR.parent / "b. Testing Set"
    if not (test_img_dir.is_dir() and test_lab_dir.is_dir()):
        return []
    samples: list[IDRiDSample] = []
    for img_path in sorted(test_img_dir.glob("*.jpg")):
        stem = img_path.stem
        per_lesion: dict[str, str] = {}
        union = None
        for lesion in LESION_TYPES:
            lab_path = test_lab_dir / _idrid_lesion_folder(lesion) / f"{stem}_{lesion}.tif"
            if not lab_path.exists():
                continue
            per_lesion[lesion] = str(lab_path)
            m = cv2.imread(str(lab_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            union = m if union is None else np.maximum(union, m)
        if union is None:
            continue
        all_dir = test_lab_dir / "0_all"
        all_dir.mkdir(exist_ok=True)
        all_path = all_dir / f"{stem}_ALL.png"
        if not all_path.exists():
            cv2.imwrite(str(all_path), union)
        samples.append(IDRiDSample(
            image_path=str(img_path),
            stem=stem,
            union_mask_path=str(all_path),
            per_lesion_mask_paths=per_lesion,
        ))
    return samples


if __name__ == "__main__":
    raise SystemExit(main())
