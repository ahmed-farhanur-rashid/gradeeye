"""
3-arm segmentation ablation with 5-fold CV on IDRiD.

Arms:
  (a) baseline   — DRIVE + IDRiD union, equal-weight loss (current `finetune_unet`)
  (b) reweighted — DRIVE + IDRiD union, per-image loss normalized by own positive-pixel count
  (c) idrid_only — IDRiD only, equal-weight loss (DRIVE branch disabled)

All arms share the same U-Net architecture (EfficientNet-B0 encoder, 4-stage
Conv-BN-ReLU decoder), the same ImageNet-pretrained encoder init, the same
input size (384×384 RGB), the same Dice+BCE loss family, and the same
training schedule (AdamW, lr=1e-4, cosine, gradient clip 1.0). The ONLY
difference between arms is the loss aggregation weight strategy + which
dataset enters the train pool.

5-fold CV is run on IDRiD's 54 images (random shuffle, seed=42). For each
fold, the U-Net is fine-tuned for N epochs and evaluated on the held-out
fold's IDRiD images. DRIVE (when used) is always added to the train pool
on every fold — DRIVE has no shared-anatomy overlap with IDRiD lesion
labels, so it never enters the validation set.

Per-lesion metrics (MA / HE / EX / SE) are computed against the model's
single sigmoid output (threshold=0.5), compared with each lesion type's
separate ground-truth mask. This measures how well a model trained on the
union mask recovers each lesion subtype.

Outputs:
  - saved/logs/segmentation_ablation/abl_3arm_<arm>.json   (per-arm detail)
  - saved/logs/segmentation_ablation/abl_3arm_summary.md   (markdown table)
  - saved/checkpoints/seg_ablation_<arm>_fold<N>.pt        (per-fold ckpt)

Library code only — no CLI. Run via:
    python -m src.segmentation.ablation_3arm
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.segmentation import build_unet_vessel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# IDRiD dataset discovery (with per-lesion-type labels)
# ─────────────────────────────────────────────────────────────────────────────

IDRID_IMG_DIR = Path("data/raw/A. Segmentation/1. Original Images/a. Training Set")
IDRID_LAB_DIR = Path("data/raw/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set")
LESION_TYPES = ("MA", "HE", "EX", "SE")  # microaneurysms / haemorrhages / exudates / soft exudates


@dataclass(frozen=True)
class IDRiDSample:
    """One IDRiD image with its per-lesion ground-truth masks."""
    image_path: str
    stem: str
    union_mask_path: str  # cached MA∪HE∪EX∪SE
    per_lesion_mask_paths: dict[str, str] = field(default_factory=dict)


def collect_idrid_samples() -> list[IDRiDSample]:
    """Walk IDRiD's training set and return one IDRiDSample per image.

    The unified (OR'd) mask is cached as `_ALL.png` so we don't recompute
    it on every fold.
    """
    if not IDRID_IMG_DIR.is_dir() or not IDRID_LAB_DIR.is_dir():
        raise FileNotFoundError(
            f"IDRiD not found at {IDRID_IMG_DIR} / {IDRID_LAB_DIR}. "
            f"Aborting ablation — at minimum IDRiD must be on disk."
        )

    samples: list[IDRiDSample] = []
    all_dir = IDRID_LAB_DIR / "0_all"
    all_dir.mkdir(exist_ok=True)

    for img_path in sorted(IDRID_IMG_DIR.glob("*.jpg")):
        stem = img_path.stem  # "IDRiD_01"
        per_lesion: dict[str, str] = {}
        union = None
        for lesion in LESION_TYPES:
            lab_path = (
                IDRID_LAB_DIR / f"{_idrid_lesion_folder(lesion)}" / f"{stem}_{lesion}.tif"
            )
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
        samples.append(IDRiDSample(
            image_path=str(img_path),
            stem=stem,
            union_mask_path=str(all_path),
            per_lesion_mask_paths=per_lesion,
        ))
    return samples


def _idrid_lesion_folder(lesion: str) -> str:
    return {
        "MA": "1. Microaneurysms",
        "HE": "2. Haemorrhages",
        "EX": "3. Hard Exudates",
        "SE": "4. Soft Exudates",
    }[lesion]


def collect_drive_samples() -> list[tuple[str, str, str | None]]:
    """Return [(image_path, mask_path, fov_path_or_None)] for DRIVE training set.

    Mirrors `_collect_drive_idrid_pairs` from `_03_segmentation.py` but
    isolated here so the ablation script doesn't depend on it.
    """
    pairs: list[tuple[str, str, str | None]] = []
    drive_dir = Path("data/raw/drive")
    if not drive_dir.is_dir():
        return pairs
    img_dir = drive_dir / "training" / "images"
    lab_dir = drive_dir / "training" / "1st_manual"
    fov_dir = drive_dir / "training" / "mask"
    if not (img_dir.is_dir() and lab_dir.is_dir()):
        return pairs
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix != ".tif":
            continue
        lab_name = img_path.name.replace("_training.tif", "_manual1.gif")
        fov_name = img_path.name.replace("_training.tif", "_training_mask.gif")
        lab_path = lab_dir / lab_name
        fov_path = fov_dir / fov_name
        if lab_path.exists():
            pairs.append((str(img_path), str(lab_path), str(fov_path) if fov_path.exists() else None))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Loss variants
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _normalize(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def _load_pair(image_path: str, mask_path: str, fov_path: str | None,
               size: int = 384) -> tuple[torch.Tensor, torch.Tensor] | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # U-Net downsamples by 32 → match loss to its output resolution.
    mask = cv2.resize(mask, (size // 2, size // 2), interpolation=cv2.INTER_NEAREST)
    if fov_path is not None and os.path.exists(fov_path):
        fov = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
        if fov is not None:
            fov = cv2.resize(fov, (size // 2, size // 2), interpolation=cv2.INTER_NEAREST)
            mask = (mask * (fov > 127)).astype(np.uint8)

    return _normalize(img), torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3))
    den = (p + target).sum(dim=(1, 2, 3)) + eps
    return (1 - (num / den)).mean()


def baseline_loss(logits: torch.Tensor, target: torch.Tensor, bce: nn.Module) -> torch.Tensor:
    """Equal-weight 0.5·BCE + 0.5·Dice, summed uniformly across the batch
    (current `finetune_unet` behavior)."""
    return 0.5 * bce(logits, target) + 0.5 * _dice_loss(logits, target)


def reweighted_loss(logits: torch.Tensor, target: torch.Tensor, bce: nn.Module,
                    eps: float = 1e-3) -> torch.Tensor:
    """Per-image loss normalized by that image's positive-pixel count.

    Each image in the batch contributes:
        L_i = ( 0.5·BCE(L_i, T_i) + 0.5·Dice(L_i, T_i) ) / ( |T_i|_1 + ε )

    and we sum across the batch. This makes each image contribute
    `1 unit of loss` regardless of how many positive pixels it has —
    DRIVE's dense vessel masks no longer drown out IDRiD's sparse
    lesion masks. ε=1e-3 prevents divide-by-zero on all-zero masks.
    """
    # Per-image BCE (reduction='none' then mean over spatial dims)
    bce_per_img = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).mean(dim=(1, 2, 3))

    # Per-image Dice loss
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    den = (p + target).sum(dim=(1, 2, 3)) + 1e-6
    dice_per_img = 1 - (2 * inter / den)

    pos_count = target.sum(dim=(1, 2, 3))  # (B,)
    per_image = 0.5 * bce_per_img + 0.5 * dice_per_img
    return (per_image / (pos_count + eps)).sum()


def idrid_only_loss(logits: torch.Tensor, target: torch.Tensor, bce: nn.Module) -> torch.Tensor:
    """Same as baseline but only used on IDRiD samples (no DRIVE in train pool).
    Keep the function shape consistent so the training loop is arm-agnostic."""
    return baseline_loss(logits, target, bce)


LOSS_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor]] = {
    "baseline": baseline_loss,
    "reweighted": reweighted_loss,
    "idrid_only": idrid_only_loss,
}


# ─────────────────────────────────────────────────────────────────────────────
# Train + eval
# ─────────────────────────────────────────────────────────────────────────────

def train_one_arm(
    arm: str,
    train_pairs: Iterable[tuple[str, str, str | None]],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> nn.Module:
    """Fine-tune a fresh U-Net on `train_pairs` using the arm's loss variant."""
    if arm not in LOSS_FNS:
        raise ValueError(f"Unknown arm {arm!r}. Valid: {list(LOSS_FNS)}")
    loss_fn = LOSS_FNS[arm]

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_unet_vessel(pretrained=True).to(DEVICE)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_pairs = list(train_pairs)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_steps = 0
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
        print(f"  [{arm}] epoch {epoch + 1}/{epochs}: loss={running / max(1, n_steps):.4f}")
    return model


@torch.no_grad()
def predict_mask(model: nn.Module, image_path: str, size: int = 384) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((size // 2, size // 2), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = _normalize(img).unsqueeze(0).to(DEVICE)
    with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu",
                        dtype=torch.bfloat16):
        logits = model(x)
    if logits.shape[-1] != size // 2:
        logits = F.interpolate(logits, size=(size // 2, size // 2),
                                mode="bilinear", align_corners=False)
    prob = torch.sigmoid(logits.float()).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def _safe_div(a: float, b: float, eps: float = 1e-6) -> float:
    if b < eps:
        return 0.0
    return a / b


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute Dice and IoU on binary masks. Returns (0, 0) on empty inputs."""
    p = pred.astype(bool)
    t = target.astype(bool)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    p_sum = p.sum()
    t_sum = t.sum()
    dice = _safe_div(2 * inter, p_sum + t_sum)
    iou = _safe_div(inter, union)
    return dice, iou


def evaluate_on_idrid_fold(model: nn.Module, fold_samples: list[IDRiDSample]) -> dict:
    """Score the model on a held-out IDRiD fold. Reports Dice/IoU per lesion
    type (MA / HE / EX / SE) plus a union mask score."""
    metrics: dict[str, list[float]] = {
        "union": [0.0, 0.0],  # [dice, iou]
        "MA": [0.0, 0.0],
        "HE": [0.0, 0.0],
        "EX": [0.0, 0.0],
        "SE": [0.0, 0.0],
    }
    n = len(fold_samples)
    if n == 0:
        return metrics

    for sample in fold_samples:
        pred = predict_mask(model, sample.image_path)
        # Union metric: pred vs union GT
        gt_union = cv2.imread(sample.union_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_union is not None:
            gt_union = cv2.resize(gt_union, (pred.shape[1], pred.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            d, i = dice_iou(pred, gt_union)
            metrics["union"][0] += d
            metrics["union"][1] += i
        # Per-lesion metrics
        for lesion, lab_path in sample.per_lesion_mask_paths.items():
            gt = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            d, i = dice_iou(pred, gt)
            metrics[lesion][0] += d
            metrics[lesion][1] += i

    for k in metrics:
        metrics[k] = [m / n for m in metrics[k]]
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    name: str
    use_drive: bool
    loss_variant: str  # key in LOSS_FNS
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    seed: int = 42


ARMS: list[AblationConfig] = [
    AblationConfig(name="baseline", use_drive=True, loss_variant="baseline", epochs=8),
    AblationConfig(name="reweighted", use_drive=True, loss_variant="reweighted", epochs=8),
    AblationConfig(name="idrid_only", use_drive=False, loss_variant="idrid_only", epochs=8),
]


def run_arm(arm: AblationConfig, idrid_samples: list[IDRiDSample],
            drive_pairs: list[tuple[str, str, str | None]],
            n_folds: int = 5,
            ckpt_dir: str = "saved/checkpoints/seg_ablation",
            results_dir: str = "saved/logs/segmentation_ablation") -> dict:
    print(f"\n{'=' * 70}\nARM: {arm.name}  (drive={arm.use_drive}, loss={arm.loss_variant})\n{'=' * 70}")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=arm.seed)
    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(idrid_samples), start=1):
        print(f"\n--- {arm.name} fold {fold_idx}/{n_folds} ---")
        idrid_train = [idrid_samples[i] for i in train_idx]
        idrid_val = [idrid_samples[i] for i in val_idx]

        # Build train pool: IDRiD always; DRIVE only if arm uses it.
        train_pairs: list[tuple[str, str, str | None]] = (
            [(s.image_path, s.union_mask_path, None) for s in idrid_train]
        )
        if arm.use_drive:
            train_pairs.extend(drive_pairs)

        ckpt_path = os.path.join(ckpt_dir, f"{arm.name}_fold{fold_idx}.pt")
        model = train_one_arm(
            arm=arm.loss_variant,
            train_pairs=train_pairs,
            epochs=arm.epochs,
            batch_size=arm.batch_size,
            lr=arm.lr,
            seed=arm.seed + fold_idx,
        )
        torch.save(model.state_dict(), ckpt_path)

        metrics = evaluate_on_idrid_fold(model, idrid_val)
        print(f"  fold {fold_idx} metrics: {metrics}")
        fold_metrics.append({
            "fold": fold_idx,
            "n_train": len(idrid_train),
            "n_train_with_drive": len(train_pairs),
            "n_val": len(idrid_val),
            "metrics": metrics,
        })

    # Aggregate mean ± std across folds
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for label in ("union", "MA", "HE", "EX", "SE"):
        dices = [m["metrics"][label][0] for m in fold_metrics]
        ious = [m["metrics"][label][1] for m in fold_metrics]
        agg[label] = {
            "dice": (float(np.mean(dices)), float(np.std(dices, ddof=1)) if len(dices) > 1 else 0.0),
            "iou": (float(np.mean(ious)), float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0),
        }

    summary = {
        "arm": arm.name,
        "use_drive": arm.use_drive,
        "loss_variant": arm.loss_variant,
        "epochs": arm.epochs,
        "batch_size": arm.batch_size,
        "lr": arm.lr,
        "random_seed": arm.seed,
        "n_folds": n_folds,
        "n_idrid_total": len(idrid_samples),
        "n_drive_total": len(drive_pairs),
        "per_fold": fold_metrics,
        "aggregate_mean_std": agg,
    }
    out_path = os.path.join(results_dir, f"abl_3arm_{arm.name}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wrote {out_path}")
    return summary


def write_summary_table(results: list[dict], path: str) -> None:
    lines = ["# Segmentation 3-Arm Ablation — 5-fold CV on IDRiD",
             "",
             "All arms share: U-Net (EfficientNet-B0 encoder, ImageNet-pretrained), "
             "Dice+BCE loss, 384×384 input, AdamW lr=1e-4, cosine schedule, "
             "gradient clip 1.0. Only the loss aggregation + dataset pool differ.",
             "",
             "Metrics: mean ± std across 5 folds. "
             "Per-lesion Dice/IoU is computed against the model's single "
             "sigmoid output (threshold=0.5), compared to each lesion type's "
             "separate ground truth (MA / HE / EX / SE) and the union mask.",
             "",
             "| Arm | n_train (drive+idrid) | "
             "Union Dice | Union IoU | MA Dice | MA IoU | "
             "HE Dice | HE IoU | EX Dice | EX IoU | SE Dice | SE IoU |",
             "|-----|----------------------|"
             "-----------|-----------|---------|---------|"
             "---------|---------|---------|---------|---------|---------|",
             ]
    for r in results:
        a = r["aggregate_mean_std"]
        n_train = r["per_fold"][0]["n_train_with_drive"]
        row = [
            r["arm"],
            str(n_train),
            f"{a['union']['dice'][0]:.3f}±{a['union']['dice'][1]:.3f}",
            f"{a['union']['iou'][0]:.3f}±{a['union']['iou'][1]:.3f}",
            f"{a['MA']['dice'][0]:.3f}±{a['MA']['dice'][1]:.3f}",
            f"{a['MA']['iou'][0]:.3f}±{a['MA']['iou'][1]:.3f}",
            f"{a['HE']['dice'][0]:.3f}±{a['HE']['dice'][1]:.3f}",
            f"{a['HE']['iou'][0]:.3f}±{a['HE']['iou'][1]:.3f}",
            f"{a['EX']['dice'][0]:.3f}±{a['EX']['dice'][1]:.3f}",
            f"{a['EX']['iou'][0]:.3f}±{a['EX']['iou'][1]:.3f}",
            f"{a['SE']['dice'][0]:.3f}±{a['SE']['dice'][1]:.3f}",
            f"{a['SE']['iou'][0]:.3f}±{a['SE']['iou'][1]:.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"\nWrote summary table: {path}")


def main() -> int:
    print("Collecting IDRiD samples...")
    idrid_samples = collect_idrid_samples()
    print(f"  IDRiD: {len(idrid_samples)} images")
    drive_pairs = collect_drive_samples()
    print(f"  DRIVE: {len(drive_pairs)} images")

    results = []
    for arm in ARMS:
        results.append(run_arm(arm, idrid_samples, drive_pairs))

    summary_path = "saved/logs/segmentation_ablation/abl_3arm_summary.md"
    write_summary_table(results, summary_path)

    # Print summary to stdout for visibility
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        a = r["aggregate_mean_std"]
        print(f"\n{r['arm']}:")
        for label in ("union", "MA", "HE", "EX", "SE"):
            d = a[label]["dice"]
            i = a[label]["iou"]
            print(f"  {label:>5s}: Dice={d[0]:.3f}±{d[1]:.3f}  IoU={i[0]:.3f}±{i[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
