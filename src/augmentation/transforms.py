"""
Augmentation strategy per plan Section 3.

USE: full 0-360 rotation, h/v flip, mild zoom (90-110%), mild
brightness/contrast jitter, small translation, MixUp (optional/ablatable).

AVOID (deliberately not implemented here — see plan for rationale):
  - CutMix: pastes local patches across images, can create anatomically
    incoherent training images.
  - Elastic deformation: warps vessel/lesion geometry.
  - Aggressive hue/saturation shift: pushes outside realistic fundus palette.
  - GAN-generated synthetic augmentation: hallucinated lesion risk.
  - Aggressive random erasing / large-block cutout: can erase the single
    microaneurysm that determines a Mild vs. No-DR label.

Augmentation strength is lighter during EyePACS phase (large dataset, less
regularization pressure needed) and can go slightly heavier during APTOS
fine-tune (small dataset, higher overfitting risk) — controlled via the
`strength` param.
"""
import random

import numpy as np
import torch
from torchvision import transforms as T


def build_train_transforms(strength: str = "light") -> T.Compose:
    """
    strength: "light" (EyePACS phase) or "heavy" (APTOS fine-tune phase).

    IMPORTANT: DRDataset (src/data/datasets.py) already runs crop/color-
    correction/normalization and hands back a normalized float32 CHW
    tensor — NOT a PIL image or raw ndarray. These transforms must operate
    directly on that tensor, so we deliberately do not include ToTensor()
    or anything expecting PIL input.

    ColorJitter's brightness/contrast operate additively/multiplicatively
    on already-normalized (mean-subtracted) values here, which is a mild
    approximation of jitter in true pixel space — acceptable since jitter
    strength is small (0.1-0.15) and the goal is regularization, not exact
    photometric realism.
    """
    if strength not in ("light", "heavy"):
        raise ValueError(f"strength must be 'light' or 'heavy', got {strength!r}")

    if strength == "light":
        zoom_range = (0.95, 1.05)
        translate_frac = 0.03
        brightness, contrast = 0.1, 0.1
    else:  # heavy
        zoom_range = (0.90, 1.10)
        translate_frac = 0.06
        brightness, contrast = 0.15, 0.15

    return T.Compose([
        T.RandomRotation(degrees=180),  # sampling +-180 covers the full 0-360 range
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(
            degrees=0,  # rotation already handled above
            translate=(translate_frac, translate_frac),
            scale=zoom_range,
        ),
        T.ColorJitter(brightness=brightness, contrast=contrast),
    ])


def build_eval_transforms() -> T.Compose:
    """No augmentation for val/test — deterministic evaluation. Identity pass-through."""
    return T.Compose([])


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """
    Whole-image linear-blend MixUp. Label follows the blend ratio.

    Semantically defensible here because DR grades are ordinal — blending
    adjacent severities isn't nonsensical the way it would be for unrelated
    object classes (plan Section 3).

    images: (B, C, H, W) float tensor.
    labels: (B,) long tensor of ordinal class indices 0-4.

    Returns mixed_images, labels_a, labels_b, lam — caller combines losses
    as: lam * loss(pred, labels_a) + (1 - lam) * loss(pred, labels_b)
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[perm]
    labels_a, labels_b = labels, labels[perm]
    return mixed_images, labels_a, labels_b, lam


def maybe_apply_mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2,
                       p: float = 0.5, enabled: bool = True):
    """
    Stochastic MixUp wrapper for use inside the training loop.
    Returns (images, labels_a, labels_b, lam). When MixUp doesn't fire,
    labels_a == labels_b == labels and lam == 1.0, so the caller's loss
    combination collapses to the normal single-label loss.
    """
    if not enabled or random.random() > p:
        return images, labels, labels, 1.0
    return mixup_batch(images, labels, alpha=alpha)
