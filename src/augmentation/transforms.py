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


class _MaskSafeCompose:
    """
    Apply a 2-stage pipeline to a (C, H, W) tensor where C is 3 or 4:
      1. `geometric` runs on ALL channels (RGB + mask must move together
         for rotation/flip/affine/resize).
      2. `color` runs on RGB only (so ColorJitter can stay 3-channel) and
         the mask is re-attached afterwards.

    This avoids the trap of splitting RGB from mask *before* applying
    geometric ops — if the geometric pipeline includes a Resize (e.g. Swin
    needs 256×256 but preprocessed images are 384×384), splitting first
    leaves the mask at 384 while RGB is at 256, then concat fails.
    """

    def __init__(self, geometric: T.Compose, color: T.Compose,
                 extra_channels: int = 1):
        self.geometric = geometric
        self.color = color
        self.extra_channels = extra_channels

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        C = t.shape[0]
        # Stage 1: geometric transforms apply to all channels.
        t = self.geometric(t)
        # Stage 2: color transforms apply to RGB only.
        if C == 3 + self.extra_channels:
            extras = t[3:]
            rgb = t[:3]
            rgb = self.color(rgb)
            return torch.cat([rgb, extras], dim=0)
        # No extras — run color over the full tensor (3-channel case).
        return self.color(t)


def build_train_transforms(strength: str = "light", img_size: int | None = None) -> T.Compose:
    """
    strength: "light" (EyePACS phase) or "heavy" (APTOS fine-tune phase).
    img_size: if set, resize the (already preprocessed) input to this size
              first. Useful when the backbone's native input resolution
              differs from preprocessing (e.g. SwinV2-Tiny@256 wants 256,
              but preprocessing writes 384 — without a resize, the model
              immediately asserts on input size mismatch).

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

    geometric_list = []
    if img_size is not None:
        # Must run BEFORE rotation/affine so the random crop/scale ops
        # operate on the right tensor size. Applies to ALL 4 channels so
        # the mask is resized alongside the RGB image.
        geometric_list.append(T.Resize((img_size, img_size), antialias=True))
    geometric_list.extend([
        T.RandomRotation(degrees=180),  # sampling +-180 covers the full 0-360 range
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(
            degrees=0,  # rotation already handled above
            translate=(translate_frac, translate_frac),
            scale=zoom_range,
        ),
    ])

    color_list = [
        T.ColorJitter(brightness=brightness, contrast=contrast),
    ]

    # NOTE: GaussianBlur and RandomErasing were intentionally NOT included
    # here. The training tensors are ALREADY ImageNet-normalized (mean-subtracted,
    # divided by std) by the time these transforms run, so:
    #   - GaussianBlur on normalized tensors applies wrong-scale smoothing
    #     and effectively injects noise (blurring a mean-centered image changes
    #     the semantic content).
    #   - RandomErasing on normalized tensors creates large negative-valued
    #     patches (mean-subtracted values) that don't correspond to any real
    #     fundus appearance, harming generalization.
    # Empirical test (revert 2026-07-28): adding either produced a 0.16 QWK
    # drop on Messidor2 zero-shot (0.61 -> 0.45). The augmented training
    # looked healthy on EyePACS val but failed to generalize.

    geometric = T.Compose(geometric_list)
    color = T.Compose(color_list)
    # Wrap with mask-safe compose: ColorJitter expects 3 channels; we have
    # 4 when seg_dir is enabled. Geometric transforms run first on ALL
    # channels (mask must move with image and be resized with image), then
    # color transforms split RGB from mask and apply color jitter to RGB
    # only.
    return _MaskSafeCompose(geometric, color, extra_channels=1)


def build_eval_transforms(img_size: int | None = None) -> T.Compose:
    """No augmentation for val/test — deterministic evaluation. Identity pass-through.
    If `img_size` is set, resize to that spatial size before returning."""
    if img_size is None:
        return T.Compose([])
    return T.Compose([T.Resize((img_size, img_size), antialias=True)])


def camera_color_jitter(images: torch.Tensor, channel_std: float = 0.08,
                         gamma_range: tuple[float, float] = (0.85, 1.15),
                         p: float = 1.0) -> torch.Tensor:
    """
    Simulate inter-camera color variation per-batch (cheap stand-in for
    RandStainNA which requires reference stain matrices we don't have).

    Operates on already-ImageNet-normalized tensors (mean-centered, std-scaled).
    Two perturbations:

    1. Per-channel multiplicative gain (simulates camera-amplifier / white
       balance differences between Topcon TRC NW6 [Messidor2], Canon CR
       [EyePACS], and various APTOS cameras):
         imgs *=  1 + N(0, channel_std)   per channel, sample-wise

    2. Per-image gamma correction (simulates illumination intensity variation):
         imgs = sign(imgs) * |imgs| ^ gamma  where gamma ~ U(0.85, 1.15)

    Why both?
    - Channel gain alone = white-balance noise only.
    - Gamma alone = illumination noise only.
    - Together they cover the dominant axes of cross-camera variation in
      fundus imagery without needing reference stain matrices.

    Important: shape (B, C, H, W), normalized values (mean-subtracted), so
    we DO NOT clip to [0, 255] (normalized tensors can be negative). Output
    stays in normalized space — what goes into the backbone is still
    on-stats.

    Args:
        images: (B, C, H, W) float tensor (already ImageNet-normalized).
        channel_std: per-channel multiplicative jitter (default 0.08 = ~8%).
        gamma_range: uniform per-image gamma (default 0.85-1.15).
        p: probability of applying (default 1.0, jitter is light enough to
           apply always).

    Returns: same shape, augmented in-place copy.
    """
    if random.random() > p:
        return images

    images = images.clone()
    B, C, H, W = images.shape
    device = images.device

    # 1. Per-channel, per-image multiplicative gain
    gain = 1.0 + torch.randn(B, C, 1, 1, device=device) * channel_std
    images = images * gain

    # 2. Per-image gamma (preserve sign for mean-centered values)
    gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(
        gamma_range[0], gamma_range[1])
    # sign-preserving power: sign(x) * |x|^gamma
    # For mean-centered tensors ~N(0,1), values are small, so the power is
    # well-behaved. Clamp magnitude away from 0 to avoid gradient issues.
    sign = torch.sign(images)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    magnitude = images.abs().clamp(min=1e-6)
    images = sign * magnitude.pow(gamma)

    return images


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

    # Memory-efficient: in-place lerp avoids allocating 2 temporaries
    # the size of the batch (caused OOM at bs=384 with 384×384 images).
    mixed_images = images.clone()
    mixed_images.lerp_(images[perm], 1 - lam)
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
