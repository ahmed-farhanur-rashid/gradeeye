"""
DDR lesion-segmentation dataset.

Reads the DDR lesion_segmentation subset (757 fundus images) and returns
(image, mask) pairs where the mask is the union of all lesion TIFs for that
image (binary: any-lesion / no-lesion).

Layout expected on disk (matches /home/farhan/my-projects/gradeeye/data/ddr):

    DDR-dataset/lesion_segmentation/
        train/
            image/<id>.jpg
            label/EX/<id>.tif, HE/<id>.tif, MA/<id>.tif, SE/<id>.tif
        valid/
            image/<id>.jpg
            segmentation label/EX/<id>.tif, ...
        test/
            image/<id>.jpg
            label/EX/<id>.tif, ...

Note: train/test use `label/`, valid uses `segmentation label/` (with space).
We normalize to a single convention internally.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DDR_ROOT_DEFAULT = (
    "/home/farhan/my-projects/gradeeye/data/ddr/DDR-dataset/lesion_segmentation"
)
LESION_TYPES = ("EX", "HE", "MA", "SE")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_label_dir(split_dir: Path) -> Path:
    """train/test use `label/`, valid uses `segmentation label/`."""
    candidate = split_dir / "label"
    if candidate.exists():
        return candidate
    candidate = split_dir / "segmentation label"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No lesion label directory found under {split_dir}")


def list_ddr_ids(split: str, root: str = DDR_ROOT_DEFAULT) -> list[str]:
    """List image ids for a split (train/valid/test)."""
    split_dir = Path(root) / split
    image_dir = split_dir / "image"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing {image_dir}")
    return sorted(p.stem for p in image_dir.glob("*.jpg"))


class DDRLesionDataset(Dataset):
    """
    Returns (image_tensor, mask_tensor) where:
      - image_tensor: float32 (3, H, W) ImageNet-normalized RGB
      - mask_tensor:  float32 (1, H, W) binary {0,1}

    The fused mask is the OR of all available lesion TIFs (EX/HE/MA/SE).
    Images missing a lesion TIF contribute nothing to that lesion class —
    they're treated as zero for that type, which is correct because DDR
    only annotates lesions that are present.
    """

    def __init__(
        self,
        split: str,
        img_size: int = 384,
        root: str = DDR_ROOT_DEFAULT,
        augment: bool = False,
        ids: Sequence[str] | None = None,
    ):
        self.split = split
        self.root = Path(root)
        self.img_size = img_size
        self.augment = augment

        split_dir = self.root / split
        self.image_dir = split_dir / "image"
        self.label_dir = _resolve_label_dir(split_dir)

        if ids is None:
            ids = list_ddr_ids(split, str(self.root))
        self.ids = list(ids)

        if augment:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size,
                                   border_mode=cv2.BORDER_CONSTANT, fill=0,
                                   fill_mask=0),
                    A.Affine(
                        scale=(0.85, 1.15),
                        translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                        rotate=(-20, 20),
                        shear=(-8, 8),
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=0,
                        p=0.8,
                    ),
                    A.HorizontalFlip(p=0.5),
                    # Heavy color augmentation to bridge DDR (dark, blue-tinted)
                    # → APTOS/EyePACS/Messidor-2 (brighter, varied) domain shift.
                    A.RandomBrightnessContrast(brightness_limit=0.35,
                                                contrast_limit=0.35, p=0.8),
                    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25,
                                          val_shift_limit=20, p=0.6),
                    A.RandomGamma(gamma_limit=(70, 130), p=0.5),
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 5)),
                        A.MedianBlur(blur_limit=5),
                        A.GaussNoise(std_range=(0.03, 0.08)),
                    ], p=0.3),
                    A.CoarseDropout(num_holes_range=(1, 4),
                                     hole_height_range=(8, 32),
                                     hole_width_range=(8, 32),
                                     fill=0, fill_mask=0, p=0.2),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size,
                                   border_mode=cv2.BORDER_CONSTANT, fill=0,
                                   fill_mask=0),
                ]
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _load_image(self, image_id: str) -> np.ndarray:
        path = self.image_dir / f"{image_id}.jpg"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_fused_mask(self, image_id: str) -> np.ndarray:
        """OR across all lesion types. Returns uint8 {0,255} at native res."""
        mask = None
        h, w = None, None
        for lesion in LESION_TYPES:
            path = self.label_dir / lesion / f"{image_id}.tif"
            if not path.exists():
                continue
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if mask is None:
                mask = (m > 0).astype(np.uint8) * 255
                h, w = mask.shape
            else:
                if m.shape != mask.shape:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, (m > 0).astype(np.uint8) * 255)
        if mask is None:
            # No lesion annotation at all -> empty mask
            if h is None:
                # Shouldn't happen — at least the image exists
                raise RuntimeError(f"No lesion masks found for {image_id}")
            mask = np.zeros((h, w), dtype=np.uint8)
        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[idx]
        image = self._load_image(image_id)
        mask = self._load_fused_mask(image_id)

        # Albumentations expects uint8 HWC for image, uint8 HW for mask.
        out = self.transform(image=image, mask=mask)
        image = out["image"]  # uint8 HWC at img_size
        mask = out["mask"]    # uint8 HW at img_size

        # Normalize image to float32 (3,H,W) with ImageNet stats.
        image = image.astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()  # (3,H,W)

        # Mask to float32 {0,1}, add channel dim
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor