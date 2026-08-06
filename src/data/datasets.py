"""
PyTorch Dataset classes per source (EyePACS/APTOS/Messidor2).

All three are 5-class (0-4 ICDR) ordinal label sets. Expects a CSV/DataFrame
with columns: image_path, label (int 0-4).

IMPORTANT (per user requirement): every dataset used here must be exactly
5-class ICDR (0-4). See `src/data/_00_manifests.py` and
`src/data/stratified_split.py` for where label counts are validated.

Normalization:
  Defaults to ImageNet mean/std (RGB, [0,1] scale) — the pretrained backbones
  (ConvNeXt, EffNetV2, Swin) were all trained with these stats, so using a
  per-dataset mean/std silently undoes part of the pretraining.

  Per-dataset normalization is still supported via the optional `norm_mean`
  / `norm_std` arguments for back-compat with old experiments, but new runs
  should leave them as defaults.

Optional 4th-channel segmentation:
  If `seg_dir` is set, the dataset looks for a sibling mask file at
  `<seg_dir>/<filename>.png` for each image and concatenates it as a 4th
  channel. Used by Option A segmentation (pre-computed vessel masks).
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.normalize import normalize_image

NUM_CLASSES = 5  # 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR

# ImageNet normalization stats (RGB channel order, [0,1] pixel scale).
# These are the standard stats for ImageNet-pretrained timm backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DRDataset(Dataset):
    """
    Generic DR grading dataset. Reads preprocessed images from disk, applies
    ImageNet (or custom) normalization, optionally concatenates a 4th-channel
    segmentation mask, then hands the tensor to the given torchvision
    transform for augmentation.

    The full preprocessing pipeline (crop/pad/resize, color correction,
    optional anisotropic filter) is run offline by
    `python -m src.data.cli preprocess --dataset <source>` (see
    `src/data/_01_preprocess.py`) and the resulting PNGs are what this
    dataset reads.
    """

    def __init__(
        self,
        csv_path: str,
        norm_stats: dict | None = None,
        transform=None,
        seg_dir: str | None = None,
    ):
        """
        norm_stats: optional dict with keys "mean" and "std" (each a 3-tuple
                    of RGB channel values). If None (default), ImageNet stats
                    are used. Kept for back-compat with per-dataset norms.
        seg_dir:    if set, expected to contain segmentation mask PNGs with
                    the same basenames as the images. Loaded lazily and
                    concatenated as a 4th channel.
        """
        self.df = pd.read_csv(csv_path)
        self._validate_labels()

        if norm_stats is None:
            self.norm_mean = IMAGENET_MEAN
            self.norm_std = IMAGENET_STD
        else:
            self.norm_mean = norm_stats["mean"]
            self.norm_std = norm_stats["std"]
        self.transform = transform
        self.seg_dir = seg_dir

    def _validate_labels(self):
        labels = self.df["label"].unique()
        if not set(labels).issubset(set(range(NUM_CLASSES))):
            raise ValueError(
                f"Dataset contains labels outside 5-class range [0-4]: {sorted(labels)}. "
                f"This project requires exactly 5-class ICDR datasets throughout."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["label"])

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # cv2.imread loads BGR; pretrained ImageNet backbones expect RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = normalize_image(img, self.norm_mean, self.norm_std)

        # normalize_image returns float32 HWC in RGB-normalized space;
        # convert to a uint8-range-free tensor directly rather than routing
        # back through PIL, since values are already normalized.
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (3, H, W)

        # Optional segmentation mask as 4th channel (Option A).
        if self.seg_dir is not None:
            seg_path = os.path.join(self.seg_dir, os.path.basename(image_path))
            mask = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # Missing mask -> zeros. Better to fail loudly, but zeros let
                # training continue if a single file is corrupted.
                mask = np.zeros(img_tensor.shape[1:], dtype=np.uint8)
            else:
                if mask.shape != img_tensor.shape[1:]:
                    mask = cv2.resize(mask, (img_tensor.shape[2], img_tensor.shape[1]),
                                      interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
            img_tensor = torch.cat([img_tensor, mask_tensor], dim=0)  # (4, H, W)

        if self.transform is not None:
            # transform pipelines built in augmentation/transforms.py expect
            # tensor-compatible ops (RandomAffine etc. work on tensors in
            # recent torchvision); if a PIL-based transform is swapped in,
            # convert accordingly.
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

    def get_labels(self) -> np.ndarray:
        return self.df["label"].to_numpy()


def build_manifest_csv(image_dir: str, labels_csv: str, out_csv: str,
                        image_col: str = "id_code", label_col: str = "diagnosis",
                        image_ext: str = ".png") -> str:
    """
    Utility to build the image_path,label manifest CSV format DRDataset
    expects, from a raw (image_id, label) labels file as typically shipped
    with these Kaggle datasets.
    """
    df = pd.read_csv(labels_csv)
    df["image_path"] = df[image_col].apply(lambda x: os.path.join(image_dir, f"{x}{image_ext}"))
    out_df = df[["image_path", label_col]].rename(columns={label_col: "label"})

    labels = out_df["label"].unique()
    if not set(labels).issubset(set(range(NUM_CLASSES))):
        raise ValueError(
            f"Labels file {labels_csv} contains classes outside 0-4: {sorted(labels)}. "
            f"Refusing to build manifest — this project requires exactly 5-class datasets."
        )

    out_df.to_csv(out_csv, index=False)
    return out_csv
