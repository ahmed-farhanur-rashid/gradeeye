"""
PyTorch Dataset classes per source (EyePACS/APTOS/Messidor2).

All three are 5-class (0-4 ICDR) ordinal label sets. Expects a CSV/DataFrame
with columns: image_path, label (int 0-4).

IMPORTANT (per user requirement): every dataset used here must be exactly
5-class ICDR (0-4). See scripts/download_datasets.py and
src/data/stratified_split.py for where label counts are validated.
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.crop_and_resize import crop_pad_resize
from src.preprocessing.color_correction import color_correction_pipeline
from src.preprocessing.anisotropic_filter import apply_anisotropic_filter
from src.preprocessing.normalize import normalize_image

NUM_CLASSES = 5  # 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR


class DRDataset(Dataset):
    """
    Generic DR grading dataset. Reads raw images from disk, applies the
    full preprocessing pipeline (crop/pad/resize -> color correction ->
    optional anisotropic filter -> per-source normalization), then hands
    the result to the given torchvision transform for augmentation.
    """

    def __init__(self, csv_path: str, norm_stats: dict, transform=None,
                 use_anisotropic_filter: bool = False, use_all_channel_clahe: bool = False):
        self.df = pd.read_csv(csv_path)
        self._validate_labels()

        self.norm_mean = norm_stats["mean"]
        self.norm_std = norm_stats["std"]
        self.transform = transform
        self.use_anisotropic_filter = use_anisotropic_filter
        self.use_all_channel_clahe = use_all_channel_clahe

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

        img = crop_pad_resize(img)
        img = color_correction_pipeline(img, use_all_channel_clahe=self.use_all_channel_clahe)
        img = apply_anisotropic_filter(img, enabled=self.use_anisotropic_filter)
        img = normalize_image(img, self.norm_mean, self.norm_std)

        # normalize_image returns float32 HWC in BGR-normalized space;
        # convert to a uint8-range-free tensor directly rather than routing
        # back through PIL, since values are already normalized.
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()

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
