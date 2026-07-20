"""
Per-source normalization: mean/std computed independently for EyePACS,
APTOS, and Messidor-2 (plan Section 2 step 6 / handoff note 6).

Do NOT pool statistics across sources — each dataset was captured with
different camera hardware and clinic lighting conditions, so a shared
normalization would bias the network toward whichever source dominates.
"""
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def compute_dataset_stats(image_paths: list[str], sample_limit: int | None = None) -> dict:
    """
    Compute per-channel mean/std (in [0, 1] scale, RGB order matching
    pretrained ImageNet backbones) over a list of preprocessed image paths.

    sample_limit: if set, subsample this many images for speed on very
    large datasets (e.g. EyePACS's 35k images) rather than reading all.
    """
    paths = image_paths
    if sample_limit is not None and len(paths) > sample_limit:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(paths), size=sample_limit, replace=False)
        paths = [paths[i] for i in idx]

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in tqdm(paths, desc="Computing normalization stats"):
        img = cv2.imread(path)
        if img is None:
            continue
        # Convert BGR (cv2 default) → RGB to match training-time channel order.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float64) / 255.0
        pixel_sum += img.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)
        pixel_count += img.shape[0] * img.shape[1]

    mean = pixel_sum / pixel_count
    variance = (pixel_sq_sum / pixel_count) - mean ** 2
    std = np.sqrt(np.clip(variance, a_min=1e-8, a_max=None))

    return {"mean": mean.tolist(), "std": std.tolist(), "n_images_sampled": len(paths)}


def save_stats(stats: dict, source_name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{source_name}_norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    return out_path


def load_stats(source_name: str, stats_dir: str) -> dict:
    path = os.path.join(stats_dir, f"{source_name}_norm_stats.json")
    with open(path) as f:
        return json.load(f)


def normalize_image(img: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """
    Normalize an RGB uint8 image to a float32 tensor-ready array using the
    given per-source mean/std (both in RGB order, [0,1] scale).

    Caller (DRDataset) is responsible for BGR→RGB conversion before calling.
    """
    img = img.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    return (img - mean_arr) / std_arr
