"""
Compute per-source normalization stats (mean/std) over PREPROCESSED images
(crop/pad/resize + color correction already applied), per plan Section 2
step 6. Run this after build_manifests.py and before training.

Usage:
    python scripts/compute_norm_stats.py --dataset all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd

from src.preprocessing.color_correction import color_correction_pipeline
from src.preprocessing.crop_and_resize import crop_pad_resize
from src.preprocessing.normalize import compute_dataset_stats, save_stats


def preprocess_and_cache_sample(manifest_csv: str, source_name: str, sample_limit: int = 2000):
    """
    Since compute_dataset_stats reads raw files from disk directly (not
    through DRDataset), and normalization stats must be computed on
    PREPROCESSED images (post crop/color-correction), we preprocess a
    sample to a temp cache dir first, then compute stats on that cache.
    """
    df = pd.read_csv(manifest_csv)
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=42)

    cache_dir = f"data/processed/_norm_cache_{source_name}"
    os.makedirs(cache_dir, exist_ok=True)

    cached_paths = []
    for i, row in df.iterrows():
        img = cv2.imread(row["image_path"])
        if img is None:
            continue
        img = crop_pad_resize(img)
        img = color_correction_pipeline(img)
        out_path = os.path.join(cache_dir, f"{i}.png")
        cv2.imwrite(out_path, img)
        cached_paths.append(out_path)

    return cached_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["eyepacs", "aptos", "messidor2", "all"], default="all")
    parser.add_argument("--sample-limit", type=int, default=2000)
    args = parser.parse_args()

    manifest_map = {
        "eyepacs": "data/processed/eyepacs_manifest.csv",
        "aptos": "data/processed/aptos_manifest.csv",
        "messidor2": "data/processed/messidor2_manifest.csv",
    }
    targets = list(manifest_map) if args.dataset == "all" else [args.dataset]

    for name in targets:
        manifest_path = manifest_map[name]
        if not os.path.exists(manifest_path):
            print(f"Skipping {name}: {manifest_path} not found (run build_manifests.py first).")
            continue

        print(f"Computing normalization stats for {name}...")
        cached_paths = preprocess_and_cache_sample(manifest_path, name, args.sample_limit)
        stats = compute_dataset_stats(cached_paths)
        out_path = save_stats(stats, name, "data/processed")
        print(f"  mean={stats['mean']}, std={stats['std']}, n={stats['n_images_sampled']}")
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
