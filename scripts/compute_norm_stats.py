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
from src.preprocessing.normalize import compute_dataset_stats, save_stats

def get_sample_paths(manifest_csv: str, sample_limit: int = 2000):
    df = pd.read_csv(manifest_csv)
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=42)
    return df["image_path"].tolist()


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
        sample_paths = get_sample_paths(manifest_path, args.sample_limit)
        stats = compute_dataset_stats(sample_paths)
        out_path = save_stats(stats, name, "data/processed")
        print(f"  mean={stats['mean']}, std={stats['std']}, n={stats['n_images_sampled']}")
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
