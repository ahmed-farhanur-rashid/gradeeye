import argparse
import os
import sys

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.crop_and_resize import crop_pad_resize
from src.preprocessing.color_correction import color_correction_pipeline
from src.preprocessing.anisotropic_filter import apply_anisotropic_filter


# Module-level worker function so it's picklable for ProcessPoolExecutor.
# cv2 releases the GIL inside its C kernels for most ops, so threads help,
# but Python-side overhead (pandas row unpacking, dict lookups) doesn't.
# Processes give true parallelism; ~2-4x faster than threads on multi-core.
def _process_one(args):
    raw_path, out_dir, use_all_channel_clahe, use_anisotropic = args
    img = cv2.imread(raw_path)
    if img is None:
        return None

    img = crop_pad_resize(img)
    img = color_correction_pipeline(img, use_all_channel_clahe=use_all_channel_clahe)
    img = apply_anisotropic_filter(img, enabled=use_anisotropic)

    filename = os.path.basename(raw_path)
    if not filename.lower().endswith(".png"):
        filename = os.path.splitext(filename)[0] + ".png"

    out_path = os.path.join(out_dir, filename)
    cv2.imwrite(out_path, img)
    return (raw_path, out_path)


def preprocess_dataset(manifest_path: str, source_name: str, use_anisotropic: bool = False, use_all_channel_clahe: bool = False):
    if not os.path.exists(manifest_path):
        print(f"Skipping {source_name}: {manifest_path} not found.")
        return

    df = pd.read_csv(manifest_path)
    out_dir = f"data/processed/{source_name}"
    os.makedirs(out_dir, exist_ok=True)

    import concurrent.futures
    import multiprocessing

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Preprocessing {source_name} ({len(df)} images, {n_workers} workers)...")

    # ProcessPoolExecutor: workers run in separate Python processes so cv2's
    # C kernels and Python-side pandas ops both run in parallel.
    # ProcessPoolExecutor doesn't have imap_unordered; use as_completed for
    # streaming progress, then build path_map keyed by raw_path.
    work_items = [(p, out_dir, use_all_channel_clahe, use_anisotropic)
                  for p in df["image_path"].tolist()]

    path_map = {}  # raw_path -> processed_path
    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one, item): item for item in work_items}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = fut.result()
            if result is None:
                failed += 1
                continue
            raw_path, out_path = result
            path_map[raw_path] = out_path

    # Rebuild the manifest with the new processed paths, preserving order.
    new_paths = [path_map.get(p) for p in df["image_path"]]
    df["image_path"] = new_paths
    df.to_csv(manifest_path, index=False)
    if failed:
        print(f"  WARNING: {failed} images failed to read")
    print(f"Saved processed images to {out_dir} and updated {manifest_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anisotropic", action="store_true", help="Enable anisotropic filtering ablation")
    parser.add_argument("--all-channel-clahe", action="store_true", help="Enable all-channel CLAHE ablation")
    parser.add_argument("--dataset", choices=["eyepacs", "aptos", "messidor2", "all"], default="all", help="Dataset to preprocess")
    args = parser.parse_args()

    manifest_map = {
        "eyepacs": "data/processed/eyepacs_manifest.csv",
        "aptos": "data/processed/aptos_manifest.csv",
        "messidor2": "data/processed/messidor2_manifest.csv",
    }
    
    targets = list(manifest_map.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        path = manifest_map[name]
        preprocess_dataset(path, name, use_anisotropic=args.anisotropic, use_all_channel_clahe=args.all_channel_clahe)

if __name__ == "__main__":
    main()
