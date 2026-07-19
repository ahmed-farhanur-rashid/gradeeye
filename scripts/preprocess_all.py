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

def preprocess_dataset(manifest_path: str, source_name: str, use_anisotropic: bool = False, use_all_channel_clahe: bool = False):
    if not os.path.exists(manifest_path):
        print(f"Skipping {source_name}: {manifest_path} not found.")
        return
    
    df = pd.read_csv(manifest_path)
    out_dir = f"data/processed/{source_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    new_paths = []
    print(f"Preprocessing {source_name}...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        raw_path = row["image_path"]
        img = cv2.imread(raw_path)
        if img is None:
            new_paths.append(raw_path)
            continue
            
        img = crop_pad_resize(img)
        img = color_correction_pipeline(img, use_all_channel_clahe=use_all_channel_clahe)
        img = apply_anisotropic_filter(img, enabled=use_anisotropic)
        
        filename = os.path.basename(raw_path)
        if not filename.lower().endswith(".png"):
            filename = os.path.splitext(filename)[0] + ".png"
            
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, img)
        new_paths.append(out_path)
        
    df["image_path"] = new_paths
    df.to_csv(manifest_path, index=False)
    print(f"Saved processed images to {out_dir} and updated {manifest_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anisotropic", action="store_true", help="Enable anisotropic filtering ablation")
    parser.add_argument("--all-channel-clahe", action="store_true", help="Enable all-channel CLAHE ablation")
    args = parser.parse_args()

    manifest_map = {
        "eyepacs": "data/processed/eyepacs_manifest.csv",
        "aptos": "data/processed/aptos_manifest.csv",
        "messidor2": "data/processed/messidor2_manifest.csv",
    }
    for name, path in manifest_map.items():
        preprocess_dataset(path, name, use_anisotropic=args.anisotropic, use_all_channel_clahe=args.all_channel_clahe)

if __name__ == "__main__":
    main()
