"""
Build the common (image_path, label) manifest CSV per source, from each
source's raw label file format. Then runs stratified_split (or eval_only
for Messidor-2). Validates 5-class at every step per user requirement.

Usage:
    python scripts/build_manifests.py --dataset all
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.stratified_split import eval_only_split, stratified_split

NUM_CLASSES = 5


def build_eyepacs_manifest():
    """
    EyePACS ships trainLabels.csv with columns: image, level (0-4).
    Images are typically .jpeg.
    """
    raw_dir = "data/raw/eyepacs"
    labels_csv = os.path.join(raw_dir, "trainLabels.csv")
    if not os.path.exists(labels_csv):
        print(f"Skipping EyePACS: {labels_csv} not found (run download_datasets.py first).")
        return None

    df = pd.read_csv(labels_csv)
    df["image_path"] = df["image"].apply(lambda x: os.path.join(raw_dir, "train", f"{x}.jpeg"))
    out_df = df[["image_path", "level"]].rename(columns={"level": "label"})

    found = set(out_df["label"].unique())
    if found != set(range(NUM_CLASSES)):
        raise ValueError(f"EyePACS labels {sorted(found)} != expected 5-class {list(range(NUM_CLASSES))}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/eyepacs_manifest.csv"
    out_df.to_csv(out_path, index=False)
    print(f"EyePACS manifest: {len(out_df)} images -> {out_path}")
    return out_path


def build_aptos_manifest():
    """APTOS ships train.csv with columns: id_code, diagnosis (0-4). Images are .png."""
    raw_dir = "data/raw/aptos"
    labels_csv = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(labels_csv):
        print(f"Skipping APTOS: {labels_csv} not found (run download_datasets.py first).")
        return None

    df = pd.read_csv(labels_csv)
    df["image_path"] = df["id_code"].apply(lambda x: os.path.join(raw_dir, "train_images", f"{x}.png"))
    out_df = df[["image_path", "diagnosis"]].rename(columns={"diagnosis": "label"})

    found = set(out_df["label"].unique())
    if found != set(range(NUM_CLASSES)):
        raise ValueError(f"APTOS labels {sorted(found)} != expected 5-class {list(range(NUM_CLASSES))}")

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/aptos_manifest.csv"
    out_df.to_csv(out_path, index=False)
    print(f"APTOS manifest: {len(out_df)} images -> {out_path}")
    return out_path


def build_messidor2_manifest():
    """
    Messidor-2 DR grades CSV column names vary by mirror version — inspect
    the actual downloaded CSV and adjust image_col/label_col below if this
    raises a KeyError. Commonly: 'image_id' and 'adjudicated_dr_grade'.
    """
    raw_dir = "data/raw/messidor2"
    candidates = [f for f in os.listdir(raw_dir) if f.endswith(".csv")] if os.path.isdir(raw_dir) else []
    if not candidates:
        print(f"Skipping Messidor-2: no CSV found in {raw_dir} (run download_datasets.py first).")
        return None

    labels_csv = os.path.join(raw_dir, candidates[0])
    df = pd.read_csv(labels_csv)
    print(f"Messidor-2 raw columns found: {list(df.columns)}")

    # Try common column name variants.
    image_col_candidates = ["image_id", "image", "id_code", "Image"]
    label_col_candidates = ["adjudicated_dr_grade", "dr_grade", "diagnosis", "label", "grade"]

    image_col = next((c for c in image_col_candidates if c in df.columns), None)
    label_col = next((c for c in label_col_candidates if c in df.columns), None)

    if image_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect image/label columns in {labels_csv}. "
            f"Columns present: {list(df.columns)}. Edit build_messidor2_manifest() "
            f"in this script to specify the correct column names manually."
        )

    df["image_path"] = df[image_col].apply(lambda x: os.path.join(raw_dir, "messidor-2", "images", f"{x}"))
    out_df = df[["image_path", label_col]].rename(columns={label_col: "label"})
    out_df = out_df.dropna(subset=["label"])  # drop ungradable images (no label)
    out_df["label"] = out_df["label"].astype(int)

    found = set(out_df["label"].unique())
    if found != set(range(NUM_CLASSES)):
        raise ValueError(
            f"Messidor-2 labels {sorted(found)} != expected 5-class {list(range(NUM_CLASSES))}. "
            f"If this mirror ships DME/gradability columns alongside DR grade, make sure "
            f"label_col points at the DR severity grade, not a different field."
        )

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/messidor2_manifest.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Messidor-2 manifest: {len(out_df)} images -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["eyepacs", "aptos", "messidor2", "all"], default="all")
    args = parser.parse_args()

    builders = {
        "eyepacs": build_eyepacs_manifest,
        "aptos": build_aptos_manifest,
        "messidor2": build_messidor2_manifest,
    }
    targets = list(builders) if args.dataset == "all" else [args.dataset]

    manifests = {}
    for name in targets:
        manifests[name] = builders[name]()

    # Splits are now built separately via scripts/build_splits.py AFTER
    # scripts/preprocess_all.py updates the manifests to point to the processed images.


if __name__ == "__main__":
    main()
