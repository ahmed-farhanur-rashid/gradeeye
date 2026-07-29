"""
Build 80/10/10 train/val/test splits for each source (EyePACS, APTOS, Messidor2).

Inputs:  data/processed/{eyepacs,aptos,messidor2}_manifest.csv
Outputs: data/splits/{source}_{train,val,test}.csv  (overwrites existing)

Stratified split — preserves the original class distribution per source.
Same seed (42) used across all sources for reproducibility.

Why we're regenerating the splits:
- We want all three sources to follow the same train/val/test split ratio
  so they can be evaluated comparably and combined during multi-domain
  training.
- Previously EyePACS only had train/val (no held-out test); APTOS had
  train/val/test; Messidor2 was eval-only. This script unifies them.
- The fraction ratio (80/10/10) was chosen by the user; APTOS test set
  goes from 367 to ~368 (rounded) — close enough to be a non-event.

Run:
    python scripts/build_multidomain_splits.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCES = ["eyepacs", "aptos", "messidor2"]
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
SEED = 42


def stratified_split_3way(df: pd.DataFrame, seed: int = SEED):
    """80/10/10 stratified split. Returns (train_df, val_df, test_df)."""
    labels = df["label"].to_numpy()

    # First split: test set (10%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))

    # Second split: val (10% of full) from remaining
    trainval_labels = labels[trainval_idx]
    relative_val_frac = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)  # val/(train+val)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_idx, val_idx = next(sss2.split(np.zeros(len(trainval_labels)), trainval_labels))

    return (
        df.iloc[trainval_idx].iloc[train_idx].reset_index(drop=True),
        df.iloc[trainval_idx].iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def main():
    out_dir = "data/splits"
    os.makedirs(out_dir, exist_ok=True)
    # Don't blow away non-default split files (if any) — only write what we generate
    for source in SOURCES:
        manifest = f"data/processed/{source}_manifest.csv"
        if not os.path.exists(manifest):
            print(f"[skip] {source}: no manifest at {manifest}")
            continue
        df = pd.read_csv(manifest)
        if "label" not in df.columns:
            print(f"[skip] {source}: manifest missing 'label' column")
            continue

        train_df, val_df, test_df = stratified_split_3way(df)
        train_df.to_csv(f"{out_dir}/{source}_train.csv", index=False)
        val_df.to_csv(f"{out_dir}/{source}_val.csv", index=False)
        test_df.to_csv(f"{out_dir}/{source}_test.csv", index=False)

        print(f"{source:>10s}: total {len(df):>6d} "
              f"-> train {len(train_df):>6d}  val {len(val_df):>6d}  test {len(test_df):>6d}")
        # Class distribution sanity check
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            counts = split_df["label"].value_counts().sort_index().tolist()
            print(f"           {split_name} classes: {counts}")


if __name__ == "__main__":
    main()