"""
Stratified train/val/test splitting per source, using sklearn
StratifiedShuffleSplit.

Per plan Section 1:
  - EyePACS: pretrain only (no held-out test needed here — final test is
    on APTOS). Split into train/val is still useful for monitoring Phase
    1/2 training.
  - APTOS: fine-tune + PRIMARY test — held-out stratified split.
  - Messidor-2: external validation ONLY — never split into train, the
    entire set is eval-only (plan is explicit: zero gradient updates).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

NUM_CLASSES = 5


def _validate_5class(df: pd.DataFrame, source_name: str):
    labels = df["label"].unique()
    if not set(labels).issubset(set(range(NUM_CLASSES))):
        raise ValueError(
            f"{source_name}: labels outside 5-class range [0-4]: {sorted(labels)}. "
            f"Every dataset in this project must be exactly 5-class ICDR."
        )


def stratified_split(manifest_csv: str, source_name: str, out_dir: str,
                      train_frac: float = 0.7, val_frac: float = 0.15,
                      test_frac: float = 0.15, seed: int = 42) -> dict:
    """
    3-way stratified split (train/val/test) for a source that gets trained
    on (EyePACS, APTOS). Writes 3 CSVs to out_dir and returns their paths.
    """
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    df = pd.read_csv(manifest_csv)
    _validate_5class(df, source_name)

    labels = df["label"].to_numpy()

    # First split off test set.
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))

    # Then split remaining train/val.
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    trainval_labels = labels[trainval_idx]
    relative_val_frac = val_frac / (train_frac + val_frac)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_idx, val_idx = next(sss2.split(np.zeros(len(trainval_labels)), trainval_labels))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = os.path.join(out_dir, f"{source_name}_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        paths[split_name] = out_path

    return paths


def train_val_split(manifest_csv: str, source_name: str, out_dir: str,
                     train_frac: float = 0.85, seed: int = 42) -> dict:
    """
    2-way stratified split (train/val only, no test) — for pretrain-only
    sources like EyePACS, where final evaluation happens on a different
    source (APTOS) rather than a held-out EyePACS split.
    """
    df = pd.read_csv(manifest_csv)
    _validate_5class(df, source_name)

    labels = df["label"].to_numpy()
    val_frac = 1.0 - train_frac

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        out_path = os.path.join(out_dir, f"{source_name}_{split_name}.csv")
        split_df.to_csv(out_path, index=False)
        paths[split_name] = out_path

    return paths


def eval_only_split(manifest_csv: str, source_name: str, out_dir: str) -> dict:
    """
    For Messidor-2: no train/val split at all, the entire set is eval-only.
    Just validates 5-class and copies through as a single "test" CSV so the
    eval scripts have a consistent interface with the other sources.
    """
    df = pd.read_csv(manifest_csv)
    _validate_5class(df, source_name)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{source_name}_test.csv")
    df.to_csv(out_path, index=False)
    return {"test": out_path}
