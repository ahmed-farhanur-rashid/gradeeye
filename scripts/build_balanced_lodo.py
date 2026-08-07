"""Step 0 + Step 2 — EyePACS one-pool + 3:1 LODO layout (folder form).

Pipeline:
  1. EyePACS = union of raw/trainLabels + raw/testLabels, image_path resolved.
  2. For each source (eyepacs, aptos, messidor2), build a 3:1-balanced
     subset (cap each class at 3x that source's rarest class count, sample
     down only).
  3. For each LODO fold:
        test/<heldout>/          = full 3:1-balanced held-out source
        val/<a>_<b>/             = 10% stratified slice of pooled
        train/<a>_<b>/           = 90% stratified pooled
     where a, b are the 2 non-held-out sources.

Folder convention:
  data/temp/eyepacs/{images,eyepacs_3to1.csv}     (intermediate, 3:1 capped)
  data/temp/aptos/{images,aptos_3to1.csv}
  data/temp/messidor2/{images,messidor2_3to1.csv}
  data/train/<sourceA>_<sourceB>/
  data/val/<sourceA>_<sourceB>/
  data/test/<heldout>/

The "images" subdirs in temp/ contain the raw image files (paths recorded
in the CSVs). The actual preprocessing pipeline reads from these temp CSVs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path("/home/farhan/my-projects/gradeeye")
RAW = ROOT / "data/raw"
TEMP = ROOT / "data/temp"
TRAIN_DIR = ROOT / "data/train"
VAL_DIR = ROOT / "data/val"
TEST_DIR = ROOT / "data/test"

NUM_CLASSES = 5
RATIO = 3
SEED = 42
VAL_FRAC = 0.10


def _safe_rmtree(p: Path) -> None:
    if p.exists():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def load_eyepacs_pool() -> pd.DataFrame:
    train = pd.read_csv(RAW / "eyepacs/trainLabels.csv")
    test = pd.read_csv(RAW / "eyepacs/testLabels.csv.csv")
    df = pd.concat([
        train.rename(columns={"image": "image_id", "level": "label"}),
        test.rename(columns={"image": "image_id", "level": "label"}),
    ], ignore_index=True)
    train_dir = RAW / "eyepacs/train"
    test_dir = RAW / "eyepacs/test"
    train_imgs = {p.name for p in train_dir.iterdir() if p.is_file()}
    test_imgs = {p.name for p in test_dir.iterdir() if p.is_file()}

    def resolve(row):
        name = f"{row.image_id}.jpeg"
        if name in train_imgs:
            return f"data/raw/eyepacs/train/{name}"
        if name in test_imgs:
            return f"data/raw/eyepacs/test/{name}"
        return None

    df["image_path"] = df.apply(resolve, axis=1)
    missing = df["image_path"].isna().sum()
    if missing:
        print(f"  EyePACS missing: {missing}, dropping")
        df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    df["source"] = "eyepacs"
    return df[["image_path", "label", "source"]]


def load_aptos_train() -> pd.DataFrame:
    train = pd.read_csv(RAW / "aptos/train.csv")
    df = train.rename(columns={"id_code": "image_id", "diagnosis": "label"}).copy()
    df["image_path"] = df["image_id"].apply(
        lambda i: f"data/raw/aptos/train_images/{i}.png"
    )
    df["exists"] = df["image_path"].apply(lambda p: (ROOT / p).exists())
    print(f"  APTOS: keep {df.exists.sum()}/{len(df)}")
    df = df[df["exists"]].drop(columns="exists").reset_index(drop=True)
    df["source"] = "aptos"
    return df[["image_path", "label", "source"]]


def load_messidor2() -> pd.DataFrame:
    md = pd.read_csv(RAW / "messidor2/messidor-2/messidor_data.csv")
    df = md.rename(columns={"image_id": "image_id",
                             "adjudicated_dr_grade": "label"}).copy()
    df = df[df["adjudicated_gradable"] == 1].reset_index(drop=True)
    df["image_path"] = df["image_id"].apply(
        lambda i: f"data/raw/messidor2/messidor-2/images/{i}"
    )
    df["exists"] = df["image_path"].apply(lambda p: (ROOT / p).exists())
    print(f"  Messidor-2: keep {df.exists.sum()}/{len(df)}")
    df = df[df["exists"]].drop(columns="exists").reset_index(drop=True)
    df["source"] = "messidor2"
    return df[["image_path", "label", "source"]]


def build_3to1(df: pd.DataFrame, source: str) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    rare = int(counts.min())
    cap = rare * RATIO
    print(f"  {source}: rarest={rare}, cap={cap}, before={counts.to_dict()}")
    parts = []
    for cls in range(NUM_CLASSES):
        sub = df[df["label"] == cls]
        n_keep = min(cap, len(sub))
        if n_keep < len(sub):
            sub = sub.sample(n=n_keep, random_state=SEED)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out_counts = out["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    print(f"  {source}: after={out_counts.to_dict()}, total={len(out)}")
    return out


def write_temp(source: str, df: pd.DataFrame) -> None:
    src_dir = TEMP / source
    _safe_rmtree(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    csv_path = src_dir / f"{source}_3to1.csv"
    df[["image_path", "label"]].to_csv(csv_path, index=False)
    print(f"  wrote {csv_path} ({len(df)} rows)")


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 90/10 train/val."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(df)), df["label"].to_numpy()))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def write_split(out_dir: Path, df: pd.DataFrame, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "manifest.csv"
    df[["image_path", "label"]].to_csv(csv_path, index=False)
    print(f"  wrote {csv_path} ({len(df)} rows, {label})")


def main() -> None:
    print("Loading sources...")
    eyepacs = load_eyepacs_pool()
    print(f"  EyePACS pool: {len(eyepacs)}")
    aptos = load_aptos_train()
    print(f"  APTOS: {len(aptos)}")
    messidor = load_messidor2()
    print(f"  Messidor-2: {len(messidor)}")

    sources = {"eyepacs": eyepacs, "aptos": aptos, "messidor2": messidor}
    balanced: dict[str, pd.DataFrame] = {}

    print("\n=== Build 3:1 subsets ===")
    for src, df in sources.items():
        sub = build_3to1(df, src)
        balanced[src] = sub
        write_temp(src, sub)

    # Clean slates for train/val/test
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        _safe_rmtree(d)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Build LODO fold folders ===")
    summary = []
    for holdout in ("eyepacs", "aptos", "messidor2"):
        train_src = [s for s in ("eyepacs", "aptos", "messidor2") if s != holdout]
        a, b = train_src
        pool = pd.concat([balanced[a], balanced[b]], ignore_index=True)
        train_df, val_df = split_train_val(pool)

        train_dir = TRAIN_DIR / f"{a}_{b}"
        val_dir = VAL_DIR / f"{a}_{b}"
        test_dir = TEST_DIR / holdout

        write_split(train_dir, train_df, f"train {a}+{b}")
        write_split(val_dir, val_df, f"val {a}+{b}")
        write_split(test_dir, balanced[holdout], f"test {holdout} (3:1 cap)")

        summary.append({
            "holdout": holdout,
            "train": f"{a}_{b}",
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(balanced[holdout])),
            "per_class_train": train_df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_dict(),
            "per_class_val": val_df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_dict(),
            "per_class_test": balanced[holdout]["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).to_dict(),
        })

    out = ROOT / "data" / "lodo_balanced_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
