"""Step 3 final — rewrite manifest image_path columns to point at processed PNGs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/home/farhan/my-projects/gradeeye")
PROCESSED_ROOT = ROOT / "data/processed"

SOURCE_PATTERNS = {
    "eyepacs": "data/raw/eyepacs/",
    "aptos": "data/raw/aptos/",
    "messidor2": "data/raw/messidor2/",
}

FOLD_MANIFESTS = [
    "data/train/aptos_messidor2/manifest.csv",
    "data/val/aptos_messidor2/manifest.csv",
    "data/test/eyepacs/manifest.csv",
    "data/train/eyepacs_messidor2/manifest.csv",
    "data/val/eyepacs_messidor2/manifest.csv",
    "data/test/aptos/manifest.csv",
    "data/train/eyepacs_aptos/manifest.csv",
    "data/val/eyepacs_aptos/manifest.csv",
    "data/test/messidor2/manifest.csv",
]


def main() -> None:
    rel_path_map: dict[str, str] = {}
    for m in FOLD_MANIFESTS:
        path = ROOT / m
        df = pd.read_csv(path)
        new_paths = []
        for p in df["image_path"]:
            if "/" in p:
                # raw path: data/raw/<source>/...; rewrite to processed
                for src, marker in SOURCE_PATTERNS.items():
                    if marker in p:
                        fname = Path(p).name.replace(".jpeg", ".png")
                        new_paths.append(f"data/processed/{src}/{fname}")
                        break
                else:
                    new_paths.append(p)  # already processed?
            else:
                new_paths.append(p)
        df["image_path"] = new_paths
        df.to_csv(path, index=False)
        print(f"  rewrote {path} ({len(df)} rows)")

    # Sanity check: confirm every image_path exists on disk
    missing = 0
    for m in FOLD_MANIFESTS:
        df = pd.read_csv(ROOT / m)
        for p in df["image_path"]:
            if not (ROOT / p).exists():
                missing += 1
                if missing <= 5:
                    print(f"  MISSING: {p}")
    print(f"Total missing: {missing}")


if __name__ == "__main__":
    main()
