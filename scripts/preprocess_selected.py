"""Step 3 — preprocess only the images referenced by the new LODO folder layout.

Run the actual preprocessing pipeline (crop_pad_resize → color_correction)
on the union of selected images. Skip the manifest rewrite so empty
intermediate states can't drop rows.

After preprocessing succeeds, run a separate `rewrite_manifests.py` step to
swap image_path from raw -> processed.
"""
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path

import cv2

from src.preprocessing.crop_and_resize import crop_pad_resize
from src.preprocessing.color_correction import color_correction_pipeline
from src.preprocessing.anisotropic_filter import apply_anisotropic_filter

ROOT = Path("/home/farhan/my-projects/gradeeye")
PROCESSED_ROOT = ROOT / "data/processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

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


def _process_one(args: tuple[str, str]) -> str | None:
    raw_path, out_path = args
    img = cv2.imread(raw_path)
    if img is None:
        return None
    img = crop_pad_resize(img)
    img = color_correction_pipeline(img, use_all_channel_clahe=False)
    img = apply_anisotropic_filter(img, enabled=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path


def main() -> None:
    import sys
    import pandas as pd
    selected_paths: set[str] = set()
    for m in FOLD_MANIFESTS:
        df = pd.read_csv(ROOT / m)
        selected_paths.update(df["image_path"].tolist())
    print(f"Union of selected images: {len(selected_paths)}", flush=True)

    buckets: dict[str, list[str]] = defaultdict(list)
    for p in selected_paths:
        for src, marker in SOURCE_PATTERNS.items():
            if marker in p:
                buckets[src].append(p)
                break
    for src, lst in buckets.items():
        print(f"  {src}: {len(lst)} images", flush=True)

    n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {n_workers} workers", flush=True)
    counts: dict[str, int] = {}
    for src, lst in buckets.items():
        out_dir = PROCESSED_ROOT / src
        out_dir.mkdir(parents=True, exist_ok=True)
        work = [
            (p, str(out_dir / Path(p).name.replace(".jpeg", ".png")))
            for p in lst
        ]
        print(f"  starting {src}: {len(work)} items", flush=True)
        succeeded = 0
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_one, w) for w in work]
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                result = fut.result()
                if result is not None:
                    succeeded += 1
                else:
                    failed += 1
                if (i + 1) % 500 == 0:
                    print(f"  {src}: {i+1}/{len(work)} ok={succeeded} fail={failed}", flush=True)
        counts[src] = succeeded
        actual = sum(1 for _ in out_dir.iterdir())
        print(f"  {src}: preprocessed {succeeded}/{len(lst)} (failed={failed}); on-disk={actual}", flush=True)
        sys.stdout.flush()

    (PROCESSED_ROOT / "preprocess_selected_summary.json").write_text(
        json.dumps({"total": len(selected_paths), "by_source": counts}, indent=2)
    )
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
