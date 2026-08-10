"""
Evaluate lesion-mask detection rate on OOD samples (APTOS, EyePACS, Messidor-2).

Picks stratified samples from each source's test manifest, runs the trained
U-Net, and reports the fraction of samples with non-empty masks, broken
down by source and by DR grade. This is the gate: if detection rate is too
low for Severe/PDR cases, the mask channel isn't useful and we should fall
back to a non-learned alternative (e.g. Sobel edges).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.segmentation.ddr_dataset import list_ddr_ids
from src.segmentation.generate_masks import (
    InferenceDataset, load_segmenter, predict_masks,
)


def stratified_sample(manifest_csv: str, per_class: int = 10) -> list[tuple[str, int]]:
    df = pd.read_csv(manifest_csv)
    rows = []
    for label in sorted(df["label"].unique()):
        sub = df[df["label"] == label]
        n = min(per_class, len(sub))
        rows.extend((p, int(label)) for p in sub.iloc[:n]["image_path"])
    return rows


def evaluate(model, samples, device, threshold=0.5, img_size=384,
             batch_size=16, save_dir=None):
    paths = [p for p, _ in samples]
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    ds = InferenceDataset(paths, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    results = []
    with torch.no_grad():
        idx = 0
        for batch in loader:
            images, names = batch
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            for i in range(probs.shape[0]):
                p = probs[i, 0]
                frac = float((p > threshold).mean())
                label = samples[idx][1]
                src_path = samples[idx][0]
                name = Path(src_path).stem
                src_source = Path(src_path).parent.name
                results.append({
                    "source": src_source,
                    "label": label,
                    "image": src_path,
                    "mask_pos_frac": frac,
                    "any_positive": frac > 0,
                    "max_prob": float(p.max()),
                })
                if save_dir and (frac > 0 or idx % 10 == 0):
                    mask = (p > threshold).astype(np.uint8) * 255
                    img = cv2.imread(src_path)
                    overlay = img.copy()
                    overlay[mask > 127] = (0, 255, 0)
                    side = np.hstack([
                        img,
                        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                        overlay,
                    ])
                    cv2.imwrite(str(Path(save_dir) /
                                    f"{src_source}_{name}_l{label}.png"), side)
                idx += 1
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--save-dir", default=None,
                        help="If set, write side-by-side visualizations here")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_segmenter(args.checkpoint, device)
    print(f"loaded {args.checkpoint} val_iou={ckpt.get('val_iou'):.4f}")

    all_results = []
    for source in ("eyepacs", "aptos", "messidor2"):
        manifest = f"data/test/{source}/manifest.csv"
        samples = stratified_sample(manifest, per_class=args.per_class)
        print(f"\n{source}: {len(samples)} samples")
        results = evaluate(model, samples, device, threshold=args.threshold,
                            img_size=args.img_size, save_dir=args.save_dir)
        all_results.extend(results)
        # Per-source summary
        df = pd.DataFrame(results)
        for label in sorted(df["label"].unique()):
            sub = df[df["label"] == label]
            print(f"  label {label}: {sub['any_positive'].sum()}/{len(sub)} "
                  f"detected, mean_frac={sub['mask_pos_frac'].mean():.4f}, "
                  f"max_prob_mean={sub['max_prob'].mean():.3f}")
        print(f"  overall: {df['any_positive'].sum()}/{len(df)} detected")

    # Overall summary
    df = pd.DataFrame(all_results)
    print(f"\nOVERALL: {df['any_positive'].sum()}/{len(df)} ({100*df['any_positive'].mean():.1f}%) detected")
    # Severe/PDR detection rate (the most important class)
    severe_pdr = df[df["label"] >= 3]
    if len(severe_pdr):
        rate = 100 * severe_pdr["any_positive"].mean()
        print(f"Severe/PDR (label 3-4): {severe_pdr['any_positive'].sum()}/{len(severe_pdr)} ({rate:.1f}%) detected")

    if args.save_dir:
        df.to_csv(Path(args.save_dir) / "ood_eval.csv", index=False)


if __name__ == "__main__":
    main()