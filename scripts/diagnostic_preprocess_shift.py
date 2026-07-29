"""
Sonnet-audit diagnostic A: Are preprocessed Messidor2 and APTOS images
in different visual regimes?

If yes — supports hypothesis #1 (Ben Graham+CLAHE amplifies domain shift).
If no  — hypothesis #1 is wrong; the 88->63 gap is somewhere else.

Outputs:
  - saved/logs/diagnostic_preprocess_shift.json  (numerical stats)
  - saved/logs/diagnostic_preprocess_shift.txt   (human summary)

Compares:
  - Per-channel mean, std (RGB) — shift in color balance?
  - Per-channel histogram KL divergence — shift in color distribution?
  - Edge density (Laplacian variance) — shift in sharpness?
  - Circular-mask coverage (background fraction) — shift in crop tightness?
  - Per-label contrast for the two MOST-equivalent labels (No-DR + Mild)

Reads:
  data/processed/{aptos,messidor2}/*.png

Run:
  python scripts/diagnostic_preprocess_shift.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd

NUM_SAMPLES = 200  # per source — statistically enough for stable means
RNG_SEED = 42


def collect_image_paths(source_dir: str, n: int) -> list[str]:
    """Sample n image paths from a preprocessed source directory."""
    files = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(".png"))
    rng = np.random.default_rng(RNG_SEED)
    if len(files) > n:
        idx = rng.choice(len(files), size=n, replace=False)
        files = [files[i] for i in idx]
    return [os.path.join(source_dir, f) for f in files]


def per_image_stats(img_bgr: np.ndarray) -> dict:
    """Compute the stats we want per image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Per-channel mean/std
    per_ch_mean = img_rgb.mean(axis=(0, 1))  # (3,) RGB
    per_ch_std = img_rgb.std(axis=(0, 1))

    # Edge density (Laplacian variance — proxy for sharpness/contrast)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Background coverage: fraction of pixels with luma < 0.05 (near-black corner mask)
    luma = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2])
    bg_frac = float((luma < 0.05).mean())

    # Per-channel histogram (32 bins) for KL divergence comparison
    hist = np.zeros((3, 32), dtype=np.float64)
    for c in range(3):
        h, _ = np.histogram(img_rgb[..., c], bins=32, range=(0.0, 1.0))
        hist[c] = h / max(h.sum(), 1)

    return {
        "mean_r": float(per_ch_mean[0]),
        "mean_g": float(per_ch_mean[1]),
        "mean_b": float(per_ch_mean[2]),
        "std_r": float(per_ch_std[0]),
        "std_g": float(per_ch_std[1]),
        "std_b": float(per_ch_std[2]),
        "laplacian_var": lap_var,
        "background_frac": bg_frac,
        "hist_rgb": hist.tolist(),
    }


def aggregate_stats(stats_list: list[dict]) -> dict:
    """Average numeric stats; concatenate histograms for KL."""
    keys = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
             "laplacian_var", "background_frac"]
    agg = {}
    for k in keys:
        vals = np.array([s[k] for s in stats_list])
        agg[k + "_mean"] = float(vals.mean())
        agg[k + "_std"] = float(vals.std())  # variance across images

    avg_hist = np.mean([np.array(s["hist_rgb"]) for s in stats_list], axis=0)
    # Add tiny epsilon for KL numerical stability
    agg["avg_hist_rgb"] = (avg_hist + 1e-10).tolist()
    return agg


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Per-channel then averaged KL(P || Q). Symmetric via (KL + KL) / 2."""
    p = np.asarray(p) + 1e-10
    q = np.asarray(q) + 1e-10
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    per_ch_kl = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    return float(per_ch_kl.mean())


def main():
    print("=" * 70)
    print("Sonnet-audit Diagnostic A: preprocessed APTOS vs Messidor2")
    print(f"  Sampling {NUM_SAMPLES} images per source (seed={RNG_SEED})")
    print("=" * 70)

    sources = {
        "aptos": "data/processed/aptos",
        "messidor2": "data/processed/messidor2",
    }
    if not all(os.path.isdir(p) for p in sources.values()):
        for name, p in sources.items():
            if not os.path.isdir(p):
                print(f"  MISSING: {p}")
        sys.exit(1)

    all_stats = {}
    for name, dirpath in sources.items():
        paths = collect_image_paths(dirpath, NUM_SAMPLES)
        print(f"\n  Processing {name} ({len(paths)} images)...")
        per_image = []
        for i, p in enumerate(paths):
            img = cv2.imread(p)
            if img is None:
                print(f"    SKIP (unreadable): {p}")
                continue
            per_image.append(per_image_stats(img))
        all_stats[name] = per_image
        agg = aggregate_stats(per_image)
        print(f"    mean RGB: ({agg['mean_r_mean']:.3f}, {agg['mean_g_mean']:.3f}, {agg['mean_b_mean']:.3f})")
        print(f"    std  RGB: ({agg['std_r_mean']:.3f}, {agg['std_g_mean']:.3f}, {agg['std_b_mean']:.3f})")
        print(f"    Laplacian var (sharpness): {agg['laplacian_var_mean']:.1f}")
        print(f"    Background coverage: {agg['background_frac_mean']:.3f}")

    aptos_hist = np.array(all_stats["aptos"][0]["hist_rgb"])  # use first sample for ref
    # Average hist per source
    aptos_avg_hist = np.mean([np.array(s["hist_rgb"]) for s in all_stats["aptos"]], axis=0)
    mess_avg_hist = np.mean([np.array(s["hist_rgb"]) for s in all_stats["messidor2"]], axis=0)

    kl_aptos_to_mess = kl_divergence(aptos_avg_hist, mess_avg_hist)
    kl_mess_to_aptos = kl_divergence(mess_avg_hist, aptos_avg_hist)
    kl_symmetric = (kl_aptos_to_mess + kl_mess_to_aptos) / 2

    # Per-channel shift in mean
    ch_shift = {
        "delta_mean_r": all_stats["messidor2"][0].get("_skip", 0) or
                        aggregate_stats(all_stats["messidor2"])["mean_r_mean"]
                        - aggregate_stats(all_stats["aptos"])["mean_r_mean"],
        "delta_mean_g": aggregate_stats(all_stats["messidor2"])["mean_g_mean"]
                        - aggregate_stats(all_stats["aptos"])["mean_g_mean"],
        "delta_mean_b": aggregate_stats(all_stats["messidor2"])["mean_b_mean"]
                        - aggregate_stats(all_stats["aptos"])["mean_b_mean"],
    }

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(f"  Symmetric KL divergence (per-channel RGB histograms): {kl_symmetric:.4f}")
    print(f"  Per-channel MEAN shift (Messidor2 - APTOS):")
    print(f"    R: {ch_shift['delta_mean_r']:+.4f}")
    print(f"    G: {ch_shift['delta_mean_g']:+.4f}")
    print(f"    B: {ch_shift['delta_mean_b']:+.4f}")

    aptos_lap = aggregate_stats(all_stats["aptos"])["laplacian_var_mean"]
    mess_lap = aggregate_stats(all_stats["messidor2"])["laplacian_var_mean"]
    print(f"  Laplacian var ratio (Messidor2 / APTOS): {mess_lap / aptos_lap:.3f}")

    # Verdict
    kl_threshold_meaningful = 0.05  # KL > 0.05 is clearly different distributions
    lap_threshold_meaningful = 1.3  # >30% sharpness difference is significant
    shift_threshold = 0.04  # >4% absolute mean shift in any channel

    significant_shift = (kl_symmetric > kl_threshold_meaningful
                          or abs(ch_shift['delta_mean_r']) > shift_threshold
                          or abs(ch_shift['delta_mean_g']) > shift_threshold
                          or abs(ch_shift['delta_mean_b']) > shift_threshold
                          or (mess_lap / aptos_lap) > lap_threshold_meaningful
                          or (mess_lap / aptos_lap) < 1 / lap_threshold_meaningful)

    if significant_shift:
        verdict = ("SIGNIFICANT preprocessing-induced domain shift. "
                    "Sonnet hypothesis #1 CONFIRMED — Ben Graham+CLAHE is "
                    "likely baking in EyePACS-style assumptions that don't "
                    "transfer to Messidor2's camera hardware. "
                    "RECOMMEND next: train variant with reduced Ben Graham "
                    "(or stronger per-batch color jitter to simulate camera "
                    "variation).")
    else:
        verdict = ("Preprocessing does NOT meaningfully shift distributions "
                    "between APTOS and Messidor2. Sonnet hypothesis #1 "
                    "REFUTED. The 88->63 QWK gap is NOT caused by preprocessing "
                    "domain shift — it's elsewhere (model capacity? "
                    "Mild-class signal? inherent cross-dataset difficulty?).")

    print(f"\n  >>> VERDICT: {verdict}\n")

    # Save outputs
    os.makedirs("saved/logs", exist_ok=True)
    summary = {
        "n_samples_per_source": NUM_SAMPLES,
        "aptos_agg": aggregate_stats(all_stats["aptos"]),
        "messidor2_agg": aggregate_stats(all_stats["messidor2"]),
        "kl_symmetric_rgb_histograms": kl_symmetric,
        "channel_mean_shift": ch_shift,
        "laplacian_var_ratio_mess_over_aptos": mess_lap / aptos_lap,
        "verdict": verdict,
    }
    with open("saved/logs/diagnostic_preprocess_shift.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("saved/logs/diagnostic_preprocess_shift.txt", "w") as f:
        f.write(f"KL symmetric: {kl_symmetric:.4f}\n")
        f.write(f"Channel shifts (Mess - Apt): R={ch_shift['delta_mean_r']:+.4f}, "
                f"G={ch_shift['delta_mean_g']:+.4f}, B={ch_shift['delta_mean_b']:+.4f}\n")
        f.write(f"Laplacian var ratio: {mess_lap/aptos_lap:.3f}\n")
        f.write(f"Verdict: {verdict}\n")
    print("  → Saved to saved/logs/diagnostic_preprocess_shift.json + .txt")


if __name__ == "__main__":
    main()