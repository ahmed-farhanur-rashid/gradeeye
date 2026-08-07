"""LODO collapse diagnostic — class distributions, confusion matrices,
checkpoint/label sanity, matched-subsample integrity.

No new training. Loads each fold's checkpoint once, evaluates it on the held-
out test_full and test_matched manifests, and writes the diagnostic JSONs
into docs/lodo_diagnostics/.

Read-only with respect to training/configs. No modifications to data files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.eval.metrics import compute_all_metrics
from src.eval.probability_calibration import NUM_CLASSES, NUM_THRESHOLDS
from src.models.corn import corn_predict_probas
from src.training.checkpoint_io import load_model_from_checkpoint

ROOT = Path("/home/farhan/my-projects/gradeeye")
SPLITS = ROOT / "data/splits"
CKPT_ROOT = ROOT / "saved/checkpoints"
DOCS = ROOT / "docs" / "lodo_diagnostics"
DOCS.mkdir(parents=True, exist_ok=True)
FOLDS = ("eyepacs", "aptos", "messidor2")


def class_dist(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
    pct = (counts / max(counts.sum(), 1) * 100).round(4)
    return {
        "n": int(counts.sum()),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "pct": {str(k): float(v) for k, v in pct.items()},
    }


def evaluate_manifest(model, manifest_path: Path, img_size: int, device: str,
                     batch_size: int = 64) -> dict:
    dataset = DRDataset(str(manifest_path),
                        transform=build_eval_transforms(img_size=img_size))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=device.startswith("cuda"),
    )
    logits_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    class_probs = corn_predict_probas(torch.from_numpy(logits)).numpy()
    preds = class_probs.argmax(axis=1)
    metrics = compute_all_metrics(labels, preds, class_probs)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    return {
        "manifest": str(manifest_path),
        "n": int(len(labels)),
        "metrics": {k: (float(v) if not isinstance(v, (int, str)) else v)
                    for k, v in metrics.items()},
        "confusion_matrix": cm.tolist(),
        "labels_dist": class_dist(pd.DataFrame({"label": labels})),
        "preds_dist": class_dist(pd.DataFrame({"label": preds})),
    }


def matched_integrity(fold: str) -> dict:
    """Compare test_full vs test_matched distributions."""
    full = pd.read_csv(SPLITS / f"lodo_{fold}_test_full.csv")
    matched = pd.read_csv(SPLITS / f"lodo_{fold}_test_matched.csv")
    full_dist = class_dist(full)
    matched_dist = class_dist(matched)
    return {
        "full": full_dist,
        "matched": matched_dist,
        "matched_over_full_pct": {
            str(k): round(matched_dist["counts"][str(k)] / max(full_dist["counts"][str(k)], 1) * 100, 4)
            for k in range(NUM_CLASSES)
        },
        "max_per_class_delta_pct": max(
            abs(full_dist["pct"][str(k)] - matched_dist["pct"][str(k)])
            for k in range(NUM_CLASSES)
        ),
    }


def pool_dist(fold: str) -> dict:
    train = pd.read_csv(SPLITS / f"lodo_{fold}_train.csv")
    val = pd.read_csv(SPLITS / f"lodo_{fold}_val.csv")
    return class_dist(pd.concat([train, val], ignore_index=True))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict = {"fold_reports": {}, "checkpoint_sanity": {}, "matched_integrity": {}}

    for fold in FOLDS:
        ckpt = CKPT_ROOT / fold / "convnext_tiny" / f"lodo_{fold}_convnext_tiny_best.pt"
        print(f"\n=== {fold}: {ckpt} ===")
        model, config = load_model_from_checkpoint(str(ckpt), device, use_ema=True)
        img_size = config.get("model", {}).get("img_size")
        arch = config.get("model", {}).get("arch")
        in_chans = config.get("model", {}).get("in_chans")
        # Checkpoint sanity (must be identical model/arch/class-mapping for both full+matched eval).
        report["checkpoint_sanity"][fold] = {
            "checkpoint_path": str(ckpt),
            "arch": arch,
            "img_size": img_size,
            "in_chans": in_chans,
            "num_thresholds_config": config.get("model", {}).get("num_thresholds"),
            "head_output_dim": int(model.head.net[-1].out_features)
                if hasattr(model, "head") else None,
        }

        # Diagnostic distribution rows
        train_pool_dist = pool_dist(fold)
        report["matched_integrity"][fold] = matched_integrity(fold)

        # Run both evals with identical checkpoint.
        full_eval = evaluate_manifest(
            model, SPLITS / f"lodo_{fold}_test_full.csv", img_size, device
        )
        matched_eval = evaluate_manifest(
            model, SPLITS / f"lodo_{fold}_test_matched.csv", img_size, device
        )
        report["fold_reports"][fold] = {
            "train_pool_dist": train_pool_dist,
            "test_full": full_eval,
            "test_matched": matched_eval,
            "qwk_delta_matched_minus_full": round(
                matched_eval["metrics"]["qwk"] - full_eval["metrics"]["qwk"], 6),
            "accuracy_delta_matched_minus_full": round(
                matched_eval["metrics"]["accuracy"] - full_eval["metrics"]["accuracy"], 6),
        }

    out_path = DOCS / "lodo_collapse_diagnostic.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved diagnostic report → {out_path}")

    # Also produce a markdown summary.
    md_lines = ["# LODO Collapse Diagnostic\n",
                "ConvNeXt-Tiny, 3 folds, single checkpoint per fold evaluated on test_full + test_matched.\n"]
    md_lines.append("## Class distributions (% per grade)\n")
    md_lines.append("| Fold | Pool dist (train+val) | test_full dist | test_matched dist | matched→full sample pct (per class) |")
    md_lines.append("|------|----------------------|----------------|-------------------|--------------------------------------|")
    for fold in FOLDS:
        d = report["matched_integrity"][fold]
        pool = report["fold_reports"][fold]["train_pool_dist"]
        pool_str = ", ".join(f"{k}:{pool['pct'][k]:.1f}" for k in map(str, range(NUM_CLASSES)))
        full_str = ", ".join(f"{k}:{d['full']['pct'][k]:.1f}" for k in map(str, range(NUM_CLASSES)))
        matched_str = ", ".join(f"{k}:{d['matched']['pct'][k]:.1f}" for k in map(str, range(NUM_CLASSES)))
        pct_per = ", ".join(f"{k}:{d['matched_over_full_pct'][k]:.1f}" for k in map(str, range(NUM_CLASSES)))
        md_lines.append(f"| {fold} | {pool_str} | {full_str} | {matched_str} | {pct_per} |")

    md_lines.append("\n## Metrics on held-out test\n")
    md_lines.append("| Fold | split | n | QWK | Acc | MacroF1 | MacroRecall |")
    md_lines.append("|------|-------|---|------|------|---------|-------------|")
    for fold in FOLDS:
        for split_name in ("test_full", "test_matched"):
            r = report["fold_reports"][fold][split_name]
            m = r["metrics"]
            md_lines.append(f"| {fold} | {split_name} | {r['n']} | {m['qwk']:.4f} | "
                            f"{m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['macro_recall']:.4f} |")

    md_lines.append("\n## Confusion matrices (rows=true, cols=pred)\n")
    class_short = ["NoDR", "Mild", "Mod", "Sev", "PDR"]
    for fold in FOLDS:
        for split_name in ("test_full", "test_matched"):
            r = report["fold_reports"][fold][split_name]
            cm = r["confusion_matrix"]
            md_lines.append(f"\n### {fold} / {split_name}  (n={r['n']})\n")
            md_lines.append("| true\\pred | " + " | ".join(class_short) + " |")
            md_lines.append("|" + "---|" * (NUM_CLASSES + 1))
            for i, row in enumerate(cm):
                md_lines.append(f"| **{class_short[i]}** | " + " | ".join(str(v) for v in row) + " |")

    md_lines.append("\n## Per-class recall (test_full vs test_matched)\n")
    md_lines.append("| Fold | split | NoDR | Mild | Mod | Sev | PDR |")
    md_lines.append("|------|-------|------|------|------|------|------|")
    for fold in FOLDS:
        for split_name in ("test_full", "test_matched"):
            r = report["fold_reports"][fold][split_name]
            m = r["metrics"]
            row = [fold, split_name]
            for c in ("No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"):
                row.append(f"{m.get(f'recall_{c}', 0):.3f}")
            md_lines.append("| " + " | ".join(row) + " |")

    md_path = DOCS / "lodo_collapse_diagnostic.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved markdown summary → {md_path}")


if __name__ == "__main__":
    main()
