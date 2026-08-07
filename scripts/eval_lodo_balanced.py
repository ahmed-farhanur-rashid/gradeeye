"""Step 6 — evaluate the 3 LODO balanced checkpoints on their held-out test sets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

from src.augmentation.transforms import build_eval_transforms
from src.data._05_dataset import DRDataset
from src.models.corn import corn_predict_probas
from src.training.checkpoint_io import load_model_from_checkpoint

ROOT = Path("/home/farhan/my-projects/gradeeye")
DOCS = ROOT / "docs/lodo_balanced_diagnostics"
DOCS.mkdir(parents=True, exist_ok=True)

FOLDS = ("eyepacs", "aptos", "messidor2")
CKPT_ROOT = ROOT / "saved/checkpoints"


def evaluate(model, manifest: Path, img_size: int, device: str,
              batch_size: int = 64) -> dict:
    ds = DRDataset(str(manifest), transform=build_eval_transforms(img_size=img_size))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=device.startswith("cuda"),
    )
    preds_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            probas = corn_predict_probas(logits).cpu().numpy()
            preds_all.append(probas.argmax(axis=1))
            labels_all.append(labels.numpy())
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(labels_all)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(5)), zero_division=0)
    return {
        "n": int(len(y_true)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_per_class": {str(c): float(recall_per_class[c]) for c in range(5)},
        "f1_per_class": {str(c): float(f1_per_class[c]) for c in range(5)},
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict[str, dict] = {}

    for fold in FOLDS:
        ckpt = CKPT_ROOT / fold / "convnext_tiny" / f"lodo_{fold}_convnext_tiny_best.pt"
        test_manifest = ROOT / "data" / "test" / fold / "manifest.csv"
        print(f"\n=== {fold}: ckpt={ckpt.name}, test={test_manifest} ===")
        model, cfg = load_model_from_checkpoint(str(ckpt), device, use_ema=True)
        img_size = cfg["model"]["img_size"]
        # Find val manifest
        val_pair = {"eyepacs": "aptos_messidor2", "aptos": "eyepacs_messidor2",
                    "messidor2": "eyepacs_aptos"}[fold]
        val_manifest = ROOT / "data" / "val" / val_pair / "manifest.csv"
        train_pair = val_pair
        train_manifest = ROOT / "data" / "train" / train_pair / "manifest.csv"

        # Train/val metrics from training log; here we only re-evaluate test.
        test_eval = evaluate(model, test_manifest, img_size, device)
        results[fold] = {
            "checkpoint": str(ckpt),
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "test_manifest": str(test_manifest),
            "test": test_eval,
        }
        print(f"  Test n={test_eval['n']}  QWK={test_eval['qwk']:.4f}  "
              f"Acc={test_eval['accuracy']:.4f}  MacroF1={test_eval['macro_f1']:.4f}")

    out = DOCS / "lodo_balanced_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")

    # Markdown summary
    lines = ["# LODO Balanced 3-fold Results", ""]
    lines.append("| Fold | Test n | QWK | Acc | MacroF1 | MacroRecall |")
    lines.append("|------|--------|------|------|---------|-------------|")
    for fold in FOLDS:
        r = results[fold]["test"]
        lines.append(f"| {fold} | {r['n']} | {r['qwk']:.4f} | {r['accuracy']:.4f} | "
                      f"{r['macro_f1']:.4f} | {r['macro_recall']:.4f} |")

    lines.append("\n## Per-class recall")
    lines.append("| Fold | NoDR | Mild | Mod | Sev | PDR |")
    lines.append("|------|------|------|------|------|------|")
    for fold in FOLDS:
        r = results[fold]["test"]["recall_per_class"]
        lines.append(f"| {fold} | {r['0']:.3f} | {r['1']:.3f} | {r['2']:.3f} | "
                      f"{r['3']:.3f} | {r['4']:.3f} |")

    lines.append("\n## Per-class F1")
    lines.append("| Fold | NoDR | Mild | Mod | Sev | PDR |")
    lines.append("|------|------|------|------|------|------|")
    for fold in FOLDS:
        r = results[fold]["test"]["f1_per_class"]
        lines.append(f"| {fold} | {r['0']:.3f} | {r['1']:.3f} | {r['2']:.3f} | "
                      f"{r['3']:.3f} | {r['4']:.3f} |")

    lines.append("\n## Confusion matrices (rows=true, cols=pred)")
    class_short = ["NoDR", "Mild", "Mod", "Sev", "PDR"]
    for fold in FOLDS:
        cm = results[fold]["test"]["confusion_matrix"]
        lines.append(f"\n### {fold} (n={results[fold]['test']['n']})\n")
        lines.append("| true\\pred | " + " | ".join(class_short) + " |")
        lines.append("|" + "---|" * (len(class_short) + 1))
        for i, row in enumerate(cm):
            lines.append(f"| **{class_short[i]}** | " + " | ".join(str(v) for v in row) + " |")

    md_path = DOCS / "lodo_balanced_eval.md"
    md_path.write_text("\n".join(lines))
    print(f"Saved → {md_path}")


if __name__ == "__main__":
    main()
