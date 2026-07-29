# GradeEye — Diabetic Retinopathy Grading Pipeline

5-class (ICDR 0-4) diabetic retinopathy grading: backbones (ConvNeXt-Tiny /
EfficientNetV2-S) + CBAM attention (last 2 stages) + CORN ordinal regression
head + optional 4th-channel anatomical segmentation mask (DRIVE vessels ∪
IDRiD lesions, fine-tuned U-Net).

**Every dataset used in this project is validated as exactly 5-class
(0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR) at multiple
points in the pipeline — download, manifest-building, and dataset
construction all hard-fail on any other label cardinality.**

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

Kaggle API credentials required for dataset downloads — set up
`~/.kaggle/kaggle.json` per https://www.kaggle.com/docs/api. For the
EyePACS and APTOS competitions specifically, you must also click
"I Understand and Accept" on each competition's Kaggle page before the
API will allow downloads.

## Pipeline

```bash
# 1. Download raw datasets (EyePACS, APTOS, Messidor-2) — all 5-class
python scripts/download_datasets.py --dataset all

# 2. Extract datasets
python scripts/extract_datasets.py --dataset all

# 3. Build RAW manifests
python scripts/build_manifests.py --dataset all

# 4. One-shot image preprocessing (border crops, CLAHE, etc.)
python scripts/preprocess_all.py

# 5. Build per-source 80/10/10 stratified splits + combined train/val CSVs
python scripts/build_multidomain_splits.py

# 6. (Optional) Pre-compute segmentation masks for the 4-channel pipeline.
#    If DRIVE (data/drive/) AND IDRiD (data/A. Segmentation/) are present,
#    the U-Net is fine-tuned on their union first. Then masks are generated
#    for all three grading datasets into data/processed/segmentation_combined/.
python scripts/precompute_seg_masks.py --datasets eyepacs aptos messidor2

# 7. Train (pick a run config — see configs/)
# Note: To retrain a model from scratch, first clear its checkpoints/logs:
# rm -rf saved/checkpoints/<run_name>* saved/logs/<run_name>*

# Main: 2-phase multi-domain ConvNeXt-Tiny (4-channel, segmented)
python scripts/train.py --config configs/full_method_multidomain.yaml

# Second model: 2-phase multi-domain EfficientNetV2-S (4-channel, segmented)
python scripts/train.py --config configs/full_method_multidomain_effnetv2.yaml

# 8. Evaluate a single model checkpoint on any split (use --tta for Test-Time Augmentation)
# ConvNeXt on APTOS test:
python scripts/evaluate.py \
    --checkpoint saved/checkpoints/full_method_multidomain_best.pt \
    --manifest data/splits/aptos_test.csv \
    --tta

# ConvNeXt on Messidor-2 test (external validation):
python scripts/evaluate.py \
    --checkpoint saved/checkpoints/full_method_multidomain_best.pt \
    --manifest data/splits/messidor2_test.csv \
    --tta

# 9. Ensemble evaluation (average probabilities across architectures)
python scripts/ensemble_evaluate.py \
    --checkpoints saved/checkpoints/full_method_multidomain_best.pt \
                  saved/checkpoints/full_method_multidomain_effnetv2_best.pt \
    --manifest data/splits/aptos_test.csv \
    --tta
```

## Structure

```text
gradeeye/
│
├── README.md
├── context.md                     # Session persistence (read this after compaction)
├── requirements.txt
│
├── configs/                       # Run configs (multidomain / multidomain_effnetv2 / ...)
│
├── data/
│   ├── raw/                       # Downloaded raw datasets (eyepacs/aptos/messidor2)
│   ├── processed/                 # Precomputed images, manifests, segmentation masks
│   │   └── segmentation_combined/ # Flat directory of all 93k vessel/lesion masks
│   └── splits/                    # Stratified per-source + combined train/val/test CSVs
│
├── docs/
│   └── methodology.md             # Full methodology writeup (CORN, preprocessing, 4-channel seg, multi-domain training)
│
├── saved/
│   ├── checkpoints/               # Model checkpoints (EMA weights + arch metadata)
│   └── logs/                      # Per-run training CSV logs + epoch metrics
│
├── scripts/
│   ├── build_multidomain_splits.py
│   ├── precompute_seg_masks.py    # U-Net fine-tune + mask inference
│   ├── train.py                   # Multi-phase training CLI
│   ├── evaluate.py                # Single-model eval
│   └── ensemble_evaluate.py       # Multi-model CORN probability ensemble
│
└── src/
    ├── augmentation/              # Train/eval transforms (incl. _MaskSafeCompose for 4-ch), MixUp
    ├── data/                      # DRDataset (3-ch or 4-ch via seg_dir)
    ├── eval/                      # QWK, accuracy, F1, AUC-ROC, TTA
    ├── losses/                    # CORN ordinal loss + per-threshold class weighting
    ├── models/                    # ConvNeXt-Tiny / EfficientNetV2-S + CBAM + projection head
    │   └── segmentation.py        # U-Net (EfficientNet-B0 encoder) for vessel/lesion masks
    ├── preprocessing/             # Border crop, Ben Graham, green-CLAHE, circular mask, ImageNet norm
    └── training/                  # Trainer, AdamW + cosine LR, sqrt-freq sampler, EMA
```

## Key design decisions

- **Backbones**: ConvNeXt-Tiny (primary) and EfficientNetV2-S (secondary) —
  two robust architectures for the paper's ensemble.
- **Attention**: CBAM inserted into the last 2 backbone stages only.
- **Ordinal head**: CORN (not CORAL/CORAL^±) — structural rank-consistency via
  conditional training on 4 binary sub-problems (y>0, y>1, y>2, y>3).
- **Primary metric**: Quadratic Weighted Kappa (QWK), not accuracy.
- **Multi-domain training**: EyePACS + APTOS + Messidor-2 concatenated into a
  single training set (75,282 images). Per-source stratified 80/10/10 splits.
- **Two-phase pipeline**: Frozen-head warmup (5 epochs) → full fine-tune
  (35 epochs), all on combined-domain data. Single-stage; no separate
  APTOS fine-tune phase (caused overfitting in legacy 3-phase setup).
- **Segmentation (4th channel)**: Auxiliary U-Net fine-tuned on DRIVE vessels
  ∪ IDRiD lesions → ~93k inference masks pre-computed into
  `data/processed/segmentation_combined/`. Grader backbone accepts
  4-channel input; first-conv weights initialized as RGB-mean for the
  additional channel. Toggleable via `model.in_chans: 3` + `seg_dir` removed.
- **Class imbalance**: inverse-sqrt-frequency WeightedRandomSampler in
  Phase 2 + per-threshold inverse_sqrt weights in CORN loss.
- **Domain shift**: Mitigated by multi-domain training + 4-channel anatomical
  prior, NOT by camera-jitter augmentation (tested empirically: hurts both
  APTOS and Messidor-2 zero-shot).
