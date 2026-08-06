# GradeEye — Diabetic Retinopathy Grading Pipeline

5-class (ICDR 0-4) diabetic retinopathy grading: backbones (ConvNeXt-Tiny,
ConvNeXt-Small, DeiT-III-Small, MaxViT-Tiny, SwinV2-Base) + CBAM attention
(last 2 stages) + CORN ordinal regression head + optional 4th-channel
anatomical segmentation mask (DRIVE vessels ∪ IDRiD lesions, fine-tuned
U-Net). Evaluated under LODO (Leave-One-Domain-Out) across EyePACS, APTOS,
and Messidor-2.

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

All commands live under `src/` — there is no `scripts/` directory.
The data pipeline is one CLI with subcommands; training is one entry point.

```bash
# 1. Build raw manifests from downloaded datasets (EyePACS, APTOS, Messidor-2)
python -m src.data.cli manifests --dataset all

# 2. One-shot image preprocessing (border crops, CLAHE, etc.)
python -m src.data.cli preprocess --dataset all

# 3. Build LODO splits (3 folds × 4 CSVs each)
python -m src.data.cli splits

# 4. (Optional) Pre-compute segmentation masks for the 4-channel pipeline.
#    If DRIVE (data/raw/drive/) AND IDRiD (data/raw/A. Segmentation/) are present,
#    the U-Net is fine-tuned on their union first. Then masks are generated
#    for all three grading datasets into data/processed/segmentation/<source>/.
python -m src.data.cli segmentation --datasets eyepacs aptos messidor2

# 5. Generate the LODO configs split by axis:
#    configs/models/<arch>.yaml      (5 files — what the model is)
#    configs/lodo/<holdout>.yaml     (3 files — which source is held out)
#    The 5×3 = 15 training runs are constructed at invocation time by
#    `src.train --model-config X --lodo-config Y` (deep-merge).
python -m src.data.lodo_configs
# (optionally: --archs convnext_tiny swinv2_base_window12to24_192to384)

# 6. Train. One entry point, two flags. Run-name is derived deterministically
#    as f"lodo_{holdout}_{arch}", so checkpoint/log paths are predictable.
#    To retrain from scratch: rm -rf saved/checkpoints/<run_name>* saved/logs/<run_name>*
python -m src.train \
    --model-config configs/models/convnext_tiny.yaml \
    --lodo-config configs/lodo/eyepacs.yaml

# 7. Evaluate a single checkpoint on any split (--tta for Test-Time Augmentation)
python -m src.eval.evaluate \
    --checkpoint saved/checkpoints/lodo_eyepacs_convnext_tiny_best.pt \
    --manifest data/splits/lodo_eyepacs_test_matched.csv \
    --tta

# 8. Calibrate decision thresholds (probability → ordinal class) on a holdout
python -m src.eval.calibrate_thresholds \
    --checkpoint saved/checkpoints/lodo_eyepacs_convnext_tiny_best.pt \
    --val-manifest data/splits/lodo_eyepacs_val.csv

# 9. Ensemble evaluation (average probabilities across architectures)
python -m src.ensemble.evaluate \
    --checkpoints saved/checkpoints/lodo_eyepacs_convnext_tiny_best.pt \
                  saved/checkpoints/lodo_eyepacs_swinv2_base_window12to24_192to384_best.pt \
    --manifest data/splits/lodo_eyepacs_test_matched.csv \
    --tta
```

## Structure

Pearl-style: everything lives under `src/`. No top-level `scripts/`
directory — entry points are `python -m <module>` invocations of library
code, not free-standing scripts.

```text
gradeeye/
│
├── README.md
├── context.md                     # Session persistence (read this after compaction)
├── requirements.txt
│
├── configs/                       # LODO configs split by axis
│   ├── models/                    #   5 model yamls — what the model is
│   └── lodo/                      #   3 LODO yamls — which source is held out
│
├── data/
│   ├── raw/                       # Downloaded raw datasets (eyepacs/aptos/messidor2)
│   ├── processed/                 # Precomputed images, manifests, segmentation masks
│   │   └── segmentation/          # Per-source vessel/lesion masks (one dir per source)
│   └── splits/                    # LODO CSVs: train/val/test_full/test_matched per fold
│
├── docs/
│   └── methodology.md             # Full methodology writeup (CORN, preprocessing, 4-channel seg, LODO)
│
├── saved/
│   ├── checkpoints/               # Model checkpoints (EMA weights + arch metadata)
│   └── logs/                      # Per-run training CSV logs + epoch metrics
│
└── src/
    ├── train.py                   # Unified training entry point (LODO pipeline)
    │
    ├── augmentation/              # Train/eval transforms (incl. _MaskSafeCompose for 4-ch), MixUp
    ├── data/
    │   ├── _00_manifests.py       # Build per-source (image_path, label) CSVs
    │   ├── _01_preprocess.py      # Crop/resize/CLAHE/filter pipeline
    │   ├── lodo_split.py          # LODO 5-class cross-domain split builder
    │   ├── lodo_configs.py        # Generate 5×3 = 15 LODO configs
    │   ├── cli.py                 # Unified data CLI (manifests/preprocess/segmentation/splits)
    │   ├── datasets.py            # DRDataset (3-ch or 4-ch via seg_dir)
    │   └── _03_segmentation.py    # U-Net fine-tune (DRIVE+IDRiD) + mask inference
    ├── eval/                      # Single-model eval, threshold calibration, metrics, TTA
    ├── ensemble/                  # Multi-checkpoint probability averaging + eval
    ├── losses/                    # CORN ordinal loss + per-threshold class weighting
    ├── models/                    # ConvNeXt-Tiny / DeiT-III / MaxViT / SwinV2 + CBAM + projection head
    │   └── segmentation.py        # U-Net (EfficientNet-B0 encoder) for vessel/lesion masks
    ├── preprocessing/             # Border crop, Ben Graham, green-CLAHE, circular mask, ImageNet norm
    └── training/                  # Trainer, AdamW + cosine LR, sqrt-freq sampler, EMA, checkpoint I/O
```

## Key design decisions

- **5 backbones**: ConvNeXt-Tiny / ConvNeXt-Small / DeiT-III-Small /
  MaxViT-Tiny / SwinV2-Base. The LODO matrix (5 × 3 = 15 training runs)
  lets us measure cross-domain transfer per-architecture. The 15 runs
  are constructed at invocation time from 5 model yamls (`configs/models/`)
  + 3 LODO yamls (`configs/lodo/`) — not 15 monolithic configs on disk.
- **Attention**: CBAM inserted into the last 2 backbone stages only.
- **Ordinal head**: CORN (not CORAL/CORAL^±) — structural rank-consistency via
  conditional training on 4 binary sub-problems (y>0, y>1, y>2, y>3).
- **Primary metric**: Quadratic Weighted Kappa (QWK), not accuracy.
- **LODO evaluation**: EyePACS / APTOS / Messidor-2 each held out in turn.
  Train pool = the other two. Cross-domain transfer is the real test —
  a model that nails EyePACS but collapses on Messidor-2 is overfit to
  EyePACS camera/center priors.
- **Two-phase pipeline**: Frozen-head warmup (5 epochs) → full fine-tune
  (35 epochs), all on the LODO train pool. Single config per fold; no
  separate APTOS fine-tune phase (caused overfitting in legacy 3-phase setup).
- **Segmentation (4th channel)**: Auxiliary U-Net fine-tuned on DRIVE vessels
  ∪ IDRiD lesions → per-source inference masks pre-computed into
  `data/processed/segmentation/<source>/`. Grader backbone accepts
  4-channel input; first-conv weights initialized as RGB-mean for the
  additional channel. Toggleable via `model.in_chans: 3` + `seg_dir` removed.
- **Class imbalance**: inverse-sqrt-frequency WeightedRandomSampler in
  Phase 2 + per-threshold inverse_sqrt weights in CORN loss.
- **Threshold calibration**: After training, fit per-threshold decision
  cutoffs on the LODO val set (not the test set!) via
  `src.eval.calibrate_thresholds` — needed because CORN's conditional
  probabilities are not naturally calibrated as a 5-way classifier.
- **Domain shift**: Mitigated by multi-domain training + 4-channel anatomical
  prior, NOT by camera-jitter augmentation (tested empirically: hurts both
  APTOS and Messidor-2 zero-shot).
