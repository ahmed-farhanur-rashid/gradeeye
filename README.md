# GradeEye — Diabetic Retinopathy Grading Pipeline

5-class (ICDR 0-4) diabetic retinopathy grading: ConvNeXt-Tiny backbone +
CBAM attention (last 2 stages) + CORN ordinal regression head.

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

# 2. Build manifests + stratified splits (validates 5-class again)
python scripts/build_manifests.py --dataset all

# 3. Compute per-source normalization stats (on preprocessed images)
python scripts/compute_norm_stats.py --dataset all

# 4. Train (pick a run config — see configs/)
python scripts/train.py --config configs/full_method.yaml
python scripts/train.py --config configs/baseline.yaml
python scripts/train.py --config configs/ablation_ce_weighted_cbam.yaml

# 5. Evaluate a checkpoint on any split
python src/eval/evaluate.py \
    --checkpoint saved/checkpoints/full_method_best.pt \
    --manifest data/splits/aptos_test.csv \
    --norm-stats data/processed/aptos_norm_stats.json

# 6. Generate all conference-paper figures + LaTeX tables in one shot
#    (confusion matrices, ROC curves, training curves, run-comparison
#    bar chart + table, per-class metrics tables), across the 3 run
#    configs at once:
python scripts/generate_paper_assets.py \
    --checkpoints saved/checkpoints/baseline_best.pt \
                  saved/checkpoints/ablation_ce_weighted_cbam_best.pt \
                  saved/checkpoints/full_method_best.pt \
    --test-manifest data/splits/aptos_test.csv \
    --norm-stats data/processed/aptos_norm_stats.json \
    --manifests-for-distribution EyePACS=data/processed/eyepacs_manifest.csv \
                                  APTOS=data/processed/aptos_manifest.csv \
                                  Messidor-2=data/processed/messidor2_manifest.csv
```

Outputs land in `paper_assets/figures/` (PNG + PDF) and `paper_assets/tables/`
(`.tex` files, `\usepackage{booktabs}` required in your paper's preamble).

## Structure

```
configs/            Run configs (baseline / ablation / full_method)
data/raw/            Downloaded raw datasets (eyepacs/aptos/messidor2)
data/processed/      Manifests + per-source normalization stats
data/splits/         Stratified train/val/test CSVs per source
src/preprocessing/   Border crop, color correction, anisotropic filter, normalize
src/augmentation/    Train/eval transforms, MixUp
src/models/          ConvNeXt-Tiny backbone, CBAM, projection head, CORN layer
src/losses/          CORN loss, per-threshold class weighting, CE baseline
src/data/            Dataset classes, stratified splitting
src/training/        Trainer, optimizer/scheduler, checkpointing, EMA
src/eval/            Metrics (QWK primary), evaluate.py, figures.py, latex_tables.py
scripts/             CLI entry points for each pipeline stage
scripts/generate_paper_assets.py   All figures + LaTeX tables in one run
saved/checkpoints/   Model checkpoints (self-contained: config + class names embedded)
saved/logs/          Per-run training CSV logs (per-batch AND per-epoch)
paper_assets/        Generated figures (PNG+PDF) and LaTeX tables
```

## Key design decisions (see plan for full rationale)

- **Backbone**: ConvNeXt-Tiny is locked — not swappable for ResNet/EfficientNet/ViT.
- **Attention**: CBAM inserted into the last 2 backbone stages only.
- **Ordinal head**: CORN (not CORAL) — structural rank-consistency via
  conditional training on 4 binary sub-problems (y>0, y>1, y>2, y>3).
- **Primary metric**: Quadratic Weighted Kappa (QWK), not accuracy.
- **3-phase training**: EyePACS frozen-head → EyePACS full-unfreeze →
  APTOS fine-tune (heavier augmentation, lower LR).
- **Messidor-2**: eval-only, zero gradient updates — external validation set.
- **Class imbalance**: inverse-sqrt-frequency weighting applied per CORN
  binary sub-problem, not as flat 5-class weights.
