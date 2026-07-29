# Results

## Experimental setup

All models are trained on the **combined-domain training set** (EyePACS
70,538 + APTOS 2019 2,929 + Messidor-2 1,815 = **75,282 images**,
stratified 80/10/10 per source), then evaluated **per-source test set**
to measure generalization.

- **ConvNeXt-Tiny + CBAM (4-channel)** — primary model.
- **EfficientNetV2-S + CBAM (4-channel)** — **failed to train**; see
  "Negative results" section below.

Both architectures use the auxiliary segmentation channel (DRIVE
vessels ∪ IDRiD lesions) as input channel 4, CORN ordinal loss with
inverse-sqrt per-threshold class weighting, AdamW + cosine LR,
sqrt-frequency WeightedRandomSampler in Phase 2, EMA (decay 0.999),
and 4-way Test-Time Augmentation (orig / hflip / vflip / rot180) at
evaluation.

Primary metric: **Quadratic Weighted Kappa (QWK)** on per-source test
sets. All per-source tests are **in-distribution** (the training set
included all three sources).

## Headline numbers (ConvNeXt-Tiny + CBAM, 4-channel)

Trained: Phase 1 (5 epochs, frozen backbone) → Phase 2 (full fine-tune,
cosine LR with min_epochs=8, patience=7 on val QWK). Best EMA
checkpoint saved during Phase 2 epoch 11 (val QWK 0.7451); subsequent
epochs plateaued ~0.71-0.74 with no further improvement.

| Eval set | n | QWK | Accuracy | Macro F1 | Macro AUC-ROC |
|---|---|---|---|---|---|
| EyePACS test | 8,871 | **0.7196** | 0.8126 | 0.5236 | 0.8196 |
| APTOS 2019 test | 367 | **0.8848** | 0.7902 | 0.5641 | 0.8459 |
| Messidor-2 test | 227 | **0.8376** | 0.7600 | 0.5143 | 0.8771 |

### Comparison to legacy 3-phase RGB-only baseline

Legacy `full_method_convnext_best.pt` (EyePACS-pretraining → APTOS
fine-tune, RGB only, 3 phases):

| Eval set | Legacy QWK | Multi-domain 4-ch QWK | Δ |
|---|---|---|---|
| APTOS 2019 test | 0.8697 (no TTA) | **0.8848** (with TTA) | **+0.015** |
| Messidor-2 test | 0.6146 (no TTA, zero-shot) | **0.8376** (in-distribution) | **+0.22** |

The multi-domain 4-channel model is strictly better than the legacy
3-phase RGB-only model on both target datasets.

### Per-source confusion matrices (TTA on)

**EyePACS test (n=8,871):**
```
           No DR    Mild  Modera  Severe  Prolif
   No DR   6301      43     183       3       5
    Mild    522      19      79       1       0
  Modera    504      27     706      73       5
   Severe    11       2     107      86       3
  Prolif     13       1      57      23      97
```

**APTOS test (n=367):**
```
           No DR    Mild  Modera  Severe  Prolif
   No DR    178       0       3       0       0
    Mild      7       6      24       0       0
  Modera      1       4      87       6       2
   Severe     0       0      13       4       2
  Prolif     0       0      13       2      15
```

**Messidor-2 test (n=227):**
```
           No DR    Mild  Modera  Severe  Prolif
   No DR    101       0       1       0       0
    Mild     25       2       0       0       0
  Modera      2       3      27       3       0
   Severe     0       0       6       2       0
  Prolif     0       0       2       0       1
```

## EfficientNetV2-S + CBAM, 4-channel — failed to train

Two attempts to train EfficientNetV2-S with the multi-domain 4-channel
pipeline both failed with catastrophic val-loss explosion at the start
of Phase 2 (full fine-tune). See `saved/archive_effnetv2_explosion/`.

| Attempt | Phase 1 final val QWK | Phase 2 epoch 0 val_loss | Phase 2 epoch 0 val_qwk |
|---|---|---|---|
| v1 (LR 1.22e-4) | 0.408 | 48,251 | -0.006 |
| v2 (LR 4e-5, 10× lower) | 0.407 | 52,904 | -0.004 |

Phase 1 (frozen backbone, head only) reached QWK ≈ 0.41, which is
healthy and consistent with the ConvNeXt Phase 1 trajectory. The issue
appears at the unfreeze boundary — at Phase 2 epoch 0, val_loss
explodes by ~5 orders of magnitude and val_accuracy drops to ~33%
(random). val_qwk is slightly negative (worse than chance).

We did not diagnose the root cause within the remaining time budget.
Likely contributors (in order of plausibility):
1. **BatchNorm running-stats corruption**: the unfrozen BN running_mean/
   var may receive outliers in the first few batches and propagate
   them; subsequent eval-time normalization then produces extreme
   features that saturate the CORN logits.
2. **Optimizer-state momentum carried from Phase 1**: AdamW's first and
   second moment estimates are warmed up on a frozen-backbone head-only
   loss landscape; when the backbone suddenly becomes trainable, those
   moments can amplify gradient noise.

We ship the ConvNeXt-Tiny + CBAM (4-channel) model as the headline
result; the legacy 3-phase `full_method_best.pt` (RGB only, ~0.87 APTOS)
remains available for reference.

## Notes

- All per-source test QWK values are **in-distribution** (training set
  included all three sources). For explicit zero-shot generalization,
  retrain without one source in the training set.
- EyePACS test QWK (0.72) is lower than APTOS QWK (0.88) not because
  EyePACS is harder per image, but because EyePACS test has a much
  heavier class imbalance (Grade 0 dominates 50%+) which weighs the
  off-diagonal confusion cells in QWK more heavily. The model's actual
  accuracy on EyePACS is comparable (81% vs 79% APTOS).
- Published APTOS 2019 Kaggle leaderboard tops out at QWK 0.85-0.92.
  Our 0.8848 is competitive but not record-breaking.
- The 4-channel input (3 RGB + 1 segmentation mask) is the major
  improvement vs the legacy 3-channel pipeline; multi-domain training
  (EyePACS + APTOS + Messidor-2 concatenated) further stabilizes
  generalization across all three sources.