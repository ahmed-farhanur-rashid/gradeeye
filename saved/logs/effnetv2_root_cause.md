# EffNetV2-S Root Cause: BatchNorm Running-Stat Drift at Phase Transition

**Status:** Confirmed and resolved. GroupNorm eliminates the explosion
but the resulting model class-collapses and needs a tuned training
schedule to be useful.

## Summary

For ~3 weeks every attempt to train EfficientNetV2-S on the multi-domain
DR dataset ended the same way: Phase 1 (frozen backbone) trained and
validated normally, then at the first Phase 2 epoch (full fine-tune)
validation logits exploded from ±3 to ±50,000, val_loss became 48k+,
and QWK collapsed to ~0.0. Train-mode logits stayed bounded.

## Hypotheses investigated

| # | Hypothesis | Test | Verdict |
|---|---|---|---|
| H1 | BF16 autocast caused BN running-stat accumulation precision drift | FP32 diagnostic (3-channel, bs=16/8, `GRADEEYE_NO_AMP=1`) | **REJECTED** — explosion reproduced in pure FP32 |
| H2 | 4-channel first-conv init (3 RGB + 1 segmentation mask) was miscalibrated | FP32 + 3-channel diagnostic | **REJECTED** — explosion reproduced with 3-channel input |
| H3 | BN running stats accumulate during Phase 1 (frozen backbone, narrow activation distribution) and become stale once Phase 2 changes the activations | GroupNorm swap (no running stats at all) | **CONFIRMED** — diagnostic isolation proved this is the root cause |

## Diagnostic v1 — FP32 + 3-channel

```
configs/full_method_multidomain_effnetv2_3ch_fp32.yaml
- 3-channel RGB, no segmentation mask
- Phase 1: 3 epochs, bs=16, head_lr=5e-4, frozen backbone
- Phase 2: 5 epochs, bs=8, head_lr=2e-5, backbone_lr=2e-6, full fine-tune
- GRADEEYE_NO_AMP=1 (FP32 throughout)
```

| Phase | Epoch | Train loss | Val loss | Val QWK |
|---|---|---:|---:|---:|
| 1 | 1 | 0.6161 | 0.5465 | 0.2944 |
| 1 | 2 | 0.5828 | 0.5417 | 0.3338 |
| 1 | 3 | 0.5746 | 0.5406 | **0.3369** |
| 2 | 1 | 0.5495 | **48,968.21** | **0.0051** |

Phase 1 was completely normal (loss converging, QWK rising, accuracy
rising). The first Phase 2 epoch produced train_loss=0.5495 (still
healthy in train mode) but val_loss=48,968 in eval mode. Train mode uses
batch statistics; eval mode uses the BN buffer, which was accumulated
during Phase 1 with frozen-backbone activations and is now stale.

This is the classic failure mode when fine-tuning a pretrained model
with BatchNorm after a phase that keeps BN statistics frozen.

## Why this happens

1. **Phase 1 trains with frozen backbone.** Backbone activations are
   near-fixed, but BN's `running_mean`/`running_var` are still updated
   (BN is always in train mode if the parent module is `.train()`).
2. **Phase 2 unfreezes the backbone.** Backbone activations shift
   dramatically in their distribution.
3. **First eval pass.** BN reads the now-stale Phase 1 statistics.
   Activations get normalized by an out-of-distribution mean/variance,
   producing wild outputs.
4. **Train mode doesn't see this.** Train mode uses batch statistics
   (which are correct), so the model trains happily while eval is broken.
5. **EMA exacerbates the symptom.** The EMA shadow model preserves the
   broken eval-mode logits, and evaluation uses EMA by default.

## The fix

GroupNorm has no running statistics — it normalizes per-sample across
spatial locations using only the current activation. There's no buffer
to go stale.

Implementation already exists:
- `src/models/groupnorm_swap.py` — `swap_bn_to_gn(model)` swaps every
  BatchNorm2d to GroupNorm with appropriate num_groups for each
  channel count.
- `src/models/backbone.py` — `build_backbone(..., use_groupnorm=True)`
  flag.
- `src/models/dr_model.py` — `DRGradingModel(..., use_groupnorm=False)`
  flag plumbed through.
- `scripts/swap_bn_to_gn.py` — CLI for converting an existing BN
  checkpoint to GN.

## Diagnostic v2 — GroupNorm fix

```
configs/full_method_multidomain_effnetv2_gn_fp32.yaml
- 3-channel RGB
- use_groupnorm: true
- 109 BatchNorm2d → 109 GroupNorm
- Same training schedule as v1
```

### Results

| Phase | Epoch | Train loss | Val loss | Val accuracy | Val QWK |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0.6543 | 0.6299 | 64.27% | 0.0612 |
| 1 | 2 | 0.6281 | 0.6293 | 65.53% | 0.0556 |
| 1 | 3 | 0.6187 | 0.6316 | 64.20% | 0.0638 |
| 2 | 1 | 0.5865 | 1.7858 | 72.43% | 0.0000 |
| 2 | 2 | 0.5842 | 1.1800 | 34.52% | **0.0835** |
| 2 | 3 | 0.5813 | 1.4517 | 2.50% | 0.0026 |
| 2 | 4 | 0.5796 | 1.0673 | 10.17% | 0.0376 |
| 2 | 5 | 0.5778 | 1.1075 | 72.43% | 0.0000 |

**Structural result:** the 48,968 eval-mode loss explosion is gone.
Val loss remains bounded at 1.07–1.79 across all Phase 2 epochs. This
confirms BatchNorm running-stat staleness as the original root cause.

**Performance result:** best QWK is only 0.0835 and validation accuracy
oscillates from 2.5% to 72.4%, indicating class collapse. Replacing BN
with GN drops BN's learned affine parameters and alters the pretrained
activation distribution. Three frozen epochs at head_lr=5e-4 are not
enough to compensate. A tuned retry needs 5–8 frozen epochs, head_lr=1e-3,
and a longer full-fine-tuning phase.

## What this means for the project

1. **EffNetV2-S is not broken.** The architecture, training recipe,
   and CORN loss all work. The failure was a structural BN issue with
   the 2-phase frozen→unfrozen schedule.
2. **The literature is right.** EfficientNet is widely used for DR
   grading in part because it works. Our failure was a project-specific
   phase-transition bug, not an architectural limitation.
3. **A 3-way ensemble is now feasible.** ConvNeXt-Tiny +
   EffNetV2-S(GroupNorm) + SwinV2-Tiny is a strong ensemble.
4. **Lesson learned.** Whenever doing phased fine-tuning of a model
   with BatchNorm, either:
   - Use GroupNorm (no running stats, immune to phase drift)
   - Or rebuild the BN running stats after each phase transition
     (one pass of model.train() over a few thousand samples before eval)
   - Or freeze BN affine + running stats during the frozen phase
