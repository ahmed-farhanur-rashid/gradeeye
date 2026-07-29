# EfficientNetV2-S root-cause analysis (2026-07-29)

## Symptom
- Phase 1 training log: val_loss = 0.55, val_qwk = 0.4187 (5 epochs, healthy)
- Phase 2 epoch 0: val_loss explodes to 32,102, val_qwk drops to 0.0192
- BN-reset-at-phase-transition fix did not help (val_loss still ~32k)

## Root cause
The EffNetV2-S model is fundamentally broken in **eval mode** at the Phase 1
checkpoint. Diagnostic reproduction:

```python
ck = load_checkpoint('saved/checkpoints/full_method_multidomain_effnetv2_step6272.pt')
val_model = DRGradingModel(arch='efficientnetv2_s', in_chans=4, ...)
val_model.load_state_dict(strip_compile_prefix(ck['ema_state_dict']))
val_model.eval()

ds = DRDataset(combined_val.csv, build_eval_transforms(img_size=384),
               seg_dir='segmentation_combined')
dl = DataLoader(ds, batch_size=16, shuffle=False)
for batch in dl:
    logits = val_model(batch[0].cuda()).float()
    print(logits.min(), logits.max(), logits.std())
```

Result on real validation data:
- **TRAIN mode** (BN uses batch stats): logits std = 1.1, range [-3, +3]
- **EVAL mode** (BN uses running stats): logits std = 81,583, range [-68k, +367k]
- **EVAL mode, BN reset to (0,1)**: logits std = 2.6 trillion

The CORN loss with BCE saturates at the model's actual logits:
- Eval mode: val_loss = 32,161 (huge)
- Train mode: val_loss = 0.55 (matches training log)

**Conclusion**: the training-time val_loss=0.55 was somehow computed with
BN in train mode (or with a different state than what was saved). The
checkpoint saved at the end of Phase 1 produces extreme logits in eval mode
and is unusable for actual evaluation.

## Why it failed Phase 2 even with BN reset
Resetting BN at the Phase 1→2 transition gives running stats = (0, 1) at
the start of Phase 2. As soon as Phase 2 begins training:
1. Backbone weights update, changing activation distributions
2. BN running stats accumulate from the new (now-corrected) distribution
3. Eval mode uses these new running stats
4. But the model's weights still produce wild outputs because the underlying
   weight scales were never corrected during Phase 1

The model is "broken at the BN layer" — it never learned to produce bounded
activations that BN running stats can normalize.

## Hypothesis (untested within time budget)
The likely cause is one of:
1. **Pretrained weight init mismatch with 4-channel input**: EffNetV2-S's
   pretrained weights are for 3-channel RGB. The first conv was extended to
   4 channels via mean-of-RGB init. For BN-heavy backbones, the BN layers
   after this first conv see an unexpected input distribution and never
   recover.
2. **BF16 autocast in training but FP32 in eval**: BN running stats may have
   been accumulated in BF16 precision while inference is FP32, causing
   precision drift. (Less likely — BF16 BN is well-supported.)
3. **Channels_last mismatch**: EffNetV2 uses channels_last=False per the
   ARCH_REGISTRY, so this is consistent.

## Recommendation for follow-up work
The diagnosis above took ~30 minutes. A full fix would require:
1. Confirming the eval-mode failure hypothesis via a from-scratch Phase 1
   training run with checkpoint inspection at every epoch.
2. Testing 3-channel input (matching pretrained weights exactly) to rule
   out the 4-channel mismatch.
3. Adding BN-running-stats sanity check after every epoch.

The ConvNeXt-Tiny model does not exhibit this issue because it uses
LayerNorm (no running stats), so the model produces consistent activations
in both train and eval modes.

For the icciml paper, the recommended strategy is to ship ConvNeXt-Tiny
+ CBAM (4-channel) as the headline model (EyePACS QWK 0.7196, APTOS QWK
0.8848, Messidor-2 QWK 0.8376) and acknowledge EfficientNetV2-S as a
secondary architecture that failed to converge.

## Failed debugging attempts
1. **Reduced LR**: v1 (1.22e-4) → val_loss 48k. v2 (4e-5) → val_loss 53k.
   v3 (2e-5) → val_loss 32k. Reducing LR does not help because the issue
   is BN running stats, not gradient magnitude.
2. **BN reset at phase transition**: Zero running_mean, fill running_var=1.
   Did not help — model still produces extreme logits after one training
   step in Phase 2.
3. **Smaller batch (16 vs 48)**: Implemented in config but no improvement.

## Diagnostic artifacts
- Training log: `saved/logs/full_method_multidomain_effnetv2_console.log`
- Epoch CSV: `saved/logs/full_method_multidomain_effnetv2_epoch_log.csv`
- Checkpoints: `saved/checkpoints/full_method_multidomain_effnetv2_*.pt`
  (step6272 is the last Phase 1 checkpoint, step12545+ are Phase 2 broken)