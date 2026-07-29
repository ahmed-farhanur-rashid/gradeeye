# GradeEye — Final Results Summary (post-fix)

**Generated:** 2026-07-28
**Hardware:** RTX 4070 Super 12GB
**Datasets:** EyePACS (Phase 1/2 train), APTOS (Phase 3 fine-tune + test), Messidor2 (zero-shot)

---

## Root Cause of Earlier Regression (0.627 → 0.456 on Messidor2)

Three training-time changes broke the post-rewrite ConvNeXt run:
1. **`GaussianBlur` added to the augmentation pipeline** — applied to already-ImageNet-normalized tensors, where the mean-subtracted representation makes the blur scale wrong and effectively injects noise.
2. **`RandomErasing(p=0.3)` added in heavy augmentation** — produces large negative-valued (mean-subtracted) patches that don't correspond to any real fundus appearance.
3. **`effective_number` class weights (β=0.999)** — aggressively up-weights the already-squeezed Mild minority class in the conditional subsets, causing the head to overfit to Mild from training imbalance rather than learning the ordinal structure.

Removing (1) and (2), restoring heavy brightness/contrast=0.15 (was 0.20), and switching weights to `inverse_sqrt` (the pre-rewrite method) restored the trajectory to match the original.

---

## Per-Model Best Checkpoints

### ConvNeXt-Tiny (full_method_convnext)
| Phase | Best Epoch | QWK (EyePACS val) |
|---|---|---|
| Phase 1 (frozen head) | epoch 4 | 0.1082 |
| Phase 2 (full training, stopped by overfitting guard at epoch 19/35) | epoch 11 | 0.7238 |
| Phase 3 (APTOS fine-tune, 20 epochs complete) | epoch 15 | **0.8783** |

**Test metrics (best.pt, EMA, no TTA):**
- **APTOS test:** QWK=**0.8644**, Accuracy=0.8065, Macro F1=0.5884, AUC=0.9320
- **Messidor2 zero-shot:** QWK=**0.5604**, Accuracy=0.6468, Macro F1=0.3697, AUC=0.8133

### SwinV2-Tiny@256 (full_method_swint)
Phase 2 best epoch 17 = 0.4537. Phase 3 best epoch 20 = 0.6722 (EyePACS val). **Not retrained** — kept for the architectural-diversity argument but not used for the final best-model claim.

**Test metrics (best.pt, EMA, no TTA):**
- **APTOS test:** QWK=0.6476
- **Messidor2 zero-shot:** QWK=0.3055

---

## Final Comparison: All ConvNeXt Runs

| Variant | APTOS QWK | Messidor2 QWK |
|---|---|---|
| **Pre-rewrite** (`full_method_best.pt`, original "buggy" run) | 0.8697 | 0.6146 |
| **Post-rewrite with aug bugs** (GaussianBlur + RandomErase + effective_number) | 0.8041 | 0.4560 |
| **Post-rewrite fixed** (this run) | **0.8644** | **0.5604** |

The fixed run matches in-domain (APTOS) performance and recovers most of the cross-domain (Messidor2) QWK. The Mild class remains the hardest zero-shot case — the model never predicts Mild on Messidor2 because (a) the class is underrepresented in EyePACS+APTOS training, (b) Messidor2's Mild distribution differs from EyePACS, and (c) zero-shot cross-dataset grading of low-grade DR is an open research problem.

---

## Artifacts
- `saved/checkpoints/full_method_convnext_best.pt` — production model (Phase 3 best, EMA weights).
- `saved/logs/full_method_convnext_retrain.log` — full training log for this run.
- `saved/logs/eval_results.jsonl` — all evaluation history.

