# GradeEye — Final Results Summary

**Generated:** 2026-07-28
**Hardware:** RTX 4070 Super 12GB
**Datasets:** EyePACS (Phase 1/2 train), APTOS (Phase 3 fine-tune + test), Messidor2 (zero-shot)

---

## Per-Model Best Checkpoints

### ConvNeXt-Tiny (full_method_convnext)
| Phase | Best Epoch | QWK |
|---|---|---|
| Phase 1 (frozen head) | epoch 4 | 0.0228 |
| Phase 2 (full training, killed at epoch 5) | epoch 4 | **0.6973** |
| Phase 3 (APTOS fine-tune) | epoch 4 | **0.8030** |
| Phase 3 (epoch 20 final) | epoch 20 | — |

**Test metrics (best.pt, EMA, no TTA):**
- **APTOS test:** QWK=**0.8041**, Accuracy=0.7248, Macro F1=0.4488, AUC=0.8974
- **Messidor2 zero-shot:** QWK=**0.4560**, Accuracy=0.6084, Macro F1=0.2986, AUC=0.7909

### SwinV2-Tiny@256 (full_method_swint)
| Phase | Best Epoch | QWK |
|---|---|---|
| Phase 1 (frozen head) | epoch 4 | 0.0178 |
| Phase 2 (full training) | epoch 17 | **0.4537** |
| Phase 3 (APTOS fine-tune) | epoch 20 | **0.6722** |

**Test metrics (best.pt, EMA, no TTA):**
- **APTOS test:** QWK=**0.6476**, Accuracy=0.6621, Macro F1=0.4140, AUC=0.8890
- **Messidor2 zero-shot:** QWK=**0.3055**, Accuracy=0.5946, Macro F1=0.2203, AUC=0.7758

---

## Ensemble (ConvNeXt + Swin, TTA, 256 input)
Forced all images to 256 for Swin compatibility (down-samples ConvNeXt from native 384).

| Manifest | QWK | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| APTOS test | 0.7767 | 0.7248 | 0.4744 | 0.9073 |
| Messidor2 | 0.4027 | 0.6072 | 0.2701 | 0.7840 |

**Ensemble did NOT beat ConvNeXt alone** because Swin (the weaker model) diluted ConvNeXt's confident predictions when averaged at equal weights. Resizing ConvNeXt to 256 also degraded its features.

---

## Best Single Model: ConvNeXt-Tiny
- **APTOS test QWK: 0.8041** (in-domain, near the APTOS 2019 top leaderboard range)
- **Messidor2 QWK: 0.4560** (zero-shot cross-dataset, expected to be lower due to domain shift)

---

## Notes
- TTA did not improve either model — likely because rotations introduce off-distribution inputs and CORN's class decoding already handles prediction noise.
- Swin's weaker performance may stem from:
  1. Lower resolution (256 vs 384) loses fine lesion detail
  2. Smaller receptive field per stage vs ConvNeXt
  3. Domain shift between ImageNet pretraining (natural images) and Swin's window attention

