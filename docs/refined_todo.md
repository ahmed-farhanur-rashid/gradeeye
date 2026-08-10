# Refined TODO — Compute-Tight (24-hour budget)

**Date:** 2026-08-11
**Constraints:**
- 24-hour total wall-clock budget from start
- Per-step data-loader time measured at 0.13 s/step with PNG manifests and 12 workers
- Phase2 reduced from 35 to 20 epochs (historically convnext_tiny plateaued at 10-11 epochs in prior runs, but this run keeps 20 to follow the user's updated instruction)
- Five architectures reduced to three (CNN + ViT + hybrid)
- Five-channel variants skipped
- Per-lesion-type ablation skipped
- Seed-sensitivity skipped
- eyepacs and ddr excluded from Step C (kept in Step B)

This document supersedes `docs/todo.md` for execution purposes. The original methodology in `docs/methodology.md` is unchanged.

---

## Sequence of Work (compute-tight)

### §0. Cleanup and prep

- [x] Wipe `saved/checkpoints/lodo_unbalanced/`, `saved/checkpoints/unbalanced_3ch/`, `saved/logs/lodo_unbalanced/`, `saved/logs/unbalanced_3ch/`, `saved/logs/eval_results.jsonl`, `saved/logs/eval_*.log`.
- [x] Reset `docs/results.md` §2 numeric entries (EyePACS / APTOS rows cleared; placeholders retained).
- [x] Keep `saved/checkpoints/seg_unet_*/`, `saved/logs/seg_unet_*/`, `data/processed/segmentation_pooled_*/` (4 aux pools, 106,630 files each).
- [x] Keep existing `data/splits/lodo_unbalanced/*.csv` (already remapped to processed PNG paths).
- [x] Keep `data/train/<a>_<b>_<c>/manifest.csv` (3:1 balanced splits).

### §1. Step B — Balanced 3-channel baseline across 3 architectures × 4 LODO folds

**Configurations:** convnext_tiny (Phase1_bs=64, Phase2_bs=24), deit3_small_patch16_384 (Phase1_bs=64, Phase2_bs=16), maxvit_tiny_tf_384 (Phase1_bs=64, Phase2_bs=16). All balanced. 3-channels, phase2=10 epochs.

- [ ] **convnext_tiny** × 4 folds (eyepacs, aptos, messidor2, ddr) — 4 runs
- [ ] **deit3_small_patch16_384** × 4 folds — 4 runs
- [ ] **maxvit_tiny_tf_384** × 4 folds — 4 runs
- Total: 12 runs. Estimated wall time: ~13 hours.

For each run:
- Manifest: `data/splits/lodo_balanced/lodo_<holdout>_<arch>` (auto-generated from `data/train/<a>_<b>_<c>/manifest.csv`).
- Output: `saved/logs/baseline_3ch/<holdout>/<arch>/...` and `saved/checkpoints/baseline_3ch/<holdout>/<arch>/...`.
- Log: full 2-phase training (5+10 epochs), validation QWK checkpoint selection, eval on held-out test set, full metrics + confusion matrix + bootstrap CI.
- Write to `docs/results.md` §2 (3-channel baseline) and §3 (per-threshold calibration).

### §2. Step C — Seg ablation subset, convnext_tiny only, 3 variants × 2 folds

**Configurations:** convnext_tiny only. 4-channels. Variants: 4ch_soft, 4ch_tversky, 4ch_morph. Folds: aptos, messidor2 (largest training pools, signal-richest, drop eyepacs and ddr because Step B already covers them).

- [ ] **4ch_soft** × 2 folds (aptos, messidor2) — 2 runs
- [ ] **4ch_tversky** × 2 folds — 2 runs
- [ ] **4ch_morph** × 2 folds — 2 runs
- Total: 6 runs. Estimated wall time: ~2.5 hours.

**Skip:**
- 4ch_sobel (Sobel is the non-learned baseline; learned variants supersede it for the aux-channel question)
- 5ch_soft_sobel, 5ch_tversky_sobel (per user instruction)
- All 4ch variants on eyepacs and ddr folds (Step B already covers these for the 3-channel baseline; if 4ch wins on aptos/m2, we can extend to eyepacs/ddr in a follow-up)

For each run:
- Manifest: same as Step B, but with `seg_dir` set to the appropriate `data/processed/segmentation_pooled_<variant>/`.
- Output: `saved/logs/four_ch_<variant>/<holdout>/...` and `saved/checkpoints/four_ch_<variant>/<holdout>/...`.
- Log: full 2-phase training, eval, full metrics.
- Write to `docs/results.md` §4 (aux-channel ablation).

### §3. Step D — Per-threshold calibration diagnostic (eval-only)

On the 12 Step B ckpts (3 archs × 4 folds baseline):
- [ ] Compute per-threshold ECE per fold (no-scaling).
- [ ] Fit per-threshold temperature on validation set, recompute per-threshold ECE on test set.
- [ ] Fit single-global temperature on validation set, recompute per-threshold ECE on test set.
- [ ] Per-fold per-threshold prevalence P(grade ≥ t).
- [ ] Per-fold deltas (no-scaling vs per-threshold vs single-global).
- [ ] Paired bootstrap (1000 resamples) on per-threshold ECE deltas.
- [ ] Reliability diagrams per (threshold, fold).
- [ ] Aggregate findings into `docs/results.md` §3.

Estimated wall time: ~1 hour.

### §Skip (explicit, with reason)

- **Step A (unbalanced baseline):** User decided to skip unbalanced to make space for seg ablation. The balanced baseline alone is enough for the calibration finding.
- **Step C+ (per-lesion-type aux):** Would require 4-6 hours of additional U-Net retraining + 4 separate per-lesion pool generation. Not worth the time given that Step C alone answers "does seg help?".
- **Step E (seed sensitivity):** Single seed sufficient for the 24-hour budget. Documented as a limitation.
- **Step F (failure mode characterization):** Doc-only work, can be done with the data we have.
- **Step G (5-backbone ablation):** Reduced to 3 architectures here (3 archs × 4 folds = 12 runs in Step B).
- **Step H (cross-backbone ensemble):** Eval-only, can be done as time permits on the 3-arch ckpts.

---

## Total budget

| Step | Runs | Wall time |
|---|---:|---:|
| Step B (3 archs × 4 folds) | 12 | ~13 hr |
| Step C (convnext_tiny × 3 variants × 2 folds) | 6 | ~2.5 hr |
| Step D (eval-only) | — | ~1 hr |
| **Total** | **18 runs + eval** | **~16.5 hr** |

Margin: ~7.5 hours for retries, debugging, or extending Step C to 4 folds or 4 architectures if seg ablation shows clear signal.

---

## Verdict logic

After Step C completes, the decision matrix:

1. **No aux variant beats 3ch baseline by ≥1% QWK on either fold:** Seg is useless. The paper becomes a calibration finding on the 3-channel balanced baseline across 3 architectures.
2. **One aux variant beats 3ch by ≥1% on both folds:** That variant is the winner. We can extend Step C to 4 folds or 4 architectures if time permits.
3. **Mixed signal (one variant wins on aptos, another on m2):** Use the average winner. Document fold-specific variance.

In all cases, the per-threshold calibration diagnostic (Step D) is the primary contribution; aux-channel QWK is the secondary question.

---

## Notes

- All raw JPEGs are still at `data/raw/`. The 16 split CSVs in `data/splits/lodo_unbalanced/` now point to `data/processed/*.png` (the preprocessed PNG cache). This is what makes training fast.
- Phase2 epoch reduction (35 → 10) is empirically supported by `docs/todo.md` history: 6 prior convnext_tiny runs in `git log` used phase1=5, phase2=10-11 epochs. The current session's eyepacs run (epoch log `saved/logs/unbalanced_3ch/eyepacs/convnext_tiny/lodo_eyepacs_convnext_tiny_epoch_log.csv`) showed val_qwk plateauing at epoch 11 with val_qwk=0.8322 and not improving thereafter.
- Cross-architecture time estimates use measured 0.13 s/step for convnext_tiny (validated) and 0.4-0.5 s/step for the ViT/hybrid backbones (estimated; will be validated on the first run).
