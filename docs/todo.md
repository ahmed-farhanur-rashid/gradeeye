# TODO

This document is the canonical runbook for the new agent. It tells you what to do, in what order, and exactly where to log every result.

**Read first, in this order:** `docs/methodology.md`, `docs/results.md`, `docs/gradeeye_citations.md`. The methodology is the source of truth for *what* to do. The results file is the source of truth for *where* to put numbers. The citation file is the source of truth for *what's already been published*.

## 0. Standing Rules

These rules apply to every step below. They are not optional.

**Rule 1 — No number without a logged artifact.** Every number that goes into `docs/results.md` must be backed by a file on disk (CSV row, JSON entry, or a checkpoint whose load-and-eval reproduces the number). State the file path next to the number the first time you write it. If a result cannot be reproduced from a logged artifact, it does not exist. Do not invent, estimate, or recall it.

**Rule 2 — No overwrite of existing artifacts.** Every experiment must write into an empty or uniquely-named subdirectory under `saved/logs/<experiment_name>/` and `saved/checkpoints/<experiment_name>/`. Before starting any training run, verify the target directory is empty or uniquely named. If it is non-empty, **stop and ask the user.** Do not overwrite, do not append, do not reuse.

**Rule 3 — Per-experiment isolation.** Each of the experiments listed in §1 produces its own subdirectory under `saved/logs/` and `saved/checkpoints/`. The naming convention is:

```
saved/logs/<experiment_name>/<holdout_source>/<arch>/...
saved/checkpoints/<experiment_name>/<holdout_source>/<arch>/...
```

where `<experiment_name>` is the variant identifier (`baseline_3ch`, `four_ch_soft`, `four_ch_sobel`, `four_ch_tversky`, `four_ch_morph`, `five_ch_soft_sobel`, `five_ch_tversky_sobel`, `unbalanced_3ch`, `tuned_thresholds_3ch`, `seed_sensitivity`, etc.). The directory layout is enforced; do not deviate from it.

**Rule 4 — All previously generated logs, checkpoints, and segmentation pools have been deleted from disk.** The `saved/logs/`, `saved/checkpoints/`, AND `data/processed/segmentation_pooled*/` directories were wiped before this runbook was written. Nothing previously stored is recoverable. The only authoritative numerical record of the experiments is what you produce now, logged into `saved/` *and* transcribed into `docs/results.md`. The segmentation pools must be regenerated (§1.3) before any 4-channel or 5-channel variant can be run, because the auxiliary channels are loaded from those pools at training time.

**Rule 5 — Stop and request permission before any decision that picks a winner.** §1.13 below is the permission gate. Do not proceed past it without an explicit "yes" from the user.

## 1. Sequence of Work

The experiments are ordered by dependency. Do not skip steps. Do not run steps out of order. The dependency graph is documented inline.

**Execute in this order, period:**

1. **§1.1 Setup Verification** — verify data on disk.
2. **§1.2 Configuration Verification** — smoke-test all 140 cells × 5 filenames = 700 filenames, zero collisions.
3. **§1.3 Segmentation Auxiliary Model** — train the U-Net (BCE+Dice + Tversky on DDR + IDRiD-train), generate all 4 aux pools, verify DDR test coverage. **This is the long blocking step (~4–6 hours on GPU); start it in the background as soon as §1.1/§1.2 pass.**
4. **§1.4 Step A** — Unbalanced baseline (4 runs).
5. **§1.5 Step B** — Balanced baseline (4 runs). **Main run.**
6. **§1.6 Per-threshold calibration diagnostic** on Step B checkpoints (eval-only).
7. **§1.7 Step C** — Aux-channel ablation (24 runs).
8. **§1.8 Step C+** — Per-lesion-type ablation (20 runs, optional).
9. **§1.9 Step D** — Threshold tuning (post-hoc, eval-only).
10. **§1.10 Step E** — Seed sensitivity (conditional, 0–6 runs).
11. **§1.11 Step F** — Failure mode analysis (doc work).
12. **§1.12 Step F.5** — Summary findings draft (doc work).
13. **⛔ §1.13 PERMISSION GATE — STOP, ASK USER** for the winning (variant, backbone) config.
14. **§1.14 Step G** — Headline run (up to 16 runs on the other 4 backbones). **Main contribution.**
15. **§1.15 Step H** — Cross-backbone ensemble phase (eval-only, additive on top of Step G).

**Do not display these steps out of order.** A todo-list UI that alphabetizes them is misleading; the dependency order is the run order.

### 1.1 Setup Verification

- [ ] Confirm the four source datasets are present and preprocessed.
- [ ] Confirm the 3:1 class-balanced manifests exist for each of the four LODO folds (each fold has a train, val, and test manifest).
- [ ] Confirm the grading-scheme reconciliation (5-class target) is applied uniformly across all four sources. The DDR reconciliation (drop ungradable, label 5) is documented in the methodology.
- [ ] Confirm the auxiliary-channel pools exist (segmentation soft masks for the soft / Tversky / morph variants, Sobel edges for the Sobel variants).
- [ ] Confirm the four-source pool is a documented subset of the GDRBench eight-source pool and the choice of subset is documented in the methodology.
- [ ] Confirm that every previously generated artifact under `saved/` is gone. If anything remains, stop and ask the user before continuing.

### 1.2 Configuration Verification

- [ ] Confirm the model configs at `configs/models/<arch>.yaml` cover the five architectures named in the methodology (ConvNeXt-Tiny, ConvNeXt-Small, DeiT-3 Small 384, MaxViT Tiny 384, SwinV2-Base 192→384).
- [ ] Confirm the variant configs at `configs/variants/<variant>.yaml` cover the six auxiliary-channel variants (4ch soft, 4ch Sobel, 4ch Tversky, 4ch morph, 5ch soft+Sobel, 5ch Tversky+Sobel). Each variant config must contain **only** the variant's own fields (`model.in_chans`, `phases.*.seg_dir`, `log_dir`, `checkpoint_dir`); the merge helper enforces this allow-list.
- [ ] Confirm the LODO configs at `configs/lodo_balanced/<holdout>.yaml` cover the four held-out sources (eyepacs, aptos, messidor2, ddr).
- [ ] Run a smoke test on every (backbone × variant × holdout) cell by calling the merge helper in dry-run mode and confirming (a) `log_dir` / `checkpoint_dir` resolve to the per-experiment paths and (b) no two cells write to the same filename. The expected matrix is 5 backbones × (1 no-variant baseline + 6 variants) × 4 holdouts = 140 cells, 5 filenames per cell, 700 filenames, zero collisions. **Do not start training from this step.** Only verify the resolution.
- [ ] If any config is missing or the path resolution does not isolate per-experiment, stop and ask the user before continuing.

### 1.3 Segmentation Auxiliary Model

This step regenerates the four auxiliary-channel pools that the 4ch and 5ch variants depend on. All four pools must cover **all four sources** (eyepacs, aptos, messidor2, ddr) — the previous pools covered only the first three and excluded DDR test images, which made the 4ch/5ch variants unevaluable on the DDR-holdout fold.

**Training source: DDR lesion segmentation is the primary source (383 train + 149 valid + 225 test images, 4 lesion types per image: MA / HE / EX / SE).** IDRiD (54 train + 27 test at `data/raw/A. Segmentation/`) is held out as the cross-domain evaluation set; it is optionally added to the training pool via `--use-idrid-train` if the agent wants to. IDRiD provides the gold-standard test signal because the methodology references it directly. IDRiD-only training was rejected because 54 images is too small to generalize — see the methodology §6 note in `docs/methodology.md`.

- [ ] Verify the DDR lesion-segmentation training data is on disk at `data/raw/DDR-dataset/lesion_segmentation/` (383 train images). If anything is missing, stop and ask the user.
- [ ] Verify the IDRiD lesion-segmentation data is on disk at `data/raw/A. Segmentation/` (54 train + 27 test). If anything is missing, stop and ask the user.
- [ ] Train the U-Net + EfficientNet-B4 segmentation model with **BCE + Dice** loss on DDR (383 train images), with the 4-fold in-domain val split (336 train / 47 val). IDRiD is *not* in the training pool by default. Log Dice / IoU / Tversky on the IDRiD test split (27 images, cross-domain) and the DDR test split (225 images, in-domain).
  ```bash
  PYTHONPATH=. python -m src.segmentation.train_mask \
      --loss bce_dice --epochs 30 \
      --output-dir saved/checkpoints/seg_unet_bcedice
  ```
- [ ] Train the U-Net + EfficientNet-B4 segmentation model with **Tversky** loss on the same DDR data. Same eval bands.
- [ ] Generate the soft-mask pools for both trained models over the full image corpus of all four sources: `data/processed/eyepacs/`, `data/processed/aptos/`, `data/processed/messidor2/`, `data/processed/ddr/`. Output to `data/processed/segmentation_pooled_soft/` and `data/processed/segmentation_pooled_tversky/` respectively.
- [ ] Generate the morphologically-processed soft-mask pool from the BCE+Dice soft pool. Output to `data/processed/segmentation_pooled_morph/`.
- [ ] Generate the Sobel edge-magnitude pool (no trained model required, pure image processing) for all four sources. Output to `data/processed/segmentation_pooled_sobel/`.
- [ ] **Verify coverage**: for each of the four pools, confirm that 100% of the test images of all four LODO holdouts have a corresponding auxiliary file. Specifically, `data/test/ddr/manifest.csv` (2,990 DDR images) must have aux files in all four pools.
- [ ] **Log segmentation-model results into `docs/results.md` Section 8.**

**Output directory convention:**
- Segmentation checkpoints: `saved/checkpoints/seg_unet_bcedice/best.pt`, `saved/checkpoints/seg_unet_tversky/best.pt`.
- Segmentation training logs: `saved/logs/seg_unet_bcedice/train_log.json`, `saved/logs/seg_unet_tversky/train_log.json`.
- Auxiliary pools: `data/processed/segmentation_pooled_{soft,tversky,morph,sobel}/<image_basename>.png`.

**Critical coverage check:** Run a small Python verification after each pool is generated to confirm 100% filename coverage across all four LODO test sets. If any test image is missing its auxiliary file, the corresponding aux-channel variant cannot be evaluated on that fold. Do not proceed to §1.4 until all four pools cover all four sources.

### 1.4 Step A — Unbalanced Baseline (ConvNeXt-T only, 3-Channel)

The natural baseline. ConvNeXt-Tiny 3-channel input, raw pooled class distribution (no 3:1 balancing), two-phase training schedule, validation-QWK checkpoint selection.

- [ ] Run for each of the four LODO folds. **4 runs total.**
- [ ] Select the checkpoint by validation QWK.
- [ ] Evaluate on the held-out domain's test set. Log QWK, accuracy, macro F1, per-class F1, macro AUC, and the confusion matrix.
- [ ] **Log results into `docs/results.md` Section 2.A (per-fold).**

**Output directory convention:** `saved/logs/unbalanced_3ch/<holdout>/convnext_tiny/...` and `saved/checkpoints/unbalanced_3ch/<holdout>/convnext_tiny/...`. Balancer is OFF; the train manifest is the raw pooled pre-balancing manifest.

### 1.5 Step B — Balanced Baseline (ConvNeXt-T only, 3-Channel) — THE MAIN RUN

This is the **main 3-channel run**. ConvNeXt-Tiny 3-channel input, 3:1 class-balanced train pool, two-phase training schedule, validation-QWK checkpoint selection. The per-threshold calibration analysis (§1.6) and the aux-channel ablation (§1.7) all reference this row as the reference point.

- [ ] Run for each of the four LODO folds. **4 runs total.**
- [ ] Select the checkpoint by validation QWK.
- [ ] Evaluate on the held-out domain's test set. Log QWK, accuracy, macro F1, per-class F1, macro AUC, and the confusion matrix.
- [ ] **Log results into `docs/results.md` Section 2.B (per-fold) and Section 2 (per-fold delta vs Step A).**

**Output directory convention:** `saved/logs/baseline_3ch/<holdout>/convnext_tiny/...` and `saved/checkpoints/baseline_3ch/<holdout>/convnext_tiny/...`. Balancer is ON; the train manifest is the 3:1-balanced manifest.

### 1.6 Per-Threshold Calibration (Core Diagnostic, on Step B Checkpoints)

For each of the four LODO folds, using the Step B checkpoints (the **balanced** 3-channel ConvNeXt-T run):

- [ ] On the held-out test set, compute ECE per threshold ($t = 1, 2, 3, 4$) and the aggregate ECE, with no calibration adjustment. **Log into `docs/results.md` Section 3.1.**
- [ ] Fit a temperature parameter per threshold on the validation set. Recompute ECE per threshold on the test set. **Log into `docs/results.md` Section 3.2.**
- [ ] Fit a single global temperature parameter on the validation set. Recompute ECE per threshold on the test set. **Log into `docs/results.md` Section 3.3.**
- [ ] Compute the per-fold deltas (no-scaling vs per-threshold scaling, per-threshold vs single-global). **Log into `docs/results.md` Section 3.4.**
- [ ] Generate four reliability diagrams per fold (one per threshold) and write the prose description. **Log into `docs/results.md` Section 3.5.**
- [ ] Compute the per-threshold positive-class prevalence (P(grade $\geq t$)) in each held-out source. **Log into `docs/results.md` Section 3.6.**
- [ ] Compute the per-fold max–min range across the per-threshold ECEs and the aggregate ECE. **Log into `docs/results.md` Section 3.7.**
- [ ] Run a paired bootstrap test (1000 resamples) on the per-threshold ECE deltas. **Record significance flags in Sections 3.4 and 3.7.**

**Output directory convention:** `saved/logs/calibration_3ch/<holdout>/...` and `saved/checkpoints/calibration_3ch/<holdout>/...`. The diagnostic operates on the *output* of §1.5 (it loads the selected checkpoint, fits temperatures on validation, and re-evaluates on test); it does not produce new training checkpoints.

### 1.7 Step C — Aux-Channel Ablation (ConvNeXt-T only, balanced, 4/5-channel variants)

All variants in this step inherit the **balanced** train pool from Step B. Only ConvNeXt-T is used (no backbone ablation here — that's Step G). The variants are:

| Variant | Channels | Aux |
|---------|---------:|----:|
| `four_ch_soft` | 4 | learned U-Net soft mask (BCE+Dice) |
| `four_ch_sobel` | 4 | Sobel edge map |
| `four_ch_tversky` | 4 | learned U-Net soft mask (Tversky loss) |
| `four_ch_morph` | 4 | morphological post-process of soft mask |
| `five_ch_soft_sobel` | 5 | soft + Sobel |
| `five_ch_tversky_sobel` | 5 | Tversky + Sobel |

- [ ] For each of the 6 variants × 4 LODO folds = **24 runs total.**
- [ ] Evaluate on the held-out test set. Log QWK, accuracy, macro F1, per-class F1, macro AUC.
- [ ] **Log results into `docs/results.md` Section 4:** per-variant QWK table, per-variant secondary metrics, per-class F1 grid.
- [ ] **Compute per-variant QWK delta vs Step B balanced baseline. Highlight any variant whose delta is ±1–2% — these are the candidates for seed-sensitivity confirmation in §1.10.**

**Output directory convention:** `saved/logs/<variant_name>/<holdout>/...` and `saved/checkpoints/<variant_name>/<holdout>/...`.

### 1.8 Step C+ — Per-Lesion-Type Aux-Channel Ablation (ConvNeXt-T only, balanced)

The 4-channel soft-mask variants in §1.7 collapse the four IDRiD/DDR lesion types (microaneurysms, hemorrhages, hard exudates, soft exudates) into a single "any-lesion" channel. This step disaggregates: each lesion type is produced as a separate soft-mask pool and fed as its own 4-channel variant. The motivation is the Mild-grade (grade 1) collapse documented by Poyrazer et al. (2026) — Mild is *defined* by microaneurysms, so feeding the MA channel directly may rescue Mild F1 under cross-domain shift.

- [ ] Re-train the U-Net (EfficientNet-B4) with a **multi-head output** (4 sigmoid heads, one per lesion type) on the IDRiD + DDR lesion-segmentation union. Each head is trained with BCE+Dice against its lesion's binary GT. Log per-lesion Dice on the IDRiD test set.
- [ ] Generate the four per-lesion-type soft-mask pools: `data/processed/segmentation_pooled_ma/`, `_he/`, `_ex/`, `_se/`. Each pool contains a single-channel PNG per image.
- [ ] **Verify coverage**: every RGB image in all four sources has a corresponding file in each of the four per-lesion pools.
- [ ] Add 4 new variant configs: `configs/variants/four_ch_ma.yaml`, `four_ch_he.yaml`, `four_ch_ex.yaml`, `four_ch_se.yaml`. Plus a 7-channel variant `configs/variants/seven_ch_all_lesions.yaml` (RGB + 4 lesions + Sobel).
- [ ] Run 4 per-lesion variants × 4 LODO folds = **16 runs.** Plus 1 seven-channel variant × 4 LODO folds = **4 runs.** Total **20 runs.**
- [ ] Evaluate on the held-out test set. Log QWK, per-class F1 (with Mild F1 emphasized), accuracy, macro F1, macro AUC.
- [ ] **Log results into `docs/results.md` Section 4.4 (per-lesion-type ablation).** Highlight any lesion whose channel rescued Mild F1.

**Output directory convention:** `saved/logs/four_ch_<lesion_type>/<holdout>/...` and `saved/logs/seven_ch_all_lesions/<holdout>/...`.

### 1.9 Step D — Threshold Tuning (Decision-Boundary, on Step B Checkpoints)

For each of the four LODO folds, evaluate Step B checkpoints with both default (argmax) and validation-tuned ordinal thresholds. This is a **post-hoc operation** on existing checkpoints — no new training.

- [ ] **Log results into `docs/results.md` Section 7** (QWK and per-class F1 with default vs tuned thresholds).

**Output directory convention:** `saved/logs/tuned_thresholds_3ch/<holdout>/...`.

### 1.10 Step E — Seed Sensitivity (Conditional, Only if Step C/C+ Shows ±1–2% Improvement)

**This step runs only if Step C or Step C+ identified a variant whose per-fold QWK improvement over Step B is in the 1–2% range.** The purpose is to confirm that the improvement is real and not a single-seed artifact.

- [ ] For each candidate variant identified in §1.7/§1.8, re-run the balanced ConvNeXt-T training with 2 additional seeds (3 seeds total including the original). One fold only (the most representative cross-domain fold, e.g. DDR-holdout).
- [ ] **Log results into `docs/results.md` Section 9.** If no variant showed ±1–2% improvement, this step is skipped and Section 9 is annotated "not run — no candidate variant identified."

**Output directory convention:** `saved/logs/seed_sensitivity/<variant>/<holdout>/seed_<N>/...` and `saved/checkpoints/seed_sensitivity/<variant>/<holdout>/seed_<N>/...`.

### 1.11 Step F — Single-Source Failure Mode Characterization

- [ ] For each held-out source, identify the worst-performing variant (from Steps A, B, C, C+) and describe the dominant failure mode (which classes are over/under-predicted, which image types are misclassified).
- [ ] **Log results into `docs/results.md` Section 10.**

**Output directory convention:** No new training. Read from the existing checkpoints under §1.4–§1.8.

### 1.12 Step F.5 — Summary Findings Draft

- [ ] Synthesize the table contents into 3–5 numbered findings in `docs/results.md` Section 11. Each finding must cite at least one table cell.
- [ ] Cite Poyrazer et al. (2026) and ORDER-DR (2026) where the present results converge with or diverge from their findings.

### 1.13 ⛔ PERMISSION GATE — STOP HERE ⛔

**Do not proceed past this point without explicit permission from the user.** The next step is the **headline run** (Step G), which produces the main contribution of the paper. Steps A–F exist solely to pick the configuration under which that contribution is most cleanly demonstrated.

The decision the user must make:

- **Which `(variant, backbone)` configuration** to ship in the headline run. The variant candidates are: 3-channel balanced baseline, any of the 6 aux-channel variants from Step C, or any of the 5 per-lesion variants from Step C+. The backbone is currently fixed at ConvNeXt-T (the only one we've trained); Step G re-runs the chosen variant on the other 4 backbones.
- The selection criterion is **which configuration produces the cleanest per-threshold calibration divergence finding** (methodology §5.3): a configuration where the per-threshold ECEs are jointly degraded and structurally divergent under LODO, with divergence interpretable as a function of per-threshold class distribution. Auxiliary channels that improve QWK on the balanced baseline are *candidates*, not auto-winners — the headline finding is about calibration, not QWK.

**At this gate, the agent must:**

1. Produce a one-paragraph summary of every completed table cell in `docs/results.md` Sections 2–10.
2. Propose 1–2 candidate "winning" configurations (`variant`, `backbone`) with their full QWK / per-threshold ECE / per-class F1 numbers and a one-line justification for why each is the cleanest demonstration of the per-threshold divergence hypothesis.
3. Ask the user: "Which configuration should be selected for the headline run on all 5 backbones?"

**The agent must wait for an explicit "yes" or an explicit instruction from the user before proceeding.** Do not proceed on inference.

If the user replies "do nothing further" or "stop here," the run is complete. Log a one-line summary at the bottom of `docs/results.md` Section 11 stating that no headline run was performed, and exit.

### 1.14 Step G — Headline Run (Only After 1.13 Is Approved)

**The headline run is where the main contribution of the paper is produced.** Steps A–F are selection/characterization: they identify which `(variant, backbone)` configuration will most cleanly demonstrate the per-threshold calibration divergence under LODO. The headline run *executes* that demonstration.

The headline run produces, for the winning configuration:
- The per-threshold ECE table (the core diagnostic, methodology §5.3) on the held-out test set, with per-threshold and single-global temperature scaling, for every (backbone × holdout) cell.
- The per-fold QWK with 95% bootstrap CI (the primary metric, methodology §5.1).
- The per-class F1 grid, with the Mild (grade 1) F1 highlighted for comparison to Poyrazer et al. (2026).
- The reliability diagrams per (threshold, fold) (methodology §5.3).

- [ ] Run the winning configuration (variant) on the other 4 backbones × 4 LODO folds = **16 runs.** (The ConvNeXt-T row of the winning variant is already trained in Step C; it does not need to be re-run unless the winning config differs from what was already produced.)
- [ ] If the winning config is the 3-channel balanced baseline (no aux channel), run it on the other 4 backbones × 4 LODO folds = **16 runs.** The ConvNeXt-T balanced baseline is already in Step B.
- [ ] For every (backbone × holdout) cell, run the full per-threshold calibration analysis from §1.6 (no-scaling, per-threshold temperature, single-global temperature) on the held-out test set. Log to `docs/results.md` Section 3.
- [ ] Log all training artifacts to `saved/logs/headline_run/<backbone>/<holdout>/...` and `saved/checkpoints/headline_run/<backbone>/<holdout>/...`. Log all calibration artifacts (fitted temperatures, per-bin counts) under the same directory.
- [ ] Update `docs/results.md` Section 5 with the full 5-backbone × 4-LODO-fold QWK matrix.
- [ ] Update `docs/results.md` Section 11 with the headline numbers: per-threshold ECE across backbones, the per-class F1 grid (Mild F1 highlighted), the structural-divergence finding (positive or negative).
- [ ] Produce the final one-line summary at the end of the run, answering: "Are the per-threshold calibration errors of a CORN ordinal regression head jointly degraded and structurally divergent under true leave-one-domain-out generalization across multiple DR domains, and is the divergence interpretable as a function of per-threshold class distribution?"

### 1.15 Step H — Cross-Backbone Ensemble Phase (Additive, Eval-Only)

The ensemble phase runs **after** Step G, on the 5 backbone checkpoints already produced. It does not replace any per-model analysis — it adds an additional layer on top, addressing three sub-questions:

1. **Robustness**: does the per-threshold ECE divergence structure (the headline finding from Step G) survive ensemble averaging across the 5 backbones?
2. **Architectural-disagreement axis**: for each threshold, how much do the 5 backbones disagree? Is disagreement structured by per-threshold class distribution (more disagreement at the Mild threshold, less at the PDR threshold)?
3. **Per-threshold recovery pattern**: does ensembling preferentially recover the easy thresholds (3, 4) and under-recover the hard one (1)?

The ensemble is an **equal-weight average of per-threshold logits** across the 5 backbones' checkpoints, evaluated on the same held-out test set as Step G. No new training is performed.

- [ ] For each of the 4 LODO folds, load the 5 backbone checkpoints produced in Step G. Average the per-threshold logits (equal weight). Apply softmax/sigmoid to get ensemble per-threshold probabilities. **4 ensemble eval runs total (one per fold).**
- [ ] For each fold, compute the per-threshold ECE on the ensemble's per-threshold predictions. Add a row to `docs/results.md` Section 3: "Ensemble (5-backbone equal-weight) per-threshold ECE by fold."
- [ ] Compute the per-fold QWK with 95% bootstrap CI for the ensemble. Add a row to Section 5.
- [ ] Compute the per-class F1 grid for the ensemble. Add a row to Section 5.
- [ ] **Per-backbone disagreement**: for each fold and each threshold, compute the standard deviation of the 5 backbones' per-threshold probabilities (on the test set). Log to `docs/results.md` Section 4.5 (new sub-section of Section 4): "Cross-backbone per-threshold probability disagreement by fold."
- [ ] **Per-threshold recovery pattern**: compute the ensemble's per-threshold ECE reduction vs the per-model mean per-threshold ECE for each fold. Specifically: ΔECE_per_threshold = ECE_per_model_mean − ECE_ensemble. If ΔECE is non-uniform across thresholds (e.g. larger at threshold 4, smaller at threshold 1), the per-threshold recovery structure is itself interpretable as a function of per-threshold class distribution — a secondary contribution.
- [ ] Update `docs/results.md` Section 11 with the ensemble sub-findings (3 sub-bullets under the headline finding).

**Output directory convention:** `saved/logs/ensemble_eval/<holdout>/...` for the per-fold ensemble diagnostics. **No new checkpoints.** All inputs are the Step G checkpoints.

**Why this is additive, not replacement:** the per-model per-threshold ECE table from Step G is the unit of analysis for the headline finding. The ensemble's per-threshold ECE is an *additional* data point that tests whether the structure survives variance reduction across architectures. The Step G per-model findings remain the primary contribution regardless of the ensemble result.

## 2. Logging Conventions

- **Every number that goes into `docs/results.md` must be backed by a file on disk** (CSV row, JSON entry, or a checkpoint whose load-and-eval reproduces the number). State the file path next to the number the first time you write it.
- **Tables take mean ± SD across the four LODO folds by default.** Per-fold values are reported in the per-fold row; the per-fold row is the primary cell, the Mean ± SD row is the summary.
- **Confidence intervals on QWK** are 95% bootstrap (1000 resamples) over the held-out domain's test set.
- **Statistical significance** is a paired bootstrap test on QWK across folds, 1000 resamples, against the 3-channel baseline. Significance threshold is pre-registered; do not adjust it post-hoc.
- **All artifacts (CSV, JSON, checkpoints) live under `saved/`** with a per-experiment subfolder so runs never overwrite each other.
- **Before any new training run, verify the output directory is empty or uniquely named.** If the targeted output directory is non-empty, stop and ask the user.

## 3. Forbidden Actions

- Do not write a number into `docs/results.md` that is not backed by a file on disk.
- Do not run an experiment using a configuration whose output directory is already populated by an earlier run.
- Do not cite a result from a previous lab notebook or chat memory. The previous numbers are not in `docs/results.md`; they are not results.
- Do not modify `docs/methodology.md` after the experiments are finalized. Methodology is fixed.
- Do not collapse two LODO folds into one number. The cross-domain number is the mean across the four folds, reported alongside the per-fold values.
- Do not report a single aggregate ECE as the calibration finding. Per-threshold ECE is the unit of analysis.
- Do not proceed past §1.17 without explicit user permission. The headline-run decision is the user's, not the agent's.

## 4. Deliverables Checklist

- [ ] `docs/methodology.md` — final, immutable.
- [ ] `docs/results.md` — every table populated, Section 11 written, headline-run results appended if §1.18 was approved.
- [ ] `docs/todo.md` — every checkbox ticked, or explicitly annotated "not run" with a reason.
- [ ] `saved/logs/<experiment_name>/...` — every experiment in its own subdirectory, complete with per-experiment logs.
- [ ] `saved/checkpoints/<experiment_name>/...` — every experiment in its own subdirectory, complete with checkpoints.
- [ ] One-line summary at the end of the run, answering: "Are the per-threshold calibration errors of a CORN ordinal regression head jointly degraded and structurally divergent under true leave-one-domain-out generalization across multiple DR domains, and is the divergence interpretable as a function of per-threshold class distribution?"