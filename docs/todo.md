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

**Rule 4 — All previously generated logs and checkpoints have been deleted from disk.** The `saved/logs/` and `saved/checkpoints/` directories were wiped before this runbook was written. Nothing previously stored is recoverable. The only authoritative numerical record of the experiments is what you produce now, logged into `saved/` *and* transcribed into `docs/results.md`.

**Rule 5 — Stop and request permission before any decision that picks a winner.** Section 1.16 below is the permission gate. Do not proceed past it without an explicit "yes" from the user.

## 1. Sequence of Work

The experiments are ordered by dependency. Do not skip steps. Do not run steps out of order. The dependency graph is documented inline.

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

- [ ] Train the U-Net + EfficientNet-B4 segmentation model with BCE + Dice loss. Log Dice / IoU / Tversky on its own evaluation split.
- [ ] Train the U-Net + EfficientNet-B4 segmentation model with Tversky loss. Log Dice / IoU / Tversky on its own evaluation split.
- [ ] Generate the soft-mask pools for both models over the full image corpus.
- [ ] Generate the morphologically-processed soft-mask pool.
- [ ] Generate the Sobel edge-magnitude pool.
- [ ] **Log results into `docs/results.md` Section 8.**

**Output directory convention:** `saved/logs/seg_unet_bcedice/`, `saved/logs/seg_unet_tversky/`, `saved/checkpoints/seg_unet_bcedice/`, `saved/checkpoints/seg_unet_tversky/`.

### 1.4 3-Channel Baseline (All Folds, All Backbones)

For each of the four LODO folds, for each of the five backbones:

- [ ] Run the two-phase training schedule (frozen backbone, then full fine-tune).
- [ ] Select the checkpoint by validation QWK.
- [ ] Evaluate on the held-out domain's test set. Log QWK, accuracy, macro F1, per-class F1, macro AUC, and the confusion matrix.
- [ ] **Log results into `docs/results.md` Section 2 (per-fold) and Section 5 (per-backbone).**

**Output directory convention:** `saved/logs/baseline_3ch/<holdout>/<arch>/...` and `saved/checkpoints/baseline_3ch/<holdout>/<arch>/...`. This is 20 cells (4 folds × 5 backbones), each in its own subdirectory. Do not let any two cells share a subdirectory.

### 1.5 Per-Threshold Calibration (Core Diagnostic, 3-Channel Baseline)

For each of the four LODO folds, using the best backbone per fold:

- [ ] On the held-out test set, compute ECE per threshold ($t = 1, 2, 3, 4$) and the aggregate ECE, with no calibration adjustment. **Log into `docs/results.md` Section 3.1.**
- [ ] Fit a temperature parameter per threshold on the validation set. Recompute ECE per threshold on the test set. **Log into `docs/results.md` Section 3.2.**
- [ ] Fit a single global temperature parameter on the validation set. Recompute ECE per threshold on the test set. **Log into `docs/results.md` Section 3.3.**
- [ ] Compute the per-fold deltas (no-scaling vs per-threshold scaling, per-threshold vs single-global). **Log into `docs/results.md` Section 3.4.**
- [ ] Generate four reliability diagrams per fold (one per threshold) and write the prose description. **Log into `docs/results.md` Section 3.5.**
- [ ] Compute the per-threshold positive-class prevalence (P(grade $\geq t$)) in each held-out source. **Log into `docs/results.md` Section 3.6.**
- [ ] Compute the per-fold max–min range across the per-threshold ECEs and the aggregate ECE. **Log into `docs/results.md` Section 3.7.**
- [ ] Run a paired bootstrap test (1000 resamples) on the per-threshold ECE deltas. **Record significance flags in Sections 3.4 and 3.7.**

**Output directory convention:** `saved/logs/calibration_3ch/<holdout>/...` and `saved/checkpoints/calibration_3ch/<holdout>/...`. The diagnostic operates on the *output* of §1.4 (it loads the selected checkpoint, fits temperatures on validation, and re-evaluates on test); it does not produce new training checkpoints.

### 1.6 4-Channel Soft Mask (All Folds, Best Backbone)

- [ ] For each of the four LODO folds, run the 4-channel variant with the soft-mask (BCE+Dice) auxiliary channel.
- [ ] Evaluate on the held-out domain's test set. Log QWK, accuracy, macro F1, per-class F1, macro AUC.
- [ ] **Log results into `docs/results.md` Section 4.**

**Output directory convention:** `saved/logs/four_ch_soft/<holdout>/...` and `saved/checkpoints/four_ch_soft/<holdout>/...`.

### 1.7 4-Channel Sobel (All Folds, Best Backbone)

- [ ] For each of the four LODO folds, run the 4-channel variant with the Sobel edge auxiliary channel.
- [ ] Evaluate on the held-out domain's test set. Log QWK, accuracy, macro F1, per-class F1, macro AUC.
- [ ] **Log results into `docs/results.md` Section 4.**

**Output directory convention:** `saved/logs/four_ch_sobel/<holdout>/...` and `saved/checkpoints/four_ch_sobel/<holdout>/...`.

### 1.8 4-Channel Tversky Soft Mask (All Folds, Best Backbone)

- [ ] Run for each of the four LODO folds.
- [ ] **Log results into `docs/results.md` Section 4.**

**Output directory convention:** `saved/logs/four_ch_tversky/<holdout>/...` and `saved/checkpoints/four_ch_tversky/<holdout>/...`.

### 1.9 4-Channel Morphological Soft Mask (All Folds, Best Backbone)

- [ ] Run for each of the four LODO folds.
- [ ] **Log results into `docs/results.md` Section 4.**

**Output directory convention:** `saved/logs/four_ch_morph/<holdout>/...` and `saved/checkpoints/four_ch_morph/<holdout>/...`.

### 1.10 5-Channel Soft + Sobel (All Folds, Best Backbone)

- [ ] Run for each of the four LODO folds.
- [ ] **Log results into `docs/results.md` Section 4.**

**Output directory convention:** `saved/logs/five_ch_soft_sobel/<holdout>/...` and `saved/checkpoints/five_ch_soft_sobel/<holdout>/...`.

### 1.11 5-Channel Tversky + Sobel (All Folds, Best Backbone)

- [ ] Run for each of the four LODO folds.
- [ ] **Log results into `docs/results.md` Section 4.**

**Output directory convention:** `saved/logs/five_ch_tversky_sobel/<holdout>/...` and `saved/checkpoints/five_ch_tversky_sobel/<holdout>/...`.

### 1.12 Class-Balancing Ablation

- [ ] For each of the four LODO folds, re-run the 3-channel baseline **without** the 3:1 class-balancing step (raw pooled class distribution).
- [ ] **Log results into `docs/results.md` Section 6** (QWK per fold and per-class F1).

**Output directory convention:** `saved/logs/unbalanced_3ch/<holdout>/...` and `saved/checkpoints/unbalanced_3ch/<holdout>/...`.

### 1.13 Threshold Tuning (Decision-Boundary)

- [ ] For each of the four LODO folds, evaluate the 3-channel baseline with both default (argmax) and validation-tuned thresholds.
- [ ] **Log results into `docs/results.md` Section 7.**

**Output directory convention:** `saved/logs/tuned_thresholds_3ch/<holdout>/...`. This step operates on the *output* of §1.4 and does not produce new training checkpoints.

### 1.14 Seed Sensitivity

- [ ] Re-run the 3-channel baseline, 4ch Sobel, and 5ch Tversky+Sobel across the pre-registered seed set (3 seeds minimum), on the source that is most representative of cross-domain performance.
- [ ] **Log results into `docs/results.md` Section 9.**

**Output directory convention:** `saved/logs/seed_sensitivity/<variant>/<holdout>/seed_<N>/...` and `saved/checkpoints/seed_sensitivity/<variant>/<holdout>/seed_<N>/...`. Three seeds × three variants × one holdout = nine cells, each in its own subdirectory.

### 1.15 Single-Source Failure Mode Characterization

- [ ] For each held-out source, identify the worst-performing variant and describe the dominant failure mode (which classes are over/under-predicted, which image types are misclassified).
- [ ] **Log results into `docs/results.md` Section 10.**

**Output directory convention:** No new training. Read from the existing checkpoints under §1.4–§1.11.

### 1.16 Summary Findings Draft

- [ ] Synthesize the table contents into 3–5 numbered findings in `docs/results.md` Section 11. Each finding must cite at least one table cell.
- [ ] Cite Poyrazer et al. (2026) and ORDER-DR (2026) where the present results converge with or diverge from their findings.

### 1.17 ⛔ PERMISSION GATE — STOP HERE ⛔

**Do not proceed past this point without explicit permission from the user.** The next step would be:

- Selecting a winning model variant (a specific `(arch, channel-variant, holdout)` triple).
- Selecting a winning training configuration (hyperparameters, augmentation, balancing).
- Re-running that configuration under a more thorough protocol (more seeds, more epochs, different seeds).

These decisions have academic-integrity implications: they define what is reported as the "headline result" of the work. They are not the agent's to make. They are not pre-determined by the methodology.

**At this gate, the agent must:**

1. Produce a one-paragraph summary of every completed table cell in `docs/results.md` Sections 2–10.
2. Propose 2–3 candidate "winning" configurations with their full QWK / per-threshold ECE / per-class F1 numbers.
3. Ask the user: "Which configuration (if any) should be selected for the headline run, and which seeds should be used for the headline run? Do you want any additional comparisons before the headline run?"

**The agent must wait for an explicit "yes" or an explicit instruction from the user before proceeding.** Do not proceed on inference. Do not proceed on defaults. Do not proceed because the chat session is long and the user might forget. Wait.

If the user replies "do nothing further" or "stop here," the run is complete. Log a one-line summary at the bottom of `docs/results.md` Section 11 stating that no headline run was performed, and exit.

### 1.18 Headline Run (Only After 1.17 Is Approved)

- [ ] Run the headline configuration chosen in §1.17, with the seeds chosen in §1.17.
- [ ] Log all artifacts to `saved/logs/headline_run/...` and `saved/checkpoints/headline_run/...`.
- [ ] Update `docs/results.md` Section 11 with the headline numbers.
- [ ] Produce the final one-line summary at the end of the run.

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