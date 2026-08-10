# Methodology

## 1. Problem Statement

Diabetic retinopathy (DR) is a leading cause of preventable blindness worldwide. Automated grading of fundus photographs into severity levels is a clinically meaningful screening task, but conventional training pipelines typically assume that the training and deployment populations are drawn from the same data distribution. In practice, fundus images are acquired from heterogeneous devices, patient populations, and clinical sites, which induces substantial **domain shift** between training and deployment cohorts. Models that achieve strong in-domain accuracy frequently degrade catastrophically when applied to out-of-domain imagery.

A growing body of recent work has established that domain shift matters specifically for DR grading: the GDRNet / GDRBench line of work (Che et al., 2023) introduced an eight-dataset public benchmark with explicit leave-one-domain-out and extreme single-domain-generalization protocols and showed that standard training recipes suffer large accuracy and AUC gaps under cross-domain evaluation. The DRGen work (Atwany & Yaqub, 2022) framed the cross-domain gap as a generalization problem and proposed training-time interventions to seek flatter minima. The DECO line of work (Xia et al., 2024) addressed the same gap through representation disentanglement of retinal and domain-specific latents. These contributions establish that **closing the cross-domain accuracy gap** is an active research line; methods to achieve it now exist (FundusAug, hybrid losses, domain-class-aware rebalancing, representation disentanglement, augmentation-based domain generalization).

This work does not propose a new method to close the cross-domain accuracy gap. A separate, less-studied question is whether the **probabilities** that these models produce are trustworthy under domain shift — that is, whether the predicted confidences reflect the true posterior probability of the predicted grade. Recently, El Bellaj et al. (2026) proposed an evidential ordinal regression head and reported that uncertainty under domain shift is miscalibrated, and Sheng, Dong, Wu & Liang (2026, "ORDER-DR") reported that scalar expected-calibration error on a binarized referable-DR endpoint nearly quadruples from in-distribution (0.049) to external validation (0.160) on the same disease. These are the closest existing signals that calibration degrades under cross-dataset shift in DR grading.

A converging line of evidence comes from Poyrazer, Yağcı & Erten (2026), who benchmarked three pretrained vision backbones as frozen feature extractors on APTOS development and Messidor-2 external validation and reported that **temperature scaling — the standard post-hoc calibration remedy — fails under cross-dataset shift** in DR grading: ECE was reported below 0.022 on the development set but in the range 0.086–0.149 on the external set after re-scaling, i.e. the recalibration step did not transfer. The same study independently reported that the Mild (grade 1) class is catastrophically vulnerable under shift, with F1 = 0.000 for two of three encoders and F1 = 0.153 for the best. This is the strongest single piece of prior evidence that *aggregate* calibration tools are inadequate under DR cross-dataset shift, and that the *earliest clinically meaningful severity grade* is disproportionately affected.

The present work addresses a finer-grained version of the calibration question. The CORN-style ordinal regression head (Cao, Mirjalili & Raschka, 2020) decomposes a $K$-class ordinal problem into $K-1$ independent binary sub-problems, one per ordinal threshold. Recent calibration reports in the DR literature — including those in the papers cited above — report a single aggregate expected-calibration error on the final categorical prediction, or a single scalar on a binarized referable endpoint. The aggregate hides the question: **each of the CORN sub-problems corresponds to a different class-imbalance regime, and their calibration may diverge under domain shift.** This work asks whether the per-threshold calibration of a CORN ordinal regression head is jointly degraded and structurally divergent under true leave-one-domain-out generalization across multiple DR domains, and whether that divergence is interpretable in terms of the per-threshold class distribution. The Mild (grade 1) threshold is the analytically prior target because the prior literature independently flags it under both ordinal and non-ordinal mechanisms.

## 2. Task Definition

### 2.1 Ordinal Grading

The task is formulated as **5-class ordinal classification** of fundus photographs into the international clinical DR severity scale:

| Grade | Label | Clinical description |
|------:|:-----:|:---------------------|
| 0 | No DR | No abnormalities |
| 1 | Mild NPDR | Microaneurysms only |
| 2 | Moderate NPDR | More than microaneurysms but less than severe |
| 3 | Severe NPDR | Extensive intraretinal hemorrhages |
| 4 | Proliferative DR | Neovascularization or vitreous hemorrhage |

The ordinal nature of the labels is exploited through the choice of loss function (see §4.4).

### 2.2 Out-of-Domain Evaluation Protocol

Each model is trained on a union of $N-1$ domains and evaluated on the held-out $N$-th domain. The training sources are pooled and class-balanced, then partitioned into a training set and a stratified validation set. The held-out domain contributes only to the test set. Reported metrics are computed exclusively on the held-out domain's test partition, which the model never observed in either training or model selection.

This protocol is a deliberate subset of the larger GDRBench benchmark (Che et al., 2023), restricted to the four fundus domains used here. The choice of four domains is a practical throughput constraint, not a methodological one; the same analysis extends to the full eight-dataset pool as an explicit follow-up.

The LODO protocol is **strictly stronger** than the per-domain test protocol used by DRGen (Atwany & Yaqub, 2022) and the single-pooled-split protocol used by El Bellaj et al. (2026), both of which allow training-time exposure to samples from the same population as the test set. Strict LODO is the only protocol that rules out in-distribution leakage into the test partition.

## 3. Data

### 3.1 Sources

Four publicly available fundus photograph datasets are used as the source domains. They differ in:

- **Acquisition device** (different camera manufacturers and models)
- **Resolution** (ranging from approximately 640×480 to over 3000×3000 pixels)
- **Patient population** (geographic, demographic, and referral-pattern heterogeneity)
- **Native grading scheme** (two datasets use the 5-class scale directly; one uses a coarser scheme that is mapped to the 5-class scale; one uses a 6-class scheme that includes an "ungradable" category which is filtered out to maintain 5-class consistency)

The four-source pool is a strict subset of GDRBench's eight-source pool; the four sources used here are the ones with the largest public corpora and the clearest 5-class or 5-class-mappable grading scheme. The remaining four GDRBench sources are not used in this study; their inclusion is a follow-up.

### 3.2 Class Balancing

The four sources exhibit substantially different class distributions. A naive pool would be dominated by the source whose No-DR class is largest (typically an order of magnitude more samples than the rarest class), biasing the model toward the majority class and away from the minority classes that are clinically most important to detect.

To control for this, training-pool class counts are **balanced to a 3:1 ratio** (most populous class subsampled to at most three times the rarest class), with downsampling only — no upsampling, no synthetic augmentation beyond what is described in §4.2. The 3:1 ratio is chosen over a stricter 1:1 ratio to retain a usable training set size while still materially reducing majority-class dominance.

The 3:1 ratio is itself a hyperparameter. The contribution of the balancing step is reported separately as an ablation (§7.4).

### 3.3 Grading-Scheme Reconciliation

Where the native grading scheme differs from the 5-class target, the following reconciliations are applied:

- Coarser schemes are mapped to the 5-class scale using the dataset's published correspondence table.
- Finer schemes that include an "ungradable" label are processed by **excluding** ungradable samples entirely. No image is relabeled to compensate; the ungradable class is treated as a quality-control filter rather than a sixth class.

After reconciliation, every image in the combined corpus carries a label in $\{0, 1, 2, 3, 4\}$.

### 3.4 Train / Validation / Test Split

For each LODO fold:

- **Training set:** all sources except the held-out one, post-class-balancing, stratified by label.
- **Validation set:** a stratified subsample (10%) of the same pool, drawn from the same sources as training, used for model selection (early stopping, learning-rate scheduling, threshold tuning).
- **Test set:** the held-out source's entire labeled set, optionally subsampled to a matched-size subset per source for cross-fold comparability.

The held-out source is never observed during training or model selection.

## 4. Model Architecture and Training

### 4.1 Backbone

Five convolutional and transformer backbones are evaluated as the feature extractor:

- ConvNeXt-Tiny
- ConvNeXt-Small
- DeiT-3 Small (384-resolution variant)
- MaxViT Tiny (384-resolution variant)
- SwinV2-Base (192→384 transfer)

All backbones are initialized from publicly published ImageNet-pretrained weights and adapted to the 5-class ordinal target via a domain-specific head described in §4.3.

A convolutional block attention module (CBAM) is appended after the backbone feature map to recalibrate channel-wise and spatial feature responses. The number of CBAM stages is a hyperparameter; the default is two.

The backbone family is intentionally wide to test whether the per-threshold calibration findings of §5.3 are an architectural artifact or a property of the task. Recent work (PRISM-DR, 2026) uses a CORAL ordinal head; the methodological decision to use CORN rather than CORAL is documented in §4.3.

### 4.2 Input Channels

The model accepts $C$-channel inputs, where $C \in \{3, 4, 5\}$ depending on the variant:

- **3-channel:** native RGB.
- **4-channel:** RGB plus a single auxiliary channel. The auxiliary channel is one of:
  - A lesion-segmentation soft mask (probability-valued output of a separately trained U-Net).
  - A Sobel edge-gradient magnitude map (image-derived, no training required).
- **5-channel:** RGB plus two auxiliary channels (a soft mask plus an edge map).

The auxiliary channels are pre-computed offline and stored alongside the preprocessed RGB images. They are loaded at training time and concatenated to the RGB channels at the dataset object level, before normalization.

**Coverage requirement:** every auxiliary pool must provide an auxiliary file for every RGB image in all four source domains — including the DDR test images. The DDR holdout fold requires its 2,990 test images to have soft-mask, Tversky-mask, morph-mask, and Sobel-edge auxiliary files. Coverage is verified explicitly before any 4-channel or 5-channel LODO fold is started; a missing file would render that fold unevaluable on the auxiliary-channel variants.

The auxiliary-channel question is a methodological supplement to the calibration question. The primary contribution concerns the 3-channel head; the auxiliary-channel variants are motivated by the prior literature on multi-channel lesion-aware DR grading (Lesion-Aware Ordinal Transformer, 2026) and are reported as a secondary axis.

### 4.3 Ordinal Regression Head

The head treats the $K$-class ordinal problem as $K-1$ binary cumulative-threshold predictions: for each threshold $t \in \{1, \dots, K-1\}$, the head emits a probability that the true grade is at least $t$. The final predicted grade is derived from these cumulative probabilities either by argmax of the implied categorical distribution or by tuning the thresholds on the validation set.

This formulation has two empirical motivations that are standard in the ordinal-regression literature: (i) it encodes the ordinal structure as an inductive bias rather than treating the labels as nominal, and (ii) it produces well-calibrated probabilities that enable downstream threshold tuning without retraining.

The formulation used here is the **CORN** (COnditional ORdered Normalization) decomposition (Cao, Mirjalili & Raschka, 2020), which defines the head as $K-1$ **independent binary sub-problems**, one per ordinal threshold. The independence of the sub-problems is the structural property that the per-threshold calibration analysis (§5.3) leverages.

A dropout layer precedes the head. The head's hidden dimension is a hyperparameter; the default is 512.

The decision to use CORN rather than the related CORAL ordinal head is deliberate: CORN's independent-binary decomposition admits a per-threshold calibration analysis that is not available under CORAL's rank-consistent joint-decoder formulation. The two are not interchangeable for the per-threshold calibration question this work asks.

### 4.4 Loss Function

Two loss families are considered:

- **CORN loss** (COnditional ORdered Normalization): a loss designed for the cumulative-threshold head formulation that respects the ordinal label structure and is empirically reported to outperform cross-entropy on ordinal tasks.
- **Cross-entropy** as a baseline comparator.

When class counts are imbalanced, **class-weighted** variants are used, with weights proportional to the inverse class frequency in the training pool.

### 4.5 Two-Phase Training Schedule

Training proceeds in two phases:

**Phase 1 — Frozen backbone.** The backbone weights are frozen and only the head, the CBAM module, and the input-channel adapter (for $C > 3$) are trained. This stabilizes the head before the backbone is allowed to drift and yields a useful validation signal for early hyperparameter rejection. Phase 1 runs for a small number of epochs (default 5) with a high head learning rate (default $10^{-3}$) and a short warmup.

**Phase 2 — Full fine-tuning.** The backbone is unfrozen and the full model is trained end-to-end. Phase 2 runs for a longer horizon (default 35 epochs) with a substantially smaller learning rate (default $1.2 \times 10^{-4}$ for the head, $1.2 \times 10^{-5}$ for the backbone), a cosine schedule, and a minimum-epoch floor before early stopping is permitted. Overfitting is monitored by a patience-based early-stopping rule on validation Quadratic Weighted Kappa (QWK).

Batch size, learning rates, weight decay, warmup epochs, and augmentation strength are backbone-dependent hyperparameters reflecting the differing memory footprints and learning-rate sensitivities of the candidate architectures.

### 4.6 Regularization

- **Exponential Moving Average (EMA):** an EMA copy of the model parameters is maintained with decay factor 0.999 and used for evaluation. This is a standard regularizer that improves test-time stability at negligible cost.
- **MixUp:** with a configurable probability, pairs of training samples and their labels are linearly interpolated. MixUp is applied at the input level (and propagated to the auxiliary channels) and is disabled in the last portion of training to avoid corrupting late-stage fine-tuning.
- **Augmentation:** geometric and photometric augmentations are applied at training time. The augmentation strength is a hyperparameter (default "light") that controls the magnitudes.
- **Sqrt sampling:** optionally, a per-class sampling probability proportional to $1/\sqrt{n_c}$ (rather than $1/n_c$) is used to balance class exposure during training. This is gentler than uniform class-balanced sampling and is the default in Phase 2.

### 4.7 Model Selection

The best checkpoint per fold is selected by **validation QWK**, not by validation accuracy. QWK is the standard metric for ordered clinical grading and is more sensitive to clinically meaningful errors than accuracy or macro-F1.

## 5. Evaluation Metrics

### 5.1 Primary Metric

The primary metric is **Quadratic Weighted Kappa (QWK)** on the held-out domain's test set, computed with the clinically standard integer-grading prediction. QWK is reported with confidence intervals obtained via bootstrap over the test set.

### 5.2 Secondary Metrics

- **Accuracy** (argmax predicted grade).
- **Macro precision, recall, F1** (unweighted mean across the 5 classes).
- **Per-class F1** (one row per clinical grade).
- **Macro AUC-ROC** (one-vs-rest, averaged across classes).
- **Confusion matrix** at the integer-grading prediction.

### 5.3 Per-Threshold Calibration (Core Diagnostic)

This is the central contribution of the work. The CORN head produces $K-1 = 4$ independent probability outputs, one per ordinal threshold $t \in \{1, 2, 3, 4\}$. Each of these outputs is a prediction of the binary event "true grade $\geq t$." The calibration of each threshold is a separate question. The diagnostic procedure is:

- For each threshold $t$, compute the **Expected Calibration Error (ECE)** of the binary "grade $\geq t$" prediction on the held-out domain's test set, with 10 equal-width bins. This is the per-threshold ECE.
- Fit **temperature scaling** on the validation set, separately per threshold, and re-evaluate ECE on the test set. Compare per-threshold ECE before and after temperature scaling.
- For comparison, also fit a **single global temperature scaling** parameter across all four thresholds and compare.
- Report the per-threshold ECE in tabular form (one row per threshold, one column per LODO fold) and characterize whether the per-threshold calibration errors diverge across folds as a function of the per-threshold class distribution in the held-out source.

The aggregate ECE on the categorical prediction is reported only as a summary, not as the primary metric. The per-threshold ECE is the unit of analysis.

The distinction between *per-threshold calibration* and *ordinal threshold tuning* is deliberate and important. The latter (e.g., as in ORDER-DR, 2026) places decision boundaries to maximize QWK on the expected-grade score; it does not measure whether the predicted probabilities at each threshold are themselves trustworthy. The two procedures are complementary and answer different questions.

The per-threshold temperature scaling analysis is explicitly motivated by the Poyrazer et al. (2026) finding that global temperature scaling fails under DR cross-dataset shift. The question is whether *per-threshold* temperature scaling recovers what the global procedure loses, and whether the recovery is uniform across thresholds or worse at the Mild (grade 1) threshold specifically.

### 5.4 Statistical Significance

- **Per-fold QWK differences** between variants are reported with a paired bootstrap test on the held-out domain's test set (1000 resamples).
- **Across-fold QWK differences** are reported as mean ± standard deviation across the $N$ folds and assessed with a one-sample $t$-test of the per-fold deltas against zero.
- **Per-fold per-threshold ECE differences** are reported with a paired bootstrap test on the held-out domain's test set (1000 resamples) per threshold.
- A pre-registered significance threshold is applied uniformly and is not adjusted post-hoc.

## 6. Lesion-Segmentation Auxiliary Model

The soft-mask auxiliary channel used by the 4-channel and 5-channel variants is produced by a separately trained segmentation model:

- **Architecture:** U-Net with an **EfficientNet-B4 encoder** (the same `build_unet_vessel` backbone used elsewhere in the segmentation module). The encoder is initialized from ImageNet-pretrained weights and adapted to single-channel output. No group normalization is used; standard batch norm is sufficient at the batch sizes used here. B4 is chosen over B0 for the larger receptive field and richer feature capacity; the cost is a ~4× parameter count increase and a ~2× inference-time cost per image, both of which are acceptable for an offline, once-per-source mask-generation pass.
- **Training data (primary):** **DDR lesion segmentation** subset at `data/raw/DDR-dataset/lesion_segmentation/`. The dataset provides 383 training images, 149 validation images, and 225 test images with pixel-level ground truth for the same four lesion types as IDRiD: microaneurysms (MA), hemorrhages (HE), hard exudates (EX), and soft exudates (SE). The four per-lesion masks are unioned into a single binary "any-lesion" mask for training and evaluation. DDR is the primary training source because it provides ~7× more images than IDRiD with comparable annotation density, and because it is the only source on disk that produces masks for the DDR-holdout fold's test images (so the segmentation model is trained on the same domain whose downstream grading pool it must annotate).
- **Training data (auxiliary, optional):** **IDRiD** lesion-segmentation subset at `data/raw/A. Segmentation/` (54 training + 27 test images). IDRiD is held out of the training pool by default and used only as the cross-domain evaluation set (the methodology references its gold-standard per-lesion GT). Adding IDRiD's 54 training images via `--use-idrid-train` is permitted but not the default; it provides only marginal additional diversity given the size.
- **Loss family:** either binary cross-entropy plus Dice or Tversky loss (the latter is more robust to class imbalance on small lesions). Both are trained and reported separately.
- **Output:** a single-channel probability-valued soft mask, thresholded post-training only for visualization, not for use as the auxiliary channel.

**The segmentation model is trained once per loss family and reused across all four DR-grading source domains** (eyepacs, aptos, messidor2, ddr). At inference time, the trained U-Net is applied to every image in the four DR-grading source corpora to produce soft masks that are stored as auxiliary files matching the RGB image basenames. This covers all four LODO folds, including the DDR-holdout fold whose 2,990 test images would otherwise have no auxiliary channel. Coverage is verified per pool before any 4-channel or 5-channel grading training run begins.

The segmentation model's training-time metrics (Dice, IoU, Tversky index on the IDRiD test split — cross-domain — and the DDR test split — in-domain) are reported separately from the grading metrics in `docs/results.md` Section 8.

### 6.1 Edge-Gradient Auxiliary Channel

The Sobel edge-gradient magnitude is computed directly from the grayscale image at preprocessing time using a fixed $3\times 3$ Sobel kernel. No learned parameters are involved. It is the cheapest auxiliary channel and serves as a control for the more expensive learned segmentation mask.

## 7. Experimental Design

### 7.1 Comparisons

The following variants are compared:

| Variant | Channels | Auxiliary | Purpose |
|--------:|:--------:|:---------:|:--------|
| Baseline (3-channel) | 3 | none | Reference performance |
| 4-channel soft mask | 4 | learned U-Net soft mask | Does learned segmentation help? |
| 4-channel Sobel | 4 | Sobel edge map | Does a parameter-free edge map help? |
| 4-channel Tversky | 4 | Tversky-loss soft mask | Does loss choice at the segmentation stage matter? |
| 4-channel morphological | 4 | morphologically processed soft mask | Does post-processing of the soft mask help? |
| 5-channel (soft + Sobel) | 5 | soft + Sobel | Does channel combination help? |
| 5-channel (Tversky + Sobel) | 5 | Tversky + Sobel | Does the channel combination matter? |

The 3-channel row is the primary arm of the work. The 4-channel and 5-channel rows are methodological supplements motivated by the prior literature on multi-channel lesion-aware DR grading; the calibration analysis is reported on the 3-channel row.

### 7.2 LODO Folds

Four LODO folds are run, one per held-out source. Each cell of the (variant × fold) matrix is a single training run.

### 7.3 Backbone Comparison

For the 3-channel baseline, the five backbones are compared to characterize the architectural effect on cross-domain generalization, independent of the auxiliary-channel question.

### 7.4 Ablation on Class Balancing

The 3-channel baseline is run with and without the 3:1 class balancing to characterize the contribution of the balancing step alone.

### 7.5 Reproducibility

A single seed is used by default. Sensitivity to seed is partially reported by repeating the high-priority comparisons across a small number of seeds; the seed set is pre-registered.

## 8. Related Work Positioning

The work positions itself relative to four lines of recent literature:

- **GDRNet / GDRBench (Che et al., 2023) and DRGen (Atwany & Yaqub, 2022):** establish that LODO-style cross-domain evaluation is a meaningful and populated line of work on this dataset pool. The present work uses LODO as a strict evaluation protocol but does not propose a new method to close the cross-domain accuracy gap.
- **DECO (Xia et al., 2024):** representation-disentanglement approach to closing the cross-domain gap, benchmarked on GDRBench. Same positioning as DRGen and GDRNet.
- **El Bellaj et al. (2026):** the closest prior work. Uses an evidential ordinal regression head with a single pooled train/val/test split and reports one aggregate uncertainty number. The present work differs in (a) using strict LODO rather than pooled split, (b) using CORN's independent-binary decomposition rather than a joint Dirichlet distribution, and (c) reporting calibration decomposed per threshold rather than as a single aggregate.
- **ORDER-DR (Sheng et al., 2026):** reports a single scalar ECE on a binarized referable-DR endpoint that nearly quadruples from in-distribution to external validation (0.049 → 0.160), providing independent confirmation that calibration degrades under cross-dataset shift in DR grading. Their "ordinal thresholds" are decision-boundary placements on the expected-grade score, not probability-calibration procedures at the per-threshold binary level. The present work cites ORDER-DR as motivation for the per-threshold calibration question rather than as a methodological competitor.
- **Poyrazer, Yağcı & Erten (2026):** independently demonstrates that global temperature scaling fails under DR cross-dataset shift (development ECE ≤ 0.022, external ECE 0.086–0.149 after re-scaling) and independently reproduces the Mild-grade catastrophic failure under shift (F1 = 0.000 for two of three encoders). Used as convergent evidence that aggregate calibration tools are inadequate under shift and that the Mild threshold is a prior target for finer-grained diagnosis. Not a methodological competitor (no ordinal decomposition, no LODO); cited as convergent prior evidence.

## 9. Threats to Validity

- **Source coverage:** only four publicly available sources are used. Generalization to sources not represented in this set is not claimed. The four-source pool is a strict subset of GDRBench's eight-source pool; extending the analysis to the full eight datasets is a follow-up.
- **Image resolution:** the preprocessing pipeline resizes all images to a uniform resolution. Resolution-induced information loss is not separately characterized.
- **Grading-scheme reconciliation:** the mapping from the 6-class scheme to the 5-class scheme is a published correspondence. Different mappings would yield different results.
- **Single seed:** the default protocol uses a single seed. The reproducibility-check subset is small.
- **Threshold tuning:** the cumulative-threshold thresholds are tuned on the validation set. Threshold quality depends on validation-set size, which is small for some folds.
- **Calibration reporting:** the per-threshold calibration analysis is reported on the 3-channel baseline only. The auxiliary-channel variants are not separately calibrated at the per-threshold level; doing so is a follow-up.

## 10. Ethics and Data Use

All four source datasets are publicly available for research use under their respective licenses. No patient-identifying information is included in the released data. The work does not constitute a clinical-grade diagnostic system and is not intended for clinical deployment without further validation.
