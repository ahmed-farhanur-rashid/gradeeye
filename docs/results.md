# Results and Analysis

## 1. Notation

For every cell of every table below:

- **QWK** = Quadratic Weighted Kappa on the held-out domain's test set.
- **Acc** = Accuracy at the argmax ordinal prediction.
- **F1 (macro)** = Unweighted mean of per-class F1.
- **AUC (macro)** = Unweighted mean of one-vs-rest area under the ROC curve.
- **ECE** = Expected Calibration Error with 10 equal-width bins.
- **CI** = 95% bootstrap confidence interval (1000 resamples) over the test set.
- **±** values report mean ± standard deviation across LODO folds.
- **Subscript $t$** indicates the threshold $t \in \{1, 2, 3, 4\}$ for per-threshold calibration results. ECE$_t$ is the binary "grade $\geq t$" calibration error on the held-out test set.

A bold value in the main cell denotes the best score in its row. An asterisk (\*) flags a value whose confidence interval excludes the corresponding baseline value (paired bootstrap, 1000 resamples, pre-registered significance threshold).

## 2. Primary Cross-Domain Result: 3-Channel Baseline

The 3-channel RGB baseline establishes the cross-domain reference. Values are QWK on the held-out domain's test set.

**Status (2026-08-11):** All 12 baseline runs completed (3 architectures × 4 LODO folds). Numbers below reflect the second-tier architecture comparison: ConvNeXt-Tiny achieves the highest mean cross-domain QWK (0.686), DeiT-3-Small/384 close behind (0.653), MaxViT-Tiny/384 weakest (0.496, dragged down by EyePACS). ConvNeXt's loss function is CORN ordinal regression with √-frequency class-balanced sampling in phase 2; phase 1 is a 5-epoch frozen-backbone warm-up at bs=64; phase 2 is a 20-epoch fine-tune at bs=24 with cosine schedule, mixup, EMA (decay 0.999), and CBAM attention. Class-balanced via 3:1 ratio on the 92k-image combined pool.

### 2.1 ConvNeXt-Tiny (3-channel, balanced)

| Held-out source | QWK | 95% CI | Acc | F1 (macro) | AUC (macro) |
|:----------------|----:|:-------|----:|:----------:|:-----------:|
| EyePACS | 0.4399 | _pending CI_ | 0.5540 | 0.3445 | 0.7501 |
| APTOS | 0.8324 | _pending CI_ | 0.6038 | 0.4192 | 0.7731 |
| Messidor-2 | 0.7250 | _pending CI_ | 0.7225 | 0.5706 | 0.8648 |
| DDR | 0.7466 | _pending CI_ | 0.6830 | 0.4714 | 0.8712 |
| **Mean ± SD** | 0.6860 ± 0.156 | _pending CI_ | 0.6408 ± 0.066 | 0.4514 ± 0.087 | 0.8148 ± 0.057 |

### 2.2 DeiT-3 Small @ 384 (3-channel, balanced)

| Held-out source | QWK | 95% CI | Acc | F1 (macro) | AUC (macro) |
|:----------------|----:|:-------|----:|:----------:|:-----------:|
| EyePACS | 0.4107 | _pending CI_ | 0.4045 | 0.3149 | 0.7721 |
| APTOS | **0.8643** | _pending CI_ | 0.6936 | 0.4824 | 0.8489 |
| Messidor-2 | 0.6567 | _pending CI_ | 0.6835 | 0.5071 | 0.8346 |
| DDR | 0.6801 | _pending CI_ | 0.6410 | 0.4337 | 0.8393 |
| **Mean ± SD** | 0.6530 ± 0.186 | _pending CI_ | 0.6057 ± 0.136 | 0.4345 ± 0.085 | 0.8237 ± 0.035 |

### 2.3 MaxViT-Tiny @ 384 (3-channel, balanced)

| Held-out source | QWK | 95% CI | Acc | F1 (macro) | AUC (macro) |
|:----------------|----:|:-------|----:|:----------:|:-----------:|
| EyePACS | 0.2027 | _pending CI_ | 0.6197 | 0.2447 | 0.6712 |
| APTOS | 0.8366 | _pending CI_ | 0.7373 | 0.4815 | 0.7840 |
| Messidor-2 | 0.3629 | _pending CI_ | 0.5889 | 0.3857 | 0.7584 |
| DDR | 0.5827 | _pending CI_ | 0.6266 | 0.3698 | 0.7655 |
| **Mean ± SD** | 0.4962 ± 0.275 | _pending CI_ | 0.6431 ± 0.065 | 0.3704 ± 0.097 | 0.7448 ± 0.050 |

**Confusion matrix (summed across folds, normalized by row):**

| True \\ Pred | 0 | 1 | 2 | 3 | 4 |
|:------------:|:-:|:-:|:-:|:-:|:-:|
| 0 |  |  |  |  |  |
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |

Per-fold raw confusion matrices:

- **EyePACS** (n=88,702, full test set): _pending — see `docs/refined_todo.md`._
- **APTOS** (n=3,662, full test set): _pending._
- **Messidor-2** (n=1,744, full test set): _pending._
- **DDR** (n=12,522, full test set): _pending._

## 3. Per-Threshold Calibration (Core Diagnostic)

This is the central contribution of the work. ECE is computed on the binary "grade $\geq t$" prediction at each CORN sub-problem, with 10 equal-width bins. The aggregate ECE over the categorical prediction is reported as a summary for comparability with prior work.

### 3.1 Per-Threshold ECE by Fold (No Calibration Adjustment)

ECE$_t$ is the per-threshold ECE without temperature scaling, on the held-out domain's test set.

| Held-out source | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (aggregate) |
|:----------------|:-------:|:-------:|:-------:|:-------:|:---------------:|
| EyePACS |  |  |  |  |  |
| APTOS |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |
| DDR |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |

### 3.2 Per-Threshold ECE by Fold (Per-Threshold Temperature Scaling)

ECE$_t$ is recomputed after fitting a separate temperature parameter per threshold on the validation set.

| Held-out source | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (aggregate) |
|:----------------|:-------:|:-------:|:-------:|:-------:|:---------------:|
| EyePACS |  |  |  |  |  |
| APTOS |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |
| DDR |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |

### 3.3 Per-Threshold ECE by Fold (Single Global Temperature Scaling)

For comparison: a single temperature parameter fit on the validation set across all four thresholds, evaluated per threshold on the test set.

| Held-out source | ECE$_1$ | ECE$_2$ | ECE$_3$ | ECE$_4$ | ECE (aggregate) |
|:----------------|:-------:|:-------:|:-------:|:-------:|:---------------:|
| EyePACS |  |  |  |  |  |
| APTOS |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |
| DDR |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |

### 3.4 Per-Threshold Calibration: Per-Fold Delta Comparison

Δ is the improvement in ECE from no temperature scaling (Table 3.1) to per-threshold temperature scaling (Table 3.2). Negative Δ indicates that per-threshold temperature scaling improved calibration at that threshold. The Δ between per-threshold and single-global scaling (Table 3.2 minus Table 3.3) is reported separately.

| Held-out source | ΔECE$_1$ | ΔECE$_2$ | ΔECE$_3$ | ΔECE$_4$ | ΔECE (aggregate) |
|:----------------|:--------:|:--------:|:--------:|:--------:|:----------------:|
| EyePACS |  |  |  |  |  |
| APTOS |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |
| DDR |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |

| Held-out source | ΔECE$_1$ | ΔECE$_2$ | ΔECE$_3$ | ΔECE$_4$ | ΔECE (aggregate) |
|:----------------|:--------:|:--------:|:--------:|:--------:|:----------------:|
| EyePACS |  |  |  |  |  |
| APTOS |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |
| DDR |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |

### 3.5 Reliability Diagrams (Per Threshold, Per Fold)

For each LODO fold, four reliability diagrams — one per ordinal threshold — display the predicted-probability bin versus the empirical frequency of "grade $\geq t$." Reliability diagrams are described in text below, one paragraph per fold. They are not transcribed as tables.

- Source A: …
- Source B: …
- Source C: …
- Source D: …

### 3.6 Per-Threshold Class-Distribution Context

For each held-out source, the per-threshold positive-class prevalence (fraction of samples with grade $\geq t$) is reported. The per-threshold class distribution is the dominant covariate for interpreting per-threshold calibration.

| Held-out source | P(grade $\geq$ 1) | P(grade $\geq$ 2) | P(grade $\geq$ 3) | P(grade $\geq$ 4) |
|:----------------|:------------------:|:------------------:|:------------------:|:------------------:|
| EyePACS |  |  |  |  |
| APTOS |  |  |  |  |
| Messidor-2 |  |  |  |  |
| DDR |  |  |  |  |

### 3.7 Per-Threshold Calibration Versus Single-Threshold Reporting

Comparison of the per-threshold ECE (the present contribution) to the aggregate ECE (the standard literature reporting). Reported as the per-fold disagreement between the largest and smallest per-threshold ECE.

| Held-out source | max$_t$ ECE$_t$ | min$_t$ ECE$_t$ | Range | Aggregate ECE |
|:----------------|:---------------:|:---------------:|:-----:|:-------------:|
| EyePACS |  |  |  |  |
| APTOS |  |  |  |  |
| Messidor-2 |  |  |  |  |
| DDR |  |  |  |  |
| **Mean ± SD** |  |  |  |  |

## 4. Auxiliary-Channel Comparison

This is a methodological supplement to §3. The auxiliary-channel question is reported as QWK and secondary metrics only; the per-threshold calibration analysis of §3 is not repeated for these variants. Doing so is a follow-up.

### 4.1 Per-Variant QWK by Fold

Each variant is trained on the same 3-source pool and evaluated on the held-out 4th source. Bold = best in row. Asterisk = statistically significant improvement over the 3-channel baseline (paired bootstrap).

| Held-out source | 3ch baseline | 4ch soft mask | 4ch Sobel | 4ch Tversky | 4ch morph | 5ch soft+Sobel | 5ch Tversky+Sobel |
|:----------------|:------------:|:-------------:|:---------:|:-----------:|:---------:|:--------------:|:------------------:|
| EyePACS | 0.4399 | _Step F pending_ | _skipped_ | _Step F pending_ | _Step F pending_ | _skipped_ | _skipped_ |
| APTOS | 0.8324 | **0.8543** (+2.19%) | _skipped_ | 0.7997 (–3.27%) | 0.8517 (+1.93%) | _skipped_ | _skipped_ |
| Messidor-2 | 0.7250 | 0.6793 (–4.57%) | _skipped_ | 0.6909 (–3.41%) | 0.6837 (–4.13%) | _skipped_ | _skipped_ |
| DDR | 0.7466 | _Step F pending_ | _skipped_ | _Step F pending_ | _Step F pending_ | _skipped_ | _skipped_ |
| **Mean (2 folds, APTOS+Messidor-2)** | 0.7787 | 0.7668 (–1.19%) | — | 0.7453 (–3.34%) | 0.7677 (–1.10%) | — | — |
| **Δ vs baseline (≥2% threshold)** | — | NOT met | — | NOT met | NOT met | — | — |

**Step C status (2026-08-11):** 4-channel seg ablation on convnext_tiny, 2 folds (APTOS, Messidor-2), 3 variants × 2 folds = 6 runs complete. **None of the variants meet the ≥2% QWK improvement threshold** vs the 3-channel baseline on either fold. Step F will extend this matrix to EyePACS and DDR.

### 4.2 Per-Variant Secondary Metrics (Mean Across Folds)

| Variant | QWK | Acc | F1 (macro) | AUC (macro) |
|:--------|:---:|:---:|:----------:|:-----------:|
| 3ch baseline |  |  |  |  |
| 4ch soft mask |  |  |  |  |
| 4ch Sobel |  |  |  |  |
| 4ch Tversky |  |  |  |  |
| 4ch morph |  |  |  |  |
| 5ch soft + Sobel |  |  |  |  |
| 5ch Tversky + Sobel |  |  |  |  |

### 4.3 Per-Class F1 (Mean Across Folds)

| Variant | F1 (Grade 0) | F1 (Grade 1) | F1 (Grade 2) | F1 (Grade 3) | F1 (Grade 4) |
|:--------|:------------:|:------------:|:------------:|:------------:|:------------:|
| 3ch baseline |  |  |  |  |  |
| 4ch soft mask |  |  |  |  |  |
| 4ch Sobel |  |  |  |  |  |
| 4ch Tversky |  |  |  |  |  |
| 4ch morph |  |  |  |  |  |
| 5ch soft + Sobel |  |  |  |  |  |
| 5ch Tversky + Sobel |  |  |  |  |  |

## 5. Backbone Comparison (3-Channel Baseline)

QWK on the held-out domain's test set, varying only the backbone architecture.

| Backbone | EyePACS | APTOS | Messidor-2 | DDR | Mean ± SD | Params (M) |
|:---------|:--------:|:--------:|:--------:|:--------:|:---------:|:----------:|
| ConvNeXt-Tiny | 0.4399 | 0.8324 | 0.7250 | 0.7466 | 0.6860 ± 0.171 | 28 |
| ConvNeXt-Small | _skipped — out of scope for the 24-h budget_ |  |  |  |  | 50 |
| DeiT-3 Small (384) | 0.4107 | 0.8643 | 0.6567 | 0.6801 | 0.6530 ± 0.186 | 22 |
| MaxViT Tiny (384) | 0.2027 | 0.8366 | 0.3629 | 0.5827 | 0.4962 ± 0.275 | 31 |
| SwinV2-Base (192→384) | _skipped — out of scope for the 24-h budget_ |  |  |  |  | 88 |

## 6. Class-Balancing Ablation

Effect of the 3:1 class-balancing step on the 3-channel baseline.

| Held-out source | QWK (balanced) | QWK (unbalanced) | Δ |
|:----------------|:--------------:|:----------------:|:-:|
| EyePACS |  |  |  |
| APTOS |  |  |  |
| Messidor-2 |  |  |  |
| DDR |  |  |  |
| **Mean ± SD** |  |  |  |

Per-class F1 with and without balancing (mean across folds):

| Class | F1 (balanced) | F1 (unbalanced) | Δ |
|:------|:-------------:|:---------------:|:-:|
| Grade 0 |  |  |  |
| Grade 1 |  |  |  |
| Grade 2 |  |  |  |
| Grade 3 |  |  |  |
| Grade 4 |  |  |  |

## 7. Effect of Threshold Tuning (Decision-Boundary, Not Calibration)

The validation-tuned ordinal thresholds (§4.3) place decision boundaries on the expected-grade score to maximize QWK. This is a **decision-boundary procedure**, distinct from the per-threshold calibration analysis of §3. Both are reported because the deployment policy on whether to tune thresholds is application-specific.

| Held-out source | QWK (default) | QWK (tuned) | Δ |
|:----------------|:-------------:|:-----------:|:-:|
| EyePACS |  |  |  |
| APTOS |  |  |  |
| Messidor-2 |  |  |  |
| DDR |  |  |  |
| **Mean ± SD** |  |  |  |

## 8. Segmentation Auxiliary Model

Dice / IoU / Tversky on the segmentation model's own evaluation split. Reported separately from the grading metrics.

| Loss family | Dice | IoU | Tversky |
|:------------|:----:|:---:|:-------:|
| BCE + Dice |  |  |  |
| Tversky |  |  |  |

Example soft-mask outputs (described in text, not as tables) for representative samples from each source.

## 9. Sensitivity to Seed

The highest-priority variants (3ch baseline, 4ch Sobel, 5ch Tversky+Sobel) are repeated across a small pre-registered seed set. Other variants are single-seed.

| Variant | Seed 1 | Seed 2 | Seed 3 | Mean ± SD |
|:--------|:------:|:------:|:------:|:---------:|
| 3ch baseline |  |  |  |  |
| 4ch Sobel |  |  |  |  |
| 5ch Tversky+Sobel |  |  |  |  |

## 10. Single-Source Failure Mode

For each held-out source, the model that achieves the worst QWK is identified and the failure is characterized qualitatively. The goal is to document, not to remediate, the failure.

| Held-out source | Worst variant | QWK | Dominant failure mode |
|:----------------|:-------------:|:---:|:----------------------|
| EyePACS |  |  |  |
| APTOS |  |  |  |
| Messidor-2 |  |  |  |
| DDR |  |  |  |

## 11. Summary Findings

(To be filled in after the experiments are complete. Each finding must be backed by a table cell above.)

- Finding 1: …
- Finding 2: …
- Finding 3: …
- Finding 4: …
- Finding 5: …