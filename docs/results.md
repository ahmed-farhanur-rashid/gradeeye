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

**Status (2026-08-11):** Prior numbers wiped — they were generated from a preprocessor-bypass (raw JPEG path) configuration that does not match the methodology's preprocessing pipeline. Restarting from clean state per user instruction. See `docs/refined_todo.md` for the active run plan.

| Held-out source | QWK | 95% CI | Acc | F1 (macro) | AUC (macro) |
|:----------------|----:|:-------|----:|:----------:|:-----------:|
| EyePACS |  |  |  |  |  |
| APTOS |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |
| DDR |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |

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
| EyePACS |  |  |  |  |  |  |  |
| APTOS |  |  |  |  |  |  |  |
| Messidor-2 |  |  |  |  |  |  |  |
| DDR |  |  |  |  |  |  |  |
| **Mean ± SD** |  |  |  |  |  |  |  |
| **Δ vs baseline** | — |  |  |  |  |  |  |
| **p (Δ vs 0)** | — |  |  |  |  |  |  |

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
| ConvNeXt-Tiny |  |  |  |  |  |  |
| ConvNeXt-Small |  |  |  |  |  |  |
| DeiT-3 Small (384) |  |  |  |  |  |  |
| MaxViT Tiny (384) |  |  |  |  |  |  |
| SwinV2-Base (192→384) |  |  |  |  |  |  |

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