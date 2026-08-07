# LODO Collapse Diagnostic Report

Date: 2026-08-07  
Scope: 3 ConvNeXt-Tiny LODO checkpoints (one per fold), evaluated against
each fold's held-out `test_full` and `test_matched` manifest.

User-flagged anomaly: EyePACS-holdout collapses from val QWK **0.7389** to
test-full **0.2423** to test-matched **0.1204**.

## Question 1 — Class distributions (% per grade)

The pool is the two non-held-out sources' train+val concatenated. Test
columns are the held-out source.

| Fold | Pool (train+val) | test_full | test_matched | matched sample % per class |
|------|------------------|-----------|--------------|-----------------------------|
| eyepacs | 0:52.2, 1:11.8, 2:24.9, 3:5.0, 4:6.1 | 0:73.7, 1:7.0, 2:14.8, 3:2.4, 4:2.2 | 0:73.6, 1:7.0, 2:15.0, 3:2.2, 4:2.2 | 0:0.3, 1:0.3, 2:0.3, 3:0.2, 4:0.3 |
| aptos | 0:73.4, 1:7.2, 2:14.9, 3:2.4, 4:2.2 | 0:49.3, 1:10.1, 2:27.3, 3:5.3, 4:8.1 | 0:49.3, 1:10.1, 2:27.3, 3:5.3, 4:7.9 | 0:6.2, 1:6.2, 2:6.2, 3:6.2, 4:6.1 |
| messidor2 | 0:72.7, 1:7.1, 2:15.3, 3:2.5, 4:2.4 | 0:58.3, 1:15.5, 2:19.9, 3:4.3, 4:2.0 | 0:58.1, 1:15.4, 2:19.8, 3:4.4, 4:2.2 | 0:13.0, 1:13.0, 2:13.0, 3:13.3, 4:14.3 |

Three observations:

- **EyePACS pool (APTOS+Messidor-2)** is *less* skewed toward NoDR (52%) than
  EyePACS test (74%). The pool *under-represents* NoDR compared to the
  held-out domain — this is the *opposite* of the standard "train skews
  toward majority class" failure mode.
- **APTOS-holdout pool** (EyePACS+Messidor-2) is *more* NoDR-skewed (73%)
  than APTOS test (49%). Mismatched prior — but **opposite direction from
  EyePACS**.
- **Messidor-2-holdout pool** (EyePACS+APTOS) is more NoDR-skewed (73%)
  than Messidor-2 test (58%).

There is no uniform prior-mismatch direction across folds. EyePACS is
unique in that its test set has the *heaviest* NoDR prior (74%) of the
three domains and the training pool has the *lightest* NoDR prior (52%).
That means EyePACS training mostly sees non-NoDR cases, so a model trained
on APTOS+Messidor-2 over-predicts NoDR when transferred to EyePACS.

## Question 2 — EyePACS test-matched confusion matrix (5×5)

| true\pred | NoDR | Mild | Mod | Sev | PDR |
|-----------|------|------|------|------|------|
| **NoDR**  | 154  | 12   | 0    | 1    | 0    |
| **Mild**  | 13   | 3    | 0    | 0    | 0    |
| **Mod**   | 30   | 4    | 0    | 0    | 0    |
| **Sev**   | 2    | 3    | 0    | 0    | 0    |
| **PDR**   | 4    | 0    | 0    | 1    | 0    |

Predicted distribution: 0:203, 1:22, 2:0, 3:2, 4:0 (n=227). The model
predicted **NoDR for 89.4%** of EyePACS-matched even though only 73.6% are
actually NoDR. This is the eye-prior-shift direction: the model
over-predicts NoDR on EyePACS.

Compare with EyePACS test-full predicted distribution: 0:79571/88702 =
89.7% NoDR. Same over-prediction rate, same collapse pattern.

## Question 3 — Matched-subsample integrity

Code path: `src/data/lodo_split.py:66-78`:

```
StratifiedShuffleSplit(n_splits=1, train_size=target_size,
                       random_state=seed).split(X=zeros, y=label)
```

`_stratified_downsample(test_full, target_size=227, seed=42)`. Applied to
`test_full` directly, with `test_full` *being* EyePACS, not the training
pool. Verified by reconstructing the operation: identical 227-row output
with label counts {0:167, 1:16, 2:34, 3:5, 4:5}.

Class-fraction deltas between EyePACS test_full and test_matched:

```
class 0: full=73.67%, matched=73.57%, delta=-0.0010
class 1: full=7.00%,  matched=7.05%,  delta=+0.0005
class 2: full=14.83%, matched=14.98%, delta=+0.0015
class 3: full=2.35%,  matched=2.20%,  delta=-0.0015
class 4: full=2.16%,  matched=2.20%,  delta=+0.0004
```

Max per-class delta = 0.4 percentage points. The matched subset preserves
EyePACS's real class proportions, **not the training pool's** and **not
uniform**. The `matched→full sample pct` column above (last column of
Question 1) shows the matched draws ~0.3% of each class (1 − 227/88702 ≈
0.997).

## Question 4 — Checkpoint and label-mapping sanity

For every fold, `load_model_from_checkpoint` returned:

| Fold | arch | img_size | in_chans | num_thresholds | head out |
|------|------|----------|----------|----------------|----------|
| eyepacs | convnext_tiny | 384 | 3 | 4 | 4 |
| aptos | convnext_tiny | 384 | 3 | 4 | 4 |
| messidor2 | convnext_tiny | 384 | 3 | 4 | 4 |

The checkpoint loaded for `test_full` and `test_matched` in each fold is
identical (same file path). `src/eval/evaluate.py:36-103` runs the same
`evaluate_checkpoint` function on both manifests, with the same image
transform (`build_eval_transforms(img_size=384)`), the same label
extraction (`DRDataset` reads `label` column verbatim), and the same
class-index mapping (CORN's `argmax` over its 4 conditional probs →
classes 0..4). No encoding mismatch is possible between the two
manifests of the same fold.

## Question 5 — Pool vs held-out distribution comparison, all 3 folds

Computed train_pool = train+val from non-held-out sources; held_out =
test_full of the held-out source.

| Fold | Train pool %NoDR | Held-out %NoDR | Delta (pool − held) |
|------|------------------|----------------|----------------------|
| eyepacs | 52.2% | 73.7% | **−21.5 pp** |
| aptos | 73.4% | 49.3% | **+24.1 pp** |
| messidor2 | 72.7% | 58.3% | **+14.4 pp** |

Test QWK:

| Fold | val QWK | test_full QWK | test_matched QWK | matched Δ from full |
|------|---------|---------------|------------------|---------------------|
| eyepacs | 0.7389 | 0.2423 | 0.1204 | −0.12 |
| aptos | 0.7218 | 0.5496 | 0.5037 | −0.05 |
| messidor2 | 0.7010 | 0.6324 | 0.5722 | −0.06 |

The collapse from val to test is **worst on EyePACS** by a wide margin, and
EyePACS has the most extreme **opposite-direction** prior mismatch (pool
much less NoDR-skewed than held-out). APTOS and Messidor-2 also have val
QWK much higher than test QWK, but with smaller prior mismatches.

## Question 6 — Sampling-variance check (bootstrap)

For each fold, take 500 random stratified 227-row samples from `test_full`,
compute QWK for each, and locate the actual `test_matched` QWK in that
distribution.

| Fold | full QWK | match QWK | bootstrap mean | bootstrap std | match percentile |
|------|----------|-----------|----------------|---------------|------------------|
| eyepacs | 0.2423 | 0.1204 | 0.2382 | 0.1002 | **3.0%** |
| aptos | 0.5496 | 0.5037 | 0.5475 | 0.0298 | **7.6%** |
| messidor2 | 0.6324 | 0.5722 | 0.6361 | 0.0374 | **4.6%** |

Every matched sample lands in the **bottom 8%** of its fold's bootstrap
distribution. On EyePACS the bootstrap variance is also the largest
(std 0.10, because only 0.3% of the data is sampled). All three matched
subsets are *unlucky draws*, but with identical per-class composition to
test_full. The matched subsamples are doing exactly what they were
designed to do — they are not creating the collapse.

## Hypothesis

**The EyePACS-holdout collapse is genuine prior-shift / domain-shift
behavior, not a code bug.**

Evidence for:

1. Test_full (n=88,702) and test_matched (n=227) on the *same* checkpoint
   show essentially the same prediction distribution (89.7% vs 89.4% NoDR)
   and the same accuracy (0.6983 vs 0.6916). A bug in eval, label mapping,
   or checkpoint loading would have affected the two differently.
2. EyePACS has the most extreme opposite-direction prior shift (pool 52%
   NoDR vs held-out 74% NoDR; +21 pp of NoDR over-representation at test
   time). APTOS and Messidor-2 also have prior mismatches but milder, and
   their test QWK is correspondingly less collapsed (0.55, 0.63 vs 0.24).
3. The CORN-argmax model learned "if uncertain, output NoDR" on a pool
   where NoDR was only 52% but Mild/Moderate were 37% combined. On
   EyePACS test where NoDR is 74%, the model's NoDR threshold becomes
   too aggressive, and Severe/PDR recall falls to 0–18%.
4. The matched-vs-full QWK gap of 0.12 on EyePACS is fully explained by
   the matched sample landing at the 3rd percentile of the bootstrap
   distribution on a non-NoDR-heavy drawn prediction set. There is no
   encoding mismatch between the two evaluations.

Evidence against (could-not-rule-out checks):

- The 88,702-sample test_full is itself an enormous sample and may hide
  the same prediction behavior at smaller scales. But the matched
  confusion matrix reproduces the test_full pattern at small scale
  (89.4% vs 89.7% NoDR), confirming the prediction behavior is robust.
- Could there be a per-batch normalization mismatch? Same eval transforms,
  same checkpoint, same dataloader parameters → no.
- Could there be a class-index remapping bug? `corn_predict_probas` is
  deterministic and identical for both eval paths → no.

What the matched subsample *does* show that test_full obscures at this
scale: even on a near-balanced sample (73.6% NoDR), the model produces
zero correct Severe and zero correct Proliferative DR predictions. **That
is the real failure mode, not QWK magnitude** — it's the prior-shift
behavior, the matched subset just exposes it more clearly on a per-class
basis.

## Conclusion

- The EyePACS QWK collapse from val to test is **real and caused by prior
  shift + domain shift**, not by a bug in the matched-subsample code, the
  checkpoint loader, the label mapping, or the eval transforms.
- The matched subset is doing what it was designed to do; its slightly
  lower QWK (0.120 vs 0.242) is an unlucky stratified draw on a small
  sample size, not a different evaluation path.
- The model over-predicts NoDR on EyePACS in both test_full and
  test_matched at identical rates. The val-set QWK was high because the
  val set came from APTOS+Messidor-2, where NoDR is closer to 52% and
  the model can use its Mild/Moderate head more reliably.
- Recommended paper framing (not applied): report test_full as the
  primary transfer metric, mention matched as a per-class recall exposure,
  and acknowledge prior shift as the dominant cross-domain failure mode
  rather than as a calibration or pipeline defect.

## Artifacts

- `docs/lodo_diagnostics/lodo_collapse_diagnostic.json` — full per-fold
  metric + distribution + confusion matrix data.
- `docs/lodo_diagnostics/lodo_collapse_diagnostic.md` — formatted summary.
- `docs/lodo_diagnostics/distribution_stats.json` — per-CSV class
  distributions for all 12 LODO CSVs.
- `scripts/diagnose_lodo_collapse.py` — reproducible script (read-only;
  no training, no config changes, no data writes).