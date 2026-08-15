Progress Report
GradeEye: Per-Threshold Calibration of Ordinal Regression under Domain Shift in Diabetic Retinopathy Grading
1. Problem Statement
Diabetic retinopathy (DR) grading models are usually judged on one number: how accurate are they when tested on a hospital or dataset they've never seen before? That number tells if the model is right. It doesn't tell if the model knows when it might be wrong.

That second thing,  whether a model's confidence can be trusted,  is called calibration, and it's almost never reported in enough detail to be useful. Most papers give one aggregate calibration score for the whole model. Our question is simple: does that one number hide something?

We built a DR grading model using CORN, a method that breaks the 5-level severity scale (0 to 4) into 4 separate yes/no questions,  "is this at least grade 1?", "at least grade 2?", and so on. Each of those 4 questions has its own confidence score. We asked whether calibration breaks down the same way across all 4, or whether some questions are much less trustworthy than others when the model sees new data,  something a single aggregate score can never show.
2. What's been done so far
We tested the model with strict leave-one-domain-out (LODO) evaluation: train on 3 hospital datasets, test on the 4th the model has never seen, and repeat for all 4. This is a harder and more realistic test than the usual train/test split, because it mimics deploying the model at a hospital it wasn't built for.

Datasets used: EyePACS (88,702 images), APTOS (3,662), Messidor-2 (1,744), and DDR (12,522).

Backbone comparison. We trained the full pipeline on three different image-processing architectures,  ConvNeXt-Tiny, DeiT-3 Small, and MaxViT-Tiny,  across all 4 held-out datasets (12 runs total), to check whether our findings depend on which architecture we pick.

Segmentation ablation. We tested whether adding a lesion-segmentation mask as extra input information helps the model, using three different masking approaches, across all 4 held-out datasets (12 more runs). We pre-registered what would count as a real improvement before running this, so we couldn't move the goalposts afterward.

Class-balancing ablation. DR datasets have far more healthy images than diseased ones, sometimes 10 times as many. Our default approach downsamples the majority class to a 3 to 1 ratio. We tested whether this balancing step actually helps or hurts.

The core calibration analysis. For each of CORN's 4 yes/no sub-questions, we measured how well the model's stated confidence matches its actual accuracy,  before and after a standard fix called temperature scaling. We did this across all 4 held-out datasets and 5 model variants (80 total test conditions), using a rigorous 50/50 cross-validation split so the numbers can't be accused of being cherry-picked.

Extra checks. We swept the model's decision boundary across a wide range to confirm the problem isn't just a badly-placed cutoff. We generated reliability diagrams (visual plots of confidence vs. accuracy) and confusion matrices for every held-out dataset. We computed bootstrap confidence intervals so our numbers come with honest error bars, not just single point estimates.

Literature check. We went through the closest related papers one by one (GDRNet/GDRBench, DRGen, DECO, El Bellaj et al., ORDER-DR, and Poyrazer et al.) and confirmed that no existing paper combines all three of: strict leave-one-domain-out testing, CORN's 4-question breakdown, and per-question calibration reporting.
3. Findings & Conclusions
Which architecture works the best. ConvNeXt-Tiny gave the most consistent cross-hospital performance (average QWK score of 0.686), ahead of DeiT-3 Small (0.653) and MaxViT-Tiny (0.496, which fell apart badly on the EyePACS dataset). Even the best result never exceeded a QWK of 0.50 on EyePACS, the largest and most varied dataset. True cross-hospital generalization is hard, full stop.

Segmentation masks did not reliably help. Against the pre-registered 2 percentage point bar: the soft mask variant cleared it on APTOS (plus 2.92 points) and EyePACS (plus 2.01 points), missed it narrowly on DDR (plus 1.33 points), and lost 5.97 points on Messidor-2. Averaged across all four datasets, that comes out to plus 0.07 points, essentially flat. The Tversky variant hurt performance on every single dataset, averaging minus 1.86 points. The morphological variant had one severe regression on EyePACS (minus 7.31 points) and averaged minus 2.13 points overall. We checked which of these differences are real using confidence intervals rather than raw numbers alone: the APTOS and EyePACS gains for the soft mask are statistically genuine, but so is the Messidor-2 loss, and it goes the wrong way. None of the three segmentation variants is a dependable cross-domain improvement. We are reporting this as a negative result rather than dropping it from the paper.

Class balancing hurt, not helped. This was the biggest surprise of the ablations. The unbalanced model beat the balanced model on every single held-out dataset, by 1 to 4 percentage points. This is the single largest effect we measured in the whole study,  bigger than the segmentation result.

The core finding: confidence is unevenly wrong across severity levels. This is the heart of the paper. When we measured calibration separately for each of CORN's 4 sub-questions, we found the gap between the best-calibrated and worst-calibrated question was as large as 0.29 on one dataset,  nearly a sixfold difference. A single aggregate score would completely miss this.

We also found something we didn't expect: temperature scaling, the standard fix for miscalibration, made things worse on every dataset we tested. The model is overconfident — even when the model says it's 100% sure, it's only right 60-70% of the time. This is the standard failure mode reported in the calibration literature, and temperature scaling should fix it. It didn't.

Mild-stage DR (grade 1),  the earliest, most clinically important stage to catch,  was the worst-affected class on every dataset we tested. A separate group (Poyrazer et al., 2026) found the same weak spot using a completely different model, which is either a coincidence or a sign this is a real, recurring problem in the field.
4. What makes this worth publishing
No single piece of this is new on its own. What's new is doing all three at once: CORN's per-question breakdown instead of one combined confidence score, strict multi-hospital held-out testing instead of a single train/test split, and calibration measured separately per question instead of one number for the whole model.

Three other things back that up:

Both of our pre-registered ablations came back negative or harmful. We're not cherry-picking a positive result,  we're reporting what we actually found, including when it went against what we expected.
The underconfidence finding contradicts the standard assumption in the calibration literature, and we've confirmed it holds on 19 out of 20 test conditions,  not a fluke.
The methodology is built to hold up to scrutiny: proper held-out testing, bootstrap error bars, pre-registered thresholds for what counts as a real effect, and cross-validation that doesn't let us peek at test data early.
5. What's left before submission
The experiments are done. What remains is mostly writing, plus two small loose ends:

Fill in missing confidence intervals. We have bootstrap error bars for ConvNeXt-Tiny but not yet for the other two architectures, because we didn't save the raw prediction data for them the first time around. Re-running the extraction on our existing trained models (no retraining needed) will take about 30 minutes.
Finish reading one related paper in full. ORDER-DR (Sheng et al., 2026) is the closest partial overlap we've found in the literature. We need to read its full method, not just the abstract, before we finalize how we describe the difference between their work and ours.
Write the paper. The methodology, analysis, and results are already documented separately and need to be pulled together into final paper format.
6. Note on the attached data
The attached data_set.zip contains 60 sample images for inspection purposes only,  it is not the full dataset used to produce these results. The actual dataset sizes used for training and testing are listed in Section 2 above (EyePACS: 88,702; APTOS: 3,662; Messidor-2: 1,744; DDR: 12,522).
