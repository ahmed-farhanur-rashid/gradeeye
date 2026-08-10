# GradeEye — Related Work Citation Table

All entries below were checked against primary sources (arXiv abstracts/PDFs,
MICCAI proceedings pages, or publisher pages) in this session. Confidence
notes are included where the source material was thin.

| Paper | Year | Venue | Contribution | Dataset(s) | Difference from your work |
|---|---|---|---|---|---|
| **DRGen: Domain Generalization in Diabetic Retinopathy Classification** — Atwany & Yaqub | 2022 | MICCAI 2022 | Sets joint-training-then-per-domain-test as the DG baseline; proposes seeking flatter minima (SWAD-style) during training plus a regularizer to improve cross-domain generalization. | EyePACS, APTOS, Messidor-1, Messidor-2 | Targets *closing* the domain gap via a training-time regularizer, evaluated as train-on-pool/test-per-domain — not a full LODO sweep. No ordinal decomposition, no per-threshold calibration analysis; reports standard classification metrics. Your contribution diagnoses a calibration failure mode rather than proposing a generalization fix. |
| **GDRNet / GDRBench — Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains** — Che, Cheng, Jin, Chen | 2023 | MICCAI 2023 (arXiv:2307.04378) | Introduces GDRBench, a public 8-dataset benchmark with two protocols: full leave-one-domain-out (DG test) and extreme single-domain generalization (ESDG test). Proposes GDRNet, combining fundus-specific augmentation (FundusAug), a hybrid pixel+image-level loss (DahLoss), and domain-class-aware rebalancing (DCR). | EyePACS, APTOS, Messidor, IDRiD, DeepDR, FGADR, RLDR, DDR (DDR and EyePACS reserved for ESDG target-only use, not as DG-test source domains) | **This is the paper closest to your evaluation protocol and the single most important citation to include.** Their DG-test *is* a leave-one-domain-out sweep across a near-identical dataset pool to yours (including DDR). They report per-domain aggregate metrics (accuracy/AUC/F1) as the outcome of interest, not calibration. Your per-threshold ECE / ordinal-decomposition calibration diagnosis is orthogonal to what GDRNet targets — they fix the accuracy gap, you diagnose a calibration-structure failure that persists independent of it. Must be cited explicitly or a reviewer familiar with this space will read the omission as a lit-search gap. |
| **DECO — Generalizing to Unseen Domains in Diabetic Retinopathy with Disentangled Representations** — Xia et al. | 2024 | MICCAI 2024 (arXiv:2406.06384) | Disentangles retinal representations into semantic (DR-relevant) and domain-noise latent components; augments training data by recombining semantic features with domain noise from other domains. Evaluated on GDRBench's DG and ESDG protocols. | Same GDRBench 8-dataset pool (Messidor, IDRiD, DeepDR, FGADR, APTOS, RLDR, plus DDR/EyePACS for ESDG) | Representation-disentanglement fix for the domain gap, benchmarked with GDRBench's standard DG-test LODO protocol and aggregate metrics. No calibration analysis, no ordinal-decomposition angle. Cite alongside GDRNet as part of the same benchmark lineage — useful for showing you're aware of the strongest recent DG baselines on this exact benchmark. |
| **Uncertainty-Aware Ordinal Deep Learning for Cross-Dataset Diabetic Retinopathy Grading** — El Bellaj et al. | Feb 2026 | arXiv:2602.10315 | Dirichlet evidential ordinal regression head (joint distribution over all classes, single evidential mechanism) + lesion-query attention pooling. Trains with an ordinal evidential loss with annealed regularization for calibrated confidence under domain shift. | APTOS, Messidor-2, subset of EyePACS | **Your closest prior work** — confirmed via full-text reading (per your own contribution doc). Uses a single pooled train/val/test split, not LODO — no domain is ever fully held out from training. Reports one aggregate uncertainty number per model, not broken out per severity threshold. Structurally different ordinal mechanism (joint Dirichlet distribution vs. your CORN independent-binary decomposition). This is the paper your "per-threshold vs. aggregate, LODO vs. pooled-split" framing is built to answer directly. |
| **PRISM-DR — Toward Reliable Diabetic Retinopathy Screening** | 2026 | MDPI Sensors (submitted May 2026, published Jul 2026) | Multi-objective five-grade DR grading framework using a gradient-partitioned training strategy; CORAL ordinal head (gradient-isolated from backbone) alongside cross-entropy, prototype-contrastive, and view-consistency objectives on the shared backbone. | EyePACS, DDR, IDRiD, APTOS, Messidor-2 (Fixed-Source, Multi-Target protocol) | Uses CORAL, a related but distinct rank-consistent ordinal head from CORN (different label-encoding/binary-decomposition scheme) — worth a methods-section footnote if you discuss ordinal-regression variants generally. Reports per-target-domain QWK under an FSMT protocol, not per-threshold calibration. *Confidence note: found via a single MDPI listing; verify the CORAL-vs-CORN mechanism detail against the full text before citing specifics.* |
| **Domain Generalization for Diabetic Retinopathy Grading with Phase Augmentation Framework** — Zhang & Liu | 2026 | Medical & Biological Engineering & Computing (Springer) | Augmentation-based domain generalization approach addressing device/lighting/imaging-condition variation across hospitals. | Not confirmed beyond the abstract — full dataset list was paywalled. | Augmentation-focused DG fix, not a calibration diagnosis. Lower-priority citation; include only if your related-work section surveys the augmentation-based DG sub-family specifically. *Confidence note: abstract only, dataset details unverified — do not cite specific numbers from this paper without reading the full text.* |

## Uniqueness claim — verified against a second round of searches

The question was: what makes this paper more publishable — i.e., is the
combination of (ordinal-decomposition calibration diagnosis) × (true LODO) ×
(per-threshold granularity) actually unoccupied, or does something in the
2025-2026 literature already do this? Two targeted searches were run to
pressure-test this claim rather than assume it.

**Net result: the claim holds, with one paper you need to check directly
before finalizing.**

### What the search surfaced

| Paper | Year | Relevance to your uniqueness claim |
|---|---|---|
| **ORDER-DR** — external validation of severity grading and referable-risk stratification from fundus images | Aug 2026, *Frontiers in Endocrinology* | **Closest hit — read this before finalizing.** A "validation-calibrated dual-branch ordinal-risk framework" for 5-class DR grading, explicitly addressing "threshold shift when externally evaluated" and reporting calibrated ordinal thresholds on Messidor-2 pre/post calibration. This overlaps your territory (ordinal + threshold + calibration + external/cross-dataset shift) more than anything else found. *Unconfirmed from the snippet*: whether their thresholds are calibrated **per-threshold** in the CORN sense (independent binary sub-problems) or refer to conventional ordinal-regression cut-points calibrated as a single scalar-shift correction. If it's the latter, your contribution stays distinct (you diagnose *why* per-threshold calibration diverges due to class-imbalance-per-threshold in an independent-binary decomposition; they appear to do single/dual-branch threshold recalibration). This is a "must read the full text" item, not a "must cite and move on" item — it's the one paper that could weaken your novelty claim if its method turns out to be closer than the abstract suggests. |
| **Dual-SwinOrd** | Mar 2026, MDPI *Bioengineering* | Ordinal + semantic-prior DR grading; explicitly frames the tension between accuracy and QWK-consistency. No calibration or LODO angle in the abstract — not a threat to your uniqueness claim, but worth a related-work mention as another recent ordinal-regression DR paper. |
| **Lesion-Aware Ordinal Transformer** | Mar 2026, *Biomedical and Pharmacology Journal* | Notable for an explicit aside: the authors report ECE specifically because, in their words, <cite index="66-1">some recent papers did not present it</cite> — indirect third-party confirmation that calibration reporting is still uncommon in this literature, supporting your positioning. Single-dataset evaluation, no LODO, no per-threshold breakdown — not a competitor to your contribution. |
| **G2c-Net / asymmetric bi-classifier grade-skewed domain adaptation** (cited within an ordinal label-distribution paper, IEEE TMI 2025) | 2025 | Addresses grade-skew under domain adaptation, but via classifier-discrepancy minimization, not calibration diagnosis. Different mechanism; low relevance. |
| Several 2025-2026 CORAL/CORN-adjacent ordinal DR papers (diffusion-based autoregressive ordinal regression, dual-resolution attention + CORAL) | 2025-2026 | Confirm CORN/CORAL remain active choices in this literature, but none combine LODO + per-threshold calibration decomposition. Reinforces rather than threatens your positioning. |

### Bottom line on uniqueness

Nothing found directly replicates your combination of (a) CORN's independent
per-threshold binary decomposition, (b) diagnosed under true multi-fold LODO
domain shift, (c) with calibration broken out **per threshold** rather than
as one scalar or one dual-branch correction. That three-way combination
remains your strongest publishability argument. The one action item: read
ORDER-DR's full method before you finalize your related-work section — not
because it necessarily overlaps, but because "external validation +
calibrated ordinal thresholds + Messidor-2" is close enough that a reviewer
who knows both papers will expect you to have read and positioned against
it explicitly.

## Poyrazer, Yağcı & Erten (2026) — full-text verification, canonical entry

Follow-up from ORDER-DR's reference [13]. Full text read directly (not
abstract-only). This closes the loop — no further verification needed on
this paper.

**Citation**: Poyrazer M, Yağcı H, Erten R. "How well do frozen foundation
models transfer? A calibration-focused benchmark for diabetic retinopathy
grading." *Frontiers in Medicine* 13:1815982 (2026).
doi: 10.3389/fmed.2026.1815982

### What it is

A frozen-encoder representation benchmark: three pretrained vision
backbones (MedSigLIP, RETFound, EfficientNet-B0) are used as fixed feature
extractors, with only a lightweight MLP head trained on top of each.
Development on APTOS (5-fold CV), zero-tuning external validation on
Messidor-2. Both binary referable-DR and 5-class ICDR grading evaluated.
Calibration (ECE, Brier score) treated as co-primary alongside AUC.

### Verdict: cite it — real conceptual overlap, zero mechanistic overlap

| Aspect | Poyrazer et al. | Your work |
|---|---|---|
| Research question | Does pretrained-representation quality transfer across DR datasets? | Does an ordinal decomposition's per-threshold calibration transfer across DR datasets? |
| Model | Standard softmax classification heads (binary + 5-class) on frozen embeddings | CORN — rank-consistent ordinal regression, independent binary sub-problems per threshold |
| Calibration granularity | ECE/Brier per encoder, per dataset, per task (binary or 5-class) — no ordinal decomposition exists to break out further | Per-threshold ECE within CORN's own binary sub-problems |
| Protocol | Single train (APTOS) → single external test (Messidor-2), not LODO | Multi-fold LODO |
| Key finding | Temperature scaling works on development data (ECE ≤0.022) but **fails under domain shift** (external ECE 0.086–0.149, barely moved by re-scaling) | Per-threshold miscalibration under domain shift, tied to per-threshold class imbalance |
| Grade-1 (Mild) finding | Catastrophic failure under shift: F1 = 0.000 for two of three encoders, 0.153 for the best | Persistent poor Mild recall (0.00–0.06) after balancing, with a bidirectional misclassification pattern |

No ordinal regression, no LODO, no per-threshold anything — this paper
cannot be a methodological competitor to your contribution because it
doesn't touch the mechanism your paper is about. It is not close enough
to require careful differentiation language the way El Bellaj et al. does.

### Why cite it anyway — two concrete reasons, not just "adjacent field"

1. **It independently demonstrates that scalar/global recalibration
   (temperature scaling) does not fix calibration under DR cross-dataset
   shift.** This is a citable, quantified prior result that directly
   motivates your finer-grained per-threshold approach as the necessary
   next step — a reviewer cannot say "why not just apply temperature
   scaling" when a recent paper already tested that and reported it
   failing under shift.
2. **It independently reproduces the Mild/grade-1 catastrophic failure
   pattern under domain shift**, via a completely different mechanism
   (representation transfer gap, not ordinal per-threshold imbalance).
   Two unconnected papers landing on the same clinical failure mode from
   different angles is strong convergent evidence that this is a real,
   recurring phenomenon in DR cross-dataset grading — not an artifact of
   either paper's specific pipeline. Cite this alongside your own Mild-class
   finding to strengthen that section.

### Suggested citation framing

Position alongside ORDER-DR in your motivation/related-work section:
recent work has also shown that standard post-hoc calibration techniques
(temperature scaling) fail to correct for calibration degradation under
DR cross-dataset shift, and that the earliest clinically meaningful
severity grade (Mild NPDR) is disproportionately vulnerable to this shift
— findings consistent with, but independently derived from, the
per-threshold analysis presented here.

### Status: no open items remain on this paper.

## ORDER-DR — full-text verification and citation recommendation

Initially flagged from an abstract-only search as a possible close overlap
(ordinal + threshold + calibration + Messidor-2). The full text was read
directly to check. Verdict: **cite it, but as motivating context, not as a
methodological competitor.**

### What ORDER-DR actually does, confirmed from the full text

| Aspect | ORDER-DR (Sheng, Dong, Wu, Liang — *Frontiers in Endocrinology*, Aug 2026) | Your work |
|---|---|---|
| Ordinal mechanism | Two EfficientNet-B0 branches (class-balanced CE + an ordinal-risk auxiliary loss combining cross-entropy, expected-grade distance, binary referable loss, and a distance-weighted soft-label term), fused by **fixed 0.5/0.5 probability averaging** | CORN — rank-consistent ordinal regression via **independent binary sub-problems per threshold** |
| "Calibration" reported | A single scalar ECE (10 equal-width bins) on the **collapsed binary referable-DR endpoint** (grade ≥2 vs. not): 0.049 ± 0.001 on APTOS-internal vs. 0.160 ± 0.008 on Messidor-2-external | **Per-threshold** ECE across all 4 ordinal cut-points, decomposed by CORN's independent binary structure |
| "Ordinal thresholds" | Four cut-points on an *expected-grade score* (weighted sum of class probabilities), tuned via coordinate search to maximize **validation QWK** — a decision-boundary placement procedure, not a probability-calibration procedure | Per-threshold ECE + temperature scaling (per-threshold vs. global), directly measuring probability calibration at each CORN sub-problem |
| Evaluation protocol | Single train/val split (APTOS) → one external test set (Messidor-2). Not LODO. | Multi-fold LODO across EyePACS/APTOS/Messidor-2(/DDR) |
| Datasets | APTOS (development), Messidor-2 (external validation only) | EyePACS, APTOS, Messidor-2, DDR |

**Bottom line on the method comparison**: their "ordinal thresholds" and your
"per-threshold calibration" sound similar in English but are different
operations — theirs places decision boundaries to maximize QWK; yours
measures whether predicted probabilities are trustworthy at each of CORN's
independent binary sub-problems. No overlap in mechanism, no overlap in
evaluation protocol. This is not a competitor to your core contribution.

### Why cite it anyway

1. **Real, citable evidence that calibration degrades under cross-dataset
   shift in DR grading specifically** — their own numbers show ECE nearly
   quadrupling (0.049 → 0.160) from in-distribution to external evaluation
   on the same disease, similar datasets. Useful for your introduction/
   motivation section as independent confirmation the problem is real and
   already recognized in the clinical literature, not something you
   invented to justify the paper.
2. **It strengthens your positioning rather than weakening it.** Even a
   paper built explicitly around calibration and cross-dataset threshold
   transfer reports calibration as one aggregate number on a collapsed
   binary problem. That's exactly the blind spot your per-threshold
   diagnosis argues against — you can cite ORDER-DR as evidence that
   *even calibration-focused DR papers* default to scalar/aggregate
   reporting, motivating why finer-grained diagnosis is needed.
3. **Same disease, overlapping dataset (Messidor-2), recent — a reviewer
   working in DR grading may well know this paper.** Citing it and
   correctly differentiating it (rather than omitting it) closes off a
   "did you know about X" reviewer comment before it's raised.

### Suggested citation framing

Position it in your related-work or motivation section along these lines
(paraphrase, don't lift phrasing): recent clinically-oriented work has
also flagged calibration degradation under cross-dataset shift in DR
grading — e.g., ORDER-DR reports a substantial ECE increase from internal
to external validation on Messidor-2 — but such reporting remains at the
level of a single aggregate metric on a binarized referable/non-referable
endpoint, rather than examining calibration structure within an ordinal
model's own decomposition. That is the gap this paper addresses.

### One follow-up worth checking

~~ORDER-DR's own reference [13]~~ — **RESOLVED, see the Poyrazer et al.
entry above.** Full text read; verdict is cite-but-not-a-competitor, same
category as ORDER-DR itself. No further action needed on either paper.

## Notes on using this table

- **Non-negotiable citations**: DRGen and GDRNet/GDRBench. These establish
  that LODO-style evaluation on this dataset pool is an existing, populated
  line of work — omitting them is the most likely single reason a
  knowledgeable reviewer flags a lit-search gap.
- **DECO** strengthens the "we know the current SOTA on this exact
  benchmark" signal — cite alongside GDRNet, same paragraph.
- **El Bellaj et al.** stays your primary point-by-point differentiation
  target, since it's mechanistically closest (ordinal regression + domain
  shift + calibration framing) even though its evaluation protocol is
  weaker than GDRBench-style LODO.
- **PRISM-DR and the phase-augmentation paper** are optional/secondary —
  include if you want a broader related-work section, but don't build any
  core argument on them without reading full text first, since both were
  confirmed only via partial/paywalled search results.
