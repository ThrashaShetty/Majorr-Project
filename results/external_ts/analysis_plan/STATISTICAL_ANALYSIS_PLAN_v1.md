# Statistical Analysis Plan (SAP) v1 — TotalSegmentator external evaluation

**Status:** FROZEN before the external cohort is analysed. This file is never edited. Its SHA-256 is stored
in `STATISTICAL_ANALYSIS_PLAN_v1.sha256` next to it. Any later change is made only in a separate,
dated amendment file that states what changed and why.

**Frozen on:** 2026-11-23 (local time, UTC+05:30).

## 0. Disclosure of what had been seen when this plan was written

* **Seen:** the results for **one** primary subject, **s0965** (16 slices). It was the SAM pilot used to test
  the evaluation infrastructure and the memory optimisation. Mean Dice over its 13 primary slices:
  Baseline 0.525, Enhanced v2 0.209, A5 0.525, Single-mask 0.231.
* **Not seen:** the other 89 subjects and 1,062 slices had not been evaluated.
* **Where the plan comes from:** the methods, tests, correction, bootstrap and seed are copied from the
  plan used for the internal v3 held-out validation. That plan was written before the v3 results
  (`results/v3/scripts/v3_report.py`, `results/v3/REPORT.md`). Nothing here was chosen or tuned using
  external data.
* **Frozen before this plan:** all method parameters were frozen before any external inference.
  Run config SHA-256: `3933ca8c48374cc9fa64bb137b1bed79b5c8551f92a267bf1adc98678dc64cca`.

## 1. Objective and hypotheses

Measure how the four frozen methods transfer to an external CT dataset from a different institution
(TotalSegmentator, University Hospital Basel). The MSD Task09 Spleen support pool is used, with no
re-tuning.

Methods (the exact v3 implementations):

| Method | Definition |
|---|---|
| **Baseline** | SAM ViT-B, box prompt = bounding box of the paired support mask, multimask output, best predicted IoU |
| **Enhanced v2** | `src/fob/fob_pipeline.py`, `mode="enhanced"`, `uncertainty="none"`, adaptive points, seed 42 |
| **A5** | Baseline if 0.5 ≤ (Baseline mask px / support mask px) ≤ 2.0, else Enhanced v2. The bounds are fixed and were never tuned |
| **Single-mask** | Enhanced v2 prompts with `multimask_output=False` |

"v3" is the earlier internal held-out validation experiment. It is **not** a method.

**Primary hypothesis (H1):** A5 differs from Baseline in Dice in the primary cohort, measured per patient.

**Secondary hypotheses**, the same family as v3:

* H2: A5 vs Enhanced v2.
* H3: Enhanced v2 vs Baseline.
* H4: Single-mask vs Enhanced v2.

## 2. Data, cohorts and the unit of analysis

Frozen inputs:

| Input | SHA-256 |
|---|---|
| Manifest (`results/external_ts/cohort_manifest.json`) | `b6de738e9f1e168d6e7f5e3c53997efb6b5412bb948a69261d1d0548e565ca26` |
| Worklist (`results/external_ts/eval/worklist.json`) | `726bb4ee1c23154e6e92910ab6b059a8a72ae877991b8f332b0d75975c668041` |
| Preprocessing QC (`results/external_ts/qc/full_qc.json`) | `56629d880320e6b6196f77192f46a30b4203f4cffb77bc74cea63129d35d8f22` |
| Checkpoint `sam_vit_b_01ec64.pth` | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |

Cohort counts, reconciled against the manifest and the worklist:

| Cohort | Rule (frozen) | Subjects | Evaluable subjects | Slices |
|---|---|---|---|---|
| **Primary** | Eligible subjects whose spleen mask touches neither axial end of the scan; every 5th slice from z = 0; slice kept if ≥ 50 spleen px after the existing 512×512 resize | 74 | 74 | 826 |
| **Sensitivity** | All 91 eligible subjects (17 of them boundary-touching); every 17th slice from z = 0; same slice rule | 91 | **90** (s1358 has 0 eligible step-17 slices) | 300 |

* **Overlap:** 48 slices are in both cohorts. Each is evaluated once and tagged with both cohort flags.
* **Unique evaluated slices:** 1,078.
* **Not in either analysis:** 161 step-5 slices of boundary-touching subjects. They were preprocessed but must
  not enter any analysis without a new approval.

**Unit of analysis: the patient (subject).** Slices from the same patient are correlated and are never
treated as independent observations in a test.

* **Per-patient value:** the unweighted mean over that patient's slices **in the cohort being analysed**.
* **Cohort statistic:** the unweighted mean of the per-patient values, so every patient has equal weight.

The primary and sensitivity cohorts overlap (the same patients and some of the same slices). They are
**not independent** and are never pooled or compared with each other by a statistical test.

## 3. Inclusion, exclusion and missing data

1. **When the analysis may run:** only on results that passed independent verification
   (`verify_eval_subjects.py`: every metric recomputed from the saved predictions, hashes checked, every
   slice present).
2. **No post-hoc exclusions.** Every slice in the cohort's worklist is analysed:
   * an empty prediction is kept and scored with the metric conventions in §5;
   * low-Dice slices and method collapses are kept.
3. **If evaluation is incomplete:** an incomplete subject (no completion marker, or failed verification)
   is left out of the patient-level analysis, listed by name, and the analysis is labelled
   **incomplete**. Nothing is imputed.
4. **If the final n differs** from the counts in §2, the report states the actual n and the reason.
5. **No outlier removal, re-pairing or re-thresholding.**

## 4. Primary analysis (confirmatory)

* **Population:** the primary cohort (n = 74 patients).
* **Endpoint:** the per-patient mean Dice difference, A5 − Baseline.
* **Test:** two-sided paired Wilcoxon signed-rank test on the 74 per-patient differences.
  * Differences with |d| ≤ 1e-12 are removed before the test, exactly as in v3.
  * `scipy.stats.wilcoxon` (SciPy 1.16.2) is then called with its defaults.
  * If no non-zero difference remains, p = 1.0.
* **Multiplicity:** Holm–Bonferroni over the family of 4 tests (H1–H4) at a family-wise α = 0.05.
* **Reported:**
  * the patient-mean ΔDice with a 95 % bootstrap CI;
  * the slice-pooled ΔDice with a 95 % patient-cluster bootstrap CI;
  * counts of patients better / worse / tied (tied = |Δ| ≤ 0.005);
  * raw Wilcoxon p and Holm-adjusted p.

## 5. Metrics

* **Dice** is the endpoint for every test.
* **IoU, Precision, Recall and Boundary IoU** are secondary and **descriptive only**: no p-values.
* All metrics use the project definitions in `src/metrics.py`:
  * Boundary IoU band width = max(1, round(0.02 × image diagonal));
  * both masks empty → 1.0;
  * zero denominator with a non-empty mask → 0.0.

## 6. Secondary analyses

* **H2–H4:** the same procedure as §4, on the primary cohort, inside the same Holm family.
* **Sensitivity cohort:** all four tests repeated on the 90 evaluable sensitivity patients. They form a
  **separate** Holm family of 4, are labelled *sensitivity*, and are not confirmatory.

## 7. Bootstrap (confidence intervals)

* **Generator:** a single `numpy.random.default_rng(42)` per cohort.
* **Order of comparisons:** H1, H2, H3, H4.
* **Resamples:** each comparison draws a fresh index matrix of shape (10,000 × n_patients), patients
  resampled with replacement. This matches `v3_report.py`.
* **Patient-mean ΔDice CI:** the 2.5th and 97.5th percentiles of the mean per-patient difference.
* **Slice-pooled ΔDice CI:** for each resample, (sum of slice differences over the resampled patients) /
  (number of slices over the resampled patients), then the same percentiles.
* **Descriptive metric CIs:** the same scheme (seed 42, 10,000 resamples) for each method's patient-level
  mean.

## 8. Descriptive statistics, reported per cohort and method, with n

* **Slice level:** mean, standard deviation (population SD, ddof = 0, as in v3), median, IQR, min, max.
* **Patient level:** mean of per-patient means, median, SD.
* **Counts:** n slices and n patients.
* **Time:**
  * the reported `*_ms` fields (v3 semantics: a cached query encoding reports its originally measured time);
  * separately, the measured wall-clock time per slice from the run logs.
* **Per subject:** the full table of per-patient means, plus per-patient variation (SD across slices).

## 9. Exploratory analyses (descriptive, no confirmatory claims)

1. **Baseline success / failure split** (Baseline Dice ≥ 0.5 vs < 0.5): mean Dice of every method in each part.
2. **A5 selection behaviour**, using the v3 definitions:
   * number of box-only vs Enhanced-activated slices;
   * "correct" = |Enhanced − Baseline| ≤ 0.01, or A5 chose the better of the two;
   * missed opportunities (kept box, Enhanced better by > 0.01);
   * harmful activations (chose Enhanced, Baseline better by > 0.01);
   * the oracle upper bound (mean of the better of Baseline and Enhanced).
3. **Failure modes:**
   * empty predictions per method;
   * Dice < 0.1;
   * over-segmentation (precision < 0.1 and recall > 0.9);
   * under-segmentation (recall < 0.1 with a non-empty prediction).
4. **Strata** (descriptive only): TotalSegmentator `study_type`; boundary-touching vs non-boundary
   (sensitivity cohort only); slice position (pole vs mid-spleen, by rank within the patient).
5. **Context:** a descriptive comparison with the internal v3 held-out results. The populations differ,
   so there is no pooled or cross-dataset test.

## 10. Rule for choosing visual examples (fixed in advance, to prevent cherry-picking)

Ranking is by per-patient A5 Dice in the primary cohort. Overlays are drawn for:

* the 3 highest-ranked, the 3 median and the 3 lowest-ranked patients, using each patient's middle slice
  by z-order;
* every harmful A5 activation. If there are more than 10, a random sample of 10 is drawn with seed 42.

All four methods are shown on the same slice, with the ground-truth contour.

## 11. Interpretation rules

* **"Statistically supported difference":** Holm-adjusted p < 0.05 **and** a 95 % patient-mean CI that
  excludes 0.
* **Otherwise:** "no difference shown". This is **not** evidence of equivalence, because no equivalence
  margin was pre-specified.
* **Sensitivity results** can support or qualify the primary conclusions, but cannot replace them.
* **Exploratory findings** are labelled as such and generate hypotheses only.
* **Generalisation:** the external results apply to TotalSegmentator v2.0.1 data under this frozen
  protocol. No clinical claim is made.

## 12. Software

Python 3.11.9 · numpy 2.2.6 · scipy 1.16.2 · pandas 2.3.3 · Windows 11, CPU. The analysis script will
be a new file. It reads only verified outputs and writes to a new, timestamped results folder.
