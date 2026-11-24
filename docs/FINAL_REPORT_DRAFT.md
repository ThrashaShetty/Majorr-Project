# Background-Aware Prompting for Enhanced Medical Image Segmentation Using SAM

**B.Tech Computer Science — Final-Year Project Report (DRAFT)**
Srinivas University · Academic Year 2026–27 · Project Guide: Mr. Vikramaditya Kumar
Team: Sharanesh S M (01SU23CS174) · Shravya S (01SU23CS181) · Thrasha Shetty (01SU23CS212) · Yashodha K (01SU23CS232)

> **Draft status (2026-11-24).** The external results (§8.3) are now filled in from
> `results/external_ts/analysis/phaseB_20261124T052029Z/analysis.json` (SHA-256 `762d22a7…`).
> * Every number in this draft is measured and taken from the files named beside it.
> * Results that do not exist yet are marked **[PENDING: …]**. Do not replace them with pilot numbers
>   (the single external subject s0965) or estimates.
> * Citations use the keys in [REFERENCES](REFERENCES.md). Items marked *citation pending* must be
>   resolved before submission.

---

## Abstract

The Segment Anything Model (SAM) can segment objects from simple prompts, but on medical images its
accuracy depends heavily on prompt quality, and drawing prompts by hand for every slice is slow. This
project studies **automatic prompt generation for spleen segmentation in abdominal CT**, using one
labelled *support* slice from a different patient, with SAM ViT-B used without fine-tuning.

Four methods were compared:

* **Baseline:** the support mask's bounding box, transferred to the query;
* **Enhanced v2:** a background-aware pipeline. It builds a background prototype from SAM features
  (BPPC), forms a query attention map (Background-centric Context Modeling, BCM), and samples background negative points plus one positive
  point (SPR), within body-mask constraints;
* **A5:** a rule that keeps Baseline unless its mask area is implausible relative to the support;
* **Single-mask:** Enhanced v2 with a single SAM output.

**Internal results.** On a held-out set of 33 MSD Task09 patients (156 slices), with statistics at
patient level:

* Enhanced v2 was **significantly worse** than Baseline (−0.165 mean per-patient Dice, Holm p = 6×10⁻⁴).
* A5 showed a small, **non-significant** gain over Baseline (+0.024, 95 % CI [−0.005, +0.055], Holm
  p = 0.29).

**External evaluation.** A fully frozen external evaluation on the TotalSegmentator dataset (74 primary
and 90 sensitivity patients, from a different institution) was engineered with write-once outputs,
independent verification and a pre-registered statistical analysis plan. A memory optimisation that
leaves results bit-identical cut SAM's peak memory from 3.07 GiB to 1.45 GiB. **External results
(primary cohort, 74 patients):** A5 did **not** improve on Baseline (−0.002 mean per-patient Dice, 95 % CI
[−0.021, +0.016], Holm p = 1.0). Enhanced v2 was again **significantly worse** than Baseline (−0.163,
Holm p = 4×10⁻⁸). The 90-patient sensitivity cohort showed the same pattern.

**Exploratory features.** Uncertainty maps, click-based refinement, an extended dashboard and
experiment-history export were added. Their evaluation is **[PENDING: v3 exploratory run]**.

---

## 1. Introduction

### 1.1 Background
Segmenting organs on CT supports volumetry, treatment planning and follow-up. Manual delineation is
slow and varies between annotators. SAM [Kirillov2023] is a foundation model trained on over a billion
masks of natural images. It segments any object given points, boxes or masks as prompts. Studies on
medical images show that SAM's accuracy varies strongly with the type and placement of the prompt
[Mazurowski2023]. Models fine-tuned for medicine, such as MedSAM [Ma2024], improve accuracy but still
need prompts.

### 1.2 Motivation
If prompts can be generated **automatically** from a single labelled example, SAM becomes a few-shot
segmenter that needs no training. One idea is to use the **background** of the labelled example as well
as its foreground: knowing what is *not* spleen could keep SAM from leaking into neighbouring organs.

### 1.3 Contributions
1. **An automatic support-to-query prompting pipeline** for SAM ViT-B on spleen CT, with three prompt
   strategies (Baseline, Enhanced v2, Single-mask) and a selection rule (A5).
2. **A background-aware prompt generator** (BPPC → BCM → SPR), with body-mask and soft-tissue
   positive-point constraints added in v2 after a diagnosed failure in v1.
3. **An honest, staged evaluation:**
   * development (8 patients);
   * held-out internal validation (33 patients, pre-registered tests);
   * a frozen external evaluation on a different institution's data (TotalSegmentator), with a statistical
     analysis plan frozen before any results.
4. **Verified engineering for CPU-only research:**
   * write-once, resumable evaluation;
   * independent metric recomputation;
   * hash-frozen configuration;
   * a validated, bit-identical memory optimisation of SAM's global attention.
5. **Exploratory tools:** uncertainty maps, interactive click refinement, an extended dashboard with all
   four methods, and experiment history with CSV, HTML and PDF export.

### 1.4 Report structure
* Chapter 2: the problem.
* Chapter 3: related work.
* Chapters 4–6: method, system design and implementation.
* Chapters 7–8: experiments and results.
* Chapters 9–11: discussion, conclusion and future work.

---

## 2. Problem Statement

**Given:**
* a support CT slice with its spleen mask;
* a query CT slice from a **different** patient, whose mask is not available to the method.

**Task:** generate SAM prompts automatically from the support pair and segment the spleen in the query.

**Constraints:**
* SAM ViT-B is used as released, with no fine-tuning.
* The query ground truth is used only for scoring.
* Evaluation uses Dice, IoU, Precision, Recall and Boundary IoU [Dice1945, Cheng2021], summarised
  **per patient**.

**Objectives** (as stated in the project consent form):

1. To implement a few-shot medical image segmentation framework using FoB and SAM.
2. To generate background-focused prompts using BPPC, BCM, and SPR.
3. To develop an adaptive background prompting mechanism for adjusting prompt placement or quantity based on image characteristics.
4. To investigate uncertainty-aware segmentation and highlight potentially unreliable prediction regions.
5. To provide interactive segmentation refinement through positive and negative user prompts.
6. To develop a visualization dashboard for input images, prompts, predicted masks, overlays, and uncertainty maps.
7. To evaluate segmentation performance using Dice Score, IoU, Precision, Recall, Boundary IoU, and inference time.
8. To compare baseline SAM, FoB-enhanced SAM, and the proposed enhanced framework.
9. To maintain experiment history and generate reports for performance analysis.
10. To explore cross-domain generalization across compatible medical imaging datasets, subject to data availability and project feasibility.

The status of each objective is in [OBJECTIVES_STATUS](OBJECTIVES_STATUS.md).

**Research questions** (the evaluation questions used to test objectives 2, 3, 7, 8 and 10):
* **RQ1:** Does background-aware prompting (Enhanced v2) beat simple box transfer (Baseline)?
* **RQ2:** Does the A5 selection rule beat Baseline?
* **RQ3:** Does single-mask output reduce Enhanced v2's over-segmentation?
* **RQ4:** Do the internal findings hold on external data from another institution, with every parameter
  frozen?

**Scope:**
* **In scope:** 2D axial slices, spleen only, CPU inference, research evaluation, a demonstration
  dashboard.
* **Out of scope:** clinical use, 3D segmentation, fine-tuning, multiple organs.

---

## 3. Literature Review

### 3.1 Segment Anything
SAM [Kirillov2023] has three parts:
* an image encoder: a ViT [Dosovitskiy2021], pre-trained with MAE [He2022], with windowed and global
  attention blocks [Li2022];
* a prompt encoder for points, boxes and masks;
* a lightweight mask decoder that returns three candidate masks with predicted-IoU scores, to handle
  ambiguity.

The ViT-B variant used here has 4 global-attention blocks over a 64×64 token grid.

### 3.2 SAM in medical imaging
* **[Mazurowski2023]** evaluated SAM zero-shot across many medical datasets. Performance depended
  strongly on the prompt: boxes generally beat single points, and accuracy varied widely between tasks.
* **MedSAM [Ma2024]** fine-tuned SAM on a large medical corpus and improved box-prompted accuracy.
  It still needs a prompt for each target.

Together these motivate **automatic prompt generation**.

### 3.3 Few-shot and background-aware segmentation
* **Few-shot segmentation:** label a query image using one or a few labelled support examples.
  Prototype methods summarise support features into class prototypes and compare them with query
  features.
* **This project's approach:** build a **background** prototype from SAM's own encoder features and
  use it to place *negative* prompts. The background-prompting idea comes from FoB, "Focus on
  Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric
  Prompting" [Bo2026].
* BPPC, BCM and SPR are a **training-free re-implementation inspired by FoB (Bo et al., CVPR 2026)**.
  FoB trains a background-prompt generator; this project uses frozen SAM ViT-B features and no training,
  and is neither the authors' code nor their model. The results here therefore say nothing about the
  original FoB's performance.

### 3.4 Datasets
* **MSD Task09 Spleen** [Antonelli2022, Simpson2019]: 61 portal-venous CT scans from Memorial Sloan
  Kettering. 41 have public labels.
* **TotalSegmentator** [Wasserthal2023, TS-Zenodo]: routine clinical CT from University Hospital Basel,
  with model-assisted, manually reviewed labels. Its 102-subject subset (v2.0.1) was used here.

### 3.5 Evaluation
* **Overlap metrics:** Dice [Dice1945] and IoU.
* **Boundary quality:** Boundary IoU [Cheng2021].
* **Metric choice:** guided by [Taha2015] and [MaierHein2024].
* **Statistics:** the paired Wilcoxon signed-rank test [Wilcoxon1945], Holm correction [Holm1979],
  and bootstrap confidence intervals [Efron1993].

---

## 4. Methodology

### 4.1 Preprocessing (identical for MSD and TotalSegmentator)
1. Take every 5th axial slice. The external sensitivity cohort also uses every 17th.
2. Apply the HU window [−200, 300] and scale to uint8.
3. Rotate 90° (`np.rot90`, k = 1).
4. Resize to 512×512: bilinear for images, nearest-neighbour for masks.
5. Exclude slices with fewer than 50 spleen pixels after resizing, non-binary labels, or very low
   contrast.

**MSD result:**
* 745 slices sampled, **209 kept**;
* patient-level split with seed 42: 33 support / held-out patients (156 slices) and 8 development
  patients (53 slices) (`data/split_summary.json`).

### 4.2 Support pairing
Each query is paired with the most similar support slice, by cosine similarity of 64×64 zero-mean,
unit-norm thumbnails. The support always comes from a different patient, and the query mask is never
used.

### 4.3 Methods
* **Baseline:** the bounding box of the support mask is the query prompt. SAM returns 3 masks, and the
  one with the highest predicted IoU is kept.
* **Enhanced v2:**
  1. **BPPC:** average the support's SAM encoder features (256×64×64) over the background cells, then
     L2-normalise. This is the background prototype.
  2. **BCM (Background-centric Context Modeling):** compute cosine similarity between each query feature and the prototype, then apply a
     sigmoid. The result is a background-attention map.
  3. **SPR:** negative candidates are where attention > 0.6, outside a 10-px band around the support's
     Canny edges, outside the dilated box, and **inside the patient body mask**. The body mask is grey
     level > 20, largest component, holes filled, eroded by 10 px.
  4. **Adaptive point count:** sampled with seed 42, with count and spacing set by the support's
     foreground ratio: 15 points 8 px apart, 8 points 24 px apart, or 4 points 48 px apart.
  5. **Positive point:** one, at the lowest-attention soft-tissue pixel (grey 90–170) inside the box.
  6. **Prompt:** the box, the negatives and the positive point.
* **A5:** compute ratio = (Baseline mask pixels) / (support mask pixels). If 0.5 ≤ ratio ≤ 2.0, keep
  Baseline; otherwise use Enhanced v2. The bounds were set on the 8 development patients only, then
  frozen. No ground truth is needed. **A5 selects between two fixed prompt types; it does not generate
  new prompts.**
* **Single-mask:** the Enhanced v2 prompts, with `multimask_output=False`.

### 4.4 Metrics
* **Dice** = 2TP / (2TP + FP + FN).
* **IoU** = TP / (TP + FP + FN).
* **Precision** = TP / (TP + FP).
* **Recall** = TP / (TP + FN).
* **Boundary IoU:** IoU of boundary bands of width max(1, round(0.02 × the image diagonal)).
* **Empty masks:** both empty → 1; a zero denominator otherwise → 0.

### 4.5 Statistical analysis
* **Unit of analysis:** the **patient**. Slices from one patient are correlated, so testing per slice
  would overstate the evidence.
* **Test:** per-patient mean Dice, compared with the paired Wilcoxon signed-rank test (differences with
  |d| ≤ 1e-12 dropped).
* **Correction:** Holm over 4 pre-registered comparisons:
  * A5 vs Baseline (primary);
  * A5 vs Enhanced v2;
  * Enhanced v2 vs Baseline;
  * Single-mask vs Enhanced v2.
* **Confidence intervals:** patient-cluster bootstrap, 10,000 resamples, seed 42.
* **External study:** follows **SAP v1**, frozen before any external results, with Amendment 01
  clarifying the descriptive-CI random stream. The primary (74) and sensitivity (90) cohorts are
  separate Holm families.

---

## 5. System Design

See [ARCHITECTURE_DIAGRAM](ARCHITECTURE_DIAGRAM.md) for the full diagram.

* **Research core (`src/`):**
  * preprocessing;
  * `SamEngine`: one SAM ViT-B with an embedding cache and a re-entrant lock;
  * `BaselineSegmentor`;
  * `FoBPipeline` (BPPC, BCM, SPR, adaptive);
  * metrics.
* **Evaluation scripts:**
  * `run_evaluation.py` (v1 and v2);
  * `results/v3/scripts/holdout.py` (v3);
  * `results/external_ts/scripts/` (external).
* **External pipeline:**
  * a frozen cohort manifest;
  * CTs streamed from the ZIP into PNG pairs;
  * exact QC;
  * a frozen run config holding the SHA-256 of 16 code files, the checkpoint and the worklist;
  * write-once slice results with resume;
  * subject completion markers;
  * independent verification;
  * batching with a verified backup after each batch.
* **Application:**
  * the FastAPI backend (`backend/main.py`) with SQLite logging and the original dashboard;
  * an **extended** API (`backend/extensions/`) and dashboard (`frontend/dashboard.html`) serving all
    four methods, uncertainty, refinement and history. These were added as new files, so the frozen
    hashes stay valid.

---

## 6. Implementation

### 6.1 Environment
* Windows 11 laptop with 7.34 GiB RAM.
* Python 3.11.9; PyTorch 2.10.0 (CPU build); segment-anything at commit `dca509fe`.
* Checkpoint `sam_vit_b_01ec64.pth`, SHA-256 `ec2df627…`.

### 6.2 v1 → v2 fix
* **v1 problem:** in v1, Enhanced scored a Dice of only 0.056, because SPR negatives landed in the air
  outside the body, giving whole-body masks.
* **v2 fix:** negatives restricted to the body mask, plus a soft-tissue positive point. Enhanced rose to
  0.332.
* The v1 results were kept unchanged as the "before" evidence.

### 6.3 External-evaluation engineering
* **Streaming:** CTs are read straight from the MD5-verified ZIP and never extracted.
* **QC:** all 1,239 PNG pairs were recomputed exactly by an independent check.
* **Freezing:** `run_config.json` (SHA-256 `3933ca8c…`) records the code hashes, and `worklist.json`
  (`726bb4ee…`) lists 1,078 unique slices. Changing method code would invalidate the run.
* **Write-once and resumable:** each slice's JSON is the commit point, and its prediction file is named
  by its own hash. On resume, completed slices are re-verified by hash and skipped.
* **Resource guard:** RAM of at least 1.6 GiB to start (1.0 GiB under a logged deviation) and 0.3 GiB
  while running; disk of at least 2.0 GiB.
* **Batches:** 9 batches, each followed by a power check, independent verification and a verified
  backup ZIP. Exit codes distinguish resource stops, power loss, verification and backup failures.

### 6.4 Memory optimisation
* **Measured problem:** SAM's peak working set was 3.07 GiB (3.29 GB) on CPU. The cause is the 4
  global-attention blocks: a 12 × 4,096 × 4,096 fp32 attention matrix (about 805 MB), plus copies from
  the relative-position step.
* **Fix:** `mem_opt.py` patches attention **in memory**, without editing any file, to process 512 query
  rows at a time, with the same operations in the same order.
* **Validation:**
  * 7 of 7 synthetic cases are bitwise identical;
  * re-running two already-evaluated real slices gave pixel-identical arrays and identical non-timing
    fields.
* **Effect:** peak working set **1.44–1.45 GiB (about 1.55 GB)**.
* **Runtime:** about **13.2 s per slice** for all four methods.

### 6.5 Exploratory features (Phase C; see [EXPLORATORY_FEATURES](EXPLORATORY_FEATURES.md))
* **Uncertainty:**
  * entropy and disagreement across SAM's 3 masks;
  * variance over K = 8 seeded prompt perturbations (box σ = 5 % of the side, points σ = 5 px);
  * scores averaged over a predicted-mask region dilated by 5 px.
* **Refinement:** SAM re-run with the previous low-resolution logits as `mask_input` plus the clicks. A
  deterministic simulated user clicks the centre of the largest error region.
* **Extended dashboard:**
  * method selector (Baseline / Enhanced v2 / A5 / Single-mask);
  * prompts drawn on the overlay;
  * uncertainty heatmaps;
  * click-to-refine;
  * history with compare and CSV/HTML/PDF export.

### 6.6 Testing
* **Project suite:** 93 tests pass; 2 tests that load the real checkpoint are excluded by default.
* **External-pipeline suite:** 108 tests pass. They cover resume after an interruption, guard refusal,
  batching exit codes and verification.
* **Phase C:** the new API and history tests use fake SAM components.

---

## 7. Experimental Setup

| Experiment | Data | Methods | Status |
|---|---|---|---|
| v1 | MSD, 8 development patients / 53 slices | Baseline, FoB, Enhanced (original) | Done |
| v2 | Same 53 slices | Baseline, FoB, Enhanced v2 | Done |
| v3 | MSD held-out: 33 patients / 156 slices; supports from the other 32 patients | Baseline, Enhanced v2, A5, Single-mask | Done |
| External | TotalSegmentator: primary 74 patients / 826 slices; sensitivity 90 patients / 300 slices; supports = the 156 MSD v3 slices | Same 4, frozen | Done (90 of 90 subjects, independently verified; analysed under SAP v1 + Amendment 01) |
| Exploratory | v3 slices | Uncertainty vs Dice; Dice vs 0–5 clicks (Baseline, A5) | **[PENDING: v3 exploratory run]** |

**External cohorts:**
* 102 subjects in the archive;
* 92 with a spleen mask;
* 91 eligible;
* 17 excluded from the primary cohort because the spleen touches the scan boundary.

**Common settings:** seed 42, CPU, SAM ViT-B, no fine-tuning.

---

## 8. Results

### 8.1 v1 and v2 (development set, 53 slices; slice-level mean Dice)

| Method | v1 | v2 |
|---|---|---|
| Baseline | 0.356 | 0.356 |
| FoB (background points only) | 0.012 | 0.068 (1 recorded failure) |
| Enhanced | 0.056 | 0.332 |

* v2 Enhanced vs Baseline: Wilcoxon p = 0.97, not significant.
* FoB was dropped after v2.

### 8.2 v3 held-out validation (33 patients, 156 slices) — `results/v3/REPORT.md`

**Slice-level results (mean ± SD):**

| Method | Dice | IoU | Precision | Recall | Boundary IoU |
|---|---|---|---|---|---|
| Baseline | 0.515 ± 0.385 | 0.439 ± 0.360 | 0.546 ± 0.398 | 0.580 ± 0.429 | 0.402 ± 0.336 |
| Enhanced v2 | 0.335 ± 0.284 | 0.243 ± 0.248 | 0.260 ± 0.260 | 0.701 ± 0.382 | 0.214 ± 0.230 |
| A5 | 0.529 ± 0.356 | 0.441 ± 0.341 | 0.542 ± 0.381 | 0.678 ± 0.386 | 0.400 ± 0.319 |
| Single-mask | 0.314 ± 0.292 | 0.229 ± 0.253 | 0.266 ± 0.274 | 0.538 ± 0.411 | 0.209 ± 0.236 |

Mean of the 33 per-patient means (Dice): Baseline 0.494 · Enhanced v2 0.329 · A5 0.518 · Single-mask
0.307. An oracle that picks the better of Baseline and Enhanced using the ground truth reaches 0.591, an
upper bound.

**Paired patient-level tests (n = 33):**

| Comparison | Patient-mean ΔDice [95 % CI] | Better / worse / tied | Holm p |
|---|---|---|---|
| **A5 vs Baseline (primary)** | +0.024 [−0.005, +0.055] | 12 / 6 / 15 | **0.289** (not significant) |
| A5 vs Enhanced v2 | +0.189 [+0.125, +0.251] | 26 / 5 / 2 | 3×10⁻⁵ |
| Enhanced v2 vs Baseline | −0.165 [−0.238, −0.090] | 9 / 24 / 0 | 6×10⁻⁴ |
| Single-mask vs Enhanced v2 | −0.022 [−0.053, +0.010] | 11 / 21 / 1 | 0.289 |

**A5 behaviour:**
* A5 kept Baseline on 118 slices and switched to Enhanced v2 on 38.
* It chose correctly on 126 of 156 slices (81 %).
* It made 7 harmful switches, 6 of them with an area ratio below 0.5, typically near the spleen poles.
* It missed 23 opportunities where Enhanced v2 would have been better.

**By Baseline outcome:**
* Where Baseline failed (64 slices with Dice < 0.5), A5 raised the mean Dice from 0.088 to 0.171.
* Where Baseline succeeded (92 slices), A5 lowered it from 0.812 to 0.778.

### 8.3 External evaluation — TotalSegmentator

Source: `results/external_ts/analysis/phaseB_20261124T052029Z/analysis.json`. Inputs were all 90 subjects,
independently verified (`audit/final_verification_20261124_050936.json`: all passed, maximum metric
deviation 0.0). The analysis script first reproduced the published v3 statistics exactly.

**Primary cohort (74 patients, 826 slices). Slice-level mean ± SD; last column is the mean of per-patient
means with its 95 % bootstrap CI:**

| Method | Dice | IoU | Precision | Recall | Boundary IoU | Patient-mean Dice [95 % CI] |
|---|---|---|---|---|---|---|
| Baseline | 0.460 ± 0.383 | 0.384 ± 0.349 | 0.473 ± 0.412 | 0.525 ± 0.417 | 0.345 ± 0.326 | 0.452 [0.401, 0.502] |
| Enhanced v2 | 0.296 ± 0.253 | 0.205 ± 0.214 | 0.219 ± 0.231 | 0.743 ± 0.342 | 0.179 ± 0.197 | 0.289 [0.258, 0.321] |
| A5 | 0.461 ± 0.359 | 0.375 ± 0.331 | 0.460 ± 0.394 | 0.616 ± 0.387 | 0.333 ± 0.307 | 0.450 [0.406, 0.494] |
| Single-mask | 0.296 ± 0.272 | 0.209 ± 0.222 | 0.243 ± 0.250 | 0.514 ± 0.389 | 0.186 ± 0.200 | 0.288 [0.255, 0.322] |

**Pre-registered paired patient-level tests (n = 74; Wilcoxon signed-rank, Holm over 4):**

| Comparison | Patient-mean ΔDice [95 % CI] | Better / worse / tied | Holm p |
|---|---|---|---|
| **A5 vs Baseline (primary)** | −0.002 [−0.021, +0.016] | 23 / 25 / 26 | **1.0** (not significant) |
| A5 vs Enhanced v2 | +0.161 [+0.126, +0.196] | 62 / 11 / 1 | 6×10⁻¹⁰ |
| Enhanced v2 vs Baseline | −0.163 [−0.207, −0.120] | 15 / 58 / 1 | 4×10⁻⁸ (Enhanced v2 **worse**) |
| Single-mask vs Enhanced v2 | −0.000 [−0.017, +0.016] | 33 / 40 / 1 | 1.0 |

**Sensitivity cohort (90 patients, 300 slices; separate Holm family):**

| Method | Dice | IoU | Precision | Recall | Boundary IoU | Patient-mean Dice [95 % CI] |
|---|---|---|---|---|---|---|
| Baseline | 0.484 ± 0.371 | 0.402 ± 0.342 | 0.510 ± 0.403 | 0.548 ± 0.402 | 0.351 ± 0.318 | 0.478 [0.425, 0.530] |
| Enhanced v2 | 0.297 ± 0.246 | 0.204 ± 0.208 | 0.216 ± 0.224 | 0.751 ± 0.341 | 0.172 ± 0.193 | 0.295 [0.263, 0.329] |
| A5 | 0.493 ± 0.346 | 0.400 ± 0.325 | 0.501 ± 0.386 | 0.635 ± 0.368 | 0.346 ± 0.301 | 0.488 [0.440, 0.536] |
| Single-mask | 0.298 ± 0.269 | 0.209 ± 0.220 | 0.250 ± 0.250 | 0.502 ± 0.385 | 0.185 ± 0.198 | 0.294 [0.258, 0.331] |

| Comparison | Patient-mean ΔDice [95 % CI] | Better / worse / tied | Holm p |
|---|---|---|---|
| **A5 vs Baseline** | +0.010 [−0.011, +0.032] | 24 / 13 / 53 | 0.45 (not significant) |
| A5 vs Enhanced v2 | +0.193 [+0.153, +0.232] | 73 / 11 / 6 | 2×10⁻¹¹ |
| Enhanced v2 vs Baseline | −0.183 [−0.231, −0.134] | 22 / 67 / 1 | 1×10⁻⁸ (Enhanced v2 **worse**) |
| Single-mask vs Enhanced v2 | −0.001 [−0.022, +0.020] | 46 / 43 / 1 | 0.89 |

"Tied" means |Δ| ≤ 0.005. The 4 tests in each cohort form one Holm family; the two cohorts are not pooled.

**A5 selection behaviour (exploratory; primary cohort):**
* A5 kept Baseline on 638 slices and switched to Enhanced v2 on 188.
* It chose the better method on 619 of 826 slices (75 %; internally 81 %).
* It made 54 harmful switches and missed 153 opportunities (a difference > 0.01 Dice in each case).
* An oracle that picks the better of Baseline and Enhanced v2 would reach a slice-mean Dice of 0.530, compared
  with 0.461 for A5.
* Where Baseline failed (401 slices, Dice < 0.5), A5 raised the mean Dice from 0.097 to 0.162. Where it
  succeeded (425 slices), A5 lowered it from 0.802 to 0.743. This is the same trade-off as internally.

**Failure modes (exploratory; primary cohort, slice counts out of 826):**

| Method | Dice < 0.1 | Under-segmentation (recall < 0.1) | Over-segmentation (precision < 0.1, recall > 0.9) |
|---|---|---|---|
| Baseline | 294 | 282 | 9 |
| Enhanced v2 | 223 | 110 | 128 |
| Single-mask | 296 | 230 | 33 |

Baseline's failures are almost all boxes that miss the spleen (recall < 0.1). Enhanced v2 fails less often
by missing the spleen, but much more often by over-segmenting.

* **Figures** (in `results/external_ts/figures/phaseB_20261124T052029Z/`, made by
  `scripts/make_phaseB_figures.py`):
  * [Figure 1: fig1_external_primary_dice_box.png]: per-patient mean Dice, primary cohort.
  * [Figure 2: fig2_external_sensitivity_dice_box.png]: per-patient mean Dice, sensitivity cohort.
  * [Figure 3: fig3_external_primary_paired_diff.png]: paired per-patient differences H1–H4, primary cohort.
  * [Figure 4: fig4_external_sensitivity_paired_diff.png]: paired per-patient differences H1–H4, sensitivity
    cohort.

### 8.4 Runtime and memory (engineering measurements, not accuracy results)
* Peak working set: 3.07 GiB (3.29 GB) → **1.44–1.45 GiB (about 1.55 GB)** with mem_opt, which keeps
  results bit-identical.
* About 13.2 s per slice for all 4 methods on CPU. The external run is estimated at about 3.9 h of
  compute.

### 8.5 Exploratory results (NOT confirmatory)
* Spearman ρ between the uncertainty scores (entropy, disagreement, variance) and Dice, at slice and
  patient level, for Baseline and A5: **[PENDING: v3 exploratory run]**
* Mean Dice after 0, 1, 2, 3, 4 and 5 simulated clicks, for Baseline and A5: **[PENDING: v3 exploratory run]**
* Dashboard end-to-end check with real SAM: **[PENDING: dashboard test]**

---

## 9. Discussion

* **RQ1 (background-aware prompting):** not supported internally.
  * BCM separates spleen from other soft tissue only weakly: their background-attention values
    overlap, at about 0.63–0.67 and 0.66–0.68 (LIMITATIONS.md).
  * As a result the positive point is often misplaced, and negative points can cut into the spleen.
  * Recall rises (0.70 vs 0.58) but precision collapses (0.26 vs 0.55): masks are oversized.
* **RQ2 (A5):** the point estimate is small and positive, but the confidence interval includes zero.
  * A5 helps where Baseline fails, costs a little where Baseline succeeds, and is the only variant that
    never falls far below Baseline.
  * The gain seen on the development set (+0.041) shrank on held-out data (+0.024 per patient), which is
    typical of a rule tuned on a small, harder sample.
* **RQ3 (single-mask):** it reduced oversized masks (20 → 8) but lost recall, so it did not help.
* **RQ4 (external):** the internal findings replicated on data from another institution.
  * Enhanced v2 was again significantly worse than Baseline, by −0.163 (internally −0.165). Its precision
    fell to 0.22 and it over-segmented 128 of 826 slices.
  * A5's small internal gain (+0.024, not significant) did not carry over: −0.002 in the primary cohort and
    +0.010 in the sensitivity cohort, neither significant. The narrow CI in the primary cohort,
    [−0.021, +0.016], rules out any gain larger than about 0.02 Dice.
  * All methods scored lower than internally: Baseline's patient-mean Dice was 0.452 against 0.494. This
    matches the difficulties stated before the results (different field of view and voxel size, which move
    where the transferred box lands; labels produced differently; mixed phases and protocols). In 282 of
    826 primary slices, Baseline's mask has recall < 0.1.
  * The 1.0 GiB start-RAM deviation used during the run only changed when slices were scheduled; the outputs
    are identical, and every run-history entry records it.
* **Why the negative results matter:** the project used pre-registration, a held-out set and a frozen
  external protocol, so the negative and inconclusive findings are trustworthy rather than artefacts of
  tuning.

---

## 10. Conclusion

* **Simple box transfer is a strong baseline** for automatic SAM prompting of the spleen.
* **The background-aware prompting implemented here did not improve on it.** The A5 selection rule gives
  at most a small, unproven gain on internal data.
* **External generalisation:** on 74 external patients (primary cohort), A5 was equivalent to Baseline
  (−0.002 Dice, 95 % CI [−0.021, +0.016]), and background-aware prompting (Enhanced v2) was significantly
  worse (−0.163). The pre-registered primary hypothesis is **not supported**. Box transfer remains the
  method to beat.
* **Engineering:** the project contributes a reproducible, verified, memory-efficient CPU evaluation
  pipeline and an exploratory interactive toolset.

---

## 11. Future Work

1. Improve box transfer itself, which is where Baseline fails externally (282 of 826 slices with recall < 0.1).
2. Support selection by anatomical position along the body axis, to target the pole errors.
3. Physical-spacing-aware box transfer, as a new and separately validated method.
4. Stronger background modelling: multiple prototypes, texture features.
5. Comparison with MedSAM [Ma2024] under the same frozen protocol.
6. 3D and inter-slice prompt propagation; slices that contain no spleen.
7. Calibrated uncertainty, and a user study of click refinement with real annotators.
8. A cross-modality check on MRI, for example CHAOS T2-SPIR (not done).
9. GPU inference and batching.

---

## References

See [REFERENCES](REFERENCES.md) for full entries and verification status. Open items before submission:

* **Resolved (2026-11-24):** [Bo2026] (FoB, CVPR 2026) added; [Efron1993] is Chapman & Hall 1993;
  [MaierHein2024] and [Holm1979] are verified.
* **Partially verified:** [Dosovitskiy2021] venue, [Antonelli2022] metadata, [Wasserthal2023] full
  author list.

## Appendices (to assemble)
* **A.** Reproducibility and checksums: [REPRODUCIBILITY_GUIDE](REPRODUCIBILITY_GUIDE.md).
* **B.** SAP v1 and Amendment 01.
* **C.** API reference: [API_DOCUMENTATION](API_DOCUMENTATION.md).
* **D.** Test results (2026-11-24, CPU, `pytest -p no:cacheprovider`):
  * project `tests/`: **93 passed**. The 2 real-checkpoint tests (`test_real_sam_*`) were deselected because
    they need SAM loaded;
  * external evaluation `results/external_ts/scripts/tests/`: **110 passed**, including the analysis
    refusal tests (unfinished evaluation, not finalised, output folder already exists) and the exact
    reproduction of the v3 statistics.
* **E.** Verification reports: `results/external_ts/audit/final_verification_20261124_050936.json` (all 90
  subjects passed), the per-batch `batch_*_verification_*.json` files, the final audit
  `results/external_ts/audit/final_audit_20261124T054443Z.json` (all 19 checks passed), and the verified backup
  `external_ts_tier1_20261124T055116Z.zip` (SHA-256 `b485b254…`).
