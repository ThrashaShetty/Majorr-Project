# Methodology

Everything below is taken from the implementation (`src/`, `results/v3/scripts/holdout.py`,
`results/external_ts/scripts/eval_external_ts.py`).

## 1. Few-shot setting

* Each query slice is paired with **one** support slice from a different patient.
* **Pairing** is by nearest neighbour: cosine similarity of 64×64 zero-mean, unit-norm thumbnails of
  the images (`run_evaluation._thumb`).
* **Only the query image** is used for pairing, never its mask.

## 2. Preprocessing (identical for MSD and TotalSegmentator)

1. Every 5th axial slice (array axis 2, starting at z = 0). The external sensitivity cohort also uses a
   step-17 grid.
2. Clip to [−200, 300] Hounsfield units (HU), scale to 0–255, convert to uint8.
3. Rotate 90° (`np.rot90`, k = 1), applying the same rotation to the mask.
4. Resize to 512×512: bilinear for images, nearest-neighbour for masks.
5. A slice is excluded if:
   * the shapes don't match;
   * the labels aren't binary;
   * the image standard deviation is below 1 (low contrast);
   * it has fewer than 50 spleen pixels after resizing.

## 3. Methods

### Baseline
Prompt = the bounding box of the support mask, applied unchanged to the query. SAM returns 3 masks;
the one with the highest predicted IoU is kept.

### Enhanced v2 (background-aware)

1. **BPPC (Background Prompt Prototype Construction).** Encode the support slice with SAM's image encoder (256×64×64). Average the features over
   the background cells (mask = 0, inside the un-padded area) and L2-normalise. The result is the
   **background prototype**.
2. **BCM (Background-centric Context Modeling).** Compute the cosine similarity between each query feature and the prototype, then apply a
   sigmoid. The result is an attention map: high means background-like.
3. **SPR (Structure-guided Prompt Refinement).** Candidate negative points are where attention > 0.6, outside a 10-px exclusion band around
   the support image's Canny edges. They're sampled with seed 42 and a minimum spacing set by the
   adaptive rule:
   * fewer than 5 % foreground → 15 points, 8 px apart;
   * fewer than 20 % → 8 points, 24 px apart;
   * otherwise → 4 points, 48 px apart.
4. **Constraints added in v2:**
   * negatives must lie outside the dilated box **and inside the patient's body** (grey level > 20,
     largest connected component, holes filled, eroded by 10 px);
   * one positive point is placed at the lowest-attention soft-tissue pixel (grey 90–170) inside the box.
5. **Prompt:** the support box + the negatives + the positive point. SAM returns 3 masks; the best is kept.

### A5 (adaptive selection)
Compute ratio = (Baseline mask pixels) / (support mask pixels).

* **If 0.5 ≤ ratio ≤ 2.0:** keep Baseline.
* **Otherwise:** use Enhanced v2.

The bounds were chosen on the 8 **development** patients and then frozen. They were never tuned on the
held-out or external data. The rule needs no ground truth.

### Single-mask
The same prompts as Enhanced v2, but SAM is asked for one mask (`multimask_output=False`).

### FoB (experimental, v1/v2 only)
Background points only, with no box. It is kept in the API for comparison but was not carried into v3
or the external evaluation, because its Dice was very low (0.068 in v2).

## 4. Order of operations per query (v3 and external, identical)

1. `invalidate` clears the embedding cache, so the query is encoded fresh.
2. Baseline runs.
3. Enhanced v2 runs, reusing the cached query embedding.
4. Single-mask decodes from Enhanced v2's prompts.
5. A5 is derived from the Baseline and Enhanced results.

SPR re-seeds on every call, and the caches are keyed by content, so the results don't depend on the
order in which queries are processed.

## 5. Metrics [Dice1945, Cheng2021, Taha2015, MaierHein2024]

| Metric | Definition |
|---|---|
| Dice | 2TP / (2TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Boundary IoU | IoU of inner boundary bands of width d = max(1, round(0.02 · diagonal)), obtained by erosion |

Empty-mask conventions:

* both masks empty → 1.0;
* zero denominator with the other mask non-empty → 0.0.

## 6. Statistics

* **Unit:** the patient.
* **Test:** paired Wilcoxon signed-rank [Wilcoxon1945] on per-patient mean Dice.
* **Correction:** Holm [Holm1979] over 4 pre-registered tests.
* **Confidence intervals:** patient-cluster bootstrap [Efron1993], 10,000 resamples, seed 42.

The external statistical analysis plan (SAP) is frozen in
`results/external_ts/analysis_plan/STATISTICAL_ANALYSIS_PLAN_v1.md` (SHA-256 `6a1fee26…`). Its §7 bootstrap
stream is clarified by [SAP_AMENDMENT_01_2026-11-23](SAP_AMENDMENT_01_2026-11-23.md).

## 7. Memory optimisation (external evaluation only; computation unchanged)

* **The problem:** SAM's 4 global-attention blocks materialise a 12 × 4,096 × 4,096 matrix, plus copies
  from the relative-position step, about 2.4 GB at once.
* **The fix:** `mem_opt.py` computes the same per-row operations in the same order, in blocks of 512 rows.
* **Validation:**
  * synthetic tests: bitwise identical in 7 of 7 cases;
  * real re-evaluation of two completed slices: all 7 arrays pixel-identical, all non-timing fields equal.
* **Effect:** peak working set fell from 3.07 GiB (3.29 GB) to 1.44–1.45 GiB (about 1.55 GB), in two measured runs.
