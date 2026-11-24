# Objectives Status

Status as of **2026-11-24** (external results added). The objectives are quoted exactly from the project consent form (see
[CONSENT_FORM_SUMMARY](CONSENT_FORM_SUMMARY.md)).

Key: **Done**, **Partial**, **Pending** (implemented, waiting for a run that needs SAM and RAM) and
**Not done**.

| # | Objective (consent form) | Implementation files | Evidence / results | Status |
|---|---|---|---|---|
| 1 | To implement a few-shot medical image segmentation framework using FoB and SAM. | `src/baseline.py` (SamEngine, Baseline), `src/fob/` (FoBPipeline), `run_evaluation.py` (support–query pairing), `src/preprocessing.py` | One labelled support slice from another patient prompts SAM ViT-B on the query, with no fine-tuning and no manual prompts. Run end to end in v1/v2 (53 slices) and v3 (156 slices) | **Done** |
| 2 | To generate background-focused prompts using BPPC, BCM, and SPR. | `src/fob/bppc.py`, `bcm.py`, `spr.py`, `fob_pipeline.py` (Enhanced v2) | Implemented: background prototype → attention map → negative points, with body-mask and positive-point constraints in v2. **The hypothesis that it helps is not supported:** Enhanced v2 is significantly worse than Baseline on held-out MSD (−0.165 per-patient mean Dice, Holm p = 6e-4) and on external TotalSegmentator (−0.163, Holm p = 4e-8, 74 patients) | **Done** (implemented and evaluated; negative result) |
| 3 | To develop an adaptive background prompting mechanism for adjusting prompt placement or quantity based on image characteristics. | `src/fob/adaptive.py`; the A5 rule in `results/v3/scripts/holdout.py` and the external runner | `adaptive.py` sets the **number and spacing** of background points from the support's foreground ratio: 15 points / 8 px, 8 points / 24 px, or 4 points / 48 px. This is a fixed rule, not learned, and it uses support characteristics, not the query's. A5 is a **selection rule** between two fixed prompt types, based on mask area; it does not generate new prompts. A5 vs Baseline: +0.024 per-patient mean Dice, 95 % CI [−0.005, +0.055], Holm p = 0.29 (not significant); externally −0.002, 95 % CI [−0.021, +0.016], Holm p = 1.0 (no gain) | **Partial.** Rule-based adaptation only; no significant benefit shown |
| 4 | To investigate uncertainty-aware segmentation and highlight potentially unreliable prediction regions. | `src/extensions/uncertainty.py`; `tests/test_ext_uncertainty.py`; `/ext/segment` heatmaps | Multimask entropy and disagreement, plus K-perturbation variance; heatmap and scalar score per slice. 8 synthetic tests pass. MC Dropout is impossible (SAM ViT-B has no dropout). Whether the score predicts Dice: **[PENDING: v3 exploratory run]** | **Partial.** Implemented; validation pending |
| 5 | To provide interactive segmentation refinement through positive and negative user prompts. | `src/extensions/refinement.py`; `/ext/refine`, `/ext/reset`; `tests/test_ext_refinement.py`, `test_ext_api.py` | SAM re-run with `mask_input` plus positive and negative clicks; a deterministic simulated user. 8 + 13 fake-model tests pass. Dice vs clicks (Baseline, A5): **[PENDING: v3 exploratory run]** | **Partial.** Implemented; validation pending |
| 6 | To develop a visualization dashboard for input images, prompts, predicted masks, overlays, and uncertainty maps. | `backend/extensions/`, `frontend/dashboard.html` (original `backend/main.py` and `frontend/index.html` unchanged) | Shows the input and support images, the prompts (box, positive and negative points), the predicted mask, the overlay, the uncertainty heatmaps and the metrics, with a 4-method selector and click-to-refine. Endpoints tested with fakes; JavaScript syntax-checked. Real-SAM end-to-end check: **[PENDING: dashboard test]** | **Partial.** Built; real-model test pending |
| 7 | To evaluate segmentation performance using Dice Score, IoU, Precision, Recall, Boundary IoU, and inference time. | `src/metrics.py`; v3 analysis; `results/external_ts/scripts/analyze_external_ts.py`; SAP v1 + Amendment 01 | Internal: all 5 metrics plus time reported for v1, v2 and v3. v3 median time per slice: Baseline 9.3 s, Enhanced v2 18.8 s, A5 9.9 s. Patient-level statistics. External: all 5 metrics for 4 methods, two cohorts (74 and 90 patients), pre-registered patient-level tests (`results/external_ts/analysis/phaseB_20261124T052029Z/`) | **Done** |
| 8 | To compare baseline SAM, FoB-enhanced SAM, and the proposed enhanced framework. | `run_evaluation.py` (v1/v2), `results/v3/scripts/holdout.py` (v3), the external runner | v1/v2 (53 slices): Baseline 0.356, FoB 0.012 → 0.068, Enhanced 0.056 → 0.332. v3 (33 patients): Baseline vs Enhanced v2 vs A5 vs Single-mask, with paired tests. FoB (background points only) was dropped after v2 because of very low Dice. External comparison of the 4 frozen methods (74 patients): Baseline 0.452 · Enhanced v2 0.289 · A5 0.450 · Single-mask 0.288 per-patient mean Dice | **Done** |
| 9 | To maintain experiment history and generate reports for performance analysis. | `backend/main.py` (original SQLite log); `backend/extensions/history.py`, `report.py`; `/ext/history*` | SQLite history, compare with deltas, and CSV, HTML and PDF export. 8 tests pass. Research reports: `results/v3/REPORT.md` and the analysis scripts | **Done** (code and tests). Not yet used on real dashboard runs |
| 10 | To explore cross-domain generalization across compatible medical imaging datasets, subject to data availability and project feasibility. | `results/external_ts/` (manifest, preprocessing, QC, frozen runner, mem_opt, batches) | **CT → CT across institutions:** MSD (Memorial Sloan Kettering) → TotalSegmentator (Basel). 91 subjects preprocessed with exact QC; all 90 evaluable subjects evaluated and independently verified. The internal findings replicate (Enhanced v2 worse; A5 ≈ Baseline); all methods score lower than internally (Baseline 0.452 vs 0.494). **No MRI:** the optional CHAOS T2-SPIR check was not done, so no cross-modality result | **Partial.** Cross-institution CT done; cross-modality not done |

## Honest summary

* **Delivered and evaluated:**
  * the few-shot SAM framework (1);
  * background-focused prompting (2), a negative result;
  * rule-based adaptive prompting (3), no significant gain;
  * internal and external evaluation and comparison (7, 8);
  * cross-institution CT generalisation (10), where the negative findings replicate.
* **Implemented, awaiting real-SAM validation:**
  * uncertainty (4);
  * interactive refinement (5);
  * the dashboard (6).
* **Done in code:** history and reports (9).
* **Not done:** cross-modality (MRI).
* **Wording caution:**
  * Objective 3 is met by **rule-based** adaptation: point count and spacing from the support's
    foreground ratio, plus A5 selection. Do not describe it as learned or generative adaptive prompting.
  * Objective 10 is met for **CT → CT** only.
