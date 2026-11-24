# Slides Outline (15 slides)

* **Placeholders:** **[PENDING: …]** marks a result that doesn't exist yet. Never fill one with numbers
  from the s0965 pilot.
* **Figure paths:** relative to `medical_sam_project/`.
* **Figures marked "to make":** they must be created from verified files. Nothing here is an invented
  plot.

---

### 1. Title
* Background-Aware Prompting for Enhanced Medical Image Segmentation Using SAM.
* B.Tech CSE final-year project · Srinivas University · 2026–27.
* Guide: Mr. Vikramaditya Kumar.
* Team: Sharanesh S M (01SU23CS174) · Shravya S (01SU23CS181) · Thrasha Shetty (01SU23CS212) · Yashodha K (01SU23CS232).
* **Figure:** a sample overlay, for example from `results/v3/adaptive/overlays/`.

### 2. Problem and motivation
* Manual CT organ delineation is slow and varies between annotators.
* SAM segments from prompts, but its medical accuracy depends on prompt quality [Mazurowski2023].
* **Goal:** prompts generated automatically from one labelled slice of another patient, with no
  fine-tuning.
* **Figure:** a support slice and mask → arrow → a query slice.

### 3. Objectives and research questions
* **Objectives** (consent form, shortened for the slide; full wording in `docs/CONSENT_FORM_SUMMARY.md`):
  1. A few-shot segmentation framework using FoB and SAM.
  2. Background-focused prompts (BPPC, BCM, SPR).
  3. Adaptive background prompting (placement and quantity).
  4. Uncertainty-aware segmentation.
  5. Interactive refinement with positive and negative prompts.
  6. A visualisation dashboard.
  7. Evaluation: Dice, IoU, Precision, Recall, Boundary IoU, inference time.
  8. Comparison: baseline SAM vs FoB-enhanced SAM vs the proposed framework.
  9. Experiment history and reports.
  10. Cross-domain generalisation (as feasible).
* **Research questions:**
* RQ1: does background-aware prompting beat box transfer?
* RQ2: does the A5 rule beat Baseline?
* RQ3: does a single-mask output help?
* RQ4: do the findings hold on external data from another institution? (Yes: the negative findings replicate.)
* **Figure:** none (text slide).

### 4. Background: SAM ViT-B
* Image encoder (ViT-B, 1024² → 256×64×64), prompt encoder, mask decoder (3 masks + IoU scores).
* Used zero-shot on CPU.
* **Figure:** a simplified SAM block diagram (to make; cite [Kirillov2023]).

### 5. Datasets
* **MSD Task09 Spleen:** 41 labelled CTs; 209 slices kept; split into 8 development and 33 held-out
  patients.
* **TotalSegmentator (Basel):** 91 eligible subjects; primary 74 patients / 826 slices; sensitivity 90
  patients / 300 slices.
* **Preprocessing:** HU window [−200, 300]; every 5th slice; 512×512; at least 50 spleen pixels.
* **Figure:** a table from `docs/DATASET_DOCUMENTATION.md`.

### 6. System architecture
* Preprocessing → SamEngine → 4 methods → metrics → statistics.
* Exploratory extensions → FastAPI → dashboard → history and export.
* **Figure:** the Mermaid diagram in `docs/ARCHITECTURE_DIAGRAM.md` (export it as PNG from mermaid.live).

### 7. Methods: Baseline and Enhanced v2
* **Baseline:** the support box, transferred to the query.
* **Enhanced v2:** BPPC (background prototype) → BCM (Background-centric Context Modeling: attention map) → SPR (negatives inside the body,
  plus 1 soft-tissue positive).
* **Figure:** a query overlay with box, negative and positive points (the `/ext/segment` overlay, or v3
  overlays).

### 8. Methods: A5 and Single-mask
* **A5:** ratio = Baseline mask area / support area. If 0.5–2.0, keep Baseline; otherwise use Enhanced v2.
* **Honest framing:** A5 is a **selection rule**, not adaptive prompt generation.
* **Single-mask:** Enhanced v2 prompts with one SAM output.
* **Figure:** an A5 decision flowchart (to make).

### 9. Evaluation protocol
* Development (8 patients) → held-out v3 (33 patients) → frozen external evaluation.
* Patient-level Wilcoxon; Holm over 4 tests; cluster bootstrap 10k (seed 42).
* SAP v1 frozen before any external results; write-once outputs; independent verification.
* **Figure:** a timeline of the protocol stages (to make).

### 10. Results I: development and v1 → v2 fix
* **v1:** Enhanced Dice 0.056; negatives fell in the air, giving whole-body masks.
* **v2 (body-mask fix):** Enhanced 0.332 vs Baseline 0.356 (p = 0.97).
* **Figure:** `results/charts/metrics_comparison.png` (v1) next to `results/v2/charts/metrics_comparison.png` (v2).

### 11. Results II: v3 held-out (33 patients)
* Per-patient mean Dice: Baseline 0.494 · Enhanced v2 0.329 · **A5 0.518** · Single-mask 0.307.
* A5 vs Baseline: +0.024 [−0.005, +0.055], Holm p = 0.29, **not significant**.
* Enhanced v2 vs Baseline: −0.165, Holm p = 6×10⁻⁴, **significantly worse**.
* **Figure:** [Figure 5: fig5_v3_holdout_dice_box.png] (in `results/external_ts/figures/slides_v3/`, made from
  `results/v3/raw_holdout_rows.csv`; its means match `results/v3/adaptive/summary.json`), plus the paired-test table.

### 12. Results III: external TotalSegmentator
* **Primary cohort (74 patients, 826 slices).** Per-patient mean Dice: Baseline 0.452 · Enhanced v2 0.289 ·
  A5 0.450 · Single-mask 0.288.
  * A5 vs Baseline: −0.002 [−0.021, +0.016], Holm p = 1.0, **not significant** (pre-registered primary
    hypothesis not supported).
  * Enhanced v2 vs Baseline: −0.163, Holm p = 4×10⁻⁸, **significantly worse**.
* **Sensitivity cohort (90 patients, 300 slices):** the same pattern. A5 vs Baseline +0.010 (Holm
  p = 0.45); Enhanced v2 vs Baseline −0.183.
* Message: the internal findings replicate at another institution; A5's small internal gain does not.
* **Figure:** [Figure 1: fig1_external_primary_dice_box.png] and [Figure 3: fig3_external_primary_paired_diff.png]
  (in `results/external_ts/figures/phaseB_20261124T052029Z/`).

### 13. Failure analysis and engineering
* **A5:** 7 harmful switches (spleen poles; ratio < 0.5) and 23 missed opportunities.
* **Enhanced v2:** over-segmentation (recall 0.70, precision 0.26).
* **External (primary):** Baseline mostly fails by missing the spleen (282 of 826 slices with recall < 0.1);
  Enhanced v2 over-segments 128 slices. A5 picked the better method on 75 % of slices (81 % internally).
* **Memory:** peak 3.07 → 1.45 GiB with mem_opt, bit-identical; about 13.2 s per slice on CPU.
* **Figure:** v3 failure overlays (for example case_018_slice_075), plus a memory bar chart (to make, from
  `results/external_ts/audit/mem_opt_validation.json`).

### 14. Exploratory features (not confirmatory)
* Uncertainty maps: multimask entropy and disagreement, plus K-perturbation variance.
* Click refinement: SAM with `mask_input` plus clicks; a simulated user for evaluation.
* Dashboard: all 4 methods, prompts drawn, uncertainty, click-to-refine, history, CSV/HTML/PDF export.
* Uncertainty vs Dice: **[PENDING: v3 exploratory run]**. Dice vs clicks: **[PENDING: v3 exploratory run]**.
* **Figure:** a dashboard screenshot **[PENDING: dashboard test]**.

### 15. Limitations, conclusion and future work
* **Limitations:**
  * 2D only;
  * spleen only;
  * CT only (no MRI);
  * box transfer without registration;
  * slice selection uses the ground truth;
  * CPU only.
* **Conclusion:** box transfer is a strong baseline; background-aware prompting did not beat it; A5
  gives at most a small, unproven gain internally and none externally (74 patients: −0.002 Dice).
* **Future work:** support selection by body-axis position; spacing-aware boxes; a MedSAM comparison;
  3D propagation; MRI.
* **Figure:** none.
