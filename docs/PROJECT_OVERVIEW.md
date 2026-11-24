# Project Overview

**Background-Aware Prompting for Enhanced Medical Image Segmentation Using SAM**
Srinivas University · B.Tech Computer Science · Academic Year 2026–27 · Project Guide: Mr. Vikramaditya Kumar

> Research prototype only. It is not clinically validated and is not a medical device.

## What the project does

The Segment Anything Model (SAM) [Kirillov2023] segments objects when it is given *prompts*: points,
boxes or masks. This project asks whether SAM can segment the **spleen in abdominal CT slices** using
prompts generated **automatically** from one labelled slice of a *different* patient (the "support"
slice). The question is whether modelling the **background** of that support slice helps.

It contains:

* **A research pipeline** (`src/`): CT preprocessing, a box-prompt Baseline, a background-aware
  "Enhanced" prompting pipeline (BPPC: Background Prompt Prototype Construction; BCM: Background-centric Context Modeling;
  SPR: Structure-guided Prompt Refinement) and metrics.
* **Experiments** on MSD Task09 Spleen [Antonelli2022, Simpson2019]:
  * v1 and v2 runs on 8 query patients;
  * a held-out validation (v3) on 33 patients that adds **A5**, an adaptive rule choosing between
    Baseline and Enhanced, and a **Single-mask** variant.
* **An external evaluation** on an independent dataset, TotalSegmentator [Wasserthal2023], with the
  method frozen. It is **in progress** (see below).
* **An application:** a FastAPI backend and an HTML dashboard for running and comparing segmentations.
  An **exploratory extension**, added as new files, provides:
  * all four methods;
  * prompt overlays;
  * uncertainty maps;
  * click refinement;
  * history with CSV, HTML and PDF export.

## The four evaluated methods

| Method | Short description |
|---|---|
| Baseline | SAM with the support mask's bounding box as the query prompt |
| Enhanced v2 | Baseline box, plus background points from the support background prototype, plus one positive point |
| A5 | Uses Baseline unless its mask area is implausible (outside 0.5–2× the support area), then Enhanced v2 |
| Single-mask | Enhanced v2 prompts, asking SAM for one mask instead of three |

## Status (2026-11-23)

| Work item | Status |
|---|---|
| MSD preprocessing, v1 and v2 runs | Done (`results/`, `results/v2/`) |
| v3 held-out validation (33 patients) | Done (`results/v3/REPORT.md`) |
| External cohort definition and preprocessing (TotalSegmentator, 91 subjects) | Done, independently verified |
| External evaluation | Done: 90 of 90 subjects, independently verified |
| External statistical analysis | Done under the frozen SAP v1 + Amendment 01 (`results/external_ts/analysis/phaseB_20261124T052029Z/`) |
| Dashboard view of research results | Pending |
| Exploratory features (Phase C): uncertainty, click refinement, extended dashboard (all 4 methods), history and export | Implemented as new files; tested with fake models ([EXPLORATORY_FEATURES](EXPLORATORY_FEATURES.md)). Real-SAM studies **[PENDING: v3 exploratory run]** |
| MRI cross-modality check | Not done |

## Results

From the v3 held-out validation (`results/v3/REPORT.md`, 33 patients, 156 slices):

* **A5 vs Baseline:** not shown to be better. +0.024 per-patient mean Dice, 95 % CI [−0.005, +0.055],
  Holm p = 0.29.
* **Enhanced v2 vs Baseline:** significantly worse. −0.165 per-patient mean Dice, Holm p = 6e-4.
* **A5 vs Enhanced v2:** A5 significantly better, Holm p = 3e-5.
* **Single-mask vs Enhanced v2:** no improvement, Holm p = 0.29.

From the external TotalSegmentator evaluation (primary cohort, 74 patients, 826 slices):

* **A5 vs Baseline:** no gain. −0.002 per-patient mean Dice, 95 % CI [−0.021, +0.016], Holm p = 1.0.
* **Enhanced v2 vs Baseline:** significantly worse. −0.163, Holm p = 4e-8.
* **A5 vs Enhanced v2:** A5 significantly better, Holm p = 6e-10.
* **Single-mask vs Enhanced v2:** no difference, Holm p = 1.0.
* The 90-patient sensitivity cohort shows the same pattern (A5 vs Baseline +0.010, Holm p = 0.45).

## Where to read next

[PROBLEM_STATEMENT](PROBLEM_STATEMENT.md) · [SYSTEM_ARCHITECTURE](SYSTEM_ARCHITECTURE.md) ·
[METHODOLOGY](METHODOLOGY.md) · [DATASET_DOCUMENTATION](DATASET_DOCUMENTATION.md) ·
[EXPERIMENTAL_SETUP](EXPERIMENTAL_SETUP.md) · [LIMITATIONS](LIMITATIONS.md) ·
[REPRODUCIBILITY_GUIDE](REPRODUCIBILITY_GUIDE.md) · [REFERENCES](REFERENCES.md)
