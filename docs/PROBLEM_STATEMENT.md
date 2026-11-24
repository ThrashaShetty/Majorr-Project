# Problem Statement

## Context

Manual delineation of organs on CT is slow and varies between annotators. Foundation segmentation
models such as SAM [Kirillov2023] can segment objects from simple prompts, but they were trained on
natural images. Their medical performance depends heavily on how good the prompts are
[Mazurowski2023, Ma2024]. Producing prompts by hand for every slice removes much of the benefit.

## Problem

Given:

* a **support** axial CT slice with its spleen mask;
* a **query** axial CT slice from a **different patient**, without its mask;

automatically generate SAM prompts from the support pair and segment the spleen in the query.

The constraints:

* SAM ViT-B is used **as is**. No fine-tuning is done.
* The prompts must be derived from the support pair alone. The query's ground truth may be used only
  for scoring.
* Performance is measured per slice and summarised **per patient**, with Dice, IoU, Precision,
  Recall and Boundary IoU [Dice1945, Cheng2021].

## Research questions

1. **RQ1:** Does background-aware prompting (Enhanced v2) improve on simple box transfer (Baseline)?
2. **RQ2:** Does an adaptive rule (A5) that falls back to Enhanced v2 only when the Baseline mask area
   looks implausible improve on Baseline?
3. **RQ3:** Does a single-mask output reduce Enhanced v2's over-segmentation?
4. **RQ4:** Do the internal findings hold on an **external** dataset from another institution,
   TotalSegmentator [Wasserthal2023], with every method parameter frozen?

**Answers available so far:** RQ1–RQ3 on internal held-out data (`results/v3/REPORT.md`). RQ4 on the
external TotalSegmentator cohorts (`results/external_ts/analysis/phaseB_20261124T052029Z/`): the internal
findings hold. Enhanced v2 is significantly worse than Baseline, and A5 shows no gain.

## Scope and non-goals

* **In scope:** 2D axial slices, the spleen, CPU inference, research evaluation, a demonstration dashboard.
* **Out of scope:**
  * clinical use;
  * 3D volumetric segmentation;
  * fine-tuning SAM;
  * multi-organ segmentation;
  * real-time performance guarantees.
