# Consent Form Summary

A copy of the project consent form's key content, for reference within the project documentation.

## Title
**Background-Aware Prompting for Enhanced Medical Image Segmentation Using SAM**

## Project team

| Role | Name | USN |
|---|---|---|
| Guide | Mr. Vikramaditya Kumar | — |
| Student | Sharanesh S M | 01SU23CS174 |
| Student | Shravya S | 01SU23CS181 |
| Student | Thrasha Shetty | 01SU23CS212 |
| Student | Yashodha K | 01SU23CS232 |

Institution: Srinivas University, B.Tech Computer Science, Academic Year 2026–27.

## Abstract
Medical image segmentation is an important task in medical image analysis that involves identifying and separating specific anatomical structures or regions of interest from medical images. However, existing segmentation models may produce inaccurate boundaries or over-segment the target region. This project proposes a background-aware prompting framework using the Segment Anything Model (SAM) to improve few-shot medical image segmentation.

The proposed system adopts the Focus on Background (FoB) approach, which uses Background Prompt Prototype Construction (BPPC), Background-centric Context Modeling (BCM), and Structure-guided Prompt Refinement (SPR) to generate and refine background-focused prompts. These prompts guide SAM in producing more accurate segmentation masks using a limited number of annotated support images.

The project will extend the basic framework with adaptive background prompt generation, uncertainty-aware segmentation, and interactive mask refinement. A visualization dashboard will display the input image, generated prompts, predicted segmentation mask, overlay, and uncertainty map. The system will also evaluate performance using Dice Score, Intersection over Union (IoU), Precision, Recall, Boundary IoU, and inference time. Experiment history and report generation will be included to support result analysis. Cross-domain evaluation between compatible medical imaging datasets, such as CT and MRI, will be explored where feasible.

The system will be developed using Python, PyTorch, SAM, and suitable image-processing and web technologies. Its performance will be compared with a baseline SAM implementation to assess the effectiveness of the proposed enhancements.

## Objectives
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

## Scope
The project will focus on a research prototype for two-dimensional medical image segmentation using selected public datasets with ground-truth masks. The core scope includes implementing FoB with SAM, adaptive background prompting, segmentation visualization, and quantitative evaluation.

Advanced extensions will include uncertainty-aware visualization, interactive mask correction, experiment history, and report generation. Cross-domain testing between CT and MRI may be performed if suitable datasets and compatible annotations are available.

The project is intended for academic research and demonstration. It will not provide medical diagnoses or replace professional clinical evaluation. Three-dimensional segmentation and clinical deployment are outside the initial scope.

## Expected outcomes
* **Outcome 1:** A working prototype that performs few-shot medical image segmentation using background-aware prompting and SAM.
* **Outcome 2:** Improved control over unwanted surrounding regions, with segmentation performance assessed experimentally against a baseline. Improvements will be reported only if supported by evaluation results.
* **Outcome 3:** A visualization interface displaying input images, predicted segmentation masks, overlays, and evaluation metrics.
* **Outcome 4:** A comparative experimental report using Dice Score, IoU, Precision, Recall, and inference time.
* **Outcome 5:** An optional interactive refinement and uncertainty visualization module to support further inspection of segmentation results.

---

The status of each objective is tracked in [OBJECTIVES_STATUS](OBJECTIVES_STATUS.md).
