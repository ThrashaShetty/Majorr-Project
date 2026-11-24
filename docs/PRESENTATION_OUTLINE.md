# Presentation Outline (about 20 slides)

> The 15-slide version with content and figures is [SLIDES_OUTLINE](SLIDES_OUTLINE.md).

The external results are now verified; see [SLIDES_OUTLINE](SLIDES_OUTLINE.md) slide 12 for the numbers.

1. **Title**: project, team, guide, university.
2. **Problem**: manual CT segmentation is slow; SAM needs prompts.
3. **Motivation**: automatic prompts from one labelled slice of another patient.
4. **Research gap**: SAM on medical images is prompt-sensitive [Mazurowski2023]; is background
   information useful?
5. **Objectives**: the 10 consent-form objectives ([CONSENT_FORM_SUMMARY](CONSENT_FORM_SUMMARY.md)) and RQ1–RQ4.
6. **Literature**: SAM, ViT, MedSAM, TotalSegmentator, Boundary IoU.
7. **Datasets**: MSD Task09 (41 cases; 33 + 8 split) and TotalSegmentator (74 / 90 patients).
8. **Architecture**: pipeline diagram (SYSTEM_ARCHITECTURE §1).
9. **Baseline**: box transfer.
10. **Enhanced v2**: BPPC → BCM → SPR (Background Prompt Prototype Construction → Background-centric Context Modeling →
    Structure-guided Prompt Refinement), plus the body-mask and positive-point constraints.
11. **A5 and Single-mask**: the area-ratio rule (0.5–2.0) and the one-mask variant.
12. **Experimental setup**: v1, v2, v3 and external; frozen protocol; statistical analysis plan (SAP).
13. **Results I (v3)**: per-patient Dice and paired tests (A5 vs Baseline p = 0.29, not significant).
14. **Results II (external primary)**: A5 vs Baseline −0.002 (Holm p = 1.0); Enhanced v2 significantly worse (−0.163).
15. **Results III (external sensitivity)**: same pattern (A5 +0.010, Holm p = 0.45; Enhanced v2 −0.183).
16. **Failure analysis**: harmful A5 switches at the spleen poles; Enhanced over-segmentation
    externally, Baseline misses the spleen on 282 of 826 slices.
17. **Engineering**: verified pipeline; memory optimisation 3.07 GiB (3.29 GB) → 1.44–1.45 GiB (about 1.55 GB) (validated bit-identical); resume and backups.
18. **Application demo**: dashboard and API (label what is and isn't implemented).
19. **Limitations**
20. **Conclusion and future work**: box transfer remains the method to beat, internally and externally.
