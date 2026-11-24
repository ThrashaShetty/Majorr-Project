# Final-Year Project Report — Outline

> The full draft is now in [FINAL_REPORT_DRAFT](FINAL_REPORT_DRAFT.md).

The status tags show the source for each part: **[ready]** means verified material already exists;
**[pending]** means it depends on the external evaluation and the statistical analysis plan (SAP) v1 analysis.

1. **Introduction** [ready]: motivation, SAM in medical imaging [Kirillov2023, Mazurowski2023, Ma2024],
   the contributions, the report structure.
2. **Literature review** [ready to draft]:
   * SAM and ViT [Kirillov2023, Dosovitskiy2021, Li2022, He2022];
   * SAM in medicine [Mazurowski2023, Ma2024];
   * segmentation datasets [Antonelli2022, Simpson2019, Wasserthal2023];
   * evaluation metrics [Dice1945, Cheng2021, Taha2015, MaierHein2024].
   * *The FoB source is pending (see REFERENCES).*
3. **Problem statement and objectives** [ready]: PROBLEM_STATEMENT.md, the 10 consent-form objectives (CONSENT_FORM_SUMMARY.md, status in OBJECTIVES_STATUS.md), RQ1–RQ4.
4. **System architecture** [ready]: SYSTEM_ARCHITECTURE.md.
5. **Dataset and preprocessing** [ready]: DATASET_DOCUMENTATION.md.
6. **Proposed methodology** [ready]: METHODOLOGY.md, covering Baseline, Enhanced v2, A5 and Single-mask.
7. **Implementation** [ready]:
   * code structure;
   * SamEngine;
   * the backend and dashboard;
   * the external pipeline safeguards: streaming, write-once outputs, resource guard, memory optimisation.
8. **Experimental setup** [ready]: EXPERIMENTAL_SETUP.md, SAP v1.
9. **Results and discussion:**
   * 9.1 v1/v2 (MSD, 53 slices) [ready]
   * 9.2 v3 held-out validation (33 patients) [ready]
   * 9.3 External primary cohort (74 patients) [ready]
   * 9.4 External sensitivity cohort (90 patients) [ready]
   * 9.5 Failure analysis and A5 behaviour [ready]
   * 9.6 Runtime and memory [ready: 3.07 GiB (3.29 GB) → 1.44–1.45 GiB (about 1.55 GB) peak working set, about 13.2 s per slice]
10. **Limitations** [ready]: LIMITATIONS.md.
11. **Conclusion and future scope** [ready]: FUTURE_WORK.md.
* **References:** REFERENCES.md.
* **Appendices:**
  * A. Reproducibility and checksums (REPRODUCIBILITY_GUIDE.md);
  * B. SAP v1;
  * C. API (API_DOCUMENTATION.md);
  * D. Test results;
  * E. Verification reports.

**Original contributions vs existing components.** The report must state which is which:

* **Existing:** SAM, TotalSegmentator and the MSD data.
* **The team's own:** the prompting pipeline (BPPC: Background Prompt Prototype Construction; BCM: Background Context
  Modeling; SPR: Structure-guided Prompt Refinement; with the v2 constraints), A5, the
  evaluation protocol and the external-validation infrastructure.
