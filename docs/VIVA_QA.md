# Viva: Likely Questions and Honest Answers

The answers below are kept short, and every number is a measured one. Where a result is pending, say so
plainly; don't guess.

## Problem and approach

**1. What problem does the project solve?**
It generates SAM prompts automatically from one labelled CT slice of another patient, and uses them to
segment the spleen in a new slice. SAM is not fine-tuned and no manual prompts are drawn.

**2. Why SAM at all, rather than training a U-Net?**
The question was whether a foundation model can be used **without training**, given a single labelled
example. A trained U-Net would probably be more accurate on the spleen, but it needs many labelled
volumes and GPU training. That is a different question.

**3. Why the spleen only, and CT only?**
* **Scope and data:** MSD Task09 provides 41 labelled spleen CTs, and TotalSegmentator offers an
  independent CT source with spleen labels for external testing.
* **One organ keeps the protocol clean:** frozen parameters, pre-registered statistics.
* **No MRI:** the optional MRI check (CHAOS) was not done, so we make no claim about MRI.

**4. Why CPU and ViT-B, not ViT-H or a GPU?**
* **Hardware:** the only machine is a laptop with 7.34 GiB of RAM and a CPU build of PyTorch.
* **Model size:** ViT-B is the smallest SAM. Even so, it peaked at 3.07 GiB on CPU until we optimised
  the attention.
* **Comparability:** every method uses the same model, so the comparisons are fair. Absolute accuracy
  would likely be higher with ViT-H.

## Methods

**5. Explain the Baseline.**
The support mask's bounding box is reused as the box prompt on the query. SAM returns 3 masks, and we
keep the one with the highest predicted IoU.

**6. Explain Enhanced v2.**
1. **BPPC:** average SAM's encoder features of the support over its background, giving a prototype.
2. **BCM (Background-centric Context Modeling):** compute the cosine similarity of each query feature to that prototype, giving a
   background-attention map.
3. **SPR:** sample negative points where attention is high. They must be inside the body, outside the
   box and away from edges, and are seeded with 42. Add one positive point at the least background-like
   soft-tissue pixel in the box.
4. **Prompt:** the box, the negatives and the positive point.

**7. Why did Enhanced v2 do worse than Baseline?**
* **Weak attention map:** BCM's background attention barely separates spleen from neighbouring soft
  tissue (overlapping values of about 0.63–0.68).
* **Badly placed points:** the positive point often lands off the spleen, and negatives can fall on it.
* **Result:** oversized or misplaced masks. Recall went up (0.70 vs 0.58) while precision collapsed
  (0.26 vs 0.55).
* **Measured cost:** on held-out data, −0.165 per-patient Dice, Holm p = 6×10⁻⁴. We report this as a
  negative result.

**8. What went wrong in v1, and how was it fixed?**
* **v1:** negative points landed in the air outside the body, and SAM produced whole-body masks (Dice
  0.056).
* **v2:** negatives restricted to a body mask, plus a soft-tissue positive point. Dice rose to 0.332.
* The v1 results were kept unchanged as evidence.

**9. Is A5 really adaptive?**
* **Only in a limited sense.** A5 is a **selection rule**: it keeps Baseline unless its mask area is
  outside 0.5–2× the support area, and then switches to Enhanced v2. It doesn't generate new prompts or
  learn anything.
* **The other adaptive part:** `adaptive.py`, which sets the number and spacing of points from the
  support's foreground ratio.
* We call it **adaptive selection**, not adaptive prompt generation.

**10. How were A5's thresholds chosen? Isn't that overfitting?**
* They were chosen on the 8 **development** patients only, then frozen before the 33-patient held-out
  test and the external test.
* The development gain (+0.041) **shrank** on held-out data (+0.024 per patient, not significant). That
  is exactly why a held-out set matters.

**11. Why does A5 sometimes make things worse?**
* 7 harmful switches in v3, mostly near the spleen poles.
* There, the nearest support slice is often mid-spleen, so the support area is much larger than the
  query's, and the ratio falls below 0.5. A5 then switches to the weaker Enhanced v2.

**12. What is Single-mask, and did it help?**
It uses the same prompts as Enhanced v2 but asks SAM for one mask. It reduced oversized masks (20 → 8)
but lost recall. On held-out data: −0.022 Dice vs Enhanced v2, Holm p = 0.29. Not adopted.

## Evaluation and statistics

**13. Why statistics at patient level rather than slice level?**
Slices from the same patient are correlated. Treating 156 slices as independent would overstate the
sample size and give falsely small p-values. The patient (n = 33) is the independent unit, so we
compare per-patient mean Dice.

**14. Why Wilcoxon, Holm and the bootstrap?**
* **Wilcoxon signed-rank:** a paired, non-parametric test. Per-patient Dice differences aren't normal.
* **Holm:** controls the family-wise error over the 4 pre-registered comparisons.
* **Cluster bootstrap by patient:** confidence intervals that respect the clustering.

**15. Why freeze the analysis plan before the external results?**
To stop us choosing tests, cohorts or exclusions after seeing the data. SAP v1 was hash-frozen before any
external result existed. Amendment 01 only clarifies a random-stream detail in the bootstrap, and was also
written before any results.

**16. Why is the external evaluation so heavily engineered?**
It runs about 4 hours on a fragile laptop. We needed:
* **resume without duplicates:** write-once results, re-verified by hash;
* **proof that the method didn't change:** code hashes in `run_config.json`;
* **independent metric recomputation;**
* **backups and power and RAM guards,** so that an interruption can't corrupt the results.

**17. What are the external results?**
All 90 subjects were evaluated, independently verified and analysed exactly as the frozen plan specified.
* **Primary (74 patients):** A5 vs Baseline −0.002 per-patient Dice, 95 % CI [−0.021, +0.016], Holm
  p = 1.0. **No gain**, and the CI rules out a gain larger than about 0.02.
* Enhanced v2 vs Baseline: −0.163, Holm p = 4×10⁻⁸. It is **significantly worse**, as it was internally
  (−0.165).
* **Sensitivity (90 patients):** the same pattern. A5 +0.010 (Holm p = 0.45); Enhanced v2 −0.183.
* All methods scored lower than internally (Baseline 0.452 vs 0.494), mostly because the transferred box
  misses the spleen: 282 of 826 primary slices have recall < 0.1.

**18. Why exclude 17 subjects from the primary cohort?**
Their spleen touches the scan boundary, so it is only partly imaged. That rule was fixed before
evaluation. They're included in the sensitivity cohort (90 subjects, every 17th slice), which is analysed
as a separate family.

**19. Isn't selecting slices with at least 50 spleen pixels using the ground truth?**
Yes. Every evaluated slice contains spleen, so we don't measure false positives on spleen-free slices.
This is a stated limitation, and the same rule applies to all methods.

## Engineering

**20. How did you reduce SAM's memory?**
* **Cause:** the 4 global-attention blocks build a 12 × 4,096 × 4,096 attention matrix, plus copies.
* **Fix:** `mem_opt.py` patches attention in memory to process 512 query rows at a time, with the same
  operations in the same order.
* **Validation:** bitwise identical on 7 synthetic cases and pixel-identical on 2 real slices.
* **Result:** peak working set fell from 3.07 GiB to 1.45 GiB.

**21. How do you know the method code didn't change during the run?**
`run_config.json` stores the SHA-256 of 16 code files and of the checkpoint. The runner refuses to run on
a mismatch, and we re-check them after each phase. New features were written as new files for exactly
this reason.

## Exploratory features

**22. How is uncertainty computed?**
There are two signals; neither needs dropout, which SAM ViT-B doesn't have:
1. **Multimask:** the entropy of the mean probability across SAM's 3 masks, and the disagreement between
   their binary versions.
2. **Perturbation:** re-run SAM on K = 8 seeded, slightly jittered prompts (box corners σ = 5 % of the
   side, points σ = 5 px). The variance is 4p(1 − p), where p is the fraction of runs marking the pixel.

The per-slice score is the mean over a region around the predicted mask.

**23. Does uncertainty predict errors?**
**[PENDING: v3 exploratory run.]** We will report the Spearman correlation with Dice, at slice and
patient level, whatever the sign. The scores measure sensitivity, not calibrated error probability.

**24. How does interactive refinement work?**
* The user clicks to add (positive) or remove (negative).
* SAM is re-run with the **previous low-resolution logits** as `mask_input`, plus all clicks so far and
  the original box, asking for one mask. This is SAM's own interactive convention.
* For evaluation, a simulated user clicks the centre of the largest error region, up to 5 times.

**25. Doesn't the simulated user cheat?**
It uses the ground truth to choose each click, so it shows an **upper bound** on how steerable SAM is,
not real user effort. A study with real annotators is future work.

**26. Is the dashboard part of the results?**
No. It's a demonstration tool, labelled exploratory. Every reported number comes from the frozen scripts.
The dashboard is built and tested with fake models; its test with real SAM is pending.

## Wrap-up

**27. What are the main limitations?**
* 2D slices only;
* spleen only, CT only;
* box transfer without registration;
* weak background features;
* a small internal sample (33 patients);
* ground-truth-based slice selection;
* externally, only one other CT source (TotalSegmentator), where A5 showed no gain;
* CPU only;
* exploratory features not yet validated with real SAM.

**28. What's original in this project?**
* A training-free re-implementation inspired by FoB (Bo et al., CVPR 2026) of the background-aware prompt
  pipeline (BPPC, BCM, SPR), with the v2 constraints.
* The A5 rule.
* The staged, pre-registered evaluation protocol.
* The verified external-evaluation infrastructure, including the memory optimisation.
* The exploratory toolset.

SAM, the datasets, the general few-shot prototype idea and the FoB background-prompting idea [Bo2026] are
existing work.

**29. What would you do with more time?**
1. Improve box transfer itself, which is where Baseline fails externally.
2. Choose supports by body-axis position.
3. Spacing-aware box transfer.
4. Compare with MedSAM.
5. 3D propagation.
6. An MRI check.
7. A user study of refinement.
8. GPU inference.

**30. The external results are negative. Is the project a failure?**
No. The goal was to test the idea rigorously. A pre-registered negative or null result, together with a
reproducible pipeline, is a valid finding, and it shows where prompting fails (pole slices, weak
background features).
