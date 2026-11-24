# Limitations

## Method
* **Box transfer.** The support bounding box isn't anatomically correct for a different patient or
  slice. Baseline, Enhanced v2 and A5 all inherit this. In TotalSegmentator, the field of view and
  voxel size also differ from MSD, which changes where the spleen sits and how large it is in the
  512×512 frame.
* **Weak background attention.** BCM (Background-centric Context Modeling) separates spleen from other soft tissue only weakly (attention
  about 0.63–0.67 vs 0.66–0.68). As a result the positive point often misses, and negatives can cut into
  the spleen. Enhanced v2 was significantly worse than Baseline on held-out MSD (v3).
* **Fixed intensity rules.** The prompt constraints (body mask, soft-tissue grey 90–170) are tied to
  the [−200, 300] Hounsfield-unit (HU) window.
* **A5 makes errors in both directions.** In v3 it made 7 harmful switches, mostly near the spleen
  poles, and missed 23 cases where Enhanced would have been better.
* **No registration.** SPR (Structure-guided Prompt Refinement) mixes support edges with query attention on a shared grid, assuming the
  slices are roughly aligned.
* **No fine-tuning.** SAM is used zero-shot, on 2D slices only.

## Evaluation
* **Slice selection uses the ground truth.** Slices with fewer than 50 spleen pixels are dropped using
  the label, so every evaluated slice contains spleen. Performance on slices without spleen is not
  measured.
* **Small internal sample.** v3 has 33 patients from one institution, and the A5 vs Baseline
  difference is small and not significant.
* **External labels differ.** TotalSegmentator labels were drafted by a model and then reviewed, which
  differs from MSD's labelling. Part of any drop in Dice may come from label differences.
* **External data differs physically.** It is resampled to 1.5 mm isotropic, so images are blurrier
  after upsampling. It has mixed phases and protocols, including noisy scans (for example s1389),
  arms in the field of view, and wide fields of view.
* **One external source only.** The external evaluation is complete (90 of 90 subjects), but it covers a
  single CT dataset. There, A5 showed no gain over Baseline (−0.002 Dice, primary cohort) and all methods
  scored lower than internally (Baseline 0.452 vs 0.494).
* **Run deviation.** Phase A started under the pre-declared 1.0 GiB start-RAM deviation, because free RAM
  was below the 1.6 GiB guard. This changed only when slices were scheduled, not what was computed; every
  run-history entry records it.
* **Timing:** the reported `*_ms` fields count a cached encoding's original time (the v3 convention).
  Wall-clock time is measured separately.

## Engineering
* **CPU only.** SAM peaks at 1.44–1.45 GiB (about 1.55 GB) with the memory optimisation. The full external run was
  estimated at about 4 hours on the test laptop (1,062 slices × the 13.2 s per slice measured on s0965). It completed
  on 2026-11-24 over three resumable runs. Two were stopped by the free-disk and free-RAM guards, and the final one
  used the 1.0 GiB start-RAM deviation.
* **The application doesn't show the verified research results.** The original API exposes Baseline,
  FoB and Enhanced. The exploratory extension (`/ext`) adds A5 and Single-mask.
* **The backend writes into protected files by default:** `results/overlays/` and `experiments.db`
  (see API_DOCUMENTATION).
* **No authentication.** The API is for local use only.

## Exploratory features (Phase C)
* **Not validated with real SAM yet.**
  * Uncertainty, click refinement and the extended dashboard are tested only with synthetic or fake models.
  * The v3 studies (uncertainty vs Dice; Dice vs clicks) are **[PENDING: v3 exploratory run]**.
* **Uncertainty is heuristic.** It measures sensitivity to SAM's outputs and to small prompt
  perturbations; it is not calibrated, and MC Dropout is impossible because SAM ViT-B has no dropout.
* **The simulated clicks use the ground truth,** so they give an upper bound, not real user effort.
* **The dashboard re-decodes Enhanced v2's mask** from the frozen prompt. Its equality with the frozen
  pipeline's mask is not yet confirmed on real SAM.
* **No cross-modality result.** The MRI check (CHAOS) was not done; everything is CT → CT.

## Clinical
* **Not a medical device.** The project is not clinically validated and makes no diagnostic claims.
