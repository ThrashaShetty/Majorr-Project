# Future Work

Each item below follows from a measured limitation. None has been carried out, except where an item says otherwise.

1. **Improve box transfer itself.** The external evaluation and pre-registered analysis are complete
   (2026-11-24). Externally, Baseline fails mostly by missing the spleen (282 of 826 primary slices with
   recall < 0.1), so better box placement (items 2 and 3) is the most promising direction.
2. **Better support selection.** Match support slices on anatomical position along the body axis, not
   only on image similarity. This would target the pole-of-spleen errors behind A5's harmful switches.
3. **Physical-scale-aware box transfer.** Rescale the support box by the ratio of pixel spacings before
   transferring it, so differences in field of view stop affecting Baseline and A5. This would be a
   new, separately validated method; it would not change the frozen one.
4. **Stronger background features.** Use multi-prototype or texture-aware background modelling to
   improve the weak separation of spleen from other soft tissue by BCM (Background-centric Context Modeling).
5. **Medical SAM variants.** Compare against MedSAM [Ma2024] under the same frozen protocol.
6. **3D and inter-slice consistency.** Propagate prompts across adjacent slices; assess slices that
   contain no spleen.
7. **Uncertainty.** The exploratory maps (multimask entropy and disagreement, K-perturbation variance)
   are implemented, but their link to error is untested **[PENDING: v3 exploratory run]**. Next steps:
   calibration, and a principled method. SAM ViT-B has no dropout, so MC Dropout is unavailable.
   Also, a user study of click refinement with real annotators, instead of the simulated user.
8. **Application.** A read-only dashboard view of the verified research results, and authentication if
   it is ever deployed. A5 and Single-mask in the API, and a configurable output folder, now exist in the
   exploratory extension.
9. **Performance.** GPU inference; batching support encodings across queries.
10. **Cross-modality.** Test the frozen method on abdominal MRI with spleen labels, for example CHAOS
    T2-SPIR. This was not done.
