# v1 `fob_pipeline.py`: reconstructed archive copy

`fob_pipeline_v1.py` is the version of `src/fob/fob_pipeline.py` that produced the **v1 evaluation**
in `results/` (Enhanced Dice 0.056, FoB 0.012). It is an archive copy for reproducibility only.
It is **not imported anywhere** and is not part of the running code.

## Provenance

* The project had no Git history, so the original file was not preserved on disk when the approved
  v2 fix was applied on 2026-11-22.
* This file was reconstructed by applying the exact inverse of the four approved v2 edits to the
  current `src/fob/fob_pipeline.py`, using the verbatim text of those edits as recorded in the
  working session. Those four edits were the only changes ever made to the file. The script is
  `results/evidence/reconstruct_v1.py`.
* Each v2 block was required to occur exactly once, and no v2 identifier (`body_mask`,
  `select_positive_point`, `SOFT_TISSUE_RANGE`, `pos_xy`, `n_negative`) remains. The result parses
  as valid Python.
* **Limitation:** no hash of the original file was ever recorded, so byte-identity with the original
  cannot be proven. A practical check would be to run it once and compare against
  `results/enhanced_results.csv` and `results/fob_results.csv`. That would need SAM and approval.

## What differs from v2

| v1 (this file) | v2 (`src/fob/fob_pipeline.py`) |
|---|---|
| SPR negatives anywhere outside the dilated box, which in practice was mostly air | negatives restricted to the patient body mask |
| positive point = minimum attention anywhere inside the box, often bone or air | minimum attention among soft-tissue (grey 90–170) body pixels inside the box; omitted if none |
| always 1 positive point | positive point optional; box-only fallback |
| FoB negatives unrestricted | FoB negatives restricted to the body |
