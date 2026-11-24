# System Architecture

## 1. Components

```
                ┌───────────────────────── research pipeline (src/) ─────────────────────────┐
 MSD NIfTI ──►  │ preprocessing.py ──► PNG slices + case-level split (seed 42)                │
                │        │                                                                    │
                │        ▼                                                                    │
                │ SamEngine (baseline.py): one SAM ViT-B, embedding cache, re-entrant lock    │
                │   ├─ BaselineSegmentor: support-mask box → SAM → best of 3 masks            │
                │   └─ FoBPipeline (fob/): BPPC → BCM → SPR → prompts → SAM                   │
                │ metrics.py: Dice, IoU, Precision, Recall, Boundary IoU                      │
                └─────────────────────────────────────────────────────────────────────────────┘
 run_evaluation.py (v1/v2)    results/v3/scripts/holdout.py (v3, adds A5 + Single-mask)
 results/external_ts/scripts/ (external TotalSegmentator pipeline, see §3)
 backend/main.py (FastAPI + SQLite)  ◄── HTTP ──►  frontend/index.html (dashboard)
```

| Module | File | Responsibility |
|---|---|---|
| Preprocessing | `src/preprocessing.py` | Every 5th axial slice; Hounsfield-unit (HU) window [−200, 300]; 90° rotation; 512×512; exclusion rules; patient-level split |
| SAM engine and Baseline | `src/baseline.py` | Loads SAM ViT-B (`strict=True`); shares one predictor; box-prompt Baseline |
| BPPC (Background Prompt Prototype Construction) | `src/fob/bppc.py` | Background prototype: SAM encoder features of the support slice, averaged over the background |
| BCM (Background-centric Context Modeling) | `src/fob/bcm.py` | Cosine similarity between query features and the prototype → attention map |
| SPR (Structure-guided Prompt Refinement) | `src/fob/spr.py` | Background point sampling away from support edges, seeded |
| Adaptive | `src/fob/adaptive.py` | Number of points and their spacing, from the support foreground ratio |
| Pipeline | `src/fob/fob_pipeline.py` | FoB / Enhanced orchestration, body mask, positive-point rule |
| Metrics | `src/metrics.py` | Metric definitions and the empty-mask conventions |

## 2. SAM ViT-B in brief [Kirillov2023]

* **Image encoder:** a ViT-B [Dosovitskiy2021] with windowed attention and 4 global-attention blocks
  [Li2022]. It takes a 1024×1024 input and produces a 256×64×64 embedding.
* **Prompt encoder:** encodes points, with labels 1 or 0, and boxes.
* **Mask decoder:** produces 3 candidate masks with predicted IoU scores. The project keeps the
  highest-scoring one. The Single-mask variant asks for 1 mask instead.

## 3. External evaluation pipeline (`results/external_ts/scripts/`)

| Stage | Script | Safeguards |
|---|---|---|
| Cohort manifest | `build_manifest.py` | Archive MD5 verified; each mask checked against its archive CRC (cyclic redundancy check); manifest written once with a SHA-256 sidecar |
| Preprocessing | `prep_external_ts.py` | CTs streamed from the ZIP, never extracted; existing `validate_slice_pair` reused unchanged; atomic write-once PNGs; per-subject completion markers |
| QC | `verify_pilot.py`, `verify_full.py` | Every slice independently recomputed; all 1,239 pairs exact |
| Evaluation runner | `eval_external_ts.py` | Frozen run config (code hashes); worklist; write-once slice results; resume |
| Memory optimisation | `mem_opt.py`, `run_with_mem_opt.py` | Row-chunked global attention patched in memory only; validated bit-identical on 7 synthetic cases and 2 real slices; peak working set 3.07 GiB (3.29 GB) → 1.44–1.45 GiB (about 1.55 GB) |
| Resource guard | `ts_resources.py` | Numeric checks (binary units, GiB): RAM 1.6 GiB to start, 0.3 GiB while running; disk 2.0 GiB. A missing or invalid reading counts as a stop |
| Verification | `verify_eval_subjects.py` | Metrics recomputed independently (numpy/scipy); hashes; completion markers |
| Batching and backup | `run_batches.py`, `make_backup.py` | AC check, keep-awake request, verification and verified backup after each batch |

## 4. Application

* **Backend:** `backend/main.py`, FastAPI, documented in [API_DOCUMENTATION](API_DOCUMENTATION.md).
  * It loads SAM once at startup.
  * Inference runs in a worker thread, serialised by a lock.
  * Experiments are logged in SQLite.
* **Frontend:** `frontend/index.html`, a single page that calls `/health`, `/segment`, `/results` and
  `/compare`.
* **Extension (exploratory, Phase C, new files only):** `backend/extensions/`, served with
  `uvicorn backend.extensions.serve:app`. It wraps the unchanged app and adds `/ext/*` routes with:
  * all 4 methods (Baseline, Enhanced v2, A5, Single-mask);
  * prompt overlays;
  * uncertainty maps (`src/extensions/uncertainty.py`);
  * click refinement (`src/extensions/refinement.py`);
  * history with compare and CSV/HTML/PDF export.

  Its page is `frontend/dashboard.html`, and its data goes to `app_data/`. See
  [EXPLORATORY_FEATURES](EXPLORATORY_FEATURES.md).
* **Remaining gap:** no view of the verified research results (v3 or external) in either dashboard.

The full diagram is in [ARCHITECTURE_DIAGRAM](ARCHITECTURE_DIAGRAM.md).

## 5. Data flow (external evaluation)

```
Zenodo ZIP (MD5-verified) ──stream──► prep (validate_slice_pair) ──► PNG pairs (1,239)
     frozen manifest + worklist (1,078 slices) + run_config (code hashes)
           ──► run_with_mem_opt ──► eval_external_ts.execute (one subject at a time)
                 slice JSON + content-addressed npz ──► subject CSV + .done
           ──► verify_eval_subjects ──► audit report ──► make_backup (verified zip)
```
