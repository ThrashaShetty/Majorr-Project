# Exploratory Features (Phase C)

> **These features are exploratory and NOT part of the frozen confirmatory analysis.** They were added
> after the external protocol was frozen, as **new files only**, so every code hash recorded in
> `results/external_ts/eval/run_config.json` still matches. None of them changes Baseline, Enhanced v2,
> A5 or Single-mask. Research prototype only; not for diagnosis.

| Feature | Files | Tests (synthetic or fake SAM) | Real-SAM status |
|---|---|---|---|
| Uncertainty maps (C.1) | `src/extensions/uncertainty.py` | `tests/test_ext_uncertainty.py` (8) | v3 correlation study **[PENDING: v3 exploratory run]** |
| Click refinement (C.2) | `src/extensions/refinement.py` | `tests/test_ext_refinement.py` (8) | v3 simulated-click study **[PENDING: v3 exploratory run]** |
| Extended API and dashboard (C.3) | `backend/extensions/{service,app,serve}.py`, `frontend/dashboard.html` | `tests/test_ext_api.py` (13) | Real-SAM end-to-end check **[PENDING: dashboard test]** |
| History and report export (C.4) | `backend/extensions/{history,report}.py` | `tests/test_ext_history.py` (8) | Works without SAM. Not yet exercised on real runs |
| Held v3 study script | `experiments/exploratory/v3_uncertainty_and_clicks.py` | `--check` passes (156 slices, 33 patients) | **Not run** (needs RAM) |
| MRI cross-modality check (C.5, optional) | — | — | **Not done** (no MRI dataset downloaded) |

Test totals on 2026-11-23: project suite **93 passed, 2 deselected**. The 2 deselected tests load the
real checkpoint. The external-pipeline suite: **108 passed**.

---

## 1. Uncertainty maps (`src/extensions/uncertainty.py`)

SAM ViT-B has no dropout layers, so MC Dropout is not available. Two cheap signals are computed
instead. Each gives a per-pixel map in [0, 1] and a scalar score per slice.

### (a) Multimask uncertainty: one SAM call
For a single prompt, SAM returns 3 candidate masks as logits.

* **Entropy map:**
  1. Take the mean sigmoid probability across the 3 masks, p̄.
  2. Compute the binary entropy H(p̄) = −p̄ log₂ p̄ − (1−p̄) log₂(1−p̄), in bits.
* **Disagreement map:**
  1. Let v be the fraction of the 3 binary masks that mark the pixel.
  2. The disagreement is 1 − |2v − 1|: 0 when all 3 masks agree, 1 when they are split.

### (b) Prompt-perturbation variance: K extra SAM calls (default K = 8, seed 42)
The prompt is perturbed K times with a seeded random generator:

* each box corner gets Gaussian noise with σ = 0.05 × the box side;
* each point gets Gaussian noise with σ = 5 px;
* everything is clipped to the image, and point labels are unchanged.

For each perturbed prompt, SAM's best mask is kept. Let p be the fraction of the K masks that mark the
pixel. The **variance map** is 4p(1 − p), scaled to [0, 1].

### Scalar score
* The score is the mean of the map over a **region of interest**: the union of all predicted masks,
  dilated by 5 px.
* Without that region, large empty background areas would dilute the score toward 0.
* If no foreground is predicted, the score is 0 and `roi_px = 0`. This is reported, not hidden.

### What it does not claim
* These scores measure SAM's **sensitivity** to its own outputs and to small prompt changes. They are
  not calibrated probabilities of error.
* Whether they predict Dice is the question of the v3 study: Spearman ρ, at slice level (n = 156) and
  patient level (n = 33). Results: **[PENDING: v3 exploratory run]**. They will be reported whatever
  their sign or p-value.

---

## 2. Interactive click refinement (`src/extensions/refinement.py`)

### Refinement step: `refine`
SAM is re-run with:

* the **previous low-resolution logits** as `mask_input`, shape (1, 256, 256);
* **all clicks so far**: label 1 = add, 0 = remove;
* the original box, if there was one;
* `multimask_output=False`.

This follows SAM's own interactive-demo convention.

### `mask_to_low_res`
This converts a binary mask into SAM's mask-input frame, so a refinement can start from any mask:

1. Resize so the longest side is 1024 (`ResizeLongestSide(1024)`), placed top-left.
2. Zero-pad to 1024 × 1024.
3. Reduce to 256 × 256.
4. Convert to logits of ±8.

### Simulated user: `next_click`, `simulate`
* The error region is prediction XOR ground truth.
* The click goes to the **largest connected component** of the false negatives or false positives,
  at the interior point **farthest from the region's edge** (by distance transform).
* A false-negative component gets a positive click; a false-positive component gets a negative click.
* A perfect prediction gets no click, and the Dice score is carried forward.
* The process is fully deterministic: no randomness.
* `simulate` returns Dice after 0, 1, …, N clicks (N = 5 in the v3 study).

### Evaluation
Mean Dice vs number of clicks (0–5), for Baseline and A5, on the 156 v3 slices.
**[PENDING: v3 exploratory run]**

---

## 3. Extended API and dashboard (`backend/extensions/`, `frontend/dashboard.html`)

### Design
* **Reuses the original app:** `create_ext_app()` wraps `backend.main.create_app()` **unchanged**. The
  original routes (`/health`, `/segment`, `/results`, `/compare`) keep working and share the same
  single SAM engine through an adapter.
* **`ExtService`** serves the four evaluated methods from one `SamEngine`, using the frozen classes
  unchanged:
  * **Baseline:** the support-mask box; the best of 3 masks.
  * **Enhanced v2:** the frozen `FoBPipeline(mode="enhanced")` builds the prompt (box, background
    negatives, positive point). The mask is then decoded from that prompt.
  * **A5:** ratio = (frozen Baseline mask pixels) / (support mask pixels). If 0.5 ≤ ratio ≤ 2.0, use
    Baseline; otherwise use Enhanced v2.
  * **Single-mask:** the Enhanced v2 prompt with `multimask_output=False`.
* **Data location:** everything the extended app writes (SQLite, overlays, temporary uploads) goes to
  `MEDSAM_EXT_DATA`, default `<project>/app_data/`. It never writes to `results/overlays/` or
  `experiments.db`.
* **Sessions** for click refinement live in memory only, at most 8 (the oldest is dropped).

### Running it

```powershell
cd C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project
$env:PYTHONDONTWRITEBYTECODE = "1"
# optional: $env:MEDSAM_EXT_DATA = "C:\path\to\demo_data"   (default: .\app_data)
.\.venv\Scripts\python.exe -B -m uvicorn backend.extensions.serve:app --host 127.0.0.1 --port 8000
```

1. Wait until `GET http://127.0.0.1:8000/ext/health` shows `"sam_loaded": true`. SAM loads in a
   background thread and needs about 1.5 GiB of free RAM.
2. Open `frontend/dashboard.html` in a browser. Opening the file directly works, because the server
   allows the `null` origin.
3. To use another port, add `?api=http://127.0.0.1:<port>` to the page URL.

**Using the page:**
* choose a support image, support mask, query image and, optionally, a query mask (used only for metrics);
* pick the method and K, then press **Segment**;
* click the overlay to refine: choose add or remove first. **Reset clicks** returns to the original
  prediction;
* the **History** panel lists runs, compares selected runs, and exports CSV, HTML or PDF.

The original dashboard (`frontend/index.html` with `backend.main:app`) is unchanged.

The full endpoint reference is in [API_DOCUMENTATION](API_DOCUMENTATION.md#extended-api-backendextensions).

---

## 4. Experiment history and report export (`backend/extensions/history.py`, `report.py`)

* **Storage:** one row per `/ext/segment` run, in the table `ext_runs` of `<data>/history_ext.db`. Each
  row holds:
  * the method, and the method A5 actually used, with its area ratio;
  * the SAM score and K;
  * the 5 metrics and the 3 uncertainty scores;
  * the prompts (as JSON);
  * the overlay path.
* **Refinement:** refine and reset update the same row's `n_clicks`, its clicks and its `refined_*`
  metrics, and save the refined overlay.
* **No delete endpoint.**
* **Compare:** shows the runs side by side, with the difference from the first selected run for each
  metric and uncertainty score, and marks the best run per metric. It warns when the runs use
  different query images.
* **Exports:**
  * **CSV:** all columns. Text cells that start with `= + - @` are prefixed with `'`, to block
    spreadsheet-formula injection.
  * **HTML:** self-contained. All text is escaped, and the overlays are embedded. Only image files
    inside the data folder can be embedded.
  * **PDF:** drawn with matplotlib, so no extra dependency is needed. It contains a summary table, a
    chart of initial vs after-clicks Dice, and the overlay pages.
  * Every export carries the note "exploratory, not cohort results".

---

## 5. Limitations of the exploratory features

* **Not validated on real SAM yet.**
  * Every test uses synthetic or fake models.
  * The v3 correlation and click studies, and an end-to-end dashboard run with the real checkpoint, are
    **pending**: [PENDING: v3 exploratory run], [PENDING: dashboard test].
* **The Enhanced v2 mask in the dashboard is decoded again from the frozen pipeline's prompt.** In the
  research scripts the frozen pipeline returns the mask directly. The two should be identical, but this
  has **not yet been checked on real SAM**. For Baseline, the v3 study script counts and reports any
  difference from the frozen mask.
* **Uncertainty is heuristic.** It is not calibrated and has no ground-truth meaning. It costs K + 1 SAM
  decoder calls per slice; the encoder runs once.
* **The simulated user is idealised.** It uses the ground truth to choose each click, so the click
  curves show an upper bound on how well SAM can be steered, not real user effort.
* **Single-user, local demo only.**
  * Sessions are in memory.
  * There is no authentication.
  * One SAM model is shared, so requests run one at a time.
* **No MRI.** The optional cross-modality check (C.5) was not done. All results are CT only.
