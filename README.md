# Background-Aware Prompting for Enhanced Medical Image Segmentation Using SAM

**Srinivas University · B.Tech Computer Science · Academic Year 2026–27**
Team project (four students) · Project Guide: **Mr. Vikramaditya Kumar**

> ⚠️ **Research-only disclaimer.** This is an academic research prototype. It is **not clinically
> validated** and must **not** be used for medical diagnosis, treatment planning or any clinical
> decision. No performance claims are made beyond what `run_evaluation.py` measures on your machine.

---

## 1. Introduction

The Segment Anything Model (SAM, Meta AI) segments objects from *prompts* (points, boxes, masks).
On medical images, SAM's quality depends heavily on the prompts it receives. This project builds a
complete, reproducible pipeline that studies **background-aware prompting**: using what the
*background* looks like in a labelled support slice to generate better prompts for an unlabelled
query slice. It is evaluated on the spleen CT task (Task09) of the Medical Segmentation Decathlon.

The deliverable contains a Python ML pipeline, a FastAPI backend with SQLite experiment tracking,
and an HTML/CSS/JavaScript dashboard.

## 2. Research motivation

* Manual point and box prompts are slow to produce and vary between annotators.
* Few-shot setups give a labelled *support* image. The foreground often changes between slices, but
  the surrounding anatomy (the background) is often more consistent.
* Modelling the background explicitly can tell SAM where the target is **not**. This approach is the
  Focus-on-Background (FoB) idea.

## 3. Problem statement

Given a support CT slice with its spleen mask and a query CT slice from a **different patient**,
segment the spleen in the query slice using SAM ViT-B. The prompts must be generated
automatically from the support pair, and the result is measured with Dice, IoU, Precision, Recall
and Boundary IoU.

## 4. Proposed methodology

```
support image ─┐                                    ┌─ query image
support mask ──┤                                    │
               ▼                                    ▼
      Adaptive config ◄── fg_ratio, edge density   SamPredictor.set_image
               │                                    │  (ResizeLongestSide → normalise → pad → ViT-B)
               ▼                                    ▼
   BPPC: SAM encoder on support ──► background ──► BCM: cosine(query features, prototype)
         [1,256,64,64] × (1−mask)    prototype           → sigmoid → attention [1,1,64,64]
                                     [1,256]                         │
                                                                     ▼
                                     SPR: Canny edges (support) + 10-px exclusion,
                                          attention > 0.6, seeded sampling, min spacing
                                                                     │
                                                                     ▼
                                     SAM prompt encoder + mask decoder → best of 3 masks
                                                                     │
                                                 uncertainty (optional) · metrics · overlay
```

### Three compared methods

| Method | Prompts given to SAM | Purpose |
|---|---|---|
| **Baseline SAM** | Bounding box of the support mask, transferred unchanged to the query | Simple transfer-prompt baseline |
| **FoB SAM** (experimental) | Only SPR **background points** (label 0) | Tests background-only prompting by itself |
| **Enhanced FoB SAM** | Support box **+** SPR background points *outside* the box (dilated 10 px) and *inside the patient body* **+** one positive point (label 1) at the **minimum** BCM attention among soft-tissue pixels inside the box (see §8a) | Background-aware refinement of the baseline box |

The prompts for Enhanced are fully specified above. The only information it adds over the
baseline comes from the background prototype and attention. Nothing else is introduced silently.

## 5. FoB architecture (code map)

| Module | File | Responsibility |
|---|---|---|
| BPPC | `src/fob/bppc.py` | Background prototype from support features |
| BCM | `src/fob/bcm.py` | Query-vs-prototype attention map |
| SPR | `src/fob/spr.py` | Edge-aware background point sampling |
| Adaptive | `src/fob/adaptive.py` | Number and spread of points from mask statistics |
| Pipeline | `src/fob/fob_pipeline.py` | Orchestration, SAM prompting, uncertainty |
| Baseline | `src/baseline.py` | Box-prompt baseline, shared `SamEngine`, batch evaluation |

### 6. BPPC: Background Prompt Prototype Construction
1. The support image goes through **SAM's own preprocessing**: `ResizeLongestSide(1024)`, then
   `Sam.preprocess` (pixel mean/std normalisation and zero-padding to 1024²). This is identical to
   `SamPredictor.set_image`.
2. `sam.image_encoder` produces features of shape `[1, 256, 64, 64]` for ViT-B. The shape is
   checked at runtime.
3. The mask is mapped into the same padded frame and downsampled with nearest-neighbour
   interpolation to 64×64. Padded cells are excluded from the background region.
4. `background_region = (1 − mask) · valid`. The prototype is the masked spatial mean, then
   L2-normalised, giving `[1, 256]`.
5. If no background cell exists, the code falls back to the mean over the valid region. This
   fallback is logged and recorded in `last_info`.

### 7. BCM: Background-centric Context Modeling
The query features (`predictor.features`) and the prototype are L2-normalised. Cosine similarity is
computed at each of the 64×64 positions and a sigmoid is applied, giving `[1, 1, 64, 64]`.
**High** values mean background-like; **low** values mean less background-like (likely
foreground). With temperature 1, the values lie in [0.269, 0.731]. The attention is cropped to the
un-padded region and resized bilinearly to the image size.

### 8. SPR: Structure-guided Prompt Refinement
1. Run `cv2.Canny(gray, 50, 150)`, then dilate the edges by 10 px with a 21×21 elliptical kernel.
   The result is an exclusion zone.
2. Candidates are pixels where attention > 0.6 **and** that lie outside the exclusion zone,
   collected with `cv2.findNonZero`.
3. Up to `n_points` unique points are sampled with a seeded RNG and a minimum pairwise distance.
   If there are no candidates, SPR returns an empty list.

**Coordinate frames:** the edges come from the support image and the attention comes from the
query. The support image is resized to the query size first, so both share one pixel grid. This
assumes the axial slices are roughly aligned. It is **not** image registration.

### 8a. Prompt placement constraints (v2)
The first full evaluation showed that unrestricted SPR negatives landed in the **air around the
patient**. The background prototype is dominated by air, so air had the highest attention. SAM then
read the prompt as "everything that is not air" and returned the whole body: Enhanced reached
recall ≈ 1 but precision ≈ 0.03. Two constraints in `src/fob/fob_pipeline.py` fix this:

* **Body mask** (`body_mask`): query grey > 20 → largest connected component → holes filled →
  eroded by 10 px. SPR negatives (FoB and Enhanced) must lie inside it.
* **Positive point** (`select_positive_point`): the minimum-attention pixel among in-box body pixels
  with grey level 90–170 (soft tissue). This excludes air (0) and bone (≈ 255), where the old argmin
  usually landed. If no pixel is eligible, the positive point is omitted.
* **Fallback:** if no negative survives, Enhanced uses box + positive point, or the box only. FoB
  still fails loudly with no prompts, and the failure is recorded in the CSV.

Grey levels refer to the preprocessing window [−200, 300] HU. These are fixed rules, not learned or
tuned on the query set.

### 9. Adaptive prompting
`fg_ratio = mask.sum() / mask.size` selects the prompt configuration:

| fg_ratio | n_points | spread (min distance) | strategy |
|---|---|---|---|
| < 0.05 | 15 | tight (8 px) | small_structure |
| < 0.20 | 8 | medium (24 px) | medium_structure |
| ≥ 0.20 | 4 | wide (48 px) | large_structure |

`edge_density` is the fraction of Canny edge pixels in the mask. It is reported for analysis.
To compare against a fixed configuration, use `--no-adaptive`, which sets 8 points.

### 10. SAM ViT-B integration
* The model is built with `sam_model_registry["vit_b"]`. The checkpoint is loaded with
  `torch.load(map_location="cpu", weights_only=True)` and `load_state_dict(strict=True)`, so a
  corrupt or mismatched checkpoint fails loudly.
* **One** model instance is shared by all three methods through `SamEngine`. `SamEngine` holds a
  re-entrant lock, because `SamPredictor` stores per-image state, and it caches the most recent
  image embedding.
* Prompts are passed in original image coordinates. `SamPredictor.predict` applies
  `apply_coords` / `apply_boxes` itself. With `multimask_output=True`, the mask with the highest
  predicted IoU is kept.

#### Uncertainty (MC Dropout status)
The installed `segment-anything` ViT-B has **no dropout modules with p > 0**. This was verified by
inspecting `model.named_modules()`. Calling `model.train()` therefore cannot produce MC-Dropout
uncertainty. The pipeline:
* checks for dropout layers. If there are none, it reports MC Dropout as **unavailable** and gives
  the reason. If dropout layers do exist, it enables only those, runs 5 passes, restores every
  module's original `training` flag, and confirms that the parameters did not change.
* in `auto` mode (the API default), it then computes **prompt-perturbation variance**, labelled
  explicitly as such: 5 passes with seeded jitter of points (σ = 6 px) and box (σ = 3 %), and the
  per-pixel std of SAM's mask probabilities. This measures sensitivity to the prompt. It is **not**
  Bayesian uncertainty.

## 11. Dataset preparation

1. Download **Task09_Spleen.tar** from the official Medical Segmentation Decathlon site
   (<http://medicaldecathlon.com/>, about 1.5 GB). The licence is CC-BY-SA 4.0.
2. Extract it so that this layout exists:
   ```
   data/raw/Task09_Spleen/imagesTr/spleen_*.nii.gz
   data/raw/Task09_Spleen/labelsTr/spleen_*.nii.gz
   ```
   The macOS `._*` files in the archive are ignored automatically. Raw files are opened read-only.
3. Run the preprocessing (Section 14).

**Preprocessing rules:**
* Take every 5th axial slice (axis 2).
* Clip HU to [−200, 300], scale to 0–255 and store as uint8.
* Rotate 90° for display; the same rotation is applied to the mask.
* Resize to 512×512: bilinear for images, nearest-neighbour for masks. Label 1 becomes 255.
* A slice is **excluded** if any of these hold:
  * `shape_mismatch`
  * `invalid_labels` (a value outside {0, 1})
  * `low_contrast` (std < 1 grey level)
  * `no_foreground` (fewer than 50 spleen pixels)
* **Split at the case (patient) level:** cases are permuted with seed 42; 80 % become support
  cases and 20 % become query cases. No case appears on both sides, which prevents slice-level
  leakage.
* List files contain one `image<TAB>mask` line per sample, with paths relative to the project root.
* `data/preprocessing_report.json` and `data/split_summary.json` record totals, exclusions,
  foreground ratios and the split.

## 12. Installation

Requires Python **3.10** (also verified on 3.11.9) and about 3 GB of free disk space for the
environment. Use PowerShell on Windows.

```powershell
cd medical_sam_project
py -3.10 -m venv .venv          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**PyTorch is platform-specific.** Install it first with **one** of these commands:

```powershell
# CPU only (Windows / Linux)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU, CUDA 12.6 wheels (check https://pytorch.org/get-started/locally/ for your driver)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

Then install everything else (this needs `git` for the segment-anything dependency):

```powershell
pip install -r requirements.txt
```

No single PyTorch wheel works on every OS and every CPU/CUDA combination. If the pinned version
is not available for your platform, choose a matching torch/torchvision pair from pytorch.org.

## 13. Checkpoint download

Download **`sam_vit_b_01ec64.pth`** (about 375 MB) from the official segment-anything repository
(<https://github.com/facebookresearch/segment-anything#model-checkpoints>):

```powershell
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
```

If the checkpoint is missing, all model-free code still works. The API then reports
`sam_loaded: false` and `/segment` returns **503**. Results are never faked.

## 14. Preprocessing

```powershell
python -m src.preprocessing                 # all cases
python -m src.preprocessing --max-cases 5   # quick test
```

## 15. Backend startup

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The server reads these optional environment variables: `MEDSAM_CHECKPOINT`, `MEDSAM_DEVICE`
(`cpu` / `cuda` / `auto`) and `MEDSAM_DB`. `experiments.db` is created at startup.

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status":"ok","sam_loaded":<actual state>,"device":…}` |
| `POST /segment` | multipart: `support_image`, `support_mask`, `query_image`, `method` (`baseline`, `fob` or `enhanced`), optional `query_mask` (ground truth). Without a ground truth, the accuracy metrics are `null`. |
| `GET /results` | All experiments, newest first |
| `GET /compare/{query_image_name}` | Latest experiment per method; `null` if that method has not been run |

Uploads are limited to 10 MB and must be PNG, JPG, BMP or TIFF. They are decoded and checked, then
stored under server-generated names in `tmp_uploads/<uuid>/`, which is deleted after each request.
Inference runs in a worker thread, so it stays off the event loop, and model access is serialised
by a lock.

## 16. Frontend startup

Open a second terminal:

```powershell
python -m http.server 5500 -d frontend --bind 127.0.0.1
```

Then browse to <http://localhost:5500/index.html>. Opening `frontend/index.html` directly also
works. The backend URL is set in one constant, `API_BASE`, at the top of the script.

## 17. Evaluation

```powershell
python run_evaluation.py                     # full query set
python run_evaluation.py --max-queries 20    # quick CPU run
python run_evaluation.py --device cuda --save-overlays
python run_evaluation.py --save-overlays --results-dir results/v2   # write to a separate folder
```

The script:

1. Re-checks that support and query cases do not overlap.
2. Pairs each query with the **most similar support image**, using cosine similarity on 64×64
   thumbnails of the query *image* only (`--support-strategy random` is also available).
3. Runs all three methods on the **same** samples with one SAM instance.
4. Writes one CSV row per sample, **including failures** (`status`, `error`).
5. Prints the count, successes, failures, and mean ± std for each metric.
6. Draws charts only from the successful rows.

## 18. Metrics

| Metric | Definition |
|---|---|
| Dice | 2TP / (2TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Boundary IoU | IoU of boundary bands of width d = max(1, round(0.02 · diagonal)), obtained by erosion (Cheng et al., 2021) |
| inference_ms | support prototype time (FoB methods) + query encoding + prompting/decoding. A cached embedding reuses its originally measured time. |

**Empty-mask convention:**
* If both masks are empty, the score is 1.0.
* If a denominator is 0 but the other mask is not empty, the score is 0.0.
* All metrics return finite floats in [0, 1].

## 19. Output directories

| Path | Content |
|---|---|
| `data/images`, `data/masks` | Preprocessed PNG slices |
| `data/support_list.txt`, `data/query_list.txt` | Case-level split |
| `results/{baseline,fob,enhanced}_results.csv` | Per-sample evaluation, created only by an actual run |
| `results/v2/` | Same layout (CSVs, summary, `charts/`, `overlays/`) for the run with the §8a prompt constraints; `results/` keeps the original v1 run |
| `results/evaluation_summary.json` | Aggregate statistics |
| `results/charts/metrics_comparison.png`, `metrics_boxplots.png` | Charts |
| `results/overlays/` | API overlays and uncertainty maps; per-sample overlays with `--save-overlays` |
| `experiments.db` | SQLite experiment log |

## 20. Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `sam_loaded: false` | Put the checkpoint at `checkpoints/sam_vit_b_01ec64.pth` and restart the backend |
| `No 'imagesTr' directory found` | Extract the dataset into `data/raw/Task09_Spleen/` |
| Dashboard says "backend unreachable" | Start uvicorn on port 8000, or edit `API_BASE` |
| `CUDA was requested but … False` | Install a CUDA PyTorch wheel or use `--device cpu` |
| FoB rows fail with "SPR found no background candidates" | Canny edges plus the 10 px exclusion covered every pixel with attention > 0.6. This is an honest method failure and is recorded in the CSV. |
| Out of memory | Close other applications. SAM ViT-B needs about 1.5–2 GB of RAM on CPU. |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

## 21. Hardware requirements

* **Minimum (CPU):** 4 cores, 8 GB RAM, about 5 GB of disk (environment, checkpoint and
  preprocessed data; the raw dataset is about 1.5 GB more). One ViT-B encoding of a 1024² image
  takes roughly 5–30 s on a laptop CPU. A full evaluation encodes each query once and each
  distinct support once, so it can take hours on CPU. Use `--max-queries`.
* **Recommended:** an NVIDIA GPU with at least 4 GB of VRAM and CUDA wheels.

## 21a. Measured results (CPU, 53 query slices / 8 query patients, seed 42, nearest support)

Mean ± std over successful rows. v1 = original prompts (`results/`); v2 = with the §8a constraints
(`results/v2/`). Baseline is byte-identical in both runs.

| Method | Run | ok / n | Dice | IoU | Precision | Recall | Boundary IoU |
|---|---|---|---|---|---|---|---|
| Baseline | v1 = v2 | 53/53 | 0.357 ± 0.391 | 0.299 ± 0.347 | 0.381 ± 0.415 | 0.386 ± 0.436 | 0.273 ± 0.328 |
| FoB | v1 | 53/53 | 0.012 ± 0.020 | 0.006 ± 0.011 | 0.006 ± 0.013 | 0.294 ± 0.447 | 0.004 ± 0.010 |
| FoB | v2 | 52/53 | 0.068 ± 0.101 | 0.038 ± 0.061 | 0.048 ± 0.075 | 0.276 ± 0.351 | 0.033 ± 0.054 |
| Enhanced | v1 | 53/53 | 0.056 ± 0.111 | 0.033 ± 0.081 | 0.034 ± 0.086 | 0.449 ± 0.493 | 0.024 ± 0.065 |
| Enhanced | v2 | 53/53 | 0.332 ± 0.291 | 0.243 ± 0.257 | 0.267 ± 0.274 | 0.593 ± 0.397 | 0.221 ± 0.238 |

* Enhanced v2 vs Baseline: no significant difference in Dice (Wilcoxon p = 0.97). Enhanced is better
  on 25 slices and worse on 22. It helps where the transferred box misses (Baseline Dice < 0.5:
  0.045 → 0.221, n = 31) and hurts where the box is good (0.796 → 0.489, n = 22).
* Enhanced v2 vs v1: Dice improved on 42 of 53 slices (Wilcoxon p ≈ 1e-7). Whole-body masks fell
  from 20 to 4.
* FoB v2 failed on 1 slice (`case_052_slice_085`) because no background candidate lies inside the
  body. The failure is recorded in the CSV.

## 22. Known limitations

* **Support box transfer:** a support bounding box is not anatomically correct for a different
  patient or slice. The Baseline and Enhanced methods both inherit this.
* **Enhanced is not better than Baseline overall.** Its mean Dice is similar (0.33 vs 0.36, not
  significant). It trades precision for recall and degrades slices where the box is already good.
  The BCM attention separates spleen from other soft tissue only weakly (≈ 0.63–0.67 vs 0.66–0.68).
  As a result the positive point often misses the spleen, and in-body negatives can cut into it.
* **Prompt constraints are intensity rules** (§8a), tied to the [−200, 300] HU window. Data with a
  different window needs other thresholds.
* **Background-only prompts (FoB)** tell SAM only where the object is not. They do not define the
  spleen boundary, so SAM may segment an arbitrary region. FoB may perform poorly, and this is
  measured rather than hidden.
* **Coordinate alignment:** SPR mixes support edges with query attention on a shared pixel grid.
  This assumes the slices are roughly aligned; no registration is performed.
* **SPR candidates:** on noisy or highly textured images, the edge exclusion can remove every
  candidate. This was observed on synthetic noise images, where 100 % of pixels were excluded.
  Behaviour on real CT has not yet been measured.
* **BCM range:** with the sigmoid at temperature 1, attention lies in [0.27, 0.73], so the 0.6
  threshold corresponds to a cosine similarity above about 0.41.
* **Uncertainty:** SAM ViT-B has no dropout, so MC Dropout is unavailable. Prompt-perturbation
  variance is a substitute that is explicitly labelled as such, with different semantics.
* SAM was **not fine-tuned** on medical data. It works on 2D slices only.
* **No performance claim** is made. Whether FoB or Enhanced beats the baseline must be decided
  from your own `run_evaluation.py` output.

## 23. Reproducibility

* Seed 42 is used for the case split, SPR sampling, support pairing and perturbation jitter. The
  evaluation also seeds torch and numpy. The model runs in `eval()` mode with gradients disabled.
* The `segment-anything` commit is pinned; the other packages are pinned in `requirements.txt`.
* `data/split_summary.json` records the exact case lists. The CSVs record the support slice used
  for every query.
* CPU and GPU floating-point results may differ slightly.

## 24. Academic and research-only disclaimer

This software is for education and research within a B.Tech final-year project. It is not a
medical device. It has not been clinically validated or reviewed by regulators, and its outputs
must not inform patient care. The MSD data is used under its licence. Do not upload identifiable
patient data to this tool.

## Testing

```powershell
python -m pytest tests -v
```

The unit tests use synthetic images, synthetic NIfTI volumes, a lightweight stand-in encoder with
SAM's interface and output shape, and an injected fake service for the API. They do not need the
checkpoint or any network access. One extra test runs against real SAM automatically once the
checkpoint is present.
