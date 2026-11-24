# API Documentation (backend/main.py)

FastAPI application `MedSAM Background-Aware Prompting API`, version 0.1.0. This page documents the
code **as it currently is**; nothing here describes planned behaviour.

## Starting the server

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MEDSAM_CHECKPOINT` | `checkpoints/sam_vit_b_01ec64.pth` | SAM weights |
| `MEDSAM_DEVICE` | `auto` | `cpu`, `cuda` or `auto` |
| `MEDSAM_DB` | `experiments.db` | SQLite experiment log |

> ⚠️ **Data-protection note.** By default the server writes:
> * overlay PNGs into `results/overlays/`, which already holds v1 research overlays;
> * experiment rows into `experiments.db`.
>
> Both are part of the protected project state. For demonstrations, point `MEDSAM_DB` at a separate
> file. The overlay folder can currently be changed only through `create_app(overlays_dir=...)`, not
> through an environment variable. Changing that is a backend change and needs approval.

## Endpoints

### `GET /health`
Returns `{"status": "ok", "sam_loaded": <bool>, "device": <str|null>, "detail": <load error|null>}`.
`sam_loaded` reflects the actual state of the model.

### `POST /segment` (multipart/form-data)

| Field | Required | Notes |
|---|---|---|
| `support_image` | yes | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif` or `.tiff` (other types → **415**); ≤ 10 MB; each side 16–4096 px |
| `support_mask` | yes | Read as greyscale; foreground > 0 |
| `query_image` | yes | Same limits as `support_image` |
| `method` | yes | `baseline`, `fob` or `enhanced` (case-insensitive); anything else → **422** |
| `query_mask` | no | Ground truth; without it the accuracy metrics are `null` |

**Response:**
`success`, `method`, `metrics` (`dice`, `iou`, `precision`, `recall`, `boundary_iou`, `inference_ms`),
`has_ground_truth`, `overlay_base64`, `mask_base64`, `uncertainty_base64`, `uncertainty_info`,
`sam_score`, `prompt_summary` (`n_points_used`, `strategy`, `box_xyxy`), `query_image_name`,
`experiment_id`.

**Errors:**
* **400** — empty file, undecodable image, or image dimensions outside 16–4096 px (`backend/main.py`, `read_upload`).
* **413** — file larger than 10 MB.
* **415** — unsupported file type (extension not in the allowed list).
* **422** — invalid `method` only.
* **503** — model not loaded. No segmentation is faked.
* **500** — unexpected server error, returned as a generic message.

**Handling:**
* Uploads are written under server-chosen names in `tmp_uploads/<uuid>/`, which is removed after the
  request.
* Inference runs in a worker thread, and model access is serialised by a lock.

### `GET /results`
Returns all logged experiments, newest first.

### `GET /compare/{query_image_name}`
Returns the latest experiment per method for that query image; `null` for a method not yet run.

## Not exposed by this original API

* **A5 and Single-mask:** not in `backend/main.py`. They are available in the extended API below (`/ext/segment`).
* **The verified research results** (v3 and external): not served by any endpoint.

## Security and limits

* **CORS:** restricted to `localhost` / `127.0.0.1` origins and `null` (a file opened locally);
  methods GET, POST and OPTIONS only.
* **Uploads:** the size cap is enforced while reading; images are decoded and validated; the filename
  never controls the storage path.
* **Not for exposure:** there is no authentication. The server is for local demonstration only and
  should not be exposed to a network.

## Tests

`tests/test_api.py` uses an injected fake service and temporary paths for the database, overlays and
uploads. It does not touch the protected files.

---

## Extended API (`backend/extensions/`)

> **Exploratory (Phase C), not part of the frozen confirmatory analysis.** Added as new files; `backend/main.py`
> is unchanged. See [EXPLORATORY_FEATURES](EXPLORATORY_FEATURES.md).

**Start:** `python -B -m uvicorn backend.extensions.serve:app --host 127.0.0.1 --port 8000`

* The original routes above stay available and share the same SAM engine.
* **Data folder:** `MEDSAM_EXT_DATA`, default `<project>/app_data/`. The extended app writes its SQLite files
  (`experiments_ext.db` for the original routes, `history_ext.db` for `/ext` runs), its overlays and its
  temporary uploads there.
* **Other variables:** `MEDSAM_CHECKPOINT`; `MEDSAM_DEVICE`, which defaults to `cpu` here.
* **Uploads** use the same validation as `/segment`: allowed extensions (otherwise 415); at most 10 MB
  (otherwise 413); each side 16–4096 px (otherwise 400).

### `GET /ext/health`
`{"sam_loaded", "device", "detail", "methods": ["baseline","enhanced","a5","single"], "features": [...], "note"}`.

### `POST /ext/segment` (multipart/form-data)

| Field | Required | Notes |
|---|---|---|
| `support_image`, `support_mask`, `query_image` | yes | as for `/segment` |
| `method` | yes | `baseline`, `enhanced`, `a5` or `single`. Anything else → **422** (`fob` is not offered here) |
| `query_mask` | no | ground truth, used only for metrics; must match the query size, otherwise **400** |
| `k` | no | number of perturbations, 2–16, default 8; outside that range → **422** |
| `uncertainty` | no | `true` (default) or `false` |

**Response:**
* `success`, `note`, `session_id`, `run_id`;
* `method`, `method_used` (for A5, `baseline` or `enhanced`), `a5_ratio` (A5 only), `sam_score`;
* `prompts`: `box` `[x0,y0,x1,y1]`, `points` `[[x,y],…]`, `labels` (1 positive, 0 negative);
* `metrics`: `dice`, `iou`, `precision`, `recall`, `boundary_iou`, or `null` without `query_mask`;
* `has_ground_truth`;
* PNG images in base64: `query_b64`, `support_b64` (support with its mask), `mask_b64`, and `overlay_b64`
  (the mask, with the box in yellow, positive points in green and negative points in red);
* `uncertainty`: `null` if switched off; otherwise `entropy_b64`, `variance_b64` (heatmaps),
  `entropy_score`, `disagreement_score`, `variance_score` and `k`.

**Errors:**
* **400** — mask and image sizes differ;
* **415**, **413** — as for `/segment`;
* **422** — bad method or `k`, or an empty support mask;
* **503** — SAM is not loaded, so nothing is computed.

Every successful call is recorded in the history.

### `POST /ext/refine` (JSON)
`{"session_id": str, "x": float ≥ 0, "y": float ≥ 0, "label": 0 | 1}`. It adds one click and re-runs SAM
with the previous low-resolution logits, all clicks so far, and the original box, asking for one mask.

**Response:** `session_id`, `run_id`, `clicks` (`[[x,y,label],…]`), `metrics`, `mask_b64`, `overlay_b64`
(clicks drawn as rings).

**Errors:**
* **404** — unknown or expired session;
* **422** — click outside the image, or label not 0/1;
* **503** — SAM not loaded.

### `POST /ext/reset` (JSON)
`{"session_id": str}`. Clears the clicks and restores the original prediction; the response has the
same shape as `/ext/refine`. **404** for an unknown session.

### History and export

| Route | Query parameters | Returns |
|---|---|---|
| `GET /ext/history` | `method` (one of the 4, otherwise **422**), `query` (substring of the query filename), `limit` 1–500 (default 100) | `{"note", "runs": [...]}`, newest first |
| `GET /ext/history/compare` | `ids=3,1,2`; at least 2 ids (otherwise **422**); unknown ids → **404**; non-integer ids → **422** | `reference_id` (the first id), `runs` with `values`, `refined`, `n_clicks` and `delta_vs_reference`, `best` per metric, `same_query` |
| `GET /ext/history/export.csv` | `ids` (optional; default all, up to 500) | CSV attachment, every column, with formula-like text neutralised |
| `GET /ext/history/report.html` | `ids` (optional) | a self-contained HTML report with embedded overlays |
| `GET /ext/history/report.pdf` | `ids` (optional) | a PDF report (matplotlib) |
| `GET /ext/history/{run_id}` | — | one run, or **404** |

There is no delete route; rows are kept.

### Tests
* `tests/test_ext_api.py` (13 tests) and `tests/test_ext_history.py` (8 tests).
* Both use fake SAM components and temporary data folders, so neither loads the checkpoint or touches
  protected files.
