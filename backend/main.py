"""FastAPI backend: segmentation endpoint, SQLite experiment tracking, comparisons.

Run from the project root::

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

Environment variables (optional):

* ``MEDSAM_CHECKPOINT`` - checkpoint path (default ``checkpoints/sam_vit_b_01ec64.pth``)
* ``MEDSAM_DEVICE``     - ``cpu``, ``cuda`` or ``auto`` (default ``auto``)
* ``MEDSAM_DB``         - SQLite path (default ``experiments.db``)

Research prototype only - not clinically validated, not for diagnosis.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import sqlite3
import threading
import uuid
from contextlib import asynccontextmanager, closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DB_PATH,
    OVERLAYS_DIR,
    PROJECT_ROOT,
    TMP_UPLOADS_DIR,
    configure_logging,
    resolve_project_path,
)
from src.baseline import EmptySupportMaskError, SegmentationError
from src.metrics import METRIC_KEYS
from src.preprocessing import decode_image_bytes, load_image_rgb, load_mask_binary
from src.visualizer import create_uncertainty_heatmap

configure_logging()
logger = logging.getLogger("backend")

ALLOWED_METHODS = ("baseline", "fob", "enhanced")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per file
MAX_IMAGE_SIDE = 4096

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    query_image_name TEXT,
    support_image_name TEXT,
    method TEXT,
    dice REAL,
    iou REAL,
    precision REAL,
    recall REAL,
    boundary_iou REAL,
    inference_ms REAL,
    overlay_path TEXT,
    uncertainty_path TEXT
);
"""
COLUMNS = ("id", "timestamp", "query_image_name", "support_image_name", "method", "dice", "iou",
           "precision", "recall", "boundary_iou", "inference_ms", "overlay_path", "uncertainty_path")


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class ExperimentStore:
    """Thread-safe SQLite experiment log (one short-lived connection per operation)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, record: dict[str, Any]) -> int:
        cols = [c for c in COLUMNS if c != "id"]
        sql = f"INSERT INTO experiments ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(sql, [record.get(c) for c in cols])
            return int(cur.lastrowid)

    def all(self) -> list[dict[str, Any]]:
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM experiments ORDER BY timestamp DESC, id DESC").fetchall()
        return [dict(r) for r in rows]

    def latest_by_method(self, query_image_name: str) -> dict[str, dict[str, Any] | None]:
        out: dict[str, dict[str, Any] | None] = {}
        with self._lock, closing(self._connect()) as conn:
            for m in ALLOWED_METHODS:
                row = conn.execute(
                    "SELECT * FROM experiments WHERE query_image_name = ? AND method = ? "
                    "ORDER BY timestamp DESC, id DESC LIMIT 1", (query_image_name, m)).fetchone()
                out[m] = dict(row) if row else None
        return out


# --------------------------------------------------------------------------- #
# Inference service (injectable for tests)
# --------------------------------------------------------------------------- #
class SegmentationService(Protocol):
    """Interface used by the API; the real implementation wraps SAM."""

    sam_loaded: bool
    device: str | None

    def segment(self, method: str, query_image: Path, support_image: Path, support_mask: Path,
                query_mask: Path | None) -> dict[str, Any]: ...


class SamSegmentationService:
    """Loads one SAM ViT-B model and serves all three methods from it."""

    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: str | None = None) -> None:
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.requested_device = device
        self.sam_loaded = False
        self.device: str | None = None
        self.load_error: str | None = None
        self._methods: dict[str, Any] = {}

    def load(self) -> None:
        """Load SAM; failures are recorded (sam_loaded stays False), never faked."""
        try:
            from src.baseline import BaselineSegmentor, SamEngine
            from src.fob.fob_pipeline import FoBPipeline

            engine = SamEngine(self.checkpoint_path, self.requested_device)
            self._methods = {
                "baseline": BaselineSegmentor(engine=engine),
                "fob": FoBPipeline(engine=engine, mode="fob", uncertainty="auto"),
                "enhanced": FoBPipeline(engine=engine, mode="enhanced", uncertainty="auto"),
            }
            self.device = str(engine.device)
            self.sam_loaded = True
            logger.info("SAM loaded on %s", self.device)
        except FileNotFoundError:
            self.load_error = "SAM checkpoint not found"
            logger.warning("SAM checkpoint not found at %s; /segment will return 503", self.checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"SAM failed to load ({type(exc).__name__})"
            logger.exception("SAM failed to load")

    def segment(self, method: str, query_image: Path, support_image: Path, support_mask: Path,
                query_mask: Path | None) -> dict[str, Any]:
        if not self.sam_loaded:
            raise RuntimeError("SAM is not loaded")
        return self._methods[method].predict(query_image, support_image, support_mask, query_mask)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def display_name(filename: str | None) -> str:
    """A sanitised display label derived from an uploaded filename (never used as a path)."""
    base = Path((filename or "").replace("\\", "/")).name
    base = _SAFE_NAME.sub("_", base).strip(" .")[:200]
    return base or "unnamed"


def encode_png_b64(image_rgb: np.ndarray) -> str:
    arr = image_rgb if image_rgb.ndim == 2 else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def read_upload(upload: UploadFile, field: str, grayscale: bool) -> bytes:
    """Read and validate one uploaded image (extension, size, decodability, dimensions)."""
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"{field}: unsupported file type '{ext or 'none'}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        raise HTTPException(400, f"{field}: file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{field}: file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    try:
        img = decode_image_bytes(data, grayscale=grayscale)
    except ValueError:
        raise HTTPException(400, f"{field}: not a valid image") from None
    if max(img.shape[:2]) > MAX_IMAGE_SIDE or min(img.shape[:2]) < 16:
        raise HTTPException(400, f"{field}: image dimensions {img.shape[1]}x{img.shape[0]} out of range")
    return data


def clean_metric(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #
def create_app(
    service: SegmentationService | None = None,
    db_path: str | Path | None = None,
    overlays_dir: Path = OVERLAYS_DIR,
    tmp_dir: Path = TMP_UPLOADS_DIR,
) -> FastAPI:
    """Build the FastAPI app. Pass ``service``/``db_path`` to inject test doubles."""
    db_file = resolve_project_path(db_path or os.environ.get("MEDSAM_DB", DEFAULT_DB_PATH))
    owns_service = service is None
    svc: Any = service or SamSegmentationService(
        os.environ.get("MEDSAM_CHECKPOINT", DEFAULT_CHECKPOINT), os.environ.get("MEDSAM_DEVICE", "auto"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        overlays_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        app.state.store = ExperimentStore(db_file)
        logger.info("SQLite ready at %s", db_file.name)
        if owns_service:
            await run_in_threadpool(svc.load)
        yield

    app = FastAPI(title="MedSAM Background-Aware Prompting API", version="0.1.0", lifespan=lifespan)
    app.state.service = svc
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"success": False, "detail": "Internal server error"})

    def store() -> ExperimentStore:
        return app.state.store

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "sam_loaded": bool(svc.sam_loaded), "device": getattr(svc, "device", None),
                "detail": getattr(svc, "load_error", None)}

    @app.post("/segment")
    async def segment(
        support_image: UploadFile = File(...),
        support_mask: UploadFile = File(...),
        query_image: UploadFile = File(...),
        method: str = Form(...),
        query_mask: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        method = method.strip().lower()
        if method not in ALLOWED_METHODS:
            raise HTTPException(422, f"Invalid method '{method}'. Allowed: {', '.join(ALLOWED_METHODS)}")
        files = {
            "support_image": (await read_upload(support_image, "support_image", False), support_image),
            "support_mask": (await read_upload(support_mask, "support_mask", True), support_mask),
            "query_image": (await read_upload(query_image, "query_image", False), query_image),
        }
        if query_mask is not None and (query_mask.filename or ""):
            files["query_mask"] = (await read_upload(query_mask, "query_mask", True), query_mask)
        if not svc.sam_loaded:
            raise HTTPException(503, "SAM model is not loaded (checkpoint missing or failed to load); "
                                     "no segmentation was performed")

        work = tmp_dir / uuid.uuid4().hex
        work.mkdir(parents=True, exist_ok=False)
        try:
            paths: dict[str, Path] = {}
            for field, (data, up) in files.items():
                # Server-chosen filename: the uploaded name never controls the path.
                p = work / f"{field}{Path(up.filename or '').suffix.lower()}"
                p.write_bytes(data)
                paths[field] = p
            try:
                result = await run_in_threadpool(
                    svc.segment, method, paths["query_image"], paths["support_image"],
                    paths["support_mask"], paths.get("query_mask"))
            except EmptySupportMaskError:
                raise HTTPException(422, "Support mask is empty; cannot derive prompts") from None
            except SegmentationError as exc:
                raise HTTPException(422, f"Segmentation failed: {exc}") from None
            except HTTPException:
                raise
            except Exception:
                logger.exception("Inference failed for method=%s", method)
                raise HTTPException(500, "Inference failed") from None

            exp_tag = uuid.uuid4().hex[:12]
            overlays_dir.mkdir(parents=True, exist_ok=True)
            overlay_file = overlays_dir / f"exp_{exp_tag}_{method}_overlay.png"
            cv2.imwrite(str(overlay_file), cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR))
            unc_b64, unc_rel = None, None
            if result.get("uncertainty_map") is not None:
                heat = create_uncertainty_heatmap(result["uncertainty_map"])
                unc_file = overlays_dir / f"exp_{exp_tag}_{method}_uncertainty.png"
                cv2.imwrite(str(unc_file), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
                unc_b64, unc_rel = encode_png_b64(heat), _rel(unc_file)
            metrics = {k: clean_metric(result.get(k)) for k in (*METRIC_KEYS, "inference_ms")}
            mask_u8 = (np.asarray(result["mask"]).astype(bool).astype(np.uint8)) * 255
            q_name, s_name = display_name(query_image.filename), display_name(support_image.filename)
            exp_id = store().insert({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "query_image_name": q_name, "support_image_name": s_name, "method": method,
                **{k: metrics[k] for k in (*METRIC_KEYS, "inference_ms")},
                "overlay_path": _rel(overlay_file), "uncertainty_path": unc_rel,
            })
            cfg = result.get("config") or {}
            return {
                "success": True,
                "method": method,
                "metrics": metrics,
                "has_ground_truth": "query_mask" in paths,
                "overlay_base64": encode_png_b64(result["overlay"]),
                "mask_base64": encode_png_b64(mask_u8),
                "uncertainty_base64": unc_b64,
                "uncertainty_info": result.get("uncertainty_info"),
                "sam_score": clean_metric(result.get("score")),
                "prompt_summary": {
                    "n_points_used": cfg.get("n_points_used"),
                    "strategy": (cfg.get("adaptive") or {}).get("strategy"),
                    "box_xyxy": cfg.get("box_xyxy", (result.get("prompt") or {}).get("box_xyxy")),
                },
                "query_image_name": q_name,
                "experiment_id": exp_id,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @app.get("/results")
    def results() -> list[dict[str, Any]]:
        return store().all()

    @app.get("/compare/{query_image_name}")
    def compare(query_image_name: str) -> dict[str, Any]:
        return store().latest_by_method(display_name(query_image_name))

    return app


def _rel(path: Path) -> str:
    """Project-relative POSIX path (internal absolute paths are not exposed)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


app = create_app()

__all__ = ["app", "create_app", "ExperimentStore", "SamSegmentationService", "load_image_rgb", "load_mask_binary"]
