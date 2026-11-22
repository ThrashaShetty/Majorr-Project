"""Baseline SAM: transfer the support-mask bounding box as a box prompt.

Method
------
1. Compute the tight bounding box of the support mask foreground.
2. Use that box, unchanged, as a SAM box prompt on the query image.
3. Request SAM's three multimask outputs and keep the highest-scoring one.

Research limitation
-------------------
A bounding box drawn on a *support* slice is not guaranteed to cover the spleen
on the *query* slice: the organ moves and changes size between slices and
patients. This is therefore a simple **transfer-prompt baseline**, not an
anatomically accurate prompt. Its accuracy depends on how well the support
slice is aligned with the query slice.

Shared infrastructure
---------------------
``SamEngine`` holds the single SAM ViT-B model and ``SamPredictor`` instance
used by every method, a re-entrant lock that serialises access to the
predictor's mutable image state, and a cache of the most recent image
embedding so that the same query image is not encoded repeatedly.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable

import numpy as np
import torch

from src import DEFAULT_CHECKPOINT, resolve_project_path
from src.metrics import METRIC_KEYS, compute_all_metrics, to_binary
from src.preprocessing import load_image_rgb, load_mask_binary
from src.visualizer import create_overlay

logger = logging.getLogger(__name__)

SAM_MODEL_TYPE = "vit_b"
EXPECTED_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the SAM checkpoint file does not exist."""


class SegmentationError(RuntimeError):
    """Raised when a segmentation cannot be produced for a sample."""


class EmptySupportMaskError(SegmentationError, ValueError):
    """Raised when the support mask has no foreground to derive a prompt from."""


# --------------------------------------------------------------------------- #
# Model loading and shared predictor
# --------------------------------------------------------------------------- #
def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return the requested device, or CUDA when available, else CPU."""
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return dev


def load_sam_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
    model_type: str = SAM_MODEL_TYPE,
) -> torch.nn.Module:
    """Build SAM via ``sam_model_registry[model_type]`` and load the checkpoint.

    The state dict is loaded with ``map_location`` (so GPU-saved weights load on
    CPU) and ``weights_only=True``, and with ``strict=True`` so that a wrong or
    truncated checkpoint fails loudly instead of yielding a half-random model.
    """
    from segment_anything import sam_model_registry

    path = resolve_project_path(checkpoint_path)
    if not path.is_file():
        raise CheckpointNotFoundError(
            f"SAM checkpoint not found at {path}. Download {EXPECTED_CHECKPOINT_NAME} "
            "from the official segment-anything repository into checkpoints/."
        )
    dev = resolve_device(device)
    logger.info("Building SAM %s and loading %s on %s", model_type, path.name, dev)
    model = sam_model_registry[model_type](checkpoint=None)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def image_key(image: np.ndarray) -> str:
    """Content hash used to cache image embeddings."""
    h = hashlib.sha1(np.ascontiguousarray(image).tobytes())
    h.update(str(image.shape).encode())
    return h.hexdigest()


class SamEngine:
    """Single shared SAM model + predictor with an embedding cache and a lock."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT,
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        from segment_anything import SamPredictor

        if model is None:
            if checkpoint_path is None:
                raise ValueError("Either checkpoint_path or model must be provided")
            model = load_sam_model(checkpoint_path, device)
        self.model = model
        self.device = next(model.parameters()).device
        self.predictor = SamPredictor(model)
        self.lock = threading.RLock()
        self._key: str | None = None
        self.last_encode_ms: float = 0.0

    def set_image(self, image_rgb: np.ndarray) -> float:
        """Set the predictor image (cached by content). Returns its encoding time in ms.

        For a cache hit, the originally measured encoding time is returned so that
        reported inference times stay comparable across methods.
        """
        validate_rgb(image_rgb)
        key = image_key(image_rgb)
        with self.lock:
            if key != self._key or not self.predictor.is_image_set:
                t0 = time.perf_counter()
                self.predictor.set_image(image_rgb, image_format="RGB")
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                self.last_encode_ms = (time.perf_counter() - t0) * 1000.0
                self._key = key
            return self.last_encode_ms

    def invalidate(self) -> None:
        """Forget the cached embedding (e.g. after stochastic re-encoding)."""
        with self.lock:
            self._key = None


def validate_rgb(image: np.ndarray) -> None:
    """Ensure ``image`` is an (H, W, 3) uint8 array as required by ``SamPredictor.set_image``."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image of shape (H, W, 3), got {getattr(image, 'shape', None)}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")


def mask_bounding_box(mask: np.ndarray) -> np.ndarray:
    """Return the tight XYXY box ``[x0, y0, x1, y1]`` of the mask foreground.

    Raises:
        EmptySupportMaskError: if the mask has no foreground.
    """
    m = to_binary(mask)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        raise EmptySupportMaskError("Support mask is empty; cannot derive a box prompt")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def select_best_mask(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Pick the highest predicted-IoU mask from SAM's multimask output."""
    if masks.ndim != 3 or scores.ndim != 1 or masks.shape[0] != scores.shape[0] or scores.size == 0:
        raise SegmentationError(f"Unexpected SAM output shapes masks={masks.shape} scores={scores.shape}")
    idx = int(np.argmax(scores))
    return masks[idx].astype(bool), float(scores[idx]), idx


def match_size(image_rgb: np.ndarray, mask: np.ndarray, size_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize a support image (bilinear) and mask (nearest) to ``size_hw`` if needed."""
    import cv2

    h, w = size_hw
    if image_rgb.shape[:2] != (h, w):
        image_rgb = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return image_rgb, mask


# --------------------------------------------------------------------------- #
# Baseline segmentor
# --------------------------------------------------------------------------- #
class BaselineSegmentor:
    """Baseline SAM ViT-B using the support-mask bounding box as the query box prompt."""

    method_name = "baseline"

    def __init__(
        self,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT,
        device: str | torch.device | None = None,
        engine: SamEngine | None = None,
    ) -> None:
        self.engine = engine if engine is not None else SamEngine(checkpoint_path, device)
        self.device = self.engine.device
        logger.info("BaselineSegmentor ready on %s", self.device)

    def predict_arrays(
        self,
        query_image_rgb: np.ndarray,
        support_mask: np.ndarray,
        query_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Segment ``query_image_rgb`` using the box of ``support_mask``.

        Returns:
            Dict with ``mask`` (bool HxW), metric keys (``None`` when no query
            ground truth is given), ``inference_ms``, ``overlay`` (RGB), ``score``
            and ``prompt`` details.
        """
        validate_rgb(query_image_rgb)
        h, w = query_image_rgb.shape[:2]
        if support_mask.shape[:2] != (h, w):
            _, support_mask = match_size(query_image_rgb, support_mask, (h, w))
        box = mask_bounding_box(support_mask)
        with self.engine.lock:
            encode_ms = self.engine.set_image(query_image_rgb)
            t0 = time.perf_counter()
            masks, scores, _ = self.engine.predictor.predict(box=box, multimask_output=True)
            mask, score, idx = select_best_mask(masks, scores)
            prompt_ms = (time.perf_counter() - t0) * 1000.0
        inference_ms = encode_ms + prompt_ms
        result: dict[str, Any] = {"mask": mask}
        if query_mask is not None:
            result.update(compute_all_metrics(mask, query_mask, inference_ms))
        else:
            result.update({k: None for k in METRIC_KEYS})
            result["inference_ms"] = inference_ms
        result["overlay"] = create_overlay(query_image_rgb, mask)
        result["score"] = score
        result["prompt"] = {"box_xyxy": box.tolist(), "mask_index": idx}
        result["uncertainty_map"] = None
        result["uncertainty_info"] = {"available": False, "method": None,
                                      "reason": "Uncertainty is not computed for the baseline method"}
        return result

    def predict(
        self,
        query_image_path: str | Path,
        support_image_path: str | Path,
        support_mask_path: str | Path,
        query_mask_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """File-based interface (the support image itself is not used by the baseline)."""
        if not Path(support_image_path).is_file():
            raise FileNotFoundError(f"Support image not found: {support_image_path}")
        query = load_image_rgb(query_image_path)
        support_mask = load_mask_binary(support_mask_path)
        gt = load_mask_binary(query_mask_path) if query_mask_path is not None else None
        return self.predict_arrays(query, support_mask, gt)

    def evaluate_batch(self, samples: Iterable[EvalSample], overlay_dir: Path | None = None) -> list[dict[str, Any]]:
        """Evaluate a list of samples; failures are recorded, never dropped."""
        return evaluate_samples(
            lambda s: self.predict(s.query_image, s.support_image, s.support_mask, s.query_mask),
            samples, self.method_name, overlay_dir,
        )


# --------------------------------------------------------------------------- #
# Batch evaluation helpers shared by all methods
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalSample:
    """One evaluation pairing of a query slice with a support slice."""

    query_image: Path
    query_mask: Path
    support_image: Path
    support_mask: Path


CSV_FIELDS: tuple[str, ...] = (
    "query_image", "support_image", "method", "status", "error",
    *METRIC_KEYS, "inference_ms", "sam_score", "n_points", "strategy",
)


def evaluate_samples(
    predict_fn: Callable[[EvalSample], dict[str, Any]],
    samples: Iterable[EvalSample],
    method: str,
    overlay_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run ``predict_fn`` sequentially over ``samples`` and return one row per sample.

    Failed samples produce a row with ``status='failed'`` and the error message,
    and empty metric fields, so that the number of failures is always reported.
    """
    import cv2

    rows: list[dict[str, Any]] = []
    samples = list(samples)
    for i, s in enumerate(samples, 1):
        row: dict[str, Any] = {k: None for k in CSV_FIELDS}
        row.update(query_image=s.query_image.name, support_image=s.support_image.name, method=method)
        try:
            res = predict_fn(s)
            for k in (*METRIC_KEYS, "inference_ms"):
                v = res.get(k)
                if v is None or not math.isfinite(float(v)):
                    raise SegmentationError(f"metric {k} unavailable or non-finite")
                row[k] = float(v)
            row["sam_score"] = res.get("score")
            cfg = res.get("config") or {}
            row["n_points"] = cfg.get("n_points_used")
            row["strategy"] = (cfg.get("adaptive") or {}).get("strategy")
            row["status"], row["error"] = "success", ""
            if overlay_dir is not None:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                out = overlay_dir / f"{method}_{s.query_image.stem}.png"
                cv2.imwrite(str(out), cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR))
        except Exception as exc:  # noqa: BLE001 - every failure must be recorded, not raised
            logger.warning("[%s] %s failed: %s", method, s.query_image.name, exc)
            row["status"], row["error"] = "failed", f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if i % 10 == 0 or i == len(samples):
            ok = sum(r["status"] == "success" for r in rows)
            logger.info("[%s] %d/%d done (%d ok)", method, i, len(samples), ok)
    return rows


def write_results_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write evaluation rows to CSV (all rows, including failures)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})
    return path
