"""Extension service: one SAM engine serving the four evaluated methods plus uncertainty and click refinement.

The frozen classes are used unchanged (BaselineSegmentor, FoBPipeline, SamEngine). Components are injectable
so the logic is testable without the checkpoint.

Method semantics (identical to the v3 / external evaluation):
  baseline : support-mask bounding box -> SAM multimask -> best mask
  enhanced : FoBPipeline(mode="enhanced") (Enhanced v2)
  a5       : baseline if 0.5 <= baseline_px / support_px <= 2.0, else enhanced
  single   : Enhanced v2 prompts with multimask_output=False
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.extensions import refinement as R
from src.extensions import uncertainty as U

logger = logging.getLogger("backend.extensions")
METHODS = ("baseline", "enhanced", "a5", "single")
A5_LO, A5_HI = 0.5, 2.0


def _bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        raise ValueError("support mask is empty; cannot derive prompts")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)


class ExtService:
    def __init__(self, checkpoint_path: str | Path | None = None, device: str | None = None,
                 engine: Any = None, base: Any = None, enh: Any = None, fob: Any = None) -> None:
        self.checkpoint_path, self.requested_device = checkpoint_path, device
        self.engine, self.base, self.enh, self.fob = engine, base, enh, fob
        self.sam_loaded = engine is not None
        self.device = str(getattr(engine, "device", "fake")) if engine is not None else None
        self.load_error: str | None = None
        self.lock = threading.RLock()

    def load(self) -> None:
        """Load SAM once; failures are recorded, never faked."""
        try:
            from src.baseline import BaselineSegmentor, SamEngine
            from src.fob.fob_pipeline import FoBPipeline
            eng = SamEngine(self.checkpoint_path, self.requested_device)
            self.engine, self.base = eng, BaselineSegmentor(engine=eng)
            self.enh = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none", seed=42)
            self.fob = FoBPipeline(engine=eng, mode="fob", uncertainty="auto", seed=42)
            # the original /segment endpoints keep their exact original objects and settings
            self.legacy = {"baseline": self.base, "fob": self.fob,
                           "enhanced": FoBPipeline(engine=eng, mode="enhanced", uncertainty="auto")}
            self.device, self.sam_loaded = str(eng.device), True
        except FileNotFoundError:
            self.load_error = "SAM checkpoint not found"
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"SAM failed to load ({type(exc).__name__})"
            logger.exception("SAM failed to load")

    # ------------------------------------------------------------------ segmentation
    def segment(self, method: str, q: np.ndarray, s: np.ndarray, smask: np.ndarray, gt: np.ndarray | None,
                k: int = 8, with_uncertainty: bool = True) -> dict[str, Any]:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not self.sam_loaded:
            raise RuntimeError("SAM is not loaded")
        predict = self.engine.predictor.predict
        with self.lock:
            self.engine.invalidate()
            rb = self.base.predict_arrays(q, smask, gt)
            box = _bbox(smask)
            ratio = int(np.asarray(rb["mask"]).sum()) / max(1, int((np.asarray(smask) > 0).sum()))
            use = method if method != "a5" else ("baseline" if A5_LO <= ratio <= A5_HI else "enhanced")
            if use == "baseline":
                prompt = {"box": box}
            else:
                re_ = self.enh.predict_arrays(q, s, smask, gt)
                c = re_["config"]
                prompt = {"box": np.array(c["box_xyxy"], np.float32)}
                if c["points_xy"]:
                    prompt.update(point_coords=np.array(c["points_xy"], np.float32),
                                  point_labels=np.array(c["point_labels"], np.int32))
            multi = method != "single"
            logits, scores, low = predict(multimask_output=multi, return_logits=True, **prompt)
            i = int(np.argmax(scores))
            mask = np.asarray(logits)[i] > 0
            unc = U.slice_uncertainty(predict, prompt, q.shape[:2], k=k, seed=42) if with_uncertainty else None
        return {"method": method, "method_used": "single" if method == "single" else use,
                "a5_ratio": ratio if method == "a5" else None, "prompt": prompt, "mask": mask,
                "score": float(np.asarray(scores)[i]), "low_res": np.asarray(low)[i:i + 1], "uncertainty": unc}

    def refine(self, q: np.ndarray, clicks: list[tuple[float, float, int]], low_res: np.ndarray | None,
               box: np.ndarray | None) -> dict[str, Any]:
        if not self.sam_loaded:
            raise RuntimeError("SAM is not loaded")
        with self.lock:
            self.engine.set_image(q)
            return R.refine(self.engine.predictor.predict, clicks, low_res, box)
