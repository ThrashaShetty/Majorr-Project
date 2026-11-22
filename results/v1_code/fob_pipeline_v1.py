"""End-to-end Focus-on-Background (FoB) and Enhanced FoB pipelines on SAM ViT-B.

Two clearly separated prompting strategies share the same modules:

``mode="fob"`` - **FoB SAM (experimental)**
    Prompts = SPR background points only (label 0). No foreground information is
    given to SAM. *Limitation:* background-only prompts tell SAM where the
    object is NOT; they do not define the spleen boundary, so SAM may return an
    arbitrary complementary region. This mode exists to measure that effect
    honestly, not because it is expected to perform well.

``mode="enhanced"`` - **Enhanced FoB SAM** (fully specified, reproducible)
    Prompts =
      1. the support-mask bounding box (identical to the baseline box prompt);
      2. up to ``n_points`` SPR background points (label 0) restricted to lie
         **outside** that box dilated by ``box_margin`` pixels;
      3. one positive point (label 1) at the location of **minimum** BCM
         background attention inside the box (the least background-like
         query pixel within the transferred box).
    The extra information over the baseline is therefore exactly: background
    negatives and one attention-derived positive point.

Processing order (both modes)
-----------------------------
support image + mask -> resize to query size -> adaptive config ->
BPPC prototype (support encoded with SAM preprocessing) ->
query encoded via ``SamPredictor.set_image`` (``predictor.features``) ->
BCM attention (cropped to the un-padded region) -> SPR points ->
``SamPredictor.predict(point_coords, point_labels[, box])`` -> best mask ->
optional uncertainty -> metrics (if ground truth) -> overlay.

All coordinates passed to ``SamPredictor.predict`` are in original query-image
pixel coordinates; the predictor applies ``ResizeLongestSide.apply_coords``
itself, so no manual transform is applied here.

Uncertainty
-----------
``uncertainty="mc_dropout"`` inspects the model for ``nn.Dropout*`` modules with
``p > 0``. SAM ViT-B (segment-anything 1.0) contains **none**, so MC Dropout is
reported as *unavailable* with the reason - never simulated by ``model.train()``.
``uncertainty="prompt_perturbation"`` instead runs 5 passes with seeded jitter
of the prompt coordinates and reports the per-pixel std of SAM's mask
probabilities, labelled as such. ``"auto"`` tries MC Dropout and, if
unsupported, uses prompt perturbation, recording both facts in
``uncertainty_info``. ``"none"`` disables uncertainty.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import cv2
import numpy as np
import torch
from torch import nn

from src import DEFAULT_CHECKPOINT, DEFAULT_SEED
from src.baseline import (
    EvalSample,
    SamEngine,
    SegmentationError,
    evaluate_samples,
    image_key,
    mask_bounding_box,
    match_size,
    select_best_mask,
    validate_rgb,
)
from src.fob.adaptive import get_adaptive_config
from src.fob.bcm import BackgroundContextModeling
from src.fob.bppc import BackgroundPromptPrototype
from src.fob.spr import StructureGuidedRefinement
from src.metrics import METRIC_KEYS, compute_all_metrics
from src.preprocessing import load_image_rgb, load_mask_binary
from src.visualizer import create_overlay

logger = logging.getLogger(__name__)

VALID_MODES = ("fob", "enhanced")
VALID_UNCERTAINTY = ("auto", "mc_dropout", "prompt_perturbation", "none")
DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)


# --------------------------------------------------------------------------- #
# Uncertainty estimation
# --------------------------------------------------------------------------- #
def find_dropout_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``(name, module)`` for every dropout module with ``p > 0``."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, DROPOUT_TYPES) and float(m.p) > 0.0]


def _predict_probability(engine: SamEngine, **prompt: Any) -> np.ndarray:
    """One SAM decoder pass returning the best mask's probability map (sigmoid of logits)."""
    logits, scores, _ = engine.predictor.predict(multimask_output=True, return_logits=True, **prompt)
    idx = int(np.argmax(scores))
    return 1.0 / (1.0 + np.exp(-logits[idx].astype(np.float64)))


def mc_dropout_uncertainty(
    engine: SamEngine, image_rgb: np.ndarray, prompt: dict[str, Any], n_passes: int = 5
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """MC Dropout: enable only dropout modules, run ``n_passes`` and take pixel std.

    Model parameters are never updated (no backward, ``torch.no_grad``), only
    dropout modules are switched to train mode, and every module's original
    ``training`` flag is restored afterwards.
    """
    modules = find_dropout_modules(engine.model)
    if not modules:
        return None, {
            "available": False, "method": "mc_dropout", "dropout_modules": 0,
            "reason": "SAM ViT-B has no dropout layers with p>0; MC Dropout cannot produce stochastic predictions",
        }
    encoder_has_dropout = any(n.startswith("image_encoder") for n, _ in modules)
    states = {m: m.training for m in engine.model.modules()}
    params_before = [p.detach().clone() for p in list(engine.model.parameters())[:4]]
    probs: list[np.ndarray] = []
    try:
        for _, m in modules:
            m.train()
        with torch.no_grad():
            for _ in range(n_passes):
                if encoder_has_dropout:
                    engine.invalidate()
                    engine.predictor.set_image(image_rgb, image_format="RGB")
                probs.append(_predict_probability(engine, **prompt))
    finally:
        for m, was_training in states.items():
            m.train(was_training)
        if encoder_has_dropout:
            engine.invalidate()
    for before, after in zip(params_before, list(engine.model.parameters())[:4], strict=False):
        if not torch.equal(before, after):
            raise RuntimeError("Model parameters changed during MC Dropout")
    std = np.stack(probs).std(axis=0)
    if float(std.max()) <= 1e-7:
        return None, {"available": False, "method": "mc_dropout", "dropout_modules": len(modules),
                      "reason": "Dropout modules present but no stochastic variation was observed"}
    return std.astype(np.float32), {"available": True, "method": "mc_dropout", "n_passes": n_passes,
                                    "dropout_modules": len(modules), "mean_std": float(std.mean()),
                                    "reason": "pixel-wise std of mask probabilities over MC Dropout passes"}


def prompt_perturbation_uncertainty(
    engine: SamEngine, prompt: dict[str, Any], image_hw: tuple[int, int],
    n_passes: int = 5, point_sigma: float = 6.0, box_jitter: float = 0.03, seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prompt-perturbation variance: seeded Gaussian jitter of points and box corners.

    This measures sensitivity of SAM's output to small prompt changes. It is
    NOT Bayesian / MC Dropout uncertainty and is labelled accordingly.
    """
    h, w = image_hw
    rng = np.random.default_rng(seed)
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(n_passes):
            p = dict(prompt)
            if p.get("point_coords") is not None:
                pts = p["point_coords"] + rng.normal(0.0, point_sigma, size=p["point_coords"].shape)
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                p["point_coords"] = pts.astype(np.float32)
            if p.get("box") is not None:
                b = p["box"].astype(np.float32)
                scale = np.array([b[2] - b[0], b[3] - b[1]] * 2, dtype=np.float32)
                b = b + rng.normal(0.0, box_jitter, size=4).astype(np.float32) * scale
                b = np.clip(b, [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1])
                p["box"] = np.array([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])],
                                    dtype=np.float32)
            probs.append(_predict_probability(engine, **p))
    std = np.stack(probs).std(axis=0).astype(np.float32)
    return std, {"available": True, "method": "prompt_perturbation", "n_passes": n_passes,
                 "mean_std": float(std.mean()), "point_sigma_px": point_sigma, "box_jitter_frac": box_jitter,
                 "reason": "pixel-wise std of mask probabilities under seeded prompt jitter (not MC Dropout)"}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class FoBPipeline:
    """FoB / Enhanced FoB segmentation with SAM ViT-B."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = DEFAULT_CHECKPOINT,
        device: str | torch.device | None = None,
        use_adaptive: bool = True,
        engine: SamEngine | None = None,
        mode: str = "fob",
        uncertainty: str = "auto",
        n_uncertainty_passes: int = 5,
        fixed_n_points: int = 8,
        box_margin: int = 10,
        seed: int = DEFAULT_SEED,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        if uncertainty not in VALID_UNCERTAINTY:
            raise ValueError(f"uncertainty must be one of {VALID_UNCERTAINTY}")
        self.engine = engine if engine is not None else SamEngine(checkpoint_path, device)
        self.device = self.engine.device
        self.sam_model = self.engine.model
        self.predictor = self.engine.predictor
        self.bppc = BackgroundPromptPrototype(self.sam_model)
        self.bcm = BackgroundContextModeling()
        self.spr = StructureGuidedRefinement(seed=seed)
        self.use_adaptive = use_adaptive
        self.mode = mode
        self.uncertainty = uncertainty
        self.n_uncertainty_passes = n_uncertainty_passes
        self.fixed_n_points = fixed_n_points
        self.box_margin = box_margin
        self.seed = seed
        self._proto_cache: dict[str, tuple[torch.Tensor, float, dict[str, Any]]] = {}
        logger.info("FoBPipeline ready (mode=%s, adaptive=%s, device=%s)", mode, use_adaptive, self.device)

    @property
    def method_name(self) -> str:
        return self.mode

    # ------------------------------------------------------------------ helpers
    def _prototype(self, support_rgb: np.ndarray, support_mask: np.ndarray) -> tuple[torch.Tensor, float, dict]:
        key = image_key(support_rgb) + image_key(support_mask)
        if key not in self._proto_cache:
            t0 = time.perf_counter()
            proto = self.bppc.extract(support_rgb, support_mask)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            if len(self._proto_cache) > 64:
                self._proto_cache.clear()
            self._proto_cache[key] = (proto, ms, dict(self.bppc.last_info))
        return self._proto_cache[key]

    def _query_attention(self, prototype: torch.Tensor, image_hw: tuple[int, int]) -> np.ndarray:
        """BCM attention on the query, cropped to the un-padded region, resized to the image (HxW)."""
        features = self.predictor.features
        if features is None or features.ndim != 4:
            raise SegmentationError("Predictor has no image features")
        att = self.bcm.compute(features, prototype)[0, 0].float().cpu().numpy()
        fh, fw = att.shape
        ih, iw = self.predictor.input_size
        size = self.bppc.img_size
        vh, vw = max(1, int(np.ceil(ih / size * fh))), max(1, int(np.ceil(iw / size * fw)))
        att = att[:vh, :vw]
        h, w = image_hw
        return cv2.resize(att, (w, h), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ predict
    def predict_arrays(
        self,
        query_image_rgb: np.ndarray,
        support_image_rgb: np.ndarray,
        support_mask: np.ndarray,
        query_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the FoB pipeline on in-memory arrays. See module docstring."""
        validate_rgb(query_image_rgb)
        validate_rgb(support_image_rgb)
        h, w = query_image_rgb.shape[:2]
        support_image_rgb, support_mask = match_size(support_image_rgb, support_mask, (h, w))
        support_mask = (support_mask > 0).astype(np.uint8)
        support_gray = cv2.cvtColor(support_image_rgb, cv2.COLOR_RGB2GRAY)

        adaptive = get_adaptive_config(support_mask)
        if self.use_adaptive:
            n_points, min_distance = int(adaptive["n_points"]), float(adaptive["min_distance"])
        else:
            n_points, min_distance = self.fixed_n_points, 0.0

        box = mask_bounding_box(support_mask) if self.mode == "enhanced" else None

        with self.engine.lock:
            proto, proto_ms, bppc_info = self._prototype(support_image_rgb, support_mask)
            encode_ms = self.engine.set_image(query_image_rgb)
            t0 = time.perf_counter()
            attention = self._query_attention(proto, (h, w))
            # The attention map is already at image resolution; SPR resizes it to the gray grid (no-op).
            if self.mode == "enhanced":
                assert box is not None
                outside = np.ones((h, w), dtype=bool)
                x0, y0, x1, y1 = box.astype(int)
                m = self.box_margin
                outside[max(0, y0 - m):min(h, y1 + m + 1), max(0, x0 - m):min(w, x1 + m + 1)] = False
                neg = self.spr.refine(support_gray, attention, n_points, min_distance, allowed_region=outside)
                inside = attention[y0:y1 + 1, x0:x1 + 1]
                py, px = np.unravel_index(int(np.argmin(inside)), inside.shape)
                pos = [(int(x0 + px), int(y0 + py))]
                coords = np.array(neg + pos, dtype=np.float32)
                labels = np.array([0] * len(neg) + [1], dtype=np.int32)
                prompt: dict[str, Any] = {"point_coords": coords, "point_labels": labels, "box": box}
            else:
                neg = self.spr.refine(support_gray, attention, n_points, min_distance)
                if not neg:
                    raise SegmentationError(
                        "SPR found no background candidates (attention>0.6 outside edge zone); "
                        "FoB has no prompts for this sample"
                    )
                coords = np.array(neg, dtype=np.float32)
                labels = np.zeros(len(neg), dtype=np.int32)
                prompt = {"point_coords": coords, "point_labels": labels}

            masks, scores, _ = self.predictor.predict(multimask_output=True, **prompt)
            mask, score, idx = select_best_mask(masks, scores)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            method_ms = (time.perf_counter() - t0) * 1000.0

            unc_map, unc_info, unc_ms = None, {"available": False, "method": None,
                                               "reason": "uncertainty disabled"}, 0.0
            if self.uncertainty != "none":
                tu = time.perf_counter()
                if self.uncertainty in ("auto", "mc_dropout"):
                    unc_map, unc_info = mc_dropout_uncertainty(
                        self.engine, query_image_rgb, prompt, self.n_uncertainty_passes)
                if unc_map is None and self.uncertainty in ("auto", "prompt_perturbation"):
                    mc_reason = unc_info.get("reason") if self.uncertainty == "auto" else None
                    unc_map, unc_info = prompt_perturbation_uncertainty(
                        self.engine, prompt, (h, w), self.n_uncertainty_passes, seed=self.seed)
                    if mc_reason:
                        unc_info["mc_dropout_unavailable_reason"] = mc_reason
                unc_ms = (time.perf_counter() - tu) * 1000.0

        inference_ms = proto_ms + encode_ms + method_ms
        result: dict[str, Any] = {"mask": mask}
        if query_mask is not None:
            result.update(compute_all_metrics(mask, query_mask, inference_ms))
        else:
            result.update({k: None for k in METRIC_KEYS})
            result["inference_ms"] = inference_ms
        result["overlay"] = create_overlay(query_image_rgb, mask)
        result["score"] = score
        result["uncertainty_map"] = unc_map
        result["uncertainty_info"] = unc_info
        result["config"] = {
            "mode": self.mode,
            "adaptive_enabled": self.use_adaptive,
            "adaptive": adaptive,
            "n_points_requested": n_points,
            "n_points_used": int(len(coords)),
            "min_distance": min_distance,
            "spr_candidates": self.spr.last_candidate_count,
            "points_xy": coords.astype(int).tolist(),
            "point_labels": labels.tolist(),
            "box_xyxy": None if box is None else box.tolist(),
            "mask_index": idx,
            "bppc": bppc_info,
            "attention_min": float(attention.min()),
            "attention_max": float(attention.max()),
            "timing_ms": {"support_prototype": proto_ms, "query_encode": encode_ms,
                          "prompting_and_decode": method_ms, "uncertainty": unc_ms},
        }
        return result

    def predict(
        self,
        query_image_path: str | Path,
        support_image_path: str | Path,
        support_mask_path: str | Path,
        query_mask_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """File-based prediction interface."""
        query = load_image_rgb(query_image_path)
        support = load_image_rgb(support_image_path)
        smask = load_mask_binary(support_mask_path)
        gt = load_mask_binary(query_mask_path) if query_mask_path is not None else None
        return self.predict_arrays(query, support, smask, gt)

    def evaluate_batch(self, samples: Iterable[EvalSample], overlay_dir: Path | None = None) -> list[dict[str, Any]]:
        """Evaluate samples sequentially; failed samples are recorded with their error."""
        return evaluate_samples(
            lambda s: self.predict(s.query_image, s.support_image, s.support_mask, s.query_mask),
            samples, self.method_name, overlay_dir,
        )
