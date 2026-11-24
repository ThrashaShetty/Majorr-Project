"""Per-pixel uncertainty for SAM predictions (exploratory; not part of the frozen analysis).

Two independent signals, each giving a [0, 1] map plus a scalar score per slice:

(a) multimask  - SAM's 3 multimask outputs for ONE prompt:
      entropy map      = binary entropy (bits) of the mean sigmoid probability across the 3 masks
      disagreement map = 1 - |2v - 1|, v = fraction of the 3 binary masks that mark the pixel
(b) perturbation - K seeded small prompt perturbations (box-corner jitter, point jitter):
      variance map     = 4 p (1 - p), p = fraction of the K best masks that mark the pixel

Scalar score = mean of the map over the ROI: the union of all predicted masks dilated by
ROI_DILATE_PX (so empty background does not dilute it). No predicted foreground -> score 0.0,
roi_px 0 (reported, never hidden).

`predict` is any callable with SamPredictor.predict's signature (point_coords, point_labels, box,
mask_input, multimask_output, return_logits) -> (masks, scores, low_res_logits), so everything here
is testable without loading SAM.
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

ROI_DILATE_PX = 5
PredictFn = Callable[..., tuple]


def _roi(masks: np.ndarray) -> np.ndarray:
    union = np.asarray(masks, bool).any(axis=0).astype(np.uint8)
    if not union.any():
        return union.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ROI_DILATE_PX + 1, 2 * ROI_DILATE_PX + 1))
    return cv2.dilate(union, k).astype(bool)


def _score(map01: np.ndarray, roi: np.ndarray) -> float:
    return float(map01[roi].mean()) if roi.any() else 0.0


def binary_entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def multimask_uncertainty(logits: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """logits: (M, H, W) SAM mask logits (M = 3 for multimask); scores: (M,) predicted IoUs."""
    logits = np.asarray(logits, np.float64)
    if logits.ndim != 3 or logits.shape[0] < 2:
        raise ValueError(f"need (M>=2, H, W) logits, got {logits.shape}")
    prob = 1.0 / (1.0 + np.exp(-logits))
    entropy = binary_entropy_bits(prob.mean(axis=0))
    binm = logits > 0
    vote = binm.mean(axis=0)
    disagreement = 1.0 - np.abs(2.0 * vote - 1.0)
    roi = _roi(binm)
    best = int(np.argmax(np.asarray(scores)))
    return {"entropy_map": entropy.astype(np.float32), "disagreement_map": disagreement.astype(np.float32),
            "entropy_score": _score(entropy, roi), "disagreement_score": _score(disagreement, roi),
            "roi_px": int(roi.sum()), "best_index": best, "best_mask": binm[best]}


def perturb_prompt(prompt: dict[str, Any], rng: np.random.Generator, image_hw: tuple[int, int],
                   box_jitter: float = 0.05, point_sigma: float = 5.0) -> dict[str, Any]:
    """New prompt dict (input untouched): box corners jittered by N(0, box_jitter * box side),
    points by N(0, point_sigma px); everything clipped to the image; labels unchanged."""
    h, w = image_hw
    out = dict(prompt)
    if prompt.get("box") is not None:
        x0, y0, x1, y1 = np.asarray(prompt["box"], np.float64)
        bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        j = rng.normal(0.0, box_jitter, 4) * np.array([bw, bh, bw, bh])
        nx0, ny0, nx1, ny1 = x0 + j[0], y0 + j[1], x1 + j[2], y1 + j[3]
        nx0, nx1 = sorted((np.clip(nx0, 0, w - 1), np.clip(nx1, 0, w - 1)))
        ny0, ny1 = sorted((np.clip(ny0, 0, h - 1), np.clip(ny1, 0, h - 1)))
        out["box"] = np.array([nx0, ny0, nx1, ny1], np.float32)
    if prompt.get("point_coords") is not None and len(prompt["point_coords"]):
        pts = np.asarray(prompt["point_coords"], np.float64) + rng.normal(0.0, point_sigma, np.shape(prompt["point_coords"]))
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        out["point_coords"] = pts.astype(np.float32)
        out["point_labels"] = np.asarray(prompt["point_labels"], np.int32).copy()
    return out


def perturbation_uncertainty(predict: PredictFn, prompt: dict[str, Any], image_hw: tuple[int, int], k: int = 8,
                             seed: int = 42, box_jitter: float = 0.05, point_sigma: float = 5.0) -> dict[str, Any]:
    if k < 2:
        raise ValueError("k must be >= 2")
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(k):
        masks, scores, _ = predict(multimask_output=True, **perturb_prompt(prompt, rng, image_hw, box_jitter, point_sigma))
        best.append(np.asarray(masks)[int(np.argmax(scores))].astype(bool))
    stack = np.stack(best)
    p = stack.mean(axis=0)
    var = 4.0 * p * (1.0 - p)
    roi = _roi(stack)
    return {"variance_map": var.astype(np.float32), "variance_score": _score(var, roi), "roi_px": int(roi.sum()),
            "k": k, "seed": seed, "box_jitter": box_jitter, "point_sigma": point_sigma}


def slice_uncertainty(predict: PredictFn, prompt: dict[str, Any], image_hw: tuple[int, int], k: int = 8,
                      seed: int = 42) -> dict[str, Any]:
    """Both signals for one prompt on the image already set in the predictor."""
    logits, scores, _ = predict(multimask_output=True, return_logits=True, **prompt)
    mm = multimask_uncertainty(logits, scores)
    pt = perturbation_uncertainty(predict, prompt, image_hw, k=k, seed=seed)
    return {**mm, **pt}


def heatmap(map01: np.ndarray, image_rgb: np.ndarray | None = None, alpha: float = 0.55) -> np.ndarray:
    """RGB uint8 heatmap (inferno) of a [0, 1] map, optionally blended over the image."""
    m = np.clip(np.asarray(map01, np.float64), 0, 1)
    colour = cv2.applyColorMap((m * 255).round().astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    if image_rgb is None:
        return np.ascontiguousarray(colour)
    img = np.asarray(image_rgb, np.float64)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    return np.clip((1 - alpha) * img + alpha * colour, 0, 255).astype(np.uint8)
