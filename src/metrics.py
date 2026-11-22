"""Segmentation evaluation metrics for binary masks.

All functions accept masks of any numeric/bool dtype; a pixel is foreground
when its value is > 0 (so both {0, 1} and {0, 255} masks work).

Empty-mask convention (documented)
----------------------------------
With TP/FP/FN pixel counts:

* Dice = 2TP / (2TP + FP + FN), IoU = TP / (TP + FP + FN). When the
  denominator is 0 (prediction AND ground truth are both empty) the
  prediction is perfect and the score is **1.0**.
* Precision = TP / (TP + FP). If nothing was predicted (TP + FP = 0) the
  score is 1.0 when the ground truth is also empty and 0.0 otherwise.
* Recall = TP / (TP + FN). If the ground truth is empty (TP + FN = 0) the
  score is 1.0 when nothing was predicted and 0.0 otherwise.
* Boundary IoU follows the IoU convention on the boundary bands.

Every function returns a finite Python ``float`` in [0, 1].
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def to_binary(mask: np.ndarray) -> np.ndarray:
    """Return a boolean 2-D mask (foreground = value > 0).

    Accepts ``(H, W)`` or single-channel ``(H, W, 1)`` arrays.
    """
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got shape {arr.shape}")
    if arr.dtype == bool:
        return arr
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0)
    return arr > 0


def _pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, g = to_binary(pred), to_binary(gt)
    if p.shape != g.shape:
        raise ValueError(f"Prediction shape {p.shape} != ground-truth shape {g.shape}")
    return p, g


def _counts(p: np.ndarray, g: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    return tp, fp, fn


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice similarity coefficient."""
    tp, fp, fn = _counts(*_pair(pred, gt))
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else float(2 * tp / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over union (Jaccard index)."""
    tp, fp, fn = _counts(*_pair(pred, gt))
    denom = tp + fp + fn
    return 1.0 if denom == 0 else float(tp / denom)


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel precision TP / (TP + FP)."""
    tp, fp, fn = _counts(*_pair(pred, gt))
    if tp + fp == 0:
        return 1.0 if fn == 0 else 0.0
    return float(tp / (tp + fp))


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel recall (sensitivity) TP / (TP + FN)."""
    tp, fp, fn = _counts(*_pair(pred, gt))
    if tp + fn == 0:
        return 1.0 if fp == 0 else 0.0
    return float(tp / (tp + fn))


def mask_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Return the inner boundary band of a binary mask.

    The band width ``d = max(1, round(dilation_ratio * image_diagonal))``
    follows the Boundary IoU definition (Cheng et al., CVPR 2021): the mask is
    eroded ``d`` times with a 3x3 kernel and the eroded mask is subtracted from
    the original. The image is zero-padded by one pixel first so that the image
    border is treated as background.
    """
    if dilation_ratio <= 0:
        raise ValueError("dilation_ratio must be positive")
    m = to_binary(mask).astype(np.uint8)
    h, w = m.shape
    diag = math.sqrt(h * h + w * w)
    dilation = max(1, int(round(dilation_ratio * diag)))
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)[1:-1, 1:-1]
    return (m - eroded) > 0


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """IoU between the boundary bands of prediction and ground truth."""
    p, g = _pair(pred, gt)
    pb, gb = mask_boundary(p, dilation_ratio), mask_boundary(g, dilation_ratio)
    inter = int(np.logical_and(pb, gb).sum())
    union = int(np.logical_or(pb, gb).sum())
    return 1.0 if union == 0 else float(inter / union)


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray, inference_time_ms: float) -> dict[str, float]:
    """Compute every metric for one prediction.

    Args:
        pred: Predicted mask.
        gt: Ground-truth mask with the same height/width.
        inference_time_ms: Measured inference wall-clock time in milliseconds.

    Returns:
        Dictionary with keys ``dice, iou, precision, recall, boundary_iou,
        inference_ms``.
    """
    t = float(inference_time_ms)
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"inference_time_ms must be a finite non-negative number, got {inference_time_ms}")
    p, g = _pair(pred, gt)
    return {
        "dice": dice_score(p, g),
        "iou": iou_score(p, g),
        "precision": precision_score(p, g),
        "recall": recall_score(p, g),
        "boundary_iou": boundary_iou(p, g),
        "inference_ms": t,
    }


METRIC_KEYS: tuple[str, ...] = ("dice", "iou", "precision", "recall", "boundary_iou")
