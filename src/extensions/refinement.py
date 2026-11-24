"""Interactive click refinement for SAM (exploratory; not part of the frozen analysis).

refine(...)        : re-run SAM with the previous low-res logits as `mask_input` plus all clicks so far
                     (and the original box, if any), single-mask output, as in SAM's interactive demo.
mask_to_low_res(...): a binary HxW mask -> SAM's (1, 256, 256) low-res logit input, using SAM's own frame:
                     ResizeLongestSide(1024) placed top-left, zero padded to 1024x1024, then /4.
next_click(...)    : simulated user. The error region is pred XOR gt; the largest connected component of
                     the false negatives or false positives is chosen, and the click goes to its interior point
                     farthest from the region edge. False negative -> positive click (1), false positive ->
                     negative click (0). Perfect prediction -> None.
simulate(...)      : Dice after 0, 1, ..., n clicks. Deterministic (no randomness).

`predict` = any callable with SamPredictor.predict's signature (testable without SAM).
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from src.metrics import dice_score  # frozen metric, used unchanged

SAM_LONG_SIDE = 1024
LOW_RES = 256
MASK_LOGIT = 8.0  # magnitude used when converting a binary mask to logits


def mask_to_low_res(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, bool)
    h, w = m.shape
    scale = SAM_LONG_SIDE / max(h, w)
    nh, nw = int(h * scale + 0.5), int(w * scale + 0.5)
    canvas = np.zeros((SAM_LONG_SIDE, SAM_LONG_SIDE), np.float32)
    canvas[:nh, :nw] = cv2.resize(m.astype(np.float32), (nw, nh), interpolation=cv2.INTER_LINEAR)
    low = cv2.resize(canvas, (LOW_RES, LOW_RES), interpolation=cv2.INTER_AREA)
    return np.where(low >= 0.5, MASK_LOGIT, -MASK_LOGIT).astype(np.float32)[None]


def refine(predict: Callable[..., tuple], clicks: list[tuple[float, float, int]], low_res_logits: np.ndarray | None,
           box: np.ndarray | None = None) -> dict[str, Any]:
    """One refinement step. clicks = [(x, y, label)] accumulated so far (label 1 = add, 0 = remove)."""
    kw: dict[str, Any] = {"multimask_output": False}
    if clicks:
        kw["point_coords"] = np.array([[x, y] for x, y, _ in clicks], np.float32)
        kw["point_labels"] = np.array([lab for _, _, lab in clicks], np.int32)
    if box is not None:
        kw["box"] = np.asarray(box, np.float32)
    if low_res_logits is not None:
        lr = np.asarray(low_res_logits, np.float32)
        kw["mask_input"] = lr if lr.ndim == 3 else lr[None]
    if "point_coords" not in kw and "box" not in kw:
        raise ValueError("refine needs at least one click or a box")
    masks, scores, low = predict(**kw)
    return {"mask": np.asarray(masks)[0].astype(bool), "score": float(np.asarray(scores)[0]),
            "low_res_logits": np.asarray(low)[:1] if low is not None else None}


def next_click(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int] | None:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    best = None  # (area, label, component mask)
    for region, label in ((gt & ~pred, 1), (pred & ~gt, 0)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if best is None or area > best[0]:
                best = (area, label, lab == i)
    if best is None:
        return None
    _, label, comp = best
    dist = cv2.distanceTransform(np.pad(comp, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y), label


def simulate(predict: Callable[..., tuple], initial_mask: np.ndarray, gt: np.ndarray, n_clicks: int = 5,
             box: np.ndarray | None = None, initial_low_res: np.ndarray | None = None) -> dict[str, Any]:
    """Dice trajectory from the initial prediction through n simulated clicks."""
    mask = np.asarray(initial_mask, bool)
    low = initial_low_res if initial_low_res is not None else mask_to_low_res(mask)
    clicks: list[tuple[int, int, int]] = []
    dice = [float(dice_score(mask, gt))]
    for _ in range(n_clicks):
        c = next_click(mask, gt)
        if c is None:                      # already perfect: carry the score forward
            dice.append(dice[-1])
            continue
        clicks.append(c)
        r = refine(predict, clicks, low, box)
        mask, low = r["mask"], (r["low_res_logits"] if r["low_res_logits"] is not None else mask_to_low_res(r["mask"]))
        dice.append(float(dice_score(mask, gt)))
    return {"dice_by_clicks": dice, "clicks": clicks, "final_mask": mask}
