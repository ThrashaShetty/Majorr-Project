"""Unit tests for src.metrics using synthetic masks."""

import math

import numpy as np
import pytest

from src.metrics import (
    boundary_iou,
    compute_all_metrics,
    dice_score,
    iou_score,
    mask_boundary,
    precision_score,
    recall_score,
)


def square(size=100, x0=20, y0=20, x1=60, y1=60, value=1):
    m = np.zeros((size, size), dtype=np.uint8)
    m[y0:y1, x0:x1] = value
    return m


def test_perfect_overlap_scores_one():
    m = square()
    assert dice_score(m, m) == 1.0
    assert iou_score(m, m) == 1.0
    assert precision_score(m, m) == 1.0
    assert recall_score(m, m) == 1.0
    assert boundary_iou(m, m) == 1.0


def test_dice_and_iou_known_values():
    gt = np.zeros((10, 10), np.uint8)
    gt[0:2, 0:5] = 1  # 10 pixels
    pred = np.zeros((10, 10), np.uint8)
    pred[0:2, 0:3] = 1  # 6 pixels, all inside gt
    assert dice_score(pred, gt) == pytest.approx(2 * 6 / (6 + 10))
    assert iou_score(pred, gt) == pytest.approx(6 / 10)


def test_precision_recall_known_values():
    gt = np.zeros((10, 10), np.uint8)
    gt[:, :5] = 1  # 50 px
    pred = np.zeros((10, 10), np.uint8)
    pred[:, 3:8] = 1  # 50 px, 20 overlap
    assert precision_score(pred, gt) == pytest.approx(20 / 50)
    assert recall_score(pred, gt) == pytest.approx(20 / 50)


def test_accepts_0_255_and_bool_masks():
    m255 = square(value=255)
    mbool = square().astype(bool)
    assert dice_score(m255, mbool) == 1.0


def test_disjoint_masks_score_zero():
    a = square(x0=0, y0=0, x1=10, y1=10)
    b = square(x0=50, y0=50, x1=70, y1=70)
    assert dice_score(a, b) == 0.0
    assert iou_score(a, b) == 0.0
    assert boundary_iou(a, b) == 0.0


def test_empty_mask_conventions():
    empty = np.zeros((32, 32), np.uint8)
    full = square(size=32, x0=4, y0=4, x1=20, y1=20)
    # both empty -> perfect
    for fn in (dice_score, iou_score, precision_score, recall_score, boundary_iou):
        assert fn(empty, empty) == 1.0
    # empty prediction, non-empty gt
    assert dice_score(empty, full) == 0.0
    assert precision_score(empty, full) == 0.0
    assert recall_score(empty, full) == 0.0
    # non-empty prediction, empty gt
    assert recall_score(full, empty) == 0.0
    assert precision_score(full, empty) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice_score(np.zeros((10, 10)), np.zeros((10, 11)))
    with pytest.raises(ValueError):
        dice_score(np.zeros((10,)), np.zeros((10,)))


def test_boundary_band_is_thin_ring():
    m = square(size=200, x0=50, y0=50, x1=150, y1=150)
    band = mask_boundary(m, 0.02)
    d = max(1, round(0.02 * math.hypot(200, 200)))  # 6 px
    assert band.sum() > 0
    assert not band[100, 100]  # interior excluded
    assert band[50, 100] and band[50 + d - 1, 100]
    assert not band[50 + d, 100]


def test_boundary_iou_penalises_boundary_shift_more_than_iou():
    gt = square(size=200, x0=50, y0=50, x1=150, y1=150)
    pred = square(size=200, x0=54, y0=54, x1=154, y1=154)
    assert boundary_iou(pred, gt) < iou_score(pred, gt)


def test_compute_all_metrics_keys_and_finite():
    gt = square()
    pred = square(x0=25)
    out = compute_all_metrics(pred, gt, 12.5)
    assert set(out) == {"dice", "iou", "precision", "recall", "boundary_iou", "inference_ms"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())
    assert out["inference_ms"] == 12.5
    with pytest.raises(ValueError):
        compute_all_metrics(pred, gt, float("nan"))
