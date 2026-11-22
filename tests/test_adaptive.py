"""Tests for src.fob.adaptive.get_adaptive_config."""

import numpy as np
import pytest

from src.fob.adaptive import get_adaptive_config


def mask_with_ratio(ratio, size=100):
    m = np.zeros((size, size), np.uint8)
    m.flat[: int(ratio * size * size)] = 1
    return m


@pytest.mark.parametrize(
    "ratio,n_points,spread,strategy",
    [
        (0.01, 15, "tight", "small_structure"),
        (0.049, 15, "tight", "small_structure"),
        (0.05, 8, "medium", "medium_structure"),
        (0.15, 8, "medium", "medium_structure"),
        (0.20, 4, "wide", "large_structure"),
        (0.60, 4, "wide", "large_structure"),
    ],
)
def test_rules(ratio, n_points, spread, strategy):
    cfg = get_adaptive_config(mask_with_ratio(ratio))
    assert (cfg["n_points"], cfg["spread"], cfg["strategy"]) == (n_points, spread, strategy)
    assert cfg["fg_ratio"] == pytest.approx(int(ratio * 10000) / 10000)


def test_edge_density_and_255_masks():
    m = np.zeros((128, 128), np.uint8)
    m[32:96, 32:96] = 255
    cfg = get_adaptive_config(m)
    assert 0 < cfg["edge_density"] < 0.1
    assert cfg["fg_ratio"] == pytest.approx(64 * 64 / 128 / 128)
    assert get_adaptive_config(np.zeros((64, 64), np.uint8))["edge_density"] == 0.0


def test_empty_mask_is_small_structure():
    cfg = get_adaptive_config(np.zeros((50, 50), np.uint8))
    assert cfg["fg_ratio"] == 0.0 and cfg["strategy"] == "small_structure"


def test_invalid_masks_raise():
    with pytest.raises(ValueError):
        get_adaptive_config(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError):
        get_adaptive_config(np.zeros((0, 0)))
    bad = np.zeros((8, 8), np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        get_adaptive_config(bad)
