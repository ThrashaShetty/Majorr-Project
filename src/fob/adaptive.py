"""Adaptive prompt configuration from support-mask statistics.

Rules (``fg_ratio = foreground pixels / total pixels``):

======================  ========  ======  ==================
condition               n_points  spread  strategy
======================  ========  ======  ==================
fg_ratio < 0.05         15        tight   small_structure
0.05 <= fg_ratio < 0.20 8         medium  medium_structure
fg_ratio >= 0.20        4         wide    large_structure
======================  ========  ======  ==================

``spread`` is made operational as the minimum pixel distance between sampled
SPR points (``min_distance``): tight = 8 px, medium = 24 px, wide = 48 px.

``edge_density`` = Canny edge pixels of the binary mask / total pixels; it is
reported as a descriptor of boundary complexity and does not change the rules.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

SPREAD_MIN_DISTANCE: dict[str, int] = {"tight": 8, "medium": 24, "wide": 48}


def get_adaptive_config(support_mask_binary: np.ndarray) -> dict[str, Any]:
    """Return the adaptive prompting configuration for a support mask.

    Args:
        support_mask_binary: 2-D mask; any value > 0 is foreground.

    Raises:
        ValueError: for non-2-D or empty-sized masks, or non-finite values.
    """
    m = np.asarray(support_mask_binary)
    if m.ndim == 3 and m.shape[2] == 1:
        m = m[..., 0]
    if m.ndim != 2 or m.size == 0:
        raise ValueError(f"support mask must be a non-empty 2-D array, got shape {m.shape}")
    if np.issubdtype(m.dtype, np.floating) and not np.isfinite(m).all():
        raise ValueError("support mask contains non-finite values")
    binary = (m > 0).astype(np.uint8)
    fg_ratio = float(binary.sum()) / float(binary.size)
    edges = cv2.Canny(binary * 255, 50, 150)
    edge_density = float((edges > 0).sum()) / float(binary.size)

    if fg_ratio < 0.05:
        cfg = {"n_points": 15, "spread": "tight", "strategy": "small_structure"}
    elif fg_ratio < 0.20:
        cfg = {"n_points": 8, "spread": "medium", "strategy": "medium_structure"}
    else:
        cfg = {"n_points": 4, "spread": "wide", "strategy": "large_structure"}
    cfg["min_distance"] = SPREAD_MIN_DISTANCE[cfg["spread"]]
    cfg["fg_ratio"] = fg_ratio
    cfg["edge_density"] = edge_density
    return cfg
