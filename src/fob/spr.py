"""Structure-guided Prompt Refinement (SPR).

Selects background point prompts that are (a) confidently background-like
according to the BCM attention map and (b) away from strong image structure.

Steps
-----
1. ``edges = cv2.Canny(gray, 50, 150)``.
2. Dilate edges by 10 pixels (elliptical 21x21 kernel) -> exclusion zone.
3. Resize the attention map (bilinear) to the image size (512 x 512 for this
   project's preprocessed slices).
4. Candidates = (attention > 0.6) AND outside the exclusion zone.
5. ``cv2.findNonZero`` -> candidate (x, y) coordinates.
6. Seeded random sampling of up to ``n_points`` distinct points, optionally
   enforcing a minimum pairwise distance.

Coordinate frames
-----------------
The attention map is computed on the **query** feature grid, while the edge map
comes from the **support** image (as specified by the method). The pipeline
resizes the support image to the query image size beforehand, so both maps
share one pixel grid and the returned ``(x, y)`` are used directly as query
image coordinates. This assumes the support and query axial slices are roughly
aligned (same field of view after preprocessing); it is not a registration.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

CANNY_LOW = 50
CANNY_HIGH = 150
EDGE_DILATION_PX = 10
ATTENTION_THRESHOLD = 0.6


class StructureGuidedRefinement:
    """Sample background point prompts from the attention map, avoiding edges."""

    def __init__(self, seed: int = 42, threshold: float = ATTENTION_THRESHOLD,
                 edge_dilation_px: int = EDGE_DILATION_PX) -> None:
        self.seed = int(seed)
        self.threshold = float(threshold)
        self.edge_dilation_px = int(edge_dilation_px)
        self.last_candidate_count = 0

    @staticmethod
    def _to_numpy_map(attention: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().float().cpu().numpy()
        arr = np.asarray(attention, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Attention map must squeeze to 2-D, got {arr.shape}")
        return arr

    def candidate_mask(self, support_image_gray: np.ndarray, background_attention_map) -> np.ndarray:
        """Boolean HxW map of admissible background-prompt locations."""
        gray = np.asarray(support_image_gray)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if gray.ndim != 2:
            raise ValueError(f"support_image_gray must be 2-D, got {gray.shape}")
        gray = gray.astype(np.uint8)
        h, w = gray.shape
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        k = 2 * self.edge_dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        exclusion = cv2.dilate(edges, kernel) > 0
        att = self._to_numpy_map(background_attention_map)
        att_resized = cv2.resize(att, (w, h), interpolation=cv2.INTER_LINEAR)
        return (att_resized > self.threshold) & ~exclusion

    def refine(
        self,
        support_image_gray: np.ndarray,
        background_attention_map: torch.Tensor | np.ndarray,
        n_points: int,
        min_distance: float = 0.0,
        allowed_region: np.ndarray | None = None,
    ) -> list[tuple[int, int]]:
        """Return up to ``n_points`` distinct ``(x, y)`` background points.

        Args:
            support_image_gray: Grayscale image defining the pixel grid and edges.
            background_attention_map: BCM output (``[1,1,h,w]`` tensor or 2-D array).
            n_points: Maximum number of points to return (0 -> empty list).
            min_distance: Minimum Euclidean distance between returned points.
            allowed_region: Optional boolean HxW mask further restricting candidates.

        Returns:
            List of integer ``(x, y)`` tuples inside the image; empty if no candidate exists.
        """
        if n_points < 0:
            raise ValueError("n_points must be non-negative")
        cand = self.candidate_mask(support_image_gray, background_attention_map)
        if allowed_region is not None:
            if allowed_region.shape != cand.shape:
                raise ValueError("allowed_region shape mismatch")
            cand &= allowed_region.astype(bool)
        pts = cv2.findNonZero(cand.astype(np.uint8))
        if pts is None or n_points == 0:
            self.last_candidate_count = 0 if pts is None else int(len(pts))
            if pts is None:
                logger.info("SPR: no candidate background points found")
            return []
        pts = pts.reshape(-1, 2)
        self.last_candidate_count = int(len(pts))
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(pts))
        h, w = cand.shape
        chosen: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        min_d2 = float(min_distance) ** 2
        for i in order:
            x, y = int(pts[i, 0]), int(pts[i, 1])
            if not (0 <= x < w and 0 <= y < h) or (x, y) in seen:
                continue
            if min_d2 > 0 and any((x - cx) ** 2 + (y - cy) ** 2 < min_d2 for cx, cy in chosen):
                continue
            chosen.append((x, y))
            seen.add((x, y))
            if len(chosen) >= n_points:
                break
        return chosen
