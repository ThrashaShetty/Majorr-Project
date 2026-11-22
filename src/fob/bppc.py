"""Background Prompt Prototype Construction (BPPC).

Builds a single L2-normalised feature vector describing what *background*
looks like in the support image, in SAM image-encoder feature space.

Steps
-----
1. Apply SAM's official preprocessing: ``ResizeLongestSide(img_size)`` on the
   uint8 RGB image, HWC->BCHW, then ``Sam.preprocess`` (pixel mean/std
   normalisation + zero padding to ``img_size x img_size``). This is exactly
   what ``SamPredictor.set_image`` does internally.
2. Run ``sam.image_encoder`` -> feature map ``[1, C, Hf, Wf]``
   (ViT-B: ``[1, 256, 64, 64]``; validated at runtime, not assumed).
3. Map the support mask into the same padded frame: resize to the
   ResizeLongestSide target size, zero-pad to ``img_size``, then
   nearest-neighbour downsample to ``(Hf, Wf)``. Padded cells are excluded
   from the background region because they contain no image content.
4. ``background_region = (1 - downsampled_mask) * valid_region``.
5. Prototype = sum(features * background_region) / sum(background_region),
   followed by L2 normalisation -> ``[1, C]``.

If no feature cell is background (foreground covers the whole valid area), the
prototype falls back to the mean over all valid cells and ``last_info``
records ``fallback=True``; this is logged, never hidden.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.metrics import to_binary

logger = logging.getLogger(__name__)


class BackgroundPromptPrototype:
    """Extract an L2-normalised background prototype from a support image/mask pair."""

    def __init__(self, sam_model: Any) -> None:
        """
        Args:
            sam_model: A ``segment_anything.modeling.Sam`` instance (or any object
                exposing ``image_encoder.img_size``, ``image_encoder(x)``,
                ``preprocess(x)`` and ``device``).
        """
        from segment_anything.utils.transforms import ResizeLongestSide

        self.sam_model = sam_model
        self.img_size: int = int(sam_model.image_encoder.img_size)
        self.transform = ResizeLongestSide(self.img_size)
        self.last_info: dict[str, Any] = {}

    @torch.no_grad()
    def encode(self, image_rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Encode an RGB uint8 image with SAM's preprocessing + image encoder.

        Returns:
            ``(features [1, C, Hf, Wf], input_size (h, w))`` where ``input_size``
            is the resized (pre-padding) image size inside the ``img_size`` canvas.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB (H, W, 3), got {image_rgb.shape} {image_rgb.dtype}")
        resized = self.transform.apply_image(image_rgb)
        x = torch.as_tensor(resized, device=self.sam_model.device)
        x = x.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_size = (int(x.shape[-2]), int(x.shape[-1]))
        features = self.sam_model.image_encoder(self.sam_model.preprocess(x))
        if features.ndim != 4 or features.shape[0] != 1:
            raise RuntimeError(f"Unexpected image-encoder output shape {tuple(features.shape)}")
        return features, input_size

    def background_region(
        self, mask: np.ndarray, input_size: tuple[int, int], feature_hw: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(background_region, valid_region)`` float32 arrays of shape ``feature_hw``."""
        m = to_binary(mask).astype(np.uint8)
        ih, iw = input_size
        fh, fw = feature_hw
        if ih > self.img_size or iw > self.img_size:
            raise ValueError(f"input_size {input_size} exceeds encoder size {self.img_size}")
        canvas = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        canvas[:ih, :iw] = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_NEAREST)
        valid = np.zeros_like(canvas)
        valid[:ih, :iw] = 1
        down_mask = cv2.resize(canvas, (fw, fh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        down_valid = cv2.resize(valid, (fw, fh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return (1.0 - down_mask) * down_valid, down_valid

    @torch.no_grad()
    def extract_from_features(
        self, features: torch.Tensor, support_mask_binary: np.ndarray, input_size: tuple[int, int]
    ) -> torch.Tensor:
        """Compute the background prototype from pre-computed encoder features."""
        if features.ndim != 4 or features.shape[0] != 1:
            raise ValueError(f"features must be [1, C, H, W], got {tuple(features.shape)}")
        fh, fw = int(features.shape[2]), int(features.shape[3])
        bg_np, valid_np = self.background_region(support_mask_binary, input_size, (fh, fw))
        bg = torch.from_numpy(bg_np).to(device=features.device, dtype=features.dtype)[None, None]
        area = float(bg.sum().item())
        fallback = False
        if area < 1.0:
            logger.warning("BPPC: support mask covers every valid feature cell; using valid-region mean")
            bg = torch.from_numpy(valid_np).to(device=features.device, dtype=features.dtype)[None, None]
            area = float(bg.sum().item())
            fallback = True
        bg_features = features * bg  # [1, C, Hf, Wf]
        prototype = bg_features.sum(dim=(2, 3)) / max(area, 1.0)  # [1, C]
        prototype = F.normalize(prototype, p=2, dim=1, eps=1e-12)
        self.last_info = {
            "feature_shape": list(features.shape),
            "background_cells": int(area) if not fallback else 0,
            "total_cells": fh * fw,
            "fallback": fallback,
        }
        return prototype

    @torch.no_grad()
    def extract(self, support_image_rgb: np.ndarray, support_mask_binary: np.ndarray) -> torch.Tensor:
        """Encode the support image and return its background prototype ``[1, C]``."""
        features, input_size = self.encode(support_image_rgb)
        return self.extract_from_features(features, support_mask_binary, input_size)
