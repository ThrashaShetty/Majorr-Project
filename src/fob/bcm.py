"""Background Context Modeling (BCM).

Compares every spatial location of the query feature map with the support
background prototype using cosine similarity, then squashes it with a sigmoid:

    attention[y, x] = sigmoid( cos(q[:, y, x], p) / temperature )

High values = background-like; low values = less background-like (candidate
foreground). With the default ``temperature = 1`` the output lies in
``[sigmoid(-1), sigmoid(1)] = [0.269, 0.731]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class BackgroundContextModeling:
    """Cosine-similarity attention between query features and a background prototype."""

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    @torch.no_grad()
    def compute(self, query_features: torch.Tensor, background_prototype: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: ``[1, C, H, W]`` encoder features of the query image.
            background_prototype: ``[1, C]`` prototype from BPPC.

        Returns:
            Attention map ``[1, 1, H, W]`` in (0, 1).
        """
        if query_features.ndim != 4 or query_features.shape[0] != 1:
            raise ValueError(f"query_features must be [1, C, H, W], got {tuple(query_features.shape)}")
        if background_prototype.ndim != 2 or background_prototype.shape[0] != 1:
            raise ValueError(f"background_prototype must be [1, C], got {tuple(background_prototype.shape)}")
        c, h, w = query_features.shape[1:]
        if background_prototype.shape[1] != c:
            raise ValueError(f"Channel mismatch: features C={c}, prototype C={background_prototype.shape[1]}")
        proto = background_prototype.to(device=query_features.device, dtype=query_features.dtype)
        q = F.normalize(query_features, p=2, dim=1, eps=1e-12).flatten(2)  # [1, C, HW]
        p = F.normalize(proto, p=2, dim=1, eps=1e-12)  # [1, C]
        sim = torch.einsum("bc,bcn->bn", p, q).reshape(1, 1, h, w)
        return torch.sigmoid(sim / self.temperature)
