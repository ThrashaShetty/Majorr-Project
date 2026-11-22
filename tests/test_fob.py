"""Tests for BPPC, BCM, SPR and the MC-Dropout capability check.

BPPC is exercised with a lightweight stand-in encoder that has the same
interface and output shape as SAM ViT-B (``img_size=1024`` ->
``[1, 256, 64, 64]``), so the SAM checkpoint is not needed. A real-SAM test
runs only when ``checkpoints/sam_vit_b_01ec64.pth`` exists.
"""

import numpy as np
import pytest
import torch
from torch import nn

from src import DEFAULT_CHECKPOINT
from src.fob.bcm import BackgroundContextModeling
from src.fob.bppc import BackgroundPromptPrototype
from src.fob.fob_pipeline import find_dropout_modules
from src.fob.spr import StructureGuidedRefinement


class _Encoder(nn.Module):
    img_size = 1024

    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(16)
        self.proj = nn.Conv2d(3, 256, 1)

    def forward(self, x):
        return self.proj(self.pool(x))


class FakeSam(nn.Module):
    """Mimics the Sam attributes BPPC relies on (image_encoder, preprocess, device)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.image_encoder = _Encoder()
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        return nn.functional.pad(x, (0, 1024 - w, 0, 1024 - h))


def _image(h=512, w=512):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _mask(h=512, w=512):
    m = np.zeros((h, w), np.uint8)
    m[200:300, 220:330] = 1
    return m


def test_bppc_prototype_shape_and_norm():
    bppc = BackgroundPromptPrototype(FakeSam())
    proto = bppc.extract(_image(), _mask())
    assert tuple(proto.shape) == (1, 256)
    assert torch.allclose(proto.norm(dim=1), torch.ones(1), atol=1e-5)
    assert bppc.last_info["feature_shape"] == [1, 256, 64, 64]
    assert 0 < bppc.last_info["background_cells"] < 64 * 64
    assert not bppc.last_info["fallback"]


def test_bppc_non_square_excludes_padding():
    bppc = BackgroundPromptPrototype(FakeSam())
    bg, valid = bppc.background_region(np.zeros((256, 512), np.uint8), (512, 1024), (64, 64))
    assert valid[:32].all() and not valid[32:].any()
    assert (bg == valid).all()


def test_bppc_full_foreground_falls_back_safely():
    bppc = BackgroundPromptPrototype(FakeSam())
    proto = bppc.extract(_image(), np.ones((512, 512), np.uint8))
    assert torch.isfinite(proto).all() and bppc.last_info["fallback"]


def test_bcm_attention_shape_and_range():
    feats = torch.randn(1, 256, 64, 64)
    proto = torch.randn(1, 256)
    att = BackgroundContextModeling().compute(feats, proto)
    assert tuple(att.shape) == (1, 1, 64, 64)
    assert float(att.min()) >= 1 / (1 + np.e) - 1e-5 and float(att.max()) <= 1 / (1 + np.exp(-1)) + 1e-5


def test_bcm_high_similarity_is_background_like():
    proto = torch.zeros(1, 4)
    proto[0, 0] = 1
    feats = torch.zeros(1, 4, 2, 2)
    feats[0, 0, 0, 0] = 1  # same direction as prototype
    feats[0, 1, 1, 1] = 1  # orthogonal
    att = BackgroundContextModeling().compute(feats, proto)
    assert att[0, 0, 0, 0] > att[0, 0, 1, 1]


def test_bcm_rejects_bad_shapes():
    bcm = BackgroundContextModeling()
    with pytest.raises(ValueError):
        bcm.compute(torch.randn(256, 64, 64), torch.randn(1, 256))
    with pytest.raises(ValueError):
        bcm.compute(torch.randn(1, 256, 64, 64), torch.randn(1, 128))


def test_spr_no_candidates_returns_empty():
    spr = StructureGuidedRefinement()
    gray = np.zeros((512, 512), np.uint8)
    att = np.full((64, 64), 0.3, np.float32)  # below 0.6 everywhere
    assert spr.refine(gray, att, n_points=10) == []
    assert spr.last_candidate_count == 0


def test_spr_points_valid_unique_reproducible_and_off_edges():
    gray = np.zeros((512, 512), np.uint8)
    gray[:, 256:] = 255  # one vertical edge at x=256
    att = torch.full((1, 1, 64, 64), 0.9)
    spr = StructureGuidedRefinement(seed=3)
    pts = spr.refine(gray, att, n_points=12, min_distance=20)
    assert len(pts) == 12
    assert len(set(pts)) == 12
    assert all(0 <= x < 512 and 0 <= y < 512 for x, y in pts)
    assert all(abs(x - 256) > 10 for x, _ in pts)  # outside 10-px edge exclusion zone
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in pts)
    assert pts == StructureGuidedRefinement(seed=3).refine(gray, att, 12, 20)
    d = [np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1:]]
    assert min(d) >= 20


def test_dropout_detection():
    assert find_dropout_modules(nn.Sequential(nn.Linear(2, 2))) == []
    found = find_dropout_modules(nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1), nn.Dropout(0.0)))
    assert len(found) == 1


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(), reason="SAM checkpoint not present")
def test_real_sam_shapes_and_no_dropout():
    from src.baseline import load_sam_model

    sam = load_sam_model(DEFAULT_CHECKPOINT, "cpu")
    assert find_dropout_modules(sam) == []
    bppc = BackgroundPromptPrototype(sam)
    feats, _ = bppc.encode(_image())
    assert tuple(feats.shape) == (1, 256, 64, 64)
    proto = bppc.extract_from_features(feats, _mask(), (1024, 1024))
    assert tuple(proto.shape) == (1, 256)
    att = BackgroundContextModeling().compute(feats, proto)
    assert tuple(att.shape) == (1, 1, 64, 64)
