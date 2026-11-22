"""Regression tests for Enhanced / FoB prompt placement (body-restricted negatives,
soft-tissue positive point, fallbacks, determinism) and a real-SAM smoke test.

Background: in the v1 evaluation the SPR negatives landed in the air around the
patient and SAM segmented the whole body (Enhanced Dice 0.056, recall ~1,
precision ~0.03). These tests pin the corrected behaviour.

Tests 1-5 use a fake SAM engine (no checkpoint) that records the exact prompt
handed to ``SamPredictor.predict``; the BCM attention is replaced by a
controlled map so point placement is fully determined by the test.
"""

import threading

import cv2
import numpy as np
import pytest
import torch

from src import DEFAULT_CHECKPOINT, PROJECT_ROOT
from src.fob.fob_pipeline import SOFT_TISSUE_RANGE, FoBPipeline, body_mask, select_positive_point
from tests.test_fob import FakeSam

H = W = 512
AIR, TISSUE, BONE = 0, 130, 255


class _FakePredictor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.input_size = (1024, 1024)
        self.is_image_set = False
        self.calls: list[dict] = []

    def set_image(self, image_rgb, image_format="RGB"):
        x = torch.as_tensor(cv2.resize(image_rgb, (1024, 1024))).permute(2, 0, 1)[None].float()
        with torch.no_grad():
            self.features = self.model.image_encoder(self.model.preprocess(x))
        self.is_image_set = True

    def predict(self, multimask_output=True, **prompt):
        self.calls.append({k: (None if v is None else np.array(v).copy()) for k, v in prompt.items()})
        masks = np.zeros((3, H, W), dtype=bool)
        masks[:, 200:260, 200:260] = True
        return masks, np.array([0.5, 0.9, 0.7], dtype=np.float32), None


class _FakeEngine:
    def __init__(self):
        self.model = FakeSam()
        self.predictor = _FakePredictor(self.model)
        self.device = torch.device("cpu")
        self.lock = threading.RLock()

    def set_image(self, image_rgb):
        self.predictor.set_image(image_rgb)
        return 1.0


def _phantom() -> np.ndarray:
    """Air background with an elliptical soft-tissue body and a bone ring/rod."""
    g = np.full((H, W), AIR, np.uint8)
    cv2.ellipse(g, (256, 256), (200, 150), 0, 0, 360, TISSUE, -1)
    cv2.rectangle(g, (240, 300), (272, 380), BONE, -1)  # "vertebra"
    return g


def _rgb(g):
    return np.repeat(g[..., None], 3, axis=2)


def _support_mask():
    m = np.zeros((H, W), np.uint8)
    m[220:280, 150:210] = 1
    return m


def _pipeline(mode, attention, seed=42):
    pipe = FoBPipeline(engine=_FakeEngine(), mode=mode, uncertainty="none", seed=seed)
    pipe._query_attention = lambda proto, hw: attention.copy()
    return pipe


def _last_prompt(pipe):
    return pipe.engine.predictor.calls[-1]


# 1 ------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["enhanced", "fob"])
def test_negative_points_lie_inside_body(mode):
    g = _phantom()
    att = np.full((H, W), 0.7, np.float32)  # air is the MOST background-like, as on real CT
    pipe = _pipeline(mode, att)
    res = pipe.predict_arrays(_rgb(g), _rgb(g), _support_mask())
    p = _last_prompt(pipe)
    neg = p["point_coords"][p["point_labels"] == 0].astype(int)
    assert len(neg) > 0
    body = body_mask(g)
    assert all(body[y, x] for x, y in neg), "a negative point fell outside the body mask"
    assert all(g[y, x] != AIR for x, y in neg)
    assert res["config"]["n_negative"] == len(neg)


def test_body_mask_fills_holes_and_erodes():
    g = _phantom()
    g[250:262, 250:262] = AIR  # internal air pocket (e.g. bowel gas) must be filled
    body = body_mask(g)
    assert body[256, 256]
    assert not body[256, 56 + 5]  # within 10 px of the outline -> eroded away
    assert body[256, 56 + 15]
    assert not body[5, 5]
    assert not body_mask(np.zeros((H, W), np.uint8)).any()


# 2 ------------------------------------------------------------------------------
def test_positive_point_avoids_air_and_bone():
    g = _phantom()
    box = np.array([200, 80, 300, 400], np.float32)  # spans air (top), tissue, bone
    att = np.full((H, W), 0.65, np.float32)
    att[g == BONE] = 0.30   # bone and air are the least background-like -> old argmin landed there
    att[g == AIR] = 0.20
    att[150, 230] = 0.50    # lowest eligible soft-tissue pixel
    body = body_mask(g)
    pt = select_positive_point(att, box, g, body)
    assert pt == (230, 150)
    lo, hi = SOFT_TISSUE_RANGE
    assert lo <= g[pt[1], pt[0]] <= hi

    pipe = _pipeline("enhanced", att)
    pipe.predict_arrays(_rgb(g), _rgb(g), _support_mask())
    p = _last_prompt(pipe)
    pos = p["point_coords"][p["point_labels"] == 1].astype(int)
    assert len(pos) == 1
    x, y = pos[0]
    assert lo <= g[y, x] <= hi and g[y, x] not in (AIR, BONE)


# 3 ------------------------------------------------------------------------------
def test_no_eligible_soft_tissue_omits_positive_point():
    g = _phantom()
    g[200:300, 120:240] = BONE  # whole support box region is bone
    att = np.full((H, W), 0.7, np.float32)
    box = np.array([150, 220, 209, 279], np.float32)
    assert select_positive_point(att, box, g, body_mask(g)) is None

    pipe = _pipeline("enhanced", att)
    res = pipe.predict_arrays(_rgb(g), _rgb(g), _support_mask())
    p = _last_prompt(pipe)
    assert 1 not in p["point_labels"].tolist()
    assert res["config"]["positive_point_used"] is False
    assert p["box"] is not None


# 4 ------------------------------------------------------------------------------
def test_no_negatives_falls_back_to_box_and_positive():
    g = _phantom()
    att = np.full((H, W), 0.4, np.float32)  # below the 0.6 SPR threshold -> no negatives
    pipe = _pipeline("enhanced", att)
    res = pipe.predict_arrays(_rgb(g), _rgb(g), _support_mask())
    p = _last_prompt(pipe)
    assert p["box"] is not None
    assert p["point_labels"].tolist() == [1]
    assert res["config"]["n_negative"] == 0
    assert res["mask"].any()


def test_no_negatives_and_no_positive_falls_back_to_box_only():
    g = np.zeros((H, W), np.uint8)  # pure air: empty body, no eligible pixels
    att = np.full((H, W), 0.7, np.float32)
    pipe = _pipeline("enhanced", att)
    pipe.predict_arrays(_rgb(g), _rgb(_phantom()), _support_mask())
    p = _last_prompt(pipe)
    assert p["box"] is not None
    assert p.get("point_coords") is None and p.get("point_labels") is None


def test_fob_without_body_negatives_still_fails_loudly():
    from src.baseline import SegmentationError

    g = np.zeros((H, W), np.uint8)
    pipe = _pipeline("fob", np.full((H, W), 0.7, np.float32))
    with pytest.raises(SegmentationError):
        pipe.predict_arrays(_rgb(g), _rgb(_phantom()), _support_mask())


# 5 ------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["enhanced", "fob"])
def test_same_seed_identical_prompts(mode):
    g = _phantom()
    att = np.random.default_rng(0).uniform(0.55, 0.73, (H, W)).astype(np.float32)
    prompts = []
    for _ in range(2):
        pipe = _pipeline(mode, att, seed=7)
        pipe.predict_arrays(_rgb(g), _rgb(g), _support_mask())
        prompts.append(_last_prompt(pipe))
    a, b = prompts
    assert a.keys() == b.keys()
    for k in a:
        assert (a[k] is None and b[k] is None) or np.array_equal(a[k], b[k])


# 6 ------------------------------------------------------------------------------
_Q = PROJECT_ROOT / "data/images/case_006_slice_095.png"
_S = PROJECT_ROOT / "data/images/case_053_slice_120.png"  # support paired by run_evaluation (nearest)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.is_file(), reason="SAM checkpoint not present")
def test_real_sam_enhanced_case_006_slice_095():
    if not (_Q.is_file() and _S.is_file()):
        pytest.skip("preprocessed data not present")
    pipe = FoBPipeline(mode="enhanced", uncertainty="none", device="cpu")
    res = pipe.predict(_Q, _S, PROJECT_ROOT / "data/masks" / _S.name, PROJECT_ROOT / "data/masks" / _Q.name)
    x0, y0, x1, y1 = res["config"]["box_xyxy"]
    box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    assert res["dice"] >= 0.8, res["dice"]
    assert res["mask"].sum() < 3 * box_area
    from src.preprocessing import load_image_rgb

    body = body_mask(cv2.cvtColor(load_image_rgb(_Q), cv2.COLOR_RGB2GRAY))
    for (x, y), lab in zip(res["config"]["points_xy"], res["config"]["point_labels"], strict=True):
        if lab == 0:
            assert body[y, x]
