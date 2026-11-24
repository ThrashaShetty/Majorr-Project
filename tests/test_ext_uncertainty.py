"""Synthetic tests for src/extensions/uncertainty.py (no SAM checkpoint needed)."""
import numpy as np
import pytest

from src.extensions import uncertainty as U

H = W = 64


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[int(y0):int(y1), int(x0):int(x1)] = True
    return m


class FakePredictor:
    """Mask = the prompt box filled; logits = +/-8 on that mask, with optional per-mask shrink."""

    def __init__(self, shrink=(0, 0, 0)):
        self.shrink, self.calls = shrink, 0

    def __call__(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
        self.calls += 1
        x0, y0, x1, y1 = box
        ms = np.stack([rect(x0 + s, y0 + s, x1 - s, y1 - s) for s in self.shrink])
        logits = np.where(ms, 8.0, -8.0)
        return (logits if return_logits else ms), np.array([0.9, 0.8, 0.7]), None


PROMPT = {"box": np.array([16, 16, 48, 48], np.float32), "point_coords": np.array([[30, 30]], np.float32),
          "point_labels": np.array([1], np.int32)}


def test_unanimous_masks_have_near_zero_multimask_uncertainty():
    logits, scores, _ = FakePredictor()(return_logits=True, **PROMPT)
    r = U.multimask_uncertainty(logits, scores)
    # logits +/-8 -> sigmoid 0.99966 -> binary entropy ~0.0044 bits: near zero, not exactly zero
    assert r["entropy_score"] < 0.01 and r["disagreement_score"] == 0 and r["roi_px"] > 0 and r["best_index"] == 0


def test_disagreeing_masks_raise_uncertainty_at_the_rim():
    logits, scores, _ = FakePredictor(shrink=(0, 3, 6))(return_logits=True, **PROMPT)
    r = U.multimask_uncertainty(logits, scores)
    assert r["entropy_score"] > 0.05 and r["disagreement_score"] > 0.05
    assert r["entropy_map"][32, 32] < 0.01 and r["entropy_map"][17, 32] > 0.5   # centre agreed, rim disputed


def test_empty_prediction_scores_zero_and_reports_it():
    logits = np.full((3, H, W), -8.0)
    r = U.multimask_uncertainty(logits, np.array([0.1, 0.1, 0.1]))
    assert r["entropy_score"] == 0.0 and r["roi_px"] == 0


def test_perturbation_is_seeded_and_deterministic():
    a = U.perturbation_uncertainty(FakePredictor(), PROMPT, (H, W), k=6, seed=7)
    b = U.perturbation_uncertainty(FakePredictor(), PROMPT, (H, W), k=6, seed=7)
    c = U.perturbation_uncertainty(FakePredictor(), PROMPT, (H, W), k=6, seed=8)
    assert np.array_equal(a["variance_map"], b["variance_map"]) and a["variance_score"] == b["variance_score"]
    assert not np.array_equal(a["variance_map"], c["variance_map"])
    assert 0 < a["variance_score"] <= 1 and a["variance_map"][32, 32] == 0    # stable core, unstable edge


def test_perturb_prompt_does_not_mutate_input_and_stays_in_bounds():
    before = {k: v.copy() for k, v in PROMPT.items()}
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = U.perturb_prompt(PROMPT, rng, (H, W), box_jitter=0.5, point_sigma=40)
        x0, y0, x1, y1 = p["box"]
        assert 0 <= x0 <= x1 <= W - 1 and 0 <= y0 <= y1 <= H - 1
        assert (p["point_coords"] >= 0).all() and (p["point_coords"][:, 0] <= W - 1).all()
        assert np.array_equal(p["point_labels"], PROMPT["point_labels"])
    assert all(np.array_equal(PROMPT[k], before[k]) for k in PROMPT)


def test_slice_uncertainty_combines_both_and_call_count():
    f = FakePredictor(shrink=(0, 2, 4))
    r = U.slice_uncertainty(f, PROMPT, (H, W), k=4, seed=1)
    assert {"entropy_score", "disagreement_score", "variance_score", "entropy_map", "variance_map"} <= set(r)
    assert f.calls == 1 + 4


def test_k_must_be_at_least_two():
    with pytest.raises(ValueError):
        U.perturbation_uncertainty(FakePredictor(), PROMPT, (H, W), k=1)


def test_heatmap_shapes():
    m = np.linspace(0, 1, H * W).reshape(H, W)
    assert U.heatmap(m).shape == (H, W, 3) and U.heatmap(m).dtype == np.uint8
    img = np.full((H, W, 3), 100, np.uint8)
    assert U.heatmap(m, img).shape == (H, W, 3)
