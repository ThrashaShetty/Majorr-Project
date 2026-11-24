"""Synthetic tests for src/extensions/refinement.py (no SAM checkpoint needed)."""
import numpy as np
import pytest

from src.extensions import refinement as R

H = W = 64


def disk(cx, cy, r):
    yy, xx = np.mgrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


class FakeClickModel:
    """Starts from mask_input (if given), adds a radius-6 disk at each positive click, removes one at each negative."""

    def __init__(self):
        self.calls = []

    def __call__(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
        self.calls.append({"n_points": 0 if point_coords is None else len(point_coords), "mask_input": mask_input is not None,
                           "multimask": multimask_output, "box": box is not None})
        m = np.zeros((H, W), bool)
        if mask_input is not None:  # undo the 1024/4 = 256 frame for a 64x64 image: 256 / 16 = 16x16... use resize
            import cv2
            up = cv2.resize(mask_input[0], (1024, 1024), interpolation=cv2.INTER_NEAREST)[:1024, :1024]
            m = cv2.resize((up > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        if point_coords is not None:
            for (x, y), lab in zip(point_coords, point_labels):
                d = disk(x, y, 6)
                m = (m | d) if lab == 1 else (m & ~d)
        return m[None], np.array([0.9]), R.mask_to_low_res(m)


def test_mask_to_low_res_shape_and_placement():
    m = np.zeros((40, 80), bool)
    m[:, :40] = True                                   # left half of a wide image
    lr = R.mask_to_low_res(m)
    assert lr.shape == (1, 256, 256) and lr.dtype == np.float32
    # wide image: scaled to 1024x512 top-left -> low-res occupies rows 0..127; left half -> cols 0..127
    assert lr[0, 10, 10] > 0 and lr[0, 10, 200] < 0 and lr[0, 200, 10] < 0


def test_next_click_positive_for_missed_region():
    gt = disk(32, 32, 12)
    pred = gt & (np.mgrid[:H, :W][1] < 32)            # right half missed
    x, y, lab = R.next_click(pred, gt)
    assert lab == 1 and gt[y, x] and not pred[y, x] and x > 32


def test_next_click_negative_for_extra_region():
    gt = disk(20, 20, 8)
    pred = gt | disk(48, 48, 10)                      # big false positive
    x, y, lab = R.next_click(pred, gt)
    assert lab == 0 and pred[y, x] and not gt[y, x]


def test_next_click_none_when_perfect():
    gt = disk(32, 32, 10)
    assert R.next_click(gt, gt) is None


def test_refine_passes_mask_input_points_box_single_mask():
    f = FakeClickModel()
    r = R.refine(f, [(10, 10, 1), (30, 30, 0)], R.mask_to_low_res(disk(20, 20, 5)), box=np.array([0, 0, 40, 40]))
    assert f.calls[-1] == {"n_points": 2, "mask_input": True, "multimask": False, "box": True}
    assert r["mask"].shape == (H, W) and r["low_res_logits"].shape == (1, 256, 256)


def test_refine_requires_a_prompt():
    with pytest.raises(ValueError):
        R.refine(FakeClickModel(), [], None, None)


def test_simulate_improves_and_is_deterministic():
    gt = disk(32, 32, 14)
    start = disk(24, 32, 9)                           # partial overlap
    a = R.simulate(FakeClickModel(), start, gt, n_clicks=5)
    b = R.simulate(FakeClickModel(), start, gt, n_clicks=5)
    assert a["dice_by_clicks"] == b["dice_by_clicks"] and a["clicks"] == b["clicks"]
    assert len(a["dice_by_clicks"]) == 6 and a["dice_by_clicks"][-1] > a["dice_by_clicks"][0]
    assert all(c[2] in (0, 1) for c in a["clicks"])


def test_simulate_perfect_start_stays_perfect_with_no_clicks():
    gt = disk(32, 32, 10)
    r = R.simulate(FakeClickModel(), gt, gt, n_clicks=3)
    assert r["dice_by_clicks"] == [1.0] * 4 and r["clicks"] == []
