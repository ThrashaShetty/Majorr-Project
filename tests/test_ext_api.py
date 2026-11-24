"""Tests for the exploratory extension API (backend/extensions). Fake SAM components; no checkpoint, no network."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.extensions import app as A
from backend.extensions.service import ExtService
from src.extensions.refinement import mask_to_low_res

H = W = 64


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True
    return m


class FakePredictor:
    def __init__(self):
        self.calls = []

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True, return_logits=False):
        self.calls.append({"multimask": multimask_output, "points": 0 if point_coords is None else len(point_coords),
                           "mask_input": mask_input is not None})
        m = rect(*box) if box is not None else np.zeros((H, W), bool)
        if point_coords is not None:
            yy, xx = np.mgrid[:H, :W]
            for (x, y), lab in zip(point_coords, point_labels):
                d = (xx - x) ** 2 + (yy - y) ** 2 <= 25
                m = (m | d) if lab == 1 else (m & ~d)
        n = 3 if multimask_output else 1
        ms = np.stack([m] * n)
        out = np.where(ms, 8.0, -8.0) if return_logits else ms
        return out, np.linspace(0.9, 0.7, n), np.repeat(mask_to_low_res(m), n, axis=0)


class FakeEngine:
    device = "fake"

    def __init__(self):
        self.predictor, self.invalidations, self.images_set = FakePredictor(), 0, 0

    def invalidate(self):
        self.invalidations += 1

    def set_image(self, img):
        self.images_set += 1


class FakeBase:
    def __init__(self, scale=1.0):
        self.scale = scale

    def predict_arrays(self, q, smask, gt):
        ys, xs = np.nonzero(smask)
        cx, cy, hw, hh = xs.mean(), ys.mean(), (xs.max() - xs.min()) / 2 * self.scale, (ys.max() - ys.min()) / 2 * self.scale
        return {"mask": rect(cx - hw, cy - hh, cx + hw, cy + hh), "score": 0.9}


class FakeEnh:
    def __init__(self):
        self.calls = 0

    def predict_arrays(self, q, s, smask, gt):
        self.calls += 1
        ys, xs = np.nonzero(smask)
        return {"mask": smask > 0, "config": {"box_xyxy": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                                              "points_xy": [[5, 5], [30, 30]], "point_labels": [0, 1]}}


class FakeLegacy:
    def predict(self, q, s, sm, qm):
        img = np.zeros((H, W, 3), np.uint8)
        return {"mask": np.zeros((H, W), bool), "overlay": img, "score": 0.5, "dice": None, "iou": None, "precision": None,
                "recall": None, "boundary_iou": None, "inference_ms": 1.0, "uncertainty_map": None, "uncertainty_info": None,
                "prompt": {"box_xyxy": [0, 0, 1, 1]}}


def png(arr):
    ok, buf = cv2.imencode(".png", arr)
    return buf.tobytes()


IMG = png(np.full((H, W, 3), 100, np.uint8))
SM = png((rect(20, 20, 40, 40) * 255).astype(np.uint8))
GT = png((rect(22, 22, 44, 44) * 255).astype(np.uint8))


def make(tmp_path, scale=1.0, loaded=True):
    if loaded:
        ext = ExtService(engine=FakeEngine(), base=FakeBase(scale), enh=FakeEnh())
        ext.legacy = {m: FakeLegacy() for m in ("baseline", "fob", "enhanced")}
    else:
        ext = ExtService()
    app = A.create_ext_app(ext=ext, data_dir=tmp_path / "data", autoload=False)
    return app, ext


def seg(c, method="baseline", gt=True, **form):
    files = {"support_image": ("s.png", IMG, "image/png"), "support_mask": ("sm.png", SM, "image/png"),
             "query_image": ("q.png", IMG, "image/png")}
    if gt:
        files["query_mask"] = ("gt.png", GT, "image/png")
    return c.post("/ext/segment", files=files, data={"method": method, **form})


def test_health_and_original_routes_still_served(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        h = c.get("/ext/health").json()
        assert h["sam_loaded"] and set(h["methods"]) == {"baseline", "enhanced", "a5", "single"} and "Exploratory" in h["note"]
        assert c.get("/health").json()["sam_loaded"] is True               # original route, shared engine


def test_baseline_segment_returns_everything(tmp_path):
    app, ext = make(tmp_path)
    with TestClient(app) as c:
        r = seg(c).json()
    assert r["method_used"] == "baseline" and r["prompts"]["box"] == [20.0, 20.0, 40.0, 40.0] and r["prompts"]["points"] == []
    assert r["metrics"]["dice"] is not None and 0 < r["metrics"]["dice"] < 1 and r["has_ground_truth"]
    for k in ("mask_b64", "overlay_b64", "query_b64", "support_b64"):
        assert len(r[k]) > 50
    assert r["uncertainty"]["k"] == 8 and len(r["uncertainty"]["entropy_b64"]) > 50 and ext.engine.invalidations == 1


@pytest.mark.parametrize("scale,used", [(1.0, "baseline"), (0.3, "enhanced")])
def test_a5_switches_on_area_ratio(tmp_path, scale, used):
    app, ext = make(tmp_path, scale=scale)
    with TestClient(app) as c:
        r = seg(c, "a5").json()
    assert r["method"] == "a5" and r["method_used"] == used and (ext.enh.calls == 1) == (used == "enhanced")
    assert (0.5 <= r["a5_ratio"] <= 2.0) == (used == "baseline")
    assert (len(r["prompts"]["points"]) == 2) == (used == "enhanced")


def test_single_mask_uses_single_output(tmp_path):
    app, ext = make(tmp_path)
    with TestClient(app) as c:
        r = seg(c, "single", uncertainty="false").json()
    assert r["method_used"] == "single" and r["uncertainty"] is None
    assert ext.engine.predictor.calls[0]["multimask"] is False and ext.engine.predictor.calls[0]["points"] == 2


@pytest.mark.parametrize("form,code", [({"method": "fob"}, 422), ({"method": "baseline", "k": "1"}, 422), ({"method": "baseline", "k": "17"}, 422)])
def test_validation_errors(tmp_path, form, code):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        files = {"support_image": ("s.png", IMG, "image/png"), "support_mask": ("sm.png", SM, "image/png"),
                 "query_image": ("q.png", IMG, "image/png")}
        assert c.post("/ext/segment", files=files, data=form).status_code == code


def test_empty_support_mask_wrong_type_and_size_mismatch(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        empty = png(np.zeros((H, W), np.uint8))
        f = {"support_image": ("s.png", IMG, "image/png"), "support_mask": ("sm.png", empty, "image/png"), "query_image": ("q.png", IMG, "image/png")}
        assert c.post("/ext/segment", files=f, data={"method": "baseline"}).status_code == 422
        f["support_mask"] = ("sm.gif", SM, "image/gif")
        assert c.post("/ext/segment", files=f, data={"method": "baseline"}).status_code == 415
        f["support_mask"] = ("sm.png", png(np.zeros((32, 32), np.uint8) + 255), "image/png")
        assert c.post("/ext/segment", files=f, data={"method": "baseline"}).status_code == 400


def test_not_loaded_returns_503(tmp_path):
    app, _ = make(tmp_path, loaded=False)
    with TestClient(app) as c:
        assert seg(c).status_code == 503 and c.get("/ext/health").json()["sam_loaded"] is False


def test_refine_and_reset(tmp_path):
    app, ext = make(tmp_path)
    with TestClient(app) as c:
        r = seg(c).json()
        sid, d0 = r["session_id"], r["metrics"]["dice"]
        r1 = c.post("/ext/refine", json={"session_id": sid, "x": 43, "y": 43, "label": 1}).json()
        assert r1["clicks"] == [[43.0, 43.0, 1]] and r1["metrics"]["dice"] > d0
        call = ext.engine.predictor.calls[-1]
        assert call["mask_input"] and call["multimask"] is False and call["points"] == 1 and ext.engine.images_set == 1
        r2 = c.post("/ext/refine", json={"session_id": sid, "x": 21, "y": 21, "label": 0}).json()
        assert len(r2["clicks"]) == 2
        r3 = c.post("/ext/reset", json={"session_id": sid}).json()
        assert r3["clicks"] == [] and r3["metrics"]["dice"] == d0
        assert c.post("/ext/refine", json={"session_id": sid, "x": 64, "y": 10, "label": 1}).status_code == 422
        assert c.post("/ext/refine", json={"session_id": "nope", "x": 1, "y": 1, "label": 1}).status_code == 404
        assert c.post("/ext/refine", json={"session_id": sid, "x": 1, "y": 1, "label": 2}).status_code == 422


def test_sessions_are_bounded(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        ids = [seg(c, uncertainty="false").json()["session_id"] for _ in range(A.MAX_SESSIONS + 1)]
        assert c.post("/ext/reset", json={"session_id": ids[0]}).status_code == 404
        assert c.post("/ext/reset", json={"session_id": ids[-1]}).status_code == 200


def test_writes_stay_in_the_data_dir(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        files = {"support_image": ("s.png", IMG, "image/png"), "support_mask": ("sm.png", SM, "image/png"),
                 "query_image": ("q.png", IMG, "image/png")}
        assert c.post("/segment", files=files, data={"method": "baseline"}).status_code == 200   # original route
    data = tmp_path / "data"
    assert (data / "experiments_ext.db").exists() and any((data / "overlays").iterdir())
