"""API tests with an injected test-double segmentation service and a temporary SQLite DB.

The fake service performs a deterministic threshold segmentation of the query
image. It exists only to exercise HTTP plumbing, validation, persistence and
response structure; it is never used by the real application.
"""

import sqlite3

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from src.baseline import EmptySupportMaskError
from src.metrics import METRIC_KEYS, compute_all_metrics
from src.preprocessing import load_image_rgb, load_mask_binary
from src.visualizer import create_overlay


class FakeService:
    def __init__(self, loaded=True):
        self.sam_loaded = loaded
        self.device = "cpu"
        self.load_error = None if loaded else "SAM checkpoint not found"
        self.calls = []

    def segment(self, method, query_image, support_image, support_mask, query_mask):
        self.calls.append(method)
        q = load_image_rgb(query_image)
        if load_mask_binary(support_mask).sum() == 0:
            raise EmptySupportMaskError("empty")
        mask = q[..., 0] > 127
        res = {"mask": mask, "overlay": create_overlay(q, mask), "score": 0.9,
               "uncertainty_map": np.random.default_rng(0).random(mask.shape).astype(np.float32) if method != "baseline" else None,
               "uncertainty_info": {"available": method != "baseline", "method": "prompt_perturbation"}}
        if query_mask is not None:
            res.update(compute_all_metrics(mask, load_mask_binary(query_mask), 5.0))
        else:
            res.update({k: None for k in METRIC_KEYS}, inference_ms=5.0)
        return res


def png_bytes(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def images():
    img = np.zeros((64, 64), np.uint8)
    img[16:48, 16:48] = 200
    mask = (img > 0).astype(np.uint8) * 255
    return png_bytes(img), png_bytes(mask)


@pytest.fixture()
def client(tmp_path):
    svc = FakeService()
    app = create_app(service=svc, db_path=tmp_path / "test.db",
                     overlays_dir=tmp_path / "overlays", tmp_dir=tmp_path / "tmp")
    with TestClient(app) as c:
        c.svc, c.tmp = svc, tmp_path
        yield c


def files_for(images, query_name="case_001_slice_010.png", with_gt=True):
    img, mask = images
    f = {
        "support_image": ("support.png", img, "image/png"),
        "support_mask": ("support_mask.png", mask, "image/png"),
        "query_image": (query_name, img, "image/png"),
    }
    if with_gt:
        f["query_mask"] = ("gt.png", mask, "image/png")
    return f


def test_health_reports_actual_state(tmp_path):
    for loaded in (True, False):
        app = create_app(service=FakeService(loaded), db_path=tmp_path / f"h{loaded}.db",
                         overlays_dir=tmp_path / "o", tmp_dir=tmp_path / "t")
        with TestClient(app) as c:
            r = c.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok" and r.json()["sam_loaded"] is loaded


def test_segment_success_inserts_experiment(client, images):
    r = client.post("/segment", files=files_for(images), data={"method": "fob"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True and body["method"] == "fob" and body["experiment_id"] == 1
    assert body["metrics"]["dice"] == pytest.approx(1.0)
    assert set(body["metrics"]) == {*METRIC_KEYS, "inference_ms"}
    assert body["overlay_base64"] and body["uncertainty_base64"] and body["mask_base64"]
    with sqlite3.connect(client.tmp / "test.db") as conn:
        rows = conn.execute("SELECT method, dice, query_image_name, overlay_path FROM experiments").fetchall()
    assert rows == [("fob", pytest.approx(1.0), "case_001_slice_010.png", rows[0][3])]
    stored = rows[0][3]
    assert stored.endswith("_fob_overlay.png") and ":" not in stored and not stored.startswith("/")
    assert any(p.name == stored.split("/")[-1] for p in (client.tmp / "overlays").iterdir())
    assert list((client.tmp / "tmp").iterdir()) == []  # temporary uploads cleaned up


def test_segment_without_ground_truth_returns_null_metrics(client, images):
    r = client.post("/segment", files=files_for(images, with_gt=False), data={"method": "baseline"})
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert all(m[k] is None for k in METRIC_KEYS) and m["inference_ms"] == 5.0
    assert r.json()["uncertainty_base64"] is None


def test_results_ordered_desc_and_compare(client, images):
    for method in ("baseline", "enhanced"):
        assert client.post("/segment", files=files_for(images), data={"method": method}).status_code == 200
    rows = client.get("/results").json()
    assert [r["method"] for r in rows] == ["enhanced", "baseline"]
    cmp = client.get("/compare/case_001_slice_010.png").json()
    assert set(cmp) == {"baseline", "fob", "enhanced"}
    assert cmp["fob"] is None and cmp["baseline"]["method"] == "baseline"


def test_results_empty_initially(client):
    assert client.get("/results").json() == []


def test_invalid_method_rejected(client, images):
    r = client.post("/segment", files=files_for(images), data={"method": "magic"})
    assert r.status_code == 422
    assert client.svc.calls == []


def test_invalid_uploads_rejected(client, images):
    f = files_for(images)
    f["query_image"] = ("notes.txt", b"hello", "text/plain")
    assert client.post("/segment", files=f, data={"method": "fob"}).status_code == 415
    f = files_for(images)
    f["support_image"] = ("fake.png", b"not really a png", "image/png")
    assert client.post("/segment", files=f, data={"method": "fob"}).status_code == 400
    f = files_for(images)
    f["support_mask"] = ("empty.png", b"", "image/png")
    assert client.post("/segment", files=f, data={"method": "fob"}).status_code == 400
    f = files_for(images)
    del f["query_image"]
    assert client.post("/segment", files=f, data={"method": "fob"}).status_code == 422
    assert client.svc.calls == []


def test_empty_support_mask_is_422(client, images):
    f = files_for(images)
    f["support_mask"] = ("m.png", png_bytes(np.zeros((64, 64), np.uint8)), "image/png")
    r = client.post("/segment", files=f, data={"method": "baseline"})
    assert r.status_code == 422 and "empty" in r.json()["detail"].lower()


def test_path_traversal_filename_is_sanitised(client, images):
    r = client.post("/segment", files=files_for(images, query_name="../../evil.png"), data={"method": "fob"})
    assert r.status_code == 200
    assert r.json()["query_image_name"] == "evil.png"
    assert not (client.tmp.parent / "evil.png").exists()


def test_segment_503_when_model_not_loaded(tmp_path, images):
    app = create_app(service=FakeService(loaded=False), db_path=tmp_path / "x.db",
                     overlays_dir=tmp_path / "o", tmp_dir=tmp_path / "t")
    with TestClient(app) as c:
        r = c.post("/segment", files=files_for(images), data={"method": "fob"})
        assert r.status_code == 503
        assert c.get("/results").json() == []  # nothing fabricated or recorded
