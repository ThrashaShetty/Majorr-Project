"""Phase C.4 tests: experiment history, compare and CSV/HTML/PDF export. Fake SAM; no checkpoint."""
import csv
import io

import pytest
from fastapi.testclient import TestClient

from backend.extensions import report as REP
from backend.extensions.history import HistoryStore, compare
from tests.test_ext_api import make, seg  # fake SAM components and request helper


def run_three(c):
    a = seg(c, "baseline").json()
    b = seg(c, "single", uncertainty="false").json()
    d = seg(c, "a5").json()
    return a, b, d


def test_runs_are_recorded_with_metrics_uncertainty_and_overlay(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        a, b, _ = run_three(c)
        runs = c.get("/ext/history").json()["runs"]
        assert [r["id"] for r in runs] == [3, 2, 1]                      # newest first
        one = c.get(f"/ext/history/{a['run_id']}").json()
        assert one["method"] == "baseline" and one["query_image_name"] == "q.png" and one["k"] == 8
        assert one["dice"] == pytest.approx(a["metrics"]["dice"]) and one["entropy_score"] is not None
        assert one["n_clicks"] == 0 and one["refined_dice"] is None
        assert (tmp_path / "data" / one["overlay_path"]).is_file()
        two = c.get(f"/ext/history/{b['run_id']}").json()
        assert two["entropy_score"] is None and two["k"] is None          # uncertainty off -> not recorded
        assert c.get("/ext/history/999").status_code == 404


def test_refine_and_reset_update_the_run(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        r = seg(c).json()
        c.post("/ext/refine", json={"session_id": r["session_id"], "x": 43, "y": 43, "label": 1})
        row = c.get(f"/ext/history/{r['run_id']}").json()
        assert row["n_clicks"] == 1 and row["refined_dice"] > row["dice"] and row["clicks_json"] == "[[43.0, 43.0, 1]]"
        assert (tmp_path / "data" / row["refined_overlay_path"]).is_file()
        c.post("/ext/reset", json={"session_id": r["session_id"]})
        row = c.get(f"/ext/history/{r['run_id']}").json()
        assert row["n_clicks"] == 0 and row["refined_dice"] is None and row["dice"] == pytest.approx(r["metrics"]["dice"])


def test_filters_and_limits(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        run_three(c)
        assert [r["method"] for r in c.get("/ext/history?method=single").json()["runs"]] == ["single"]
        assert len(c.get("/ext/history?limit=2").json()["runs"]) == 2
        assert len(c.get("/ext/history?query=q.p").json()["runs"]) == 3
        assert c.get("/ext/history?query=%25").json()["runs"] == []        # LIKE wildcard escaped
        assert c.get("/ext/history?method=fob").status_code == 422
        assert c.get("/ext/history?limit=0").status_code == 422


def test_compare(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        run_three(c)
        j = c.get("/ext/history/compare?ids=2,1").json()
        assert j["reference_id"] == 2 and [r["id"] for r in j["runs"]] == [2, 1] and j["same_query"]
        assert j["runs"][0]["delta_vs_reference"]["dice"] == 0
        assert j["runs"][1]["delta_vs_reference"]["entropy_score"] is None   # run 2 has no uncertainty
        assert j["best"]["dice"] in (1, 2)
        assert c.get("/ext/history/compare?ids=1").status_code == 422
        assert c.get("/ext/history/compare?ids=1,77").status_code == 404
        assert c.get("/ext/history/compare?ids=1,x").status_code == 422


def test_csv_export(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        run_three(c)
        r = c.get("/ext/history/export.csv?ids=3,1")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
        rows = list(csv.DictReader(io.StringIO(r.text)))
        assert [x["id"] for x in rows] == ["3", "1"] and "refined_dice" in rows[0]
        assert len(list(csv.DictReader(io.StringIO(c.get("/ext/history/export.csv").text)))) == 3


def test_html_and_pdf_reports(tmp_path):
    app, _ = make(tmp_path)
    with TestClient(app) as c:
        a, _, _ = run_three(c)
        c.post("/ext/refine", json={"session_id": a["session_id"], "x": 43, "y": 43, "label": 1})
        h = c.get("/ext/history/report.html")
        assert h.status_code == 200 and "NOT part of the frozen confirmatory analysis" in h.text
        assert h.text.count("data:image/png;base64,") == 4                 # 3 initial + 1 refined overlay
        p = c.get("/ext/history/report.pdf?ids=1,2")
        assert p.status_code == 200 and p.content[:5] == b"%PDF-" and len(p.content) > 2000
        empty = c.get("/ext/history/report.pdf?ids=")                     # empty ids = all runs
        assert empty.content[:5] == b"%PDF-"


def test_report_escapes_and_csv_neutralises_formulas(tmp_path):
    rows = [{"id": 1, "query_image_name": "<script>x</script>", "method": "=HYPERLINK(1)", "overlay_path": "../../etc/passwd"}]
    html = REP.to_html(rows, tmp_path)
    assert "<script>x" not in html and "&lt;script&gt;" in html and "base64," not in html   # no path escape
    rec = next(csv.DictReader(io.StringIO(REP.to_csv(rows))))
    assert rec["method"] == "'=HYPERLINK(1)"
    assert REP.to_pdf([], tmp_path)[:5] == b"%PDF-"


def test_store_unit(tmp_path):
    s = HistoryStore(tmp_path / "h.db")
    i = s.add({"method": "baseline", "query_image_name": "a", "dice": 0.5})
    j = s.add({"method": "a5", "query_image_name": "b", "dice": 0.7})
    assert s.list(ids=[]) == [] and len(s.list()) == 2
    cmp = compare(s.list(ids=[j, i])[::-1])
    assert cmp["best"]["dice"] == j and compare([])["runs"] == []
