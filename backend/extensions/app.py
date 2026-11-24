"""Extended API (exploratory): the original app from backend/main.py, unchanged, plus /ext routes.

Run from the project root (serve.py builds the app; importing this module never loads SAM):
    python -B -m uvicorn backend.extensions.serve:app --host 127.0.0.1 --port 8000
Open frontend/dashboard.html (new page; frontend/index.html is untouched).

Data: everything this app writes (SQLite log, overlays, temp uploads) goes to MEDSAM_EXT_DATA
(default <project>/app_data/), never to results/overlays or experiments.db.

/ext routes
  GET  /ext/health   model state + feature list
  POST /ext/segment  multipart: support_image, support_mask, query_image, method (baseline|enhanced|a5|single),
                     optional query_mask, k (2-16, default 8), uncertainty (true/false). Returns prompts (box,
                     points), mask, overlay with prompts drawn, uncertainty heatmaps + scores, metrics, session_id.
  POST /ext/refine   JSON {session_id, x, y, label (1 add | 0 remove)} -> refined mask, overlay, metrics.
  POST /ext/reset    JSON {session_id} -> clears the clicks, back to the original prediction.
  GET  /ext/history                    ?method=&query=&limit= (1-500)  -> recorded runs, newest first
  GET  /ext/history/compare            ?ids=3,1,2  -> side by side + deltas vs the first id
  GET  /ext/history/export.csv         ?ids= (optional; default = all, up to 500)
  GET  /ext/history/report.html        ?ids=  self-contained HTML report
  GET  /ext/history/report.pdf         ?ids=  PDF report (matplotlib)
  GET  /ext/history/{run_id}           one run
Sessions live in memory only (at most MAX_SESSIONS, oldest dropped). Each /ext/segment call is recorded in
<data>/history_ext.db with its overlay in <data>/ext_overlays/; refinement updates that run's refined_* fields.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from backend.main import clean_metric, create_app, display_name, encode_png_b64, read_upload
from backend.extensions import report as REP
from backend.extensions.history import HistoryStore, compare as compare_runs
from backend.extensions.service import METHODS, ExtService
from src import DEFAULT_CHECKPOINT, PROJECT_ROOT
from src.extensions.uncertainty import heatmap
from src.metrics import METRIC_KEYS, compute_all_metrics
from src.preprocessing import decode_image_bytes

MAX_SESSIONS = 8
EXPLORATORY_NOTE = "Exploratory features; not part of the frozen confirmatory analysis. Research prototype, not for diagnosis."


class LegacyAdapter:
    """Lets the original /segment, /results, /compare routes share the extension's single SAM engine."""

    def __init__(self, ext: ExtService) -> None:
        self.ext = ext

    @property
    def sam_loaded(self) -> bool:
        return self.ext.sam_loaded

    @property
    def device(self) -> str | None:
        return self.ext.device

    @property
    def load_error(self) -> str | None:
        return self.ext.load_error

    def segment(self, method, query_image, support_image, support_mask, query_mask):
        if not self.ext.sam_loaded:
            raise RuntimeError("SAM is not loaded")
        with self.ext.lock:
            return self.ext.legacy[method].predict(query_image, support_image, support_mask, query_mask)


def draw_prompts(img: np.ndarray, mask: np.ndarray | None, prompt: dict[str, Any], clicks=()) -> np.ndarray:
    out = np.ascontiguousarray(img.copy())
    if mask is not None:
        m = np.asarray(mask, bool)
        tint = out.copy()
        tint[m] = (0.55 * out[m] + 0.45 * np.array([0, 200, 255])).astype(np.uint8)
        out = tint
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, (0, 160, 255), 1)
    if prompt.get("box") is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in prompt["box"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 220, 0), 1)
    for (x, y), lab in zip(prompt.get("point_coords", []) if prompt.get("point_coords") is not None else [],
                           prompt.get("point_labels", []) if prompt.get("point_labels") is not None else []):
        cv2.circle(out, (int(x), int(y)), 4, (0, 220, 0) if lab == 1 else (255, 40, 40), -1)
    for x, y, lab in clicks:
        cv2.circle(out, (int(x), int(y)), 6, (0, 220, 0) if lab == 1 else (255, 40, 40), 2)
    return out


class RefineBody(BaseModel):
    session_id: str
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    label: int = Field(ge=0, le=1)


class ResetBody(BaseModel):
    session_id: str


def create_ext_app(ext: ExtService | None = None, data_dir: Path | None = None, autoload: bool = True):
    data_dir = Path(data_dir or os.environ.get("MEDSAM_EXT_DATA", PROJECT_ROOT / "app_data"))
    ext = ext or ExtService(os.environ.get("MEDSAM_CHECKPOINT", DEFAULT_CHECKPOINT), os.environ.get("MEDSAM_DEVICE", "cpu"))
    app = create_app(service=LegacyAdapter(ext), db_path=data_dir / "experiments_ext.db",
                     overlays_dir=data_dir / "overlays", tmp_dir=data_dir / "tmp_uploads")
    if autoload and not ext.sam_loaded:
        threading.Thread(target=ext.load, daemon=True).start()
    sessions: "OrderedDict[str, dict]" = OrderedDict()
    slock = threading.Lock()
    r = APIRouter(prefix="/ext")
    hist = HistoryStore(data_dir / "history_ext.db")
    ovl_dir = data_dir / "ext_overlays"

    def save_png(name: str, rgb: np.ndarray) -> str:
        ovl_dir.mkdir(parents=True, exist_ok=True)
        p = ovl_dir / name
        if not cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise HTTPException(500, "could not save overlay")
        return p.relative_to(data_dir).as_posix()

    def parse_ids(ids: str | None) -> list[int] | None:
        if ids is None or not ids.strip():
            return None
        try:
            out = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(422, "ids must be comma-separated integers") from None
        if len(out) > 500:
            raise HTTPException(422, "at most 500 ids")
        return out

    def selected(ids: str | None) -> list[dict]:
        idl = parse_ids(ids)
        rows = hist.list(ids=idl, limit=500)
        if idl is not None:
            order = {i: n for n, i in enumerate(idl)}
            rows.sort(key=lambda x: order[x["id"]])
            missing = sorted(set(idl) - {x["id"] for x in rows})
            if missing:
                raise HTTPException(404, f"unknown run id(s): {missing}")
        return rows

    def metrics_for(mask, gt):
        if gt is None:
            return {k: None for k in METRIC_KEYS}
        m = compute_all_metrics(mask, gt, 0.0)
        return {k: clean_metric(m[k]) for k in METRIC_KEYS}

    def session(sid: str) -> dict:
        with slock:
            if sid not in sessions:
                raise HTTPException(404, "unknown or expired session_id")
            return sessions[sid]

    @r.get("/health")
    def health():
        return {"sam_loaded": ext.sam_loaded, "device": ext.device, "detail": ext.load_error, "methods": list(METHODS),
                "features": ["prompt_overlay", "uncertainty_multimask", "uncertainty_perturbation", "click_refinement"],
                "note": EXPLORATORY_NOTE}

    @r.post("/segment")
    async def segment(support_image: UploadFile = File(...), support_mask: UploadFile = File(...),
                      query_image: UploadFile = File(...), method: str = Form(...),
                      query_mask: UploadFile | None = File(None), k: int = Form(8), uncertainty: bool = Form(True)):
        method = method.strip().lower()
        if method not in METHODS:
            raise HTTPException(422, f"Invalid method '{method}'. Allowed: {', '.join(METHODS)}")
        if not 2 <= k <= 16:
            raise HTTPException(422, "k must be between 2 and 16")
        s = decode_image_bytes(await read_upload(support_image, "support_image", False))
        sm = decode_image_bytes(await read_upload(support_mask, "support_mask", True), grayscale=True) > 0
        q = decode_image_bytes(await read_upload(query_image, "query_image", False))
        gt = None
        if query_mask is not None and (query_mask.filename or ""):
            gt = decode_image_bytes(await read_upload(query_mask, "query_mask", True), grayscale=True) > 0
            if gt.shape != q.shape[:2]:
                raise HTTPException(400, "query_mask size differs from query_image")
        if sm.shape != s.shape[:2]:
            raise HTTPException(400, "support_mask size differs from support_image")
        if not sm.any():
            raise HTTPException(422, "Support mask is empty; cannot derive prompts")
        if not ext.sam_loaded:
            raise HTTPException(503, "SAM model is not loaded; no segmentation was performed")
        try:
            res = await run_in_threadpool(ext.segment, method, q, s, sm, gt, k, uncertainty)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        sid = uuid.uuid4().hex
        p = res["prompt"]
        unc = res["uncertainty"]
        overlay = draw_prompts(q, res["mask"], p)
        mets = metrics_for(res["mask"], gt)
        tag = uuid.uuid4().hex[:12]
        run_id = hist.add({
            "session_id": sid, "query_image_name": display_name(query_image.filename),
            "support_image_name": display_name(support_image.filename), "method": res["method"],
            "method_used": res["method_used"], "a5_ratio": res["a5_ratio"], "sam_score": clean_metric(res["score"]),
            "k": k if unc is not None else None, **mets,
            **({key: clean_metric(unc[key]) for key in ("entropy_score", "disagreement_score", "variance_score")} if unc else {}),
            "prompt_json": json.dumps({"box": [float(v) for v in p["box"]] if p.get("box") is not None else None,
                                       "points": p["point_coords"].tolist() if p.get("point_coords") is not None else [],
                                       "labels": p["point_labels"].tolist() if p.get("point_labels") is not None else []}),
            "overlay_path": save_png(f"run_{tag}.png", overlay)})
        with slock:
            sessions[sid] = {"q": q, "gt": gt, "prompt": res["prompt"], "base_mask": res["mask"], "base_low": res["low_res"],
                             "mask": res["mask"], "low": res["low_res"], "clicks": [], "run_id": run_id, "tag": tag}
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
        return {"success": True, "note": EXPLORATORY_NOTE, "session_id": sid, "run_id": run_id, "method": res["method"],
                "method_used": res["method_used"], "a5_ratio": res["a5_ratio"], "sam_score": clean_metric(res["score"]),
                "prompts": {"box": [float(v) for v in p["box"]] if p.get("box") is not None else None,
                            "points": p["point_coords"].tolist() if p.get("point_coords") is not None else [],
                            "labels": p["point_labels"].tolist() if p.get("point_labels") is not None else []},
                "metrics": mets, "has_ground_truth": gt is not None,
                "query_b64": encode_png_b64(q), "support_b64": encode_png_b64(draw_prompts(s, sm, {})),
                "mask_b64": encode_png_b64(res["mask"].astype(np.uint8) * 255),
                "overlay_b64": encode_png_b64(overlay),
                "uncertainty": None if unc is None else {
                    "entropy_b64": encode_png_b64(heatmap(unc["entropy_map"], q)),
                    "variance_b64": encode_png_b64(heatmap(unc["variance_map"], q)),
                    "entropy_score": clean_metric(unc["entropy_score"]), "disagreement_score": clean_metric(unc["disagreement_score"]),
                    "variance_score": clean_metric(unc["variance_score"]), "k": unc["k"]}}

    def _refined_payload(sid, st):
        mets = metrics_for(st["mask"], st["gt"])
        overlay = draw_prompts(st["q"], st["mask"], st["prompt"], st["clicks"])
        path = save_png(f"run_{st['tag']}_refined.png", overlay) if st["clicks"] else None
        hist.update_refined(st["run_id"], st["clicks"], mets, path)
        return {"success": True, "session_id": sid, "run_id": st["run_id"], "clicks": [list(c) for c in st["clicks"]],
                "metrics": mets, "mask_b64": encode_png_b64(st["mask"].astype(np.uint8) * 255),
                "overlay_b64": encode_png_b64(overlay)}

    @r.post("/refine")
    async def refine(body: RefineBody):
        st = session(body.session_id)
        h, w = st["q"].shape[:2]
        if body.x >= w or body.y >= h:
            raise HTTPException(422, f"click outside the image ({w}x{h})")
        if not ext.sam_loaded:
            raise HTTPException(503, "SAM model is not loaded")
        clicks = st["clicks"] + [(float(body.x), float(body.y), int(body.label))]
        out = await run_in_threadpool(ext.refine, st["q"], clicks, st["low"], st["prompt"].get("box"))
        st.update(clicks=clicks, mask=out["mask"], low=out["low_res_logits"])
        return _refined_payload(body.session_id, st)

    @r.post("/reset")
    def reset(body: ResetBody):
        st = session(body.session_id)
        st.update(clicks=[], mask=st["base_mask"], low=st["base_low"])
        return _refined_payload(body.session_id, st)

    @r.get("/history")
    def history(method: str | None = None, query: str | None = None, limit: int = Query(100, ge=1, le=500)):
        if method is not None and method not in METHODS:
            raise HTTPException(422, f"method must be one of {', '.join(METHODS)}")
        return {"note": REP.NOTE, "runs": hist.list(method=method, query=query, limit=limit)}

    @r.get("/history/compare")
    def history_compare(ids: str = Query(...)):
        rows = selected(ids)
        if len(rows) < 2:
            raise HTTPException(422, "compare needs at least 2 run ids")
        return {"note": REP.NOTE, **compare_runs(rows)}

    @r.get("/history/export.csv")
    def history_csv(ids: str | None = None):
        return Response(REP.to_csv(selected(ids)), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="sam_ext_history.csv"'})

    @r.get("/history/report.html")
    def history_html(ids: str | None = None):
        return HTMLResponse(REP.to_html(selected(ids), data_dir))

    @r.get("/history/report.pdf")
    async def history_pdf(ids: str | None = None):
        rows = selected(ids)
        pdf = await run_in_threadpool(REP.to_pdf, rows, data_dir)
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="sam_ext_report.pdf"'})

    @r.get("/history/{run_id}")
    def history_one(run_id: int):
        row = hist.get(run_id)
        if row is None:
            raise HTTPException(404, "unknown run id")
        return row

    app.include_router(r)
    app.state.ext = ext
    app.state.history = hist
    return app
