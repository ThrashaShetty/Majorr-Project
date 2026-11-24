"""Experiment history for the exploratory /ext API (Phase C.4). NOT part of the frozen analysis.

One row per /ext/segment run, in its own SQLite file (<MEDSAM_EXT_DATA>/history_ext.db); the original
experiments table and experiments.db are never touched. Click refinement updates the run's refined_* fields.
Rows are never deleted by the API.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.metrics import METRIC_KEYS

UNC_KEYS = ("entropy_score", "disagreement_score", "variance_score")
COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"), ("timestamp", "TEXT"), ("session_id", "TEXT"),
    ("query_image_name", "TEXT"), ("support_image_name", "TEXT"), ("method", "TEXT"), ("method_used", "TEXT"),
    ("a5_ratio", "REAL"), ("sam_score", "REAL"), ("k", "INTEGER"),
    *((m, "REAL") for m in METRIC_KEYS), *((u, "REAL") for u in UNC_KEYS),
    ("n_clicks", "INTEGER"), ("clicks_json", "TEXT"), *((f"refined_{m}", "REAL") for m in METRIC_KEYS),
    ("prompt_json", "TEXT"), ("overlay_path", "TEXT"), ("refined_overlay_path", "TEXT"),
)
NAMES = tuple(c for c, _ in COLUMNS)
SCHEMA = "CREATE TABLE IF NOT EXISTS ext_runs (" + ", ".join(f"{c} {t}" for c, t in COLUMNS) + ")"
COMPARE_KEYS = (*METRIC_KEYS, *UNC_KEYS, "sam_score")


class HistoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, rec: dict[str, Any]) -> int:
        rec = {**rec, "timestamp": rec.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")}
        rec.setdefault("n_clicks", 0)
        rec.setdefault("clicks_json", "[]")
        cols = [c for c in NAMES if c != "id"]
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(f"INSERT INTO ext_runs ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                               [rec.get(c) for c in cols])
            return int(cur.lastrowid)

    def update_refined(self, run_id: int, clicks: list, metrics: dict[str, Any] | None, overlay_path: str | None) -> None:
        sets = {"n_clicks": len(clicks), "clicks_json": json.dumps([list(c) for c in clicks]),
                "refined_overlay_path": overlay_path if clicks else None}
        for m in METRIC_KEYS:
            sets[f"refined_{m}"] = (metrics or {}).get(m) if clicks else None
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(f"UPDATE ext_runs SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?", [*sets.values(), run_id])

    def get(self, run_id: int) -> dict[str, Any] | None:
        with self._lock, closing(self._connect()) as conn:
            r = conn.execute("SELECT * FROM ext_runs WHERE id = ?", (run_id,)).fetchone()
        return dict(r) if r else None

    def list(self, method: str | None = None, query: str | None = None, limit: int = 100,
             ids: list[int] | None = None) -> list[dict[str, Any]]:
        where, args = [], []
        if method:
            where.append("method = ?")
            args.append(method)
        if query:
            where.append("query_image_name LIKE ? ESCAPE '\\'")
            args.append("%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")
        if ids is not None:
            if not ids:
                return []
            where.append(f"id IN ({', '.join('?' * len(ids))})")
            args.extend(ids)
        sql = "SELECT * FROM ext_runs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
        with self._lock, closing(self._connect()) as conn:
            return [dict(r) for r in conn.execute(sql, [*args, int(limit)]).fetchall()]


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Side-by-side values plus differences relative to the first run given (reference)."""
    if not rows:
        return {"reference_id": None, "runs": [], "best": {}}
    ref = rows[0]
    runs = []
    for r in rows:
        d = {}
        for k in COMPARE_KEYS:
            a, b = r.get(k), ref.get(k)
            d[k] = None if a is None or b is None else a - b
        runs.append({"id": r["id"], "method": r["method"], "method_used": r["method_used"],
                     "query_image_name": r["query_image_name"], "values": {k: r.get(k) for k in COMPARE_KEYS},
                     "refined": {m: r.get(f"refined_{m}") for m in METRIC_KEYS}, "n_clicks": r.get("n_clicks"),
                     "delta_vs_reference": d})
    best = {}
    for m in METRIC_KEYS:
        vals = [(r["values"][m], r["id"]) for r in runs if r["values"][m] is not None]
        best[m] = max(vals)[1] if vals else None
    return {"reference_id": ref["id"], "runs": runs, "best": best,
            "same_query": len({r["query_image_name"] for r in rows}) == 1}
