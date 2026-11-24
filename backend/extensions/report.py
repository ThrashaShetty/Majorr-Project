"""CSV / HTML / PDF export of /ext experiment history (Phase C.4). Exploratory; no SAM needed.

HTML is self-contained (overlays embedded as base64 PNG; every text value escaped).
PDF is drawn with matplotlib (Agg): page 1 = summary table + Dice bar chart, then up to 4 overlays per page.
"""
from __future__ import annotations

import base64
import csv
import html
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.extensions.history import NAMES, UNC_KEYS
from src.metrics import METRIC_KEYS

NOTE = ("Exploratory dashboard runs. NOT part of the frozen confirmatory analysis; single interactive runs, "
        "not cohort results. Research prototype, not for diagnosis.")
TABLE_COLS = ("id", "timestamp", "query_image_name", "method", "method_used", *METRIC_KEYS, *UNC_KEYS,
              "n_clicks", *(f"refined_{m}" for m in METRIC_KEYS))


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _csv_safe(v: Any) -> Any:
    # neutralise spreadsheet formula injection in text cells (filenames are user supplied)
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(NAMES), extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _csv_safe(r.get(k)) for k in NAMES})
    return buf.getvalue()


def _img_b64(path: str | None, base_dir: Path) -> str | None:
    if not path:
        return None
    p = (base_dir / path).resolve()
    if base_dir.resolve() not in p.parents or not p.is_file():   # only files inside the data dir
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")


def to_html(rows: list[dict[str, Any]], base_dir: Path, title: str = "SAM exploratory run report") -> str:
    e = html.escape
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    head = "".join(f"<th>{e(c)}</th>" for c in TABLE_COLS)
    body = "".join("<tr>" + "".join(f"<td>{e(_fmt(r.get(c)))}</td>" for c in TABLE_COLS) + "</tr>" for r in rows)
    figs = []
    for r in rows:
        for key, cap in (("overlay_path", "initial"), ("refined_overlay_path", f"after {r.get('n_clicks') or 0} clicks")):
            b = _img_b64(r.get(key), base_dir)
            if b:
                figs.append(f'<figure><img alt="run {r["id"]} {e(cap)}" src="data:image/png;base64,{b}">'
                            f"<figcaption>#{r['id']} {e(str(r.get('query_image_name')))} · {e(str(r.get('method')))} · {e(cap)}</figcaption></figure>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{e(title)}</title>
<style>
:root{{--bg:#fff;--ink:#1b1f24;--muted:#5b6470;--line:#d9dde3;--warn:#a15c00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#111418;--ink:#e6e9ee;--muted:#9aa4b0;--line:#2c333c;--warn:#e3a33b}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:13px/1.45 system-ui,Segoe UI,sans-serif}}
.note{{color:var(--warn);margin:8px 0 16px}} .wrap{{overflow-x:auto}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}} th,td{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}}
.figs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;margin-top:16px}}
figure{{margin:0}} img{{width:100%;border:1px solid var(--line)}} figcaption{{font-size:12px;color:var(--muted)}}
</style></head><body>
<h1 style="font-size:18px">{e(title)}</h1>
<div class="note">{e(NOTE)}</div>
<p style="color:var(--muted)">Generated {e(now)} · {len(rows)} run(s)</p>
<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
<div class="figs">{''.join(figs)}</div>
</body></html>"""


def to_pdf(rows: list[dict[str, Any]], base_dir: Path, title: str = "SAM exploratory run report") -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    cols = ("id", "query_image_name", "method", "method_used", "dice", "iou", "boundary_iou", "entropy_score",
            "variance_score", "n_clicks", "refined_dice")
    with PdfPages(buf, metadata={"Title": title, "Subject": NOTE}) as pdf:
        for start in range(0, max(len(rows), 1), 25):
            chunk = rows[start:start + 25]
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(0.04, 0.96, title, fontsize=14, weight="bold")
            fig.text(0.04, 0.93, NOTE, fontsize=8, color="#a15c00")
            ax = fig.add_axes([0.04, 0.42 if start == 0 else 0.05, 0.92, 0.48 if start == 0 else 0.85])
            ax.axis("off")
            if chunk:
                cells = [[_fmt(r.get(c))[:28] for c in cols] for r in chunk]
                t = ax.table(cellText=cells, colLabels=list(cols), loc="upper center", cellLoc="left")
                t.auto_set_font_size(False)
                t.set_fontsize(6.5)
            else:
                ax.text(0.5, 0.5, "No runs recorded", ha="center")
            if start == 0 and rows:
                bx = fig.add_axes([0.08, 0.06, 0.88, 0.3])
                labels = [f"#{r['id']} {r['method']}" for r in rows][:40]
                d0 = [r.get("dice") or 0.0 for r in rows][:40]
                d1 = [r.get("refined_dice") if r.get("refined_dice") is not None else (r.get("dice") or 0.0) for r in rows][:40]
                x = range(len(labels))
                bx.bar([i - 0.2 for i in x], d0, width=0.4, label="initial Dice", color="#4a78b5")
                bx.bar([i + 0.2 for i in x], d1, width=0.4, label="after clicks", color="#e08a3c")
                bx.set_xticks(list(x), labels, rotation=60, ha="right", fontsize=6)
                bx.set_ylim(0, 1)
                bx.set_ylabel("Dice")
                bx.legend(fontsize=7, frameon=False)
                bx.spines[["top", "right"]].set_visible(False)
            pdf.savefig(fig)
            plt.close(fig)
        imgs = []
        for r in rows:
            for key, cap in (("overlay_path", "initial"), ("refined_overlay_path", f"{r.get('n_clicks') or 0} clicks")):
                p = r.get(key)
                if p and _img_b64(p, base_dir):
                    imgs.append(((base_dir / p), f"#{r['id']} {r['method']} {cap}"))
        for start in range(0, len(imgs), 4):
            fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
            for ax, item in zip(axes.flat, imgs[start:start + 4] + [None] * 4):
                ax.axis("off")
                if item:
                    ax.imshow(plt.imread(str(item[0])))
                    ax.set_title(item[1][:60], fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
