"""SAM evaluation runner for the TotalSegmentator external test (v3 methodology, frozen).

Per-query method logic is a statement-for-statement port of results/v3/scripts/holdout.py
(Baseline, Enhanced v2, A5 [0.5, 2.0], single-mask), see ``evaluate_query``. The ONLY
intended differences from v3:
  1. queries are TS slices (frozen cohorts), not MSD held-out slices;
  2. supports = all 156 v3 support slices (33 MSD patients). No leave-one-patient-out
     exclusion is needed because no TS query shares a patient with MSD;
  3. a slice on both grids is evaluated ONCE and tagged with both cohort flags
     (per-query results are order-independent: SPR re-seeds per call, caches are content-keyed);
  4. output layout, hashing, resume and resource guarding (infrastructure only).

Commands:
  prepare  - verify inputs, freeze run_config.json + worklist.json (support pairing). NO SAM.
  execute  - requires --execute AND --i-have-explicit-approval. Loads SAM, evaluates one subject
             at a time, resumable. NOT APPROVED YET.

Outputs (results/external_ts/eval/):
  run_config.json, worklist.json                (write-once, frozen before inference)
  slices/<subject>/<stem>.json                  (write-once commit point per slice)
  masks/<subject>/<stem>.<sha12>.npz            (content-addressed prediction masks)
  subjects/<subject>.csv + .done                (per-subject rows + hash marker)
  raw_rows.csv, raw_rows_primary.csv, raw_rows_sensitivity.csv (after all subjects)
  logs/resource_checks.jsonl, logs/run_history.jsonl
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as C  # noqa: E402
from ts_resources import GB, ResourceGuard, ResourceStop  # noqa: E402

# --------------------------------------------------------------------------- #
# Frozen v3 method parameters (copied from holdout.py / evaluation_config.json)
# --------------------------------------------------------------------------- #
A5_LO, A5_HI = 0.5, 2.0  # fixed before v3; never tuned
SEED = 42
DEVICE = "cpu"
CHECKPOINT_REL = "checkpoints/sam_vit_b_01ec64.pth"
ENHANCED_KW = {"mode": "enhanced", "uncertainty": "none", "seed": SEED}  # use_adaptive default (True), as in v3
METHODS = ("baseline", "enhanced_v2", "a5", "single_mask")

EVAL_DIR = C.RESULTS / "eval"
EXPECTED = {"primary_slices": 826, "sensitivity_slices": 300, "primary_subjects": 74, "sensitivity_evaluable_subjects": 90,
            "support_patients": 33, "support_slices": 156}

CODE_FILES = [
    "src/__init__.py", "src/preprocessing.py", "src/baseline.py", "src/metrics.py",
    "src/fob/__init__.py", "src/fob/adaptive.py", "src/fob/bcm.py", "src/fob/bppc.py", "src/fob/fob_pipeline.py", "src/fob/spr.py",
    "run_evaluation.py", "results/v3/scripts/holdout.py", "results/v3/adaptive/evaluation_config.json",
    "results/external_ts/scripts/ts_common.py", "results/external_ts/scripts/ts_resources.py",
    "results/external_ts/scripts/eval_external_ts.py",
]


class EvalError(C.TSError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=C._json_default) + "\n")


# --------------------------------------------------------------------------- #
# Inputs: support pool, frozen cohorts, worklist, pairing
# --------------------------------------------------------------------------- #
def load_support_pool(support_list: Path, query_list: Path, project_root: Path,
                      expected: tuple[int, int] | None = (33, 156)) -> list[dict[str, Any]]:
    """The v3 support pool: every line of data/support_list.txt, dev patients excluded by construction."""
    pid = lambda p: int(re.search(r"case_(\d+)_slice_", p).group(1))  # noqa: E731
    lines = [ln.split("\t") for ln in support_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dev = {pid(ln.split("\t")[0]) for ln in query_list.read_text(encoding="utf-8").splitlines() if ln.strip()}
    pool = []
    for img_rel, msk_rel in lines:
        p = pid(img_rel)
        if p in dev:
            raise EvalError(f"support slice {img_rel} belongs to a development patient")
        img, msk = project_root / img_rel, project_root / msk_rel
        if not img.is_file() or not msk.is_file():
            raise EvalError(f"support file missing: {img_rel}")
        pool.append({"image": img_rel, "mask": msk_rel, "patient": f"case_{p:03d}",
                     "image_sha256": C.file_digest(img), "mask_sha256": C.file_digest(msk)})
    n_pat = len({s["patient"] for s in pool})
    if expected and (n_pat, len(pool)) != expected:
        raise EvalError(f"support pool {n_pat} patients / {len(pool)} slices != expected {expected}")
    return pool


def collect_subjects(manifest: dict[str, Any], preprocessed: Path) -> list[dict[str, Any]]:
    """Frozen cohort facts + processed slices from the hash-verified preprocessing logs."""
    from prep_external_ts import paths_for, verify_subject
    out = []
    for r in manifest["subjects"]:
        if not r["eligible"]:
            continue
        verify_subject(preprocessed, r["subject"])
        log = json.loads(paths_for(preprocessed, r["subject"])["log"].read_text(encoding="utf-8"))
        files = {s["z"]: {"file": s["file"], "image_sha256": s["image_sha256"], "mask_sha256": s["mask_sha256"]}
                 for s in log["slices"] if s["valid"]}
        out.append({"subject": r["subject"], "primary": r["primary"], "sensitivity": r["sensitivity"],
                    "processed_step5": log["processed_step5"], "processed_step17": log["processed_step17"], "files": files})
    return out


def build_worklist(subjects: list[dict[str, Any]], expected: dict[str, int] | None = EXPECTED) -> list[dict[str, Any]]:
    """Unique query slices: primary = step-5 slices of primary subjects; sensitivity = step-17 slices of all
    eligible subjects. Step-5 slices of non-primary subjects are NOT included (not in either frozen analysis)."""
    items = []
    for s in subjects:
        prim = set(s["processed_step5"]) if s["primary"] else set()
        sens = set(s["processed_step17"]) if s["sensitivity"] else set()
        for z in sorted(prim | sens):
            f = s["files"][z]
            items.append({"subject": s["subject"], "z": z, "file": f["file"], "image_sha256": f["image_sha256"],
                          "mask_sha256": f["mask_sha256"], "in_primary": z in prim, "in_sensitivity": z in sens})
    counts = {
        "primary_slices": sum(i["in_primary"] for i in items), "sensitivity_slices": sum(i["in_sensitivity"] for i in items),
        "primary_subjects": len({i["subject"] for i in items if i["in_primary"]}),
        "sensitivity_evaluable_subjects": len({i["subject"] for i in items if i["in_sensitivity"]}),
    }
    if expected:
        bad = {k: (v, expected[k]) for k, v in counts.items() if expected.get(k) != v}
        if bad:
            raise EvalError(f"worklist counts differ from frozen protocol: {bad}")
    return items


def pair_supports(items: list[dict[str, Any]], pool: list[dict[str, Any]], query_dir: Path, project_root: Path,
                  thumb: Callable[[Path], np.ndarray]) -> None:
    """v3 'nearest' pairing: argmax cosine of 64x64 thumbnails (run_evaluation._thumb) over the 156 supports."""
    bank = np.stack([thumb(project_root / s["image"]) for s in pool])
    for it in items:
        sim = bank @ thumb(query_dir / it["file"])
        j = int(np.argmax(sim))
        it.update(support_index=j, support_image=pool[j]["image"], support_mask=pool[j]["mask"],
                  support_patient=pool[j]["patient"], support_similarity=round(float(sim[j]), 6))


def code_hashes(project_root: Path, files: Iterable[str] = CODE_FILES) -> dict[str, str]:
    return {f: C.file_digest(project_root / f) for f in files}


def make_config(*, manifest_sha: str, qc_sha: str | None, pool: list[dict[str, Any]], worklist_sha: str,
                hashes: dict[str, str], checkpoint_sha: str | None) -> dict[str, Any]:
    return {
        "config_version": 1,
        "manifest_sha256": manifest_sha, "full_qc_sha256": qc_sha, "worklist_sha256": worklist_sha,
        "checkpoint": {"path": CHECKPOINT_REL, "sha256": checkpoint_sha}, "device": DEVICE, "seed": SEED,
        "methods": list(METHODS),
        "method_settings": {
            "baseline": "src.baseline.BaselineSegmentor(engine) - support-mask box, multimask, best predicted IoU",
            "enhanced_v2": {"class": "src.fob.fob_pipeline.FoBPipeline", **ENHANCED_KW, "use_adaptive": True},
            "a5": {"rule": "keep baseline if A5_LO <= baseline_mask_px / support_mask_px <= A5_HI else enhanced_v2",
                   "lo": A5_LO, "hi": A5_HI, "tuned": False},
            "single_mask": "Enhanced v2 prompts, predictor.predict(multimask_output=False), best mask",
            "per_query_order": "invalidate -> baseline -> enhanced_v2 -> single_mask decode (as holdout.py)",
        },
        "support_pool": {"source": "data/support_list.txt", "n_patients": len({s['patient'] for s in pool}),
                         "n_slices": len(pool), "patients": sorted({s["patient"] for s in pool}),
                         "pairing": "nearest 64x64 thumbnail cosine (run_evaluation._thumb), argmax over all 156",
                         "files_digest": C.sha256_bytes("".join(s["image_sha256"] + s["mask_sha256"] for s in pool).encode())},
        "query_preprocessing": "frozen TS preprocessing (manifest + full QC); images/masks loaded with src.preprocessing.load_image_rgb/load_mask_binary",
        "code_sha256": hashes,
    }


# --------------------------------------------------------------------------- #
# Per-query logic: statement-for-statement port of holdout.py's loop body
# --------------------------------------------------------------------------- #
def evaluate_query(q: np.ndarray, gt: np.ndarray, s: np.ndarray, smask: np.ndarray, *, eng: Any, base: Any, enh: Any,
                   select_best_mask: Callable, compute_all_metrics: Callable, metric_keys: Iterable[str]
                   ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    support_px = int((smask > 0).sum())

    eng.invalidate()  # every query is encoded fresh -> honest Baseline timing
    rb = base.predict_arrays(q, smask, gt)
    re_ = enh.predict_arrays(q, s, smask, gt)
    c = re_["config"]
    box = np.array(c["box_xyxy"], np.float32)
    prompt = {"box": box}
    if c["points_xy"]:
        prompt.update(point_coords=np.array(c["points_xy"], np.float32), point_labels=np.array(c["point_labels"], np.int32))
    t0 = time.perf_counter()
    ms, sc, _ = eng.predictor.predict(multimask_output=False, **prompt)
    single_decode_ms = (time.perf_counter() - t0) * 1000
    m_single, s_single, _ = select_best_mask(ms, sc)
    rs = compute_all_metrics(m_single, gt, 0.0)

    box_px = int(rb["mask"].sum())
    ratio = box_px / support_px
    keep_box = A5_LO <= ratio <= A5_HI
    tm = c["timing_ms"]
    enh_extra = tm["support_prototype"] + tm["prompting_and_decode"]
    row = dict(support_px=support_px, gt_px=int((gt > 0).sum()), box_px=box_px, box_area_vs_support=ratio,
               a5_choice="box" if keep_box else "enhanced", box_pred_iou=rb["score"], enh_pred_iou=re_["score"],
               single_pred_iou=s_single, n_negative=c["n_negative"], positive_point_used=c["positive_point_used"],
               adaptive_strategy=c["adaptive"]["strategy"], n_points_requested=c["n_points_requested"])
    for m in metric_keys:
        row[f"baseline_{m}"] = rb[m]
        row[f"enhanced_{m}"] = re_[m]
        row[f"single_{m}"] = rs[m]
        row[f"a5_{m}"] = rb[m] if keep_box else re_[m]
    row["baseline_ms"] = rb["inference_ms"]
    row["enhanced_ms"] = re_["inference_ms"]
    row["single_decode_ms"] = single_decode_ms
    row["single_ms"] = re_["inference_ms"]
    row["a5_ms"] = rb["inference_ms"] + (0.0 if keep_box else enh_extra)
    arrays = dict(gt=gt > 0, baseline=rb["mask"], enhanced=re_["mask"], single=m_single,
                  box=box, points=np.array(c["points_xy"]), labels=np.array(c["point_labels"]))
    return row, arrays


# --------------------------------------------------------------------------- #
# Output layout, verification, resume
# --------------------------------------------------------------------------- #
def stem(item: dict[str, Any]) -> str:
    return Path(item["file"]).stem


def slice_paths(eval_dir: Path, item: dict[str, Any]) -> dict[str, Path]:
    return {"json": eval_dir / "slices" / item["subject"] / f"{stem(item)}.json",
            "mask_dir": eval_dir / "masks" / item["subject"]}


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


ITEM_KEYS = ("subject", "z", "file", "image_sha256", "mask_sha256", "in_primary", "in_sensitivity",
             "support_image", "support_mask", "support_patient")


def verify_slice(eval_dir: Path, item: dict[str, Any], config_sha: str) -> dict[str, Any]:
    p = slice_paths(eval_dir, item)
    try:
        rec = json.loads(p["json"].read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise EvalError(f"{p['json']}: unreadable slice result ({e!r})") from e
    if rec.get("config_sha256") != config_sha:
        raise EvalError(f"{p['json']}: produced under a different run_config")
    for k in ITEM_KEYS:
        if rec["item"].get(k) != item[k]:
            raise EvalError(f"{p['json']}: item field {k} differs from worklist")
    npz = p["mask_dir"] / rec["npz_file"]
    if not npz.is_file() or C.file_digest(npz) != rec["npz_sha256"]:
        raise EvalError(f"{npz}: prediction masks missing or hash mismatch")
    with np.load(npz, allow_pickle=False) as z:
        for k in ("gt", "baseline", "enhanced", "single"):
            if k not in z.files or z[k].shape != (512, 512):
                raise EvalError(f"{npz}: array {k} missing or wrong shape")
    return rec


def rows_csv_bytes(recs: list[dict[str, Any]]) -> bytes:
    rows = [{**{k: r["item"][k] for k in ("subject", "z", "file", "in_primary", "in_sensitivity", "support_image",
                                           "support_patient", "support_similarity")},
             "query_image": r["item"]["file"], "patient": r["item"]["subject"], **r["row"]} for r in recs]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def verify_subject_done(eval_dir: Path, subject: str, items: list[dict[str, Any]], config_sha: str) -> None:
    done = eval_dir / "subjects" / f"{subject}.done"
    csvp = eval_dir / "subjects" / f"{subject}.csv"
    rec = json.loads(done.read_text(encoding="utf-8"))
    if rec["config_sha256"] != config_sha:
        raise EvalError(f"{subject}: done marker from a different run_config")
    if not csvp.is_file() or C.file_digest(csvp) != rec["csv_sha256"]:
        raise EvalError(f"{subject}: subject csv missing or hash mismatch")
    recs = [verify_slice(eval_dir, it, config_sha) for it in items]
    if rows_csv_bytes(recs) != csvp.read_bytes():
        raise EvalError(f"{subject}: subject csv does not match slice results")
    if sorted(rec["slice_json_sha256"]) != sorted(C.file_digest(slice_paths(eval_dir, it)["json"]) for it in items):
        raise EvalError(f"{subject}: slice result hashes differ from done marker")


def process_subject(subject: str, items: list[dict[str, Any]], *, eval_dir: Path, config_sha: str, query_dir: Path,
                    project_root: Path, methods: tuple[Any, Any, Any], fns: dict[str, Any], guard: ResourceGuard,
                    history: Path) -> str:
    done = eval_dir / "subjects" / f"{subject}.done"
    if done.exists():
        verify_subject_done(eval_dir, subject, items, config_sha)
        return "skipped_verified"
    eng, base, enh = methods
    recs = []
    for it in items:
        p = slice_paths(eval_dir, it)
        if p["json"].exists():
            recs.append(verify_slice(eval_dir, it, config_sha))
            continue
        # inputs are re-verified immediately before use
        qi, qm = query_dir / "images" / it["file"], query_dir / "masks" / it["file"]
        si, sm = project_root / it["support_image"], project_root / it["support_mask"]
        for f, want in ((qi, it["image_sha256"]), (qm, it["mask_sha256"]),
                        (si, it["support_image_sha256"]), (sm, it["support_mask_sha256"])):
            if C.file_digest(f) != want:
                raise EvalError(f"input hash mismatch: {f}")
        q, gt = fns["load_image_rgb"](qi), fns["load_mask_binary"](qm)
        s, smask = fns["load_image_rgb"](si), fns["load_mask_binary"](sm)
        row, arrays = evaluate_query(q, gt, s, smask, eng=eng, base=base, enh=enh,
                                     select_best_mask=fns["select_best_mask"],
                                     compute_all_metrics=fns["compute_all_metrics"], metric_keys=fns["metric_keys"])
        data = npz_bytes(arrays)
        sha = C.sha256_bytes(data)
        npz_name = f"{stem(it)}.{sha[:12]}.npz"
        C.write_once(p["mask_dir"] / npz_name, data)
        rec = {"config_sha256": config_sha, "item": {k: it[k] for k in (*ITEM_KEYS, "support_similarity")},
               "row": row, "npz_file": npz_name, "npz_sha256": sha, "utc": _now()}
        C.write_once(p["json"], C.json_bytes(rec))  # commit point
        recs.append(json.loads(p["json"].read_text(encoding="utf-8")))
        guard.check(f"after {subject} z={it['z']}")  # safe checkpoint
    csv_bytes = rows_csv_bytes(recs)
    _, csv_sha = C.write_once(eval_dir / "subjects" / f"{subject}.csv", csv_bytes)
    marker = {"subject": subject, "config_sha256": config_sha, "csv_sha256": csv_sha, "n_slices": len(items),
              "slice_json_sha256": sorted(C.file_digest(slice_paths(eval_dir, it)["json"]) for it in items)}
    C.write_once(done, C.json_bytes(marker))
    return "evaluated"


def finalize(eval_dir: Path, worklist: list[dict[str, Any]], config_sha: str) -> dict[str, str]:
    by: dict[str, list] = {}
    for it in worklist:
        by.setdefault(it["subject"], []).append(it)
    recs = []
    for s, its in by.items():
        verify_subject_done(eval_dir, s, its, config_sha)
        recs += [verify_slice(eval_dir, it, config_sha) for it in its]
    out = {}
    for name, sel in (("raw_rows.csv", lambda r: True), ("raw_rows_primary.csv", lambda r: r["item"]["in_primary"]),
                      ("raw_rows_sensitivity.csv", lambda r: r["item"]["in_sensitivity"])):
        out[name] = C.write_once(eval_dir / name, rows_csv_bytes([r for r in recs if sel(r)]))[1]
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare(*, manifest_path: Path, qc_path: Path | None, preprocessed: Path, project_root: Path, eval_dir: Path,
            subjects: list[dict[str, Any]] | None = None, expected: dict[str, int] | None = EXPECTED,
            thumb: Callable[[Path], np.ndarray] | None = None, code_files: Iterable[str] = CODE_FILES,
            checkpoint: Path | None = None) -> dict[str, Any]:
    """Freeze run_config.json + worklist.json. Reads PNGs only; never loads SAM."""
    from build_manifest import load_manifest
    manifest = load_manifest(manifest_path)
    msha = C.file_digest(manifest_path)
    qc_sha = None
    if qc_path is not None:
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if not qc.get("all_passed") or qc.get("manifest_sha256") != msha:
            raise EvalError("full preprocessing QC did not pass or refers to another manifest")
        qc_sha = C.file_digest(qc_path)
    if subjects is None:
        subjects = collect_subjects(manifest, preprocessed)
    pool = load_support_pool(project_root / "data/support_list.txt", project_root / "data/query_list.txt", project_root,
                             expected=(expected["support_patients"], expected["support_slices"]) if expected else None)
    sp = manifest["support_pool"]
    if [p for p in sorted({s["patient"] for s in pool})] != sp["support_patients"] or len(pool) != sp["n_support_slices"]:
        raise EvalError("support pool differs from the one frozen in the manifest")
    items = build_worklist(subjects, expected)
    if thumb is None:
        from run_evaluation import _thumb as thumb  # the v3 pairing function, unchanged
    pair_supports(items, pool, preprocessed / "images", project_root, thumb)
    for it in items:
        it["support_image_sha256"] = pool[it["support_index"]]["image_sha256"]
        it["support_mask_sha256"] = pool[it["support_index"]]["mask_sha256"]
    wl_bytes = C.json_bytes({"items": items})
    cfg = make_config(manifest_sha=msha, qc_sha=qc_sha, pool=pool, worklist_sha=C.sha256_bytes(wl_bytes),
                      hashes=code_hashes(project_root, code_files),
                      checkpoint_sha=C.file_digest(checkpoint) if checkpoint is not None else None)
    eval_dir.mkdir(parents=True, exist_ok=True)
    C.write_once(eval_dir / "worklist.json", wl_bytes)
    _, cfg_sha = C.write_once(eval_dir / "run_config.json", C.json_bytes(cfg))
    return {"config": cfg, "config_sha256": cfg_sha, "items": items}


def load_frozen(eval_dir: Path, project_root: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    cfg_path, wl_path = eval_dir / "run_config.json", eval_dir / "worklist.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if C.file_digest(wl_path) != cfg["worklist_sha256"]:
        raise EvalError("worklist.json does not match run_config.json")
    now = code_hashes(project_root, cfg["code_sha256"].keys())
    changed = [f for f, h in cfg["code_sha256"].items() if now[f] != h]
    if changed:
        raise EvalError(f"code changed since the run_config was frozen: {changed}")
    return cfg, C.file_digest(cfg_path), json.loads(wl_path.read_text(encoding="utf-8"))["items"]


def real_methods(project_root: Path, checkpoint_sha: str | None) -> tuple[Any, Any, Any]:  # pragma: no cover - needs SAM
    import torch
    from src.baseline import BaselineSegmentor, SamEngine
    from src.fob.fob_pipeline import FoBPipeline
    ckpt = project_root / CHECKPOINT_REL
    if checkpoint_sha is not None and C.file_digest(ckpt) != checkpoint_sha:
        raise EvalError("checkpoint hash differs from run_config")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    eng = SamEngine(ckpt, DEVICE)
    return eng, BaselineSegmentor(engine=eng), FoBPipeline(engine=eng, **ENHANCED_KW)


def real_fns() -> dict[str, Any]:
    from src.baseline import select_best_mask
    from src.metrics import METRIC_KEYS, compute_all_metrics
    from src.preprocessing import load_image_rgb, load_mask_binary
    return {"select_best_mask": select_best_mask, "compute_all_metrics": compute_all_metrics, "metric_keys": METRIC_KEYS,
            "load_image_rgb": load_image_rgb, "load_mask_binary": load_mask_binary}


def execute(*, eval_dir: Path, project_root: Path, query_dir: Path, guard_start: ResourceGuard, guard_run: ResourceGuard,
            methods_factory: Callable[[], tuple[Any, Any, Any]], fns: dict[str, Any],
            subjects: list[str] | None = None) -> int:
    """Returns 0 done, 2 failure (stopped), 3 resource stop (clean). Never continues past a failure."""
    history = eval_dir / "logs" / "run_history.jsonl"
    cfg, cfg_sha, worklist = load_frozen(eval_dir, project_root)
    by: dict[str, list] = {}
    for it in worklist:
        by.setdefault(it["subject"], []).append(it)
    order = list(by) if subjects is None else subjects
    unknown = [s for s in order if s not in by]
    if unknown:
        raise EvalError(f"subjects not in worklist: {unknown}")
    append_jsonl(history, {"utc": _now(), "event": "run_start", "config_sha256": cfg_sha, "subjects": len(order),
                           "python": platform.python_version()})
    try:
        guard_start.check("before model load")
        methods = methods_factory()
        for s in order:
            guard_run.check(f"before subject {s}")
            append_jsonl(history, {"utc": _now(), "event": "subject_start", "subject": s})
            status = process_subject(s, by[s], eval_dir=eval_dir, config_sha=cfg_sha, query_dir=query_dir,
                                     project_root=project_root, methods=methods, fns=fns, guard=guard_run, history=history)
            append_jsonl(history, {"utc": _now(), "event": status, "subject": s})
    except ResourceStop as e:
        append_jsonl(history, {"utc": _now(), "event": "RESOURCE_STOP", "reason": e.reason, "check": e.record})
        print(f"RESOURCE STOP: {e.reason} - completed work is kept; resume later", file=sys.stderr)
        return 3
    except BaseException as e:  # noqa: BLE001 - record and stop, never skip
        append_jsonl(history, {"utc": _now(), "event": "FAILED", "error": repr(e), "traceback": traceback.format_exc(limit=6)})
        print(f"FAILED: {e!r} - stopping", file=sys.stderr)
        return 2
    if subjects is None:
        hashes = finalize(eval_dir, worklist, cfg_sha)
        append_jsonl(history, {"utc": _now(), "event": "finalized", "outputs": hashes})
    append_jsonl(history, {"utc": _now(), "event": "run_end"})
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="freeze run_config + worklist (no SAM)")
    ex = sub.add_parser("execute", help="run SAM evaluation (NOT approved yet)")
    ex.add_argument("--execute", action="store_true")
    ex.add_argument("--i-have-explicit-approval", action="store_true")
    ex.add_argument("--min-ram-gb-start", type=float, default=1.6)
    ex.add_argument("--min-ram-gb-run", type=float, default=0.3)
    ex.add_argument("--min-disk-gb", type=float, default=2.0)
    ex.add_argument("--subjects", nargs="*", default=None)
    a = ap.parse_args(argv)
    P = C.PROJECT_ROOT
    if a.cmd == "prepare":
        r = prepare(manifest_path=C.MANIFEST, qc_path=C.RESULTS / "qc" / "full_qc.json", preprocessed=C.PREPROCESSED,
                    project_root=P, eval_dir=EVAL_DIR, checkpoint=P / CHECKPOINT_REL)
        print(json.dumps({"config_sha256": r["config_sha256"], "worklist_items": len(r["items"])}))
        return 0
    if not (a.execute and a.i_have_explicit_approval):
        print("Refusing: inference requires --execute and --i-have-explicit-approval (approval not yet given).", file=sys.stderr)
        return 1
    log = EVAL_DIR / "logs" / "resource_checks.jsonl"
    gs = ResourceGuard(int(a.min_ram_gb_start * GB), int(a.min_disk_gb * GB), Path("C:/"), log)
    gr = ResourceGuard(int(a.min_ram_gb_run * GB), int(a.min_disk_gb * GB), Path("C:/"), log)
    cfg, _, _ = load_frozen(EVAL_DIR, P)
    return execute(eval_dir=EVAL_DIR, project_root=P, query_dir=C.PREPROCESSED, guard_start=gs, guard_run=gr,
                   methods_factory=lambda: real_methods(P, cfg["checkpoint"]["sha256"]), fns=real_fns(),
                   subjects=a.subjects)


if __name__ == "__main__":
    raise SystemExit(main())
