"""Tests for the resource guard and the TS evaluation runner - synthetic inputs, MOCKED SAM only.

No checkpoint is loaded: SamEngine / load_sam_model are patched to raise if touched.
Run: python -B -m pytest -p no:cacheprovider -q results/external_ts/scripts/tests
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import build_manifest as BM  # noqa: E402
import eval_external_ts as E  # noqa: E402
import ts_common as C  # noqa: E402
from ts_resources import GB, ResourceGuard, ResourceStop  # noqa: E402

from run_evaluation import _thumb  # noqa: E402  (v3 pairing function)
from src.baseline import select_best_mask  # noqa: E402
from src.metrics import METRIC_KEYS, compute_all_metrics  # noqa: E402
from src.preprocessing import load_image_rgb, load_mask_binary  # noqa: E402

FNS = {"select_best_mask": select_best_mask, "compute_all_metrics": compute_all_metrics, "metric_keys": METRIC_KEYS,
       "load_image_rgb": load_image_rgb, "load_mask_binary": load_mask_binary}
TEST_EXPECTED = {"primary_slices": 4, "sensitivity_slices": 4, "primary_subjects": 2, "sensitivity_evaluable_subjects": 3,
                 "support_patients": 2, "support_slices": 3}


# --------------------------------------------------------------------------- #
# guard
# --------------------------------------------------------------------------- #
def guard(ram, disk, tmp_path, min_ram=1 * GB, min_disk=2 * GB):
    seq_r = iter(ram) if isinstance(ram, list) else None
    seq_d = iter(disk) if isinstance(disk, list) else None
    rr = (lambda: next(seq_r)) if seq_r else (ram if callable(ram) else (lambda: ram))
    dd = (lambda: next(seq_d)) if seq_d else (disk if callable(disk) else (lambda: disk))
    return ResourceGuard(min_ram, min_disk, Path("C:/"), tmp_path / "res.jsonl", ram_reader=rr, disk_reader=dd)


def test_guard_healthy(tmp_path):
    g = guard(3 * GB, 10 * GB, tmp_path)
    rec = g.check("start")
    assert rec["ok"] and rec["ram_available_bytes"] == 3 * GB
    assert json.loads((tmp_path / "res.jsonl").read_text().splitlines()[0])["stage"] == "start"


@pytest.mark.parametrize("ram,disk,msg", [
    (0.5 * GB, 10 * GB, "available RAM"),
    (3 * GB, 1 * GB, "free disk"),
    (int(1 * GB) - 1, 10 * GB, "available RAM"),          # one byte under the limit
])
def test_guard_below_limits(tmp_path, ram, disk, msg):
    with pytest.raises(ResourceStop, match=msg) as ei:
        guard(int(ram), int(disk), tmp_path).check("x")
    assert ei.value.record["ok"] is False
    assert json.loads((tmp_path / "res.jsonl").read_text())["stop_reasons"]


def test_guard_exactly_at_limit_passes(tmp_path):
    assert guard(1 * GB, 2 * GB, tmp_path).check("edge")["ok"]


@pytest.mark.parametrize("bad", [None, "779907072\r", "100", float("nan"), float("inf"), -5, True, [1], object()])
def test_guard_invalid_or_missing_readings_fail_safe(tmp_path, bad):
    with pytest.raises(ResourceStop):
        guard(bad, 10 * GB, tmp_path).check("ram bad")
    with pytest.raises(ResourceStop):
        guard(3 * GB, bad, tmp_path).check("disk bad")


def test_guard_reader_exception_fails_safe(tmp_path):
    def boom():
        raise OSError("psutil unavailable")
    with pytest.raises(ResourceStop, match="reading failed"):
        guard(boom, 10 * GB, tmp_path).check("x")


def test_guard_decline_during_processing(tmp_path):
    g = guard([3 * GB, 2 * GB, 0.9 * GB], [10 * GB] * 3, tmp_path)
    g.check("a")
    g.check("b")
    with pytest.raises(ResourceStop):
        g.check("c")
    assert [h["ok"] for h in g.history] == [True, True, False]


def test_guard_rejects_bad_limits():
    with pytest.raises(ValueError):
        ResourceGuard(0, 1, Path("C:/"))
    with pytest.raises(ValueError):
        ResourceGuard(1.5, 1, Path("C:/"))  # float limit refused


def test_real_readers_are_numeric():
    from ts_resources import make_disk_reader, read_ram_available
    r, d = read_ram_available(), make_disk_reader(Path("C:/"))()
    assert isinstance(r, int) and isinstance(d, int) and r > 0 and d > 0  # read-only check


def test_bash_crlf_bug_documented():
    # The old driver's failure mode: a string with a trailing CR is NOT a valid number here.
    assert E is not None
    with pytest.raises(ResourceStop):
        ResourceGuard(1, 1, Path("C:/"), ram_reader=lambda: "100\r", disk_reader=lambda: 10).check("crlf")


# --------------------------------------------------------------------------- #
# synthetic project + preprocessed outputs
# --------------------------------------------------------------------------- #
def png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr)


def blob_image(seed: int, cx: int, cy: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    img = (rng.normal(100, 10, (512, 512))).clip(0, 255).astype(np.uint8)
    yy, xx = np.mgrid[:512, :512]
    m = ((yy - cy) ** 2 + (xx - cx) ** 2) < 40 ** 2
    img[m] = 170
    return img, np.where(m, 255, 0).astype(np.uint8)


SUBJECTS = {  # (primary, step5, step17)
    "s0001": (True, [5, 10, 85], [17, 85]),
    "s0002": (False, [5, 10], [17]),     # boundary: step-5 slices must NOT be evaluated
    "s0003": (False, [5], []),           # s1358-like: zero sensitivity slices -> no items
    "s0004": (True, [0], [0]),
}


@pytest.fixture()
def env(tmp_path):
    root = tmp_path / "project"
    sup = [("case_002", 70, 200, 300), ("case_002", 75, 210, 300), ("case_008", 10, 300, 200)]
    lines = []
    for k, (pat, z, cx, cy) in enumerate(sup):
        img, m = blob_image(k, cx, cy)
        name = f"{pat}_slice_{z:03d}.png"
        png(root / "data/images" / name, img)
        png(root / "data/masks" / name, m)
        lines.append(f"data/images/{name}\tdata/masks/{name}")
    (root / "data/support_list.txt").write_text("\n".join(lines) + "\n")
    (root / "data/query_list.txt").write_text("data/images/case_003_slice_020.png\tdata/masks/case_003_slice_020.png\n")
    (root / "code").mkdir()
    (root / "code/method.py").write_text("# frozen method code\n")
    pre = tmp_path / "preprocessed"
    subjects = []
    for i, (s, (prim, s5, s17)) in enumerate(SUBJECTS.items()):
        files = {}
        for z in sorted(set(s5) | set(s17)):
            img, m = blob_image(100 + i * 10 + z, 180 + z, 290)
            name = C.output_name(s, z)
            png(pre / "images" / name, img)
            png(pre / "masks" / name, m)
            files[z] = {"file": name, "image_sha256": C.file_digest(pre / "images" / name),
                        "mask_sha256": C.file_digest(pre / "masks" / name)}
        subjects.append({"subject": s, "primary": prim, "sensitivity": True, "processed_step5": s5,
                         "processed_step17": s17, "files": files})
    manifest = {"manifest_version": 1, "support_pool": {"support_patients": ["case_002", "case_008"], "n_support_slices": 3},
                "subjects": []}
    mpath = tmp_path / "manifest/cohort_manifest.json"
    BM.write_manifest(manifest, mpath)
    return dict(tmp=tmp_path, root=root, pre=pre, subjects=subjects, mpath=mpath, eval=tmp_path / "eval")


def do_prepare(env, **kw):
    args = dict(manifest_path=env["mpath"], qc_path=None, preprocessed=env["pre"], project_root=env["root"],
                eval_dir=env["eval"], subjects=env["subjects"], expected=TEST_EXPECTED, thumb=_thumb,
                code_files=["code/method.py"])
    args.update(kw)
    return E.prepare(**args)


# --------------------------------------------------------------------------- #
# mocked SAM
# --------------------------------------------------------------------------- #
class Calls(list):
    pass


def box_mask(box, scale=1.0):
    x0, y0, x1, y1 = (int(v) for v in box)
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 * scale, (y1 - y0) / 2 * scale
    m = np.zeros((512, 512), bool)
    m[int(cy - hh):int(cy + hh) + 1, int(cx - hw):int(cx + hw) + 1] = True
    return m


class FakePredictor:
    def __init__(self, calls):
        self.calls = calls

    def predict(self, multimask_output=True, box=None, point_coords=None, point_labels=None):
        self.calls.append(("predict", multimask_output, None if box is None else box.dtype.name,
                           None if point_coords is None else (point_coords.dtype.name, point_labels.dtype.name, len(point_coords))))
        m = box_mask(box)
        n = 3 if multimask_output else 1
        return np.stack([m] * n), np.linspace(0.9, 0.7, n).astype(np.float32), None


class FakeEngine:
    def __init__(self, calls):
        self.calls, self.predictor = calls, FakePredictor(calls)

    def invalidate(self):
        self.calls.append(("invalidate",))


class FakeBaseline:
    def __init__(self, calls, scale=1.0):
        self.calls, self.scale = calls, scale

    def predict_arrays(self, q, smask, gt):
        self.calls.append(("baseline",))
        ys, xs = np.nonzero(smask)
        m = box_mask((xs.min(), ys.min(), xs.max(), ys.max()), self.scale)
        return {"mask": m, **compute_all_metrics(m, gt, 10.0), "score": 0.95}


class FakeEnhanced:
    def __init__(self, calls, fail_on=(), interrupt_on=()):
        self.calls, self.fail_on, self.interrupt_on, self.n = calls, set(fail_on), set(interrupt_on), 0

    def predict_arrays(self, q, s, smask, gt):
        self.n += 1
        self.calls.append(("enhanced",))
        if self.n in self.fail_on:
            raise ValueError("simulated method failure")
        if self.n in self.interrupt_on:
            raise KeyboardInterrupt("simulated interruption")
        ys, xs = np.nonzero(smask)
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        m = box_mask(box, 0.8)
        cfg = {"box_xyxy": box, "points_xy": [[10, 10], [250, 250]], "point_labels": [0, 1], "n_negative": 1,
               "positive_point_used": True, "adaptive": {"strategy": "medium"}, "n_points_requested": 8,
               "timing_ms": {"support_prototype": 5.0, "prompting_and_decode": 7.0}}
        return {"mask": m, **compute_all_metrics(m, gt, 20.0), "score": 0.85, "config": cfg}


@pytest.fixture(autouse=True)
def no_real_sam(monkeypatch):
    import src.baseline as B

    def refuse(*a, **k):
        raise AssertionError("real SAM must not be loaded in tests")
    monkeypatch.setattr(B, "load_sam_model", refuse)
    monkeypatch.setattr(B.SamEngine, "__init__", refuse)


def run_exec(env, calls=None, enh=None, ram=None, subjects=None, scale=1.0):
    calls = calls if calls is not None else Calls()
    enh = enh or FakeEnhanced(calls)
    g = guard(ram if ram is not None else 3 * GB, 10 * GB, env["tmp"], min_ram=1 * GB)
    rc = E.execute(eval_dir=env["eval"], project_root=env["root"], query_dir=env["pre"], guard_start=g, guard_run=g,
                   methods_factory=lambda: (FakeEngine(calls), FakeBaseline(calls, scale), enh), fns=FNS, subjects=subjects)
    return rc, calls, enh


def history(env):
    p = env["eval"] / "logs/run_history.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()]


def outputs(env):
    return sorted((env["eval"] / "slices").rglob("*.json"))


# --------------------------------------------------------------------------- #
# cohort, slices, support pool, pairing, config
# --------------------------------------------------------------------------- #
def test_worklist_cohorts_and_slice_indexing(env):
    items = do_prepare(env)["items"]
    got = {(i["subject"], i["z"]): (i["in_primary"], i["in_sensitivity"]) for i in items}
    assert got == {("s0001", 5): (True, False), ("s0001", 10): (True, False), ("s0001", 17): (False, True),
                   ("s0001", 85): (True, True), ("s0002", 17): (False, True), ("s0004", 0): (True, True)}
    assert not any(i["subject"] == "s0003" for i in items)          # zero sensitivity slices
    assert ("s0002", 5) not in got and ("s0002", 10) not in got     # boundary step-5 slices excluded
    assert len(items) == len(set((i["subject"], i["z"]) for i in items))  # each slice once


def test_worklist_count_mismatch_refused(env):
    with pytest.raises(E.EvalError, match="worklist counts"):
        do_prepare(env, expected={**TEST_EXPECTED, "primary_slices": 826})


def test_pairing_is_v3_nearest_argmax(env):
    items = do_prepare(env)["items"]
    sup = [l.split("\t")[0] for l in (env["root"] / "data/support_list.txt").read_text().splitlines()]
    bank = np.stack([_thumb(env["root"] / s) for s in sup])
    for it in items:
        assert it["support_image"] == sup[int(np.argmax(bank @ _thumb(env["pre"] / "images" / it["file"])))]


def test_support_pool_integrity(env):
    lst = env["root"] / "data/support_list.txt"
    with pytest.raises(E.EvalError, match="support pool"):
        E.load_support_pool(lst, env["root"] / "data/query_list.txt", env["root"], expected=(33, 156))
    lst.write_text(lst.read_text() + "data/images/case_003_slice_020.png\tdata/masks/case_003_slice_020.png\n")
    with pytest.raises(E.EvalError, match="development patient"):
        E.load_support_pool(lst, env["root"] / "data/query_list.txt", env["root"], expected=None)


def test_support_pool_must_match_manifest(env):
    m = json.loads(env["mpath"].read_text())
    m["support_pool"]["support_patients"] = ["case_002"]
    p = env["tmp"] / "m2/cohort_manifest.json"
    BM.write_manifest(m, p)
    with pytest.raises(E.EvalError, match="differs from the one frozen"):
        do_prepare(env, manifest_path=p)


def test_config_frozen_and_consistent(env):
    a = do_prepare(env)
    b = do_prepare(env)  # identical inputs -> write-once no-op
    assert a["config_sha256"] == b["config_sha256"]
    cfg = a["config"]
    assert cfg["method_settings"]["a5"]["lo"] == 0.5 and cfg["method_settings"]["a5"]["hi"] == 2.0
    assert cfg["seed"] == 42 and cfg["device"] == "cpu" and cfg["methods"] == ["baseline", "enhanced_v2", "a5", "single_mask"]
    assert cfg["method_settings"]["enhanced_v2"]["uncertainty"] == "none"
    (env["root"] / "code/method.py").write_text("# changed\n")
    with pytest.raises(C.OutputConflict):
        do_prepare(env)  # a different config can never replace the frozen one
    with pytest.raises(E.EvalError, match="code changed"):
        E.load_frozen(env["eval"], env["root"])


# --------------------------------------------------------------------------- #
# method logic (ported holdout.py body)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale,expect", [(1.0, "box"), (0.5 ** 0.5, "box"), (0.3, "enhanced"), (1.9, "enhanced")])
def test_evaluate_query_a5_and_call_order(scale, expect):
    calls = Calls()
    img, m = blob_image(1, 256, 256)
    rgb = np.stack([img] * 3, -1)
    smask = (m > 0).astype(np.uint8)
    row, arr = E.evaluate_query(rgb, m > 0, rgb, smask, eng=FakeEngine(calls), base=FakeBaseline(calls, scale),
                                enh=FakeEnhanced(calls), select_best_mask=select_best_mask,
                                compute_all_metrics=compute_all_metrics, metric_keys=METRIC_KEYS)
    assert [c[0] for c in calls] == ["invalidate", "baseline", "enhanced", "predict"]
    assert calls[-1][1] is False and calls[-1][2] == "float32" and calls[-1][3] == ("float32", "int32", 2)
    ratio = row["box_px"] / row["support_px"]
    assert row["a5_choice"] == ("box" if 0.5 <= ratio <= 2.0 else "enhanced") == expect
    src = "baseline" if expect == "box" else "enhanced"
    for k in METRIC_KEYS:
        assert row[f"a5_{k}"] == row[f"{src}_{k}"]
    assert row["a5_ms"] == (10.0 if expect == "box" else 10.0 + 12.0)
    assert set(arr) == {"gt", "baseline", "enhanced", "single", "box", "points", "labels"}


def test_a5_inclusive_bounds_exact():
    for ratio, expect in ((0.5, "box"), (2.0, "box"), (0.4999, "enhanced"), (2.0001, "enhanced")):
        assert ("box" if E.A5_LO <= ratio <= E.A5_HI else "enhanced") == expect


def test_holdout_body_is_ported_statement_for_statement():
    """Every statement of holdout.py's per-query logic appears in evaluate_query (normalised)."""
    import inspect
    ported = {ln.strip() for ln in inspect.getsource(E.evaluate_query).splitlines()}
    holdout = (C.PROJECT_ROOT / "results/v3/scripts/holdout.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(holdout) if "support_px = int((smask > 0).sum())" in l)
    end = next(i for i, l in enumerate(holdout) if "np.savez_compressed" in l)
    must = ["eng.invalidate()", "rb = base.predict_arrays(q, smask, gt)", "re_ = enh.predict_arrays(q, s, smask, gt)",
            "ms, sc, _ = eng.predictor.predict(multimask_output=False, **prompt)", "m_single, s_single, _ = select_best_mask(ms, sc)",
            "rs = compute_all_metrics(m_single, gt, 0.0)", "keep_box = A5_LO <= ratio <= A5_HI",
            'row[f"a5_{m}"] = rb[m] if keep_box else re_[m]', 'row["a5_ms"] = rb["inference_ms"] + (0.0 if keep_box else enh_extra)']
    body = [l.strip() for l in holdout[start:end]]
    for stmt in must:
        assert any(stmt in l for l in body), f"not in holdout: {stmt}"
        assert any(stmt in l for l in ported), f"not ported: {stmt}"


# --------------------------------------------------------------------------- #
# execution: resume, duplicates, hashes, failures, guard integration
# --------------------------------------------------------------------------- #
def test_execute_end_to_end_and_finalize(env):
    do_prepare(env)
    rc, calls, _ = run_exec(env)
    assert rc == 0
    assert sum(c[0] == "baseline" for c in calls) == 6  # 6 unique slices, each evaluated once
    rows = (env["eval"] / "raw_rows.csv").read_text().splitlines()
    assert len(rows) == 1 + 6
    assert len((env["eval"] / "raw_rows_primary.csv").read_text().splitlines()) == 1 + 4
    assert len((env["eval"] / "raw_rows_sensitivity.csv").read_text().splitlines()) == 1 + 4
    assert history(env)[-2]["event"] == "finalized"


def test_execute_refused_without_approval_flags(capsys):
    assert E.main(["execute"]) == 1
    assert E.main(["execute", "--execute"]) == 1
    assert "approval" in capsys.readouterr().err


def test_rerun_skips_verified_no_duplicates(env):
    do_prepare(env)
    run_exec(env)
    mt = {p: p.stat().st_mtime_ns for p in outputs(env)}
    rc, calls, _ = run_exec(env)
    assert rc == 0 and not any(c[0] == "baseline" for c in calls)
    assert {p: p.stat().st_mtime_ns for p in outputs(env)} == mt


def test_interrupt_then_resume(env, tmp_path):
    do_prepare(env)
    rc, calls, _ = run_exec(env, enh=None)
    clean = {p.relative_to(env["eval"]): json.loads(p.read_text())["row"] for p in outputs(env)}
    # fresh eval dir, interrupt at the 3rd query
    env2 = dict(env, eval=tmp_path / "eval2")
    do_prepare(env2)
    c2 = Calls()
    rc, _, _ = run_exec(env2, calls=c2, enh=FakeEnhanced(c2, interrupt_on={3}))
    assert rc == 2 and len(outputs(env2)) == 2 and "KeyboardInterrupt" in history(env2)[-1]["error"]
    first = {p: p.stat().st_mtime_ns for p in outputs(env2)}
    c3 = Calls()
    rc, _, _ = run_exec(env2, calls=c3)
    assert rc == 0 and sum(c[0] == "baseline" for c in c3) == 4  # only the 4 missing slices
    assert all(p.stat().st_mtime_ns == t for p, t in first.items())
    resumed = {p.relative_to(env2["eval"]): json.loads(p.read_text())["row"] for p in outputs(env2)}
    drop = {"single_decode_ms"}
    assert {k: {f: v for f, v in r.items() if f not in drop} for k, r in resumed.items()} == \
           {k: {f: v for f, v in r.items() if f not in drop} for k, r in clean.items()}


def test_resource_decline_stops_cleanly_and_resumes(env):
    do_prepare(env)
    # start check, subject s0001 check, then per-slice checks: RAM drops below 1 GB after 2 slices
    rc, _, _ = run_exec(env, ram=[3 * GB, 3 * GB, 3 * GB, 0.5 * GB])
    assert rc == 3
    h = history(env)
    assert h[-1]["event"] == "RESOURCE_STOP" and "available RAM" in h[-1]["reason"]
    assert len(outputs(env)) == 2 and not (env["eval"] / "subjects/s0001.done").exists()
    log = [json.loads(x) for x in (env["tmp"] / "res.jsonl").read_text().splitlines()]
    assert log[-1]["ok"] is False and all(r["ok"] for r in log[:-1])
    rc, calls, _ = run_exec(env)
    assert rc == 0 and sum(c[0] == "baseline" for c in calls) == 4


def test_guard_blocks_model_load(env):
    do_prepare(env)
    loaded = []
    g = guard(0.2 * GB, 10 * GB, env["tmp"])
    rc = E.execute(eval_dir=env["eval"], project_root=env["root"], query_dir=env["pre"], guard_start=g, guard_run=g,
                   methods_factory=lambda: loaded.append(1), fns=FNS)
    assert rc == 3 and not loaded


def test_failure_stops_run_and_later_subjects_untouched(env):
    do_prepare(env)
    c = Calls()
    rc, _, _ = run_exec(env, calls=c, enh=FakeEnhanced(c, fail_on={2}))
    assert rc == 2
    h = history(env)
    assert h[-1]["event"] == "FAILED" and "simulated method failure" in h[-1]["error"]
    started = [x["subject"] for x in h if x["event"] == "subject_start"]
    assert started == ["s0001"]
    assert not list((env["eval"] / "slices").glob("s0002/*")) and not (env["eval"] / "subjects/s0001.done").exists()


def test_tampered_prediction_detected(env):
    do_prepare(env)
    run_exec(env)
    npz = next((env["eval"] / "masks/s0001").glob("*.npz"))
    npz.write_bytes(npz.read_bytes() + b"x")
    rc, calls, _ = run_exec(env)
    assert rc == 2 and "hash mismatch" in history(env)[-1]["error"]


def test_tampered_query_input_detected(env):
    do_prepare(env)
    f = env["pre"] / "images" / C.output_name("s0001", 10)
    f.write_bytes(f.read_bytes() + b"x")
    rc, _, _ = run_exec(env)
    assert rc == 2 and "input hash mismatch" in history(env)[-1]["error"]


def test_foreign_or_stale_result_never_overwritten(env):
    do_prepare(env)
    items = json.loads((env["eval"] / "worklist.json").read_text())["items"]
    p = E.slice_paths(env["eval"], items[0])["json"]
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"config_sha256": "0" * 64, "item": {}, "row": {}}))
    before = p.read_bytes()
    rc, _, _ = run_exec(env)
    assert rc == 2 and "different run_config" in history(env)[-1]["error"] and p.read_bytes() == before


def test_orphan_prediction_file_is_ignored_not_deleted(env):
    do_prepare(env)
    items = json.loads((env["eval"] / "worklist.json").read_text())["items"]
    orphan = env["eval"] / "masks" / items[0]["subject"] / f"{Path(items[0]['file']).stem}.deadbeef0000.npz"
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"left by an interrupted run")
    rc, _, _ = run_exec(env)
    assert rc == 0 and orphan.exists()
    rec = json.loads(E.slice_paths(env["eval"], items[0])["json"].read_text())
    assert rec["npz_file"] != orphan.name


def test_partial_subject_run_does_not_finalize(env):
    do_prepare(env)
    rc, _, _ = run_exec(env, subjects=["s0004"])
    assert rc == 0 and (env["eval"] / "subjects/s0004.done").exists() and not (env["eval"] / "raw_rows.csv").exists()
    with pytest.raises(E.EvalError, match="not in worklist"):
        run_exec(env, subjects=["s0003"])
