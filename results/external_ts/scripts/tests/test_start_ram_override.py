"""Tests for the explicit resource-constrained start-RAM override (Phase D). Mocked; no SAM, no real outputs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_external_ts as E  # noqa: E402
import run_batches as RB  # noqa: E402
import run_with_mem_opt as L  # noqa: E402
from ts_resources import GB  # noqa: E402


# ------------------------------------------------------------------ rule
@pytest.mark.parametrize("value,flag,expected", [
    (None, False, (1.6, False)),      # default unchanged
    (1.6, False, (1.6, False)),       # explicit default is not a deviation
    (1.6, True, (1.6, False)),
    (1.0, True, (1.0, True)),         # approved deviation
    (0.8, True, (0.8, True)),         # lowest accepted
])
def test_resolve_accepts(value, flag, expected):
    assert L.resolve_start_ram(value, flag) == expected


@pytest.mark.parametrize("value,flag,msg", [
    (1.0, False, "requires --resource-constrained-deviation"),
    (0.79, True, "below the minimum"),
    (0.3, True, "below the minimum"),
    (1.7, True, "only lowering"),
    (None, True, "without a lower"),
    (math.nan, True, "invalid"),
    (math.inf, True, "invalid"),
])
def test_resolve_refuses(value, flag, msg):
    with pytest.raises(SystemExit, match=msg):
        L.resolve_start_ram(value, flag)


def test_stop_and_disk_limits_are_fixed():
    assert (L.RUN_RAM_GB, L.DISK_GB, L.DEFAULT_START_RAM_GIB) == (0.3, 2.0, 1.6)


# ------------------------------------------------------------------ launcher
def test_launcher_refuses_before_any_work(monkeypatch):
    monkeypatch.setattr(L, "verify_optimisation", lambda: (_ for _ in ()).throw(AssertionError("must not be reached")))
    with pytest.raises(SystemExit, match="requires --resource-constrained-deviation"):
        L.main(["--subjects", "s0965", "--start-ram-gib", "1.0", "--check-only"])


def test_tag_history_adds_fields_and_restores(tmp_path):
    orig = E.append_jsonl
    restore = L.tag_history(1.0, True)
    E.append_jsonl(tmp_path / "h.jsonl", {"event": "x"})
    restore()
    assert E.append_jsonl is orig
    rec = json.loads((tmp_path / "h.jsonl").read_text())
    assert rec == {"event": "x", "start_ram_gib": 1.0, "resource_constrained_deviation": True}


@pytest.mark.parametrize("argv,start,dev", [
    ([], 1.6, False),
    (["--start-ram-gib", "1.0", "--resource-constrained-deviation"], 1.0, True),
])
def test_launcher_execute_records_threshold_everywhere(monkeypatch, tmp_path, argv, start, dev):
    import mem_opt
    monkeypatch.setattr(E, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(L, "verify_optimisation", lambda: {"evidence_sha256": "x"})
    monkeypatch.setattr(L, "preflight", lambda s: {"subjects": {}})
    monkeypatch.setattr(mem_opt, "install", lambda: {"active": True})
    monkeypatch.setattr(E, "load_frozen", lambda *a: ({"checkpoint": {"sha256": "x"}}, "cfg", []))
    seen = {}

    def fake_execute(**kw):  # stands in for the frozen runner: writes history via E.append_jsonl like the real one
        seen.update(kw)
        E.append_jsonl(tmp_path / "logs" / "run_history.jsonl", {"event": "run_start"})
        E.append_jsonl(tmp_path / "logs" / "run_history.jsonl", {"event": "run_end"})
        return 0

    monkeypatch.setattr(E, "execute", fake_execute)
    orig = E.append_jsonl
    assert L.main(["--subjects", "s0965", "--execute", "--i-have-explicit-approval", *argv]) == 0
    assert E.append_jsonl is orig                                # tagging removed after the run
    recs = [json.loads(x) for x in (tmp_path / "logs/run_history.jsonl").read_text().splitlines()]
    assert len(recs) == 3 and all(r["start_ram_gib"] == start and r["resource_constrained_deviation"] is dev for r in recs)
    assert recs[0]["guards"]["start_ram_gib"] == start and recs[0]["guards"]["run_ram_gib"] == 0.3
    assert seen["guard_start"].min_ram_bytes == int(start * GB)
    assert seen["guard_run"].min_ram_bytes == int(0.3 * GB)       # stop limit unchanged
    assert seen["guard_start"].min_disk_bytes == seen["guard_run"].min_disk_bytes == int(2.0 * GB)


# ------------------------------------------------------------------ batch driver
def test_real_launcher_argv(monkeypatch):
    captured = []
    monkeypatch.setattr(L, "main", lambda argv: captured.append(argv) or 0)
    RB.real_launcher(["s1"])
    RB.real_launcher(["s1"], 1.0, True)
    assert "--start-ram-gib" not in captured[0] and "--resource-constrained-deviation" not in captured[0]
    assert captured[1][-3:] == ["--start-ram-gib", "1.0", "--resource-constrained-deviation"]


def _env(tmp_path):
    ev = tmp_path / "eval"
    (ev / "subjects").mkdir(parents=True)
    items = [{"subject": f"s{i:04d}", "z": 0} for i in range(3)]
    (ev / "worklist.json").write_text(json.dumps({"items": items}))
    logs = []

    def launcher(b):
        for s in b:
            (ev / "subjects" / f"{s}.done").write_text("{}")
        return 0

    def verify(b, a, out, snap):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"all_passed": True}))
        return 0

    def backup(d):
        d.mkdir(parents=True, exist_ok=True)
        (d / f"b{len(list(d.iterdir()))}.zip").write_bytes(b"z")
        return 0

    return dict(backup_dest=tmp_path / "bk", eval_dir=ev, batch_dir=tmp_path / "batches", audit_dir=tmp_path / "audit",
                project_root=tmp_path, expected=None, ac=lambda: True, awake=lambda: 1, release=lambda: 1,
                launcher=launcher, verify=verify, backup=backup, log=logs.append), logs


def test_driver_tags_every_record_with_deviation(tmp_path):
    kw, logs = _env(tmp_path)
    assert RB.run(**kw, start_ram_gib=1.0, deviation=True) == 0
    assert logs and all(r["start_ram_gib"] == 1.0 and r["resource_constrained_deviation"] is True for r in logs)


def test_driver_default_is_unchanged(tmp_path):
    kw, logs = _env(tmp_path)
    assert RB.run(**kw) == 0
    assert all(r["start_ram_gib"] == 1.6 and r["resource_constrained_deviation"] is False for r in logs)


def test_driver_refuses_lower_limit_without_flag_before_any_work(tmp_path):
    kw, logs = _env(tmp_path)
    with pytest.raises(SystemExit, match="requires --resource-constrained-deviation"):
        RB.run(**kw, start_ram_gib=1.0)
    assert not logs and not (tmp_path / "batches").exists()


def test_driver_cli_refuses_early():
    with pytest.raises(SystemExit, match="requires --resource-constrained-deviation"):
        RB.main(["--backup-dest", "x", "--dry-run", "--start-ram-gib", "1.0"])
