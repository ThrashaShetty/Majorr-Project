"""Mocked tests for run_batches.py: sequencing, AC failure, interruption/resume, verification failure,
backup/mirror failure and sleep-prevention release. No SAM, no real evaluation outputs touched.
Run: python -B -m pytest -p no:cacheprovider -q results/external_ts/scripts/tests/test_run_batches.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import run_batches as RB  # noqa: E402
import ts_common as C  # noqa: E402

SUBJECTS = [f"s{i:04d}" for i in range(1, 90)]  # 89 remaining


@pytest.fixture()
def env(tmp_path):
    ev = tmp_path / "eval"
    items = [{"subject": "s0965", "z": z} for z in range(16)]
    n = 0
    for k, s in enumerate(SUBJECTS):
        for z in range(12 if k % 2 else 12):
            items.append({"subject": s, "z": z})
            n += 1
    (ev / "subjects").mkdir(parents=True)
    (ev / "worklist.json").write_text(json.dumps({"items": items}))
    (ev / "subjects/s0965.done").write_text("{}")
    st = {"calls": [], "verify": [], "backups": 0, "awake": 0, "release": 0}

    def launcher(batch):
        st["calls"].append(list(batch))
        for s in batch:
            (ev / "subjects" / f"{s}.done").write_text("{}")
        return 0

    def verify(batch, allowed, out, snap):
        st["verify"].append((list(batch), list(allowed), snap.exists()))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"all_passed": True}))
        return 0

    def backup(dest):
        st["backups"] += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"b{st['backups']:02d}.zip").write_bytes(b"zip%d" % st["backups"])
        return 0

    def awake():
        st["awake"] += 1
        return 1

    def release():
        st["release"] += 1
        return 1

    logs = []
    kw = dict(backup_dest=tmp_path / "backups", eval_dir=ev, batch_dir=tmp_path / "batches", audit_dir=tmp_path / "audit",
              project_root=tmp_path, expected=(89, n, 9), ac=lambda: True, awake=awake, release=release,
              launcher=launcher, verify=verify, backup=backup, log=logs.append)
    return dict(tmp=tmp_path, ev=ev, st=st, kw=kw, logs=logs, n=n)


def events(env):
    return [r["event"] for r in env["logs"]]


def test_plan_nine_batches_in_worklist_order(env):
    assert RB.run(**env["kw"]) == 0
    plan = json.loads((env["tmp"] / "batches/batch_plan.json").read_text())
    assert [len(b) for b in plan["batches"]] == [10] * 8 + [9]
    assert [s for b in plan["batches"] for s in b] == SUBJECTS and "s0965" not in plan["subjects"]
    assert env["st"]["calls"] == plan["batches"]
    assert env["st"]["backups"] == 9 and len(env["st"]["verify"]) == 9
    # each verification allows exactly the subjects completed before the batch (incl. s0965) and has a snapshot
    assert env["st"]["verify"][0][1] == ["s0965"] and all(v[2] for v in env["st"]["verify"])
    assert set(env["st"]["verify"][3][1]) == {"s0965", *SUBJECTS[:30]}
    assert env["st"]["awake"] == 1 and env["st"]["release"] == 1
    assert events(env)[-2:] == ["driver_complete", "sleep_prevention_released"]


def test_expected_workload_enforced(env):
    with pytest.raises(C.TSError, match="remaining workload"):
        RB.run(**{**env["kw"], "expected": (90, env["n"], 9)})
    assert not (env["tmp"] / "batches/batch_plan.json").exists()


def test_plan_write_once_and_tamper_detected(env):
    RB.run(**env["kw"])
    p = env["tmp"] / "batches/batch_plan.json"
    p.write_bytes(p.read_bytes().replace(b"s0001", b"s9999"))
    with pytest.raises(C.TSError, match="hash mismatch"):
        RB.run(**env["kw"])


def test_ac_lost_before_batch_3_stops(env):
    seq = iter([True, True, False])
    assert RB.run(**{**env["kw"], "ac": lambda: next(seq)}) == RB.EXIT["ac_lost"]
    assert len(env["st"]["calls"]) == 2 and env["st"]["backups"] == 2
    assert env["st"]["release"] == 1 and events(env)[-1] == "sleep_prevention_released"


@pytest.mark.parametrize("rc,code", [(3, "resource_stop"), (2, "runner_failed")])
def test_launcher_stop_codes(env, rc, code):
    base = env["kw"]["launcher"]
    def launcher(b):
        return rc if len(env["st"]["calls"]) == 1 and not env["st"]["calls"].append(b) else base(b)
    assert RB.run(**{**env["kw"], "launcher": launcher}) == RB.EXIT[code]
    assert len(env["st"]["verify"]) == 1 and env["st"]["backups"] == 1 and env["st"]["release"] == 1


def test_verification_failure_stops_without_backup(env):
    base = env["kw"]["verify"]
    def verify(b, a, out, snap):
        if len(env["st"]["verify"]) == 1:
            env["st"]["verify"].append("fail")
            return 1
        return base(b, a, out, snap)
    assert RB.run(**{**env["kw"], "verify": verify}) == RB.EXIT["verify_failed"]
    assert len(env["st"]["calls"]) == 2 and env["st"]["backups"] == 1  # no backup for the failed batch, no batch 3


def test_backup_failure_stops(env):
    assert RB.run(**{**env["kw"], "backup": lambda d: 1}) == RB.EXIT["backup_failed"]
    assert len(env["st"]["calls"]) == 1 and env["st"]["release"] == 1


def test_interrupt_then_resume_skips_verified_batches(env):
    base = env["kw"]["launcher"]
    def launcher(b):
        if len(env["st"]["calls"]) == 3:
            raise KeyboardInterrupt
        return base(b)
    assert RB.run(**{**env["kw"], "launcher": launcher}) == RB.EXIT["interrupted"]
    assert env["st"]["release"] == 1 and len(env["st"]["calls"]) == 3
    env["st"]["calls"].clear()
    assert RB.run(**env["kw"]) == 0
    plan = json.loads((env["tmp"] / "batches/batch_plan.json").read_text())
    assert env["st"]["calls"] == plan["batches"][3:]               # batches 1-3 skipped, 4..9 run
    assert events(env).count("batch_skipped_verified") == 3 and env["st"]["release"] == 2


def test_done_but_unverified_batch_is_rerun_and_verified(env):
    RB.load_or_create_plan(env["kw"]["batch_dir"], env["ev"], env["kw"]["expected"])  # plan exists first
    for s in SUBJECTS[:10]:
        (env["ev"] / "subjects" / f"{s}.done").write_text("{}")
    assert RB.run(**env["kw"]) == 0
    assert env["st"]["calls"][0] == SUBJECTS[:10]  # launcher still called (it verifies and skips internally)


def test_release_called_on_unexpected_exception(env):
    def verify(*a):
        raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        RB.run(**{**env["kw"], "verify": verify})
    assert env["st"]["awake"] == 1 and env["st"]["release"] == 1


def test_mirror_copy_verified_and_never_overwrites(env):
    mirror = env["tmp"] / "usb"
    assert RB.run(**{**env["kw"], "mirror_dest": mirror}) == 0
    assert sorted(p.name for p in mirror.iterdir()) == [f"b{i:02d}.zip" for i in range(1, 10)]
    src, before = env["tmp"] / "backups", set()
    (src / "new.zip").write_bytes(b"x")
    (mirror / "new.zip").write_bytes(b"other")
    with pytest.raises(C.TSError, match="exists"):
        RB.mirror_copy(src, set(p.name for p in src.iterdir()) - {"new.zip"}, mirror)
    assert (mirror / "new.zip").read_bytes() == b"other"


def test_dry_run_writes_nothing(env, capsys):
    assert RB.run(**{**env["kw"], "dry_run": True}) == 0
    assert not (env["tmp"] / "batches").exists() and not env["st"]["calls"] and env["st"]["awake"] == 0
    assert '"batches": 9' in capsys.readouterr().out


def test_cli_refuses_without_approval(capsys):
    assert RB.main(["--backup-dest", "x"]) == 1


def test_real_sleep_prevention_set_and_released():
    prev = RB.keep_awake()
    assert prev != 0                               # API call succeeded
    state = RB.release_awake()                     # returns the state we had set
    assert state & RB.ES_SYSTEM_REQUIRED and state & RB.ES_CONTINUOUS


def test_real_ac_check_returns_bool():
    assert isinstance(RB.ac_power_connected(), bool)
