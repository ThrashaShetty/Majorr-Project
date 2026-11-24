"""Sequential batch driver for the remaining TS evaluation (single process, one batch at a time).

Per batch:  AC-power check -> run_with_mem_opt (frozen runner, all existing guards) -> independent
            verify_eval_subjects (batch subjects; earlier verified subjects allowed; pre-batch snapshot
            of existing eval outputs must be unchanged) -> new verified Tier-1 backup (+ optional mirror copy).
Stops immediately (non-zero exit) on: AC lost (4), resource guard stop (3), runner failure (2),
verification failure (5), backup/mirror failure (6), interruption (130). Completed work is always kept;
re-running resumes: batches whose subjects are all done AND have a passing verification report are skipped.

Sleep prevention: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) for the lifetime of the run,
released in `finally` (Windows also drops it automatically when the process exits). Power settings are not
changed. Nothing is ever terminated or deleted.

Batch plan: written once (write-once JSON + sidecar) at results/external_ts/batches/batch_plan.json from the
subjects remaining when it is first created; later runs reuse and verify it.

Usage:
  python -B -I run_batches.py --dry-run --backup-dest C:/Anvith/FINALYEARPROJECTTHRASHA/backups/external_ts
  python -B -I run_batches.py --execute --i-have-explicit-approval --backup-dest <dir> [--mirror-dest <dir>]
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as C  # noqa: E402

RESULTS = C.RESULTS
EVAL = RESULTS / "eval"
BATCH_DIR = RESULTS / "batches"
AUDIT = RESULTS / "audit"
BATCH_SIZE = 10
EXPECTED_SUBJECTS, EXPECTED_SLICES, EXPECTED_BATCHES = 89, 1062, 9
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
EXIT = {"ok": 0, "runner_failed": 2, "resource_stop": 3, "ac_lost": 4, "verify_failed": 5, "backup_failed": 6,
        "interrupted": 130}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ------------------------------------------------------------------ real environment hooks
def ac_power_connected() -> bool:
    import psutil
    b = psutil.sensors_battery()
    return True if b is None else bool(b.power_plugged)  # no battery (desktop) -> mains


def keep_awake() -> int:
    return int(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))


def release_awake() -> int:
    return int(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS))


def real_launcher(subjects: list[str], start_ram_gib: float | None = None, deviation: bool = False) -> int:
    import run_with_mem_opt
    argv = ["--execute", "--i-have-explicit-approval", "--subjects", *subjects]
    if start_ram_gib is not None:
        argv += ["--start-ram-gib", str(start_ram_gib)]
    if deviation:
        argv += ["--resource-constrained-deviation"]
    return run_with_mem_opt.main(argv)


def resolve_start_ram(value: float | None, deviation: bool) -> tuple[float, bool]:
    """Same rule as the launcher (single source of truth in run_with_mem_opt.resolve_start_ram)."""
    import run_with_mem_opt
    return run_with_mem_opt.resolve_start_ram(value, deviation)


def real_verify(subjects: list[str], allowed: list[str], out: Path, snapshot: Path) -> int:
    import verify_eval_subjects
    return verify_eval_subjects.main(["--subjects", *subjects, "--out", str(out), "--unchanged-snapshot", str(snapshot),
                                      "--allow-existing", *allowed])


def real_backup(dest: Path) -> int:
    import make_backup
    return make_backup.main(["--dest", str(dest)])


# ------------------------------------------------------------------ plan / state
def done_subjects(eval_dir: Path) -> list[str]:
    return sorted(p.stem for p in (eval_dir / "subjects").glob("*.done")) if (eval_dir / "subjects").exists() else []


def remaining(eval_dir: Path) -> tuple[list[str], int]:
    wl = json.loads((eval_dir / "worklist.json").read_text(encoding="utf-8"))["items"]
    done = set(done_subjects(eval_dir))
    order: list[str] = []
    for w in wl:
        if w["subject"] not in done and w["subject"] not in order:
            order.append(w["subject"])
    return order, sum(1 for w in wl if w["subject"] not in done)


def make_plan(subjects: list[str], size: int = BATCH_SIZE) -> list[list[str]]:
    return [subjects[i:i + size] for i in range(0, len(subjects), size)]


def load_or_create_plan(batch_dir: Path, eval_dir: Path, expected: tuple[int, int, int] | None) -> dict[str, Any]:
    p = batch_dir / "batch_plan.json"
    if p.exists():
        data = p.read_bytes()
        side = p.with_suffix(".sha256").read_text().split()[0]
        if C.sha256_bytes(data) != side:
            raise C.TSError("batch_plan.json hash mismatch")
        return json.loads(data)
    subs, n_slices = remaining(eval_dir)
    batches = make_plan(subs)
    if expected and (len(subs), n_slices, len(batches)) != expected:
        raise C.TSError(f"remaining workload {len(subs)} subjects / {n_slices} slices / {len(batches)} batches != {expected}")
    plan = {"created_utc": now(), "subjects": subs, "n_slices": n_slices, "batch_size": BATCH_SIZE,
            "batches": batches, "already_done_at_creation": done_subjects(eval_dir),
            "worklist_sha256": C.file_digest(eval_dir / "worklist.json")}
    data = C.json_bytes(plan)
    C.write_once(p, data)
    C.write_once(p.with_suffix(".sha256"), f"{C.sha256_bytes(data)}  {p.name}\n".encode())
    return plan


def passing_report(audit: Path, k: int) -> Path | None:
    for r in sorted(audit.glob(f"batch_{k:02d}_verification_*.json")):
        try:
            if json.loads(r.read_text(encoding="utf-8")).get("all_passed") is True:
                return r
        except Exception:  # noqa: BLE001
            continue
    return None


def snapshot_eval(eval_dir: Path, out: Path, project_root: Path) -> None:
    lines = []
    for f in sorted(eval_dir.rglob("*")):
        if f.is_file() and "logs" not in f.relative_to(eval_dir).parts:
            lines.append(f"{C.file_digest(f)}  {f.relative_to(project_root).as_posix()}\n")
    C.write_once(out, "".join(lines).encode())


def mirror_copy(src_dir: Path, names_before: set[str], mirror: Path) -> list[str]:
    """Copy the NEW backup files to the mirror (tmp + rename), verify hashes, never overwrite."""
    new = sorted(n for n in os.listdir(src_dir) if n not in names_before and not n.startswith(".tmp_"))
    mirror.mkdir(parents=True, exist_ok=True)
    for n in new:
        dst = mirror / n
        if dst.exists():
            raise C.TSError(f"mirror target exists: {dst}")
        tmp = mirror / f".tmp_{n}.{os.getpid()}"
        shutil.copyfile(src_dir / n, tmp)
        if C.file_digest(tmp) != C.file_digest(src_dir / n):
            raise C.TSError(f"mirror copy hash mismatch for {n} (temp kept at {tmp})")
        os.replace(tmp, dst)
    return new


# ------------------------------------------------------------------ driver
def run(*, backup_dest: Path, mirror_dest: Path | None = None, dry_run: bool = False,
        eval_dir: Path = EVAL, batch_dir: Path = BATCH_DIR, audit_dir: Path = AUDIT, project_root: Path = C.PROJECT_ROOT,
        expected: tuple[int, int, int] | None = (EXPECTED_SUBJECTS, EXPECTED_SLICES, EXPECTED_BATCHES),
        ac: Callable[[], bool] = ac_power_connected, awake: Callable[[], int] = keep_awake,
        release: Callable[[], int] = release_awake, launcher: Callable[[list[str]], int] = real_launcher,
        verify: Callable[[list[str], list[str], Path, Path], int] = real_verify,
        backup: Callable[[Path], int] = real_backup, log: Callable[[dict], None] | None = None,
        start_ram_gib: float | None = None, deviation: bool = False) -> int:
    start_gib, is_dev = resolve_start_ram(start_ram_gib, deviation)  # validated before anything else
    hist = eval_dir / "logs" / "run_history.jsonl"
    def emit(rec: dict) -> None:
        rec = {"utc": now(), "driver": "run_batches", **rec, "start_ram_gib": start_gib, "resource_constrained_deviation": is_dev}
        (log or (lambda r: append(hist, r)))(rec)
        print(json.dumps(rec), flush=True)

    if dry_run:
        subs, n = remaining(eval_dir)
        plan = make_plan(subs)
        print(json.dumps({"dry_run": True, "remaining_subjects": len(subs), "remaining_slices": n, "batches": len(plan),
                          "batch_sizes": [len(b) for b in plan], "ac_power": ac(), "plan_file_exists": (batch_dir / "batch_plan.json").exists()}, indent=1))
        return 0
    plan = load_or_create_plan(batch_dir, eval_dir, expected)
    emit({"event": "driver_start", "batches": len(plan["batches"]), "subjects": len(plan["subjects"])})
    awake_state = awake()
    emit({"event": "sleep_prevention_on", "previous_state": awake_state})
    try:
        for k, batch in enumerate(plan["batches"], 1):
            done = set(done_subjects(eval_dir))
            rep = passing_report(audit_dir, k)
            if set(batch) <= done and rep is not None:
                emit({"event": "batch_skipped_verified", "batch": k, "report": rep.name})
                continue
            if not ac():
                emit({"event": "STOP", "batch": k, "reason": "AC power not connected"})
                return EXIT["ac_lost"]
            stamp = now()
            snap = batch_dir / f"pre_batch_{k:02d}_{stamp}.sha256"
            snapshot_eval(eval_dir, snap, project_root)
            emit({"event": "batch_start", "batch": k, "subjects": batch, "snapshot": snap.name})
            rc = launcher(batch)
            gc.collect()
            if rc != 0:
                key = "resource_stop" if rc == 3 else "runner_failed"
                emit({"event": "STOP", "batch": k, "reason": key, "launcher_rc": rc})
                return EXIT[key]
            allowed = sorted(set(done_subjects(eval_dir)) - set(batch))
            out = audit_dir / f"batch_{k:02d}_verification_{stamp}.json"
            vrc = verify(batch, allowed, out, snap)
            if vrc != 0:
                emit({"event": "STOP", "batch": k, "reason": "verification failed", "report": out.name})
                return EXIT["verify_failed"]
            before = set(os.listdir(backup_dest)) if backup_dest.exists() else set()
            if backup(backup_dest) != 0:
                emit({"event": "STOP", "batch": k, "reason": "backup failed"})
                return EXIT["backup_failed"]
            mirrored = None
            if mirror_dest is not None:
                try:
                    mirrored = mirror_copy(backup_dest, before, mirror_dest)
                except Exception as e:  # noqa: BLE001
                    emit({"event": "STOP", "batch": k, "reason": f"mirror failed: {e!r}"})
                    return EXIT["backup_failed"]
            emit({"event": "batch_complete", "batch": k, "verification": out.name, "mirrored": mirrored})
        emit({"event": "driver_complete"})
        return EXIT["ok"]
    except KeyboardInterrupt:
        emit({"event": "STOP", "reason": "interrupted (KeyboardInterrupt)"})
        return EXIT["interrupted"]
    finally:
        r = release()
        emit({"event": "sleep_prevention_released", "result": r})


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup-dest", type=Path, required=True)
    ap.add_argument("--mirror-dest", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--i-have-explicit-approval", action="store_true")
    ap.add_argument("--start-ram-gib", type=float, default=None, help="start RAM limit in GiB (default 1.6)")
    ap.add_argument("--resource-constrained-deviation", action="store_true")
    a = ap.parse_args(argv)
    start_gib, is_dev = resolve_start_ram(a.start_ram_gib, a.resource_constrained_deviation)  # refuse early
    if not a.dry_run and not (a.execute and a.i_have_explicit_approval):
        print("Refusing: requires --execute and --i-have-explicit-approval (or --dry-run)", file=sys.stderr)
        return 1
    launcher = (lambda b: real_launcher(b, start_gib if is_dev else None, is_dev))
    return run(backup_dest=a.backup_dest, mirror_dest=a.mirror_dest, dry_run=a.dry_run, launcher=launcher,
               start_ram_gib=a.start_ram_gib, deviation=a.resource_constrained_deviation)


if __name__ == "__main__":
    raise SystemExit(main())
