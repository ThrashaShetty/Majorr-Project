"""Launcher: install the validated in-memory attention optimisation, then run the FROZEN evaluation runner.

Nothing on disk is modified: eval_external_ts.py, run_config.json, src/ and segment_anything stay as they are
(the runner's load_frozen() still verifies every frozen code hash). The launcher:
  1. verifies the validation evidence file (exact SHA-256, A_all_identical) and that mem_opt's upstream and
     patched source hashes equal those recorded in the evidence;
  2. installs mem_opt (process-local monkeypatch of segment_anything Attention.forward);
  3. records the patch + evidence hashes in eval/logs/run_history.jsonl;
  4. calls eval_external_ts.execute() with the SAME guards as eval_external_ts.main():
     start 1.6 GB RAM, run 0.3 GB RAM, disk 2.0 GB on C:\\ (thresholds are not configurable here);
  5. logs process RSS / peak working set / pagefile at every guard check to eval/logs/launcher_memory.jsonl.

Usage:
  python -B -I run_with_mem_opt.py --check-only --subjects s0965
  python -B -I run_with_mem_opt.py --execute --i-have-explicit-approval --subjects s0965
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_external_ts as E  # noqa: E402
import ts_common as C  # noqa: E402
from ts_resources import GB, ResourceGuard, ResourceStop  # noqa: E402

EVIDENCE = C.RESULTS / "audit" / "mem_opt_validation.json"
EVIDENCE_SHA256 = "f4c70af9b029480e29f97aadb9536b6b15ad32460b4a2d897981871ea86c0689"
START_RAM_GB, RUN_RAM_GB, DISK_GB = 1.6, 0.3, 2.0  # identical to eval_external_ts.main defaults (binary GiB)
DEFAULT_START_RAM_GIB = START_RAM_GB
MIN_OVERRIDE_START_RAM_GIB = 0.8  # lowest start limit accepted even with --resource-constrained-deviation
MEM_LOG = E.EVAL_DIR / "logs" / "launcher_memory.jsonl"


def resolve_start_ram(value: float | None, deviation: bool) -> tuple[float, bool]:
    """Return (start_ram_gib, is_deviation). Default 1.6 GiB. A lower value is a resource-constrained
    deviation and is accepted only with the explicit flag and only down to MIN_OVERRIDE_START_RAM_GIB.
    The stop (0.3 GiB) and disk (2.0 GiB) limits are never configurable."""
    import math
    if value is None:
        if deviation:
            raise SystemExit("--resource-constrained-deviation given without a lower --start-ram-gib")
        return DEFAULT_START_RAM_GIB, False
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SystemExit(f"invalid --start-ram-gib {value!r}")
    if value > DEFAULT_START_RAM_GIB:
        raise SystemExit(f"--start-ram-gib {value} exceeds the default {DEFAULT_START_RAM_GIB}; only lowering is supported")
    if value == DEFAULT_START_RAM_GIB:
        return DEFAULT_START_RAM_GIB, False
    if value < MIN_OVERRIDE_START_RAM_GIB:
        raise SystemExit(f"--start-ram-gib {value} is below the minimum override {MIN_OVERRIDE_START_RAM_GIB}")
    if not deviation:
        raise SystemExit(f"--start-ram-gib {value} < {DEFAULT_START_RAM_GIB} requires --resource-constrained-deviation")
    return float(value), True


def tag_history(start_ram_gib: float, deviation: bool):
    """Add the actual start limit and deviation marker to EVERY record written via E.append_jsonl in this
    process (run_history.jsonl events from the frozen runner included) without editing the runner file.
    Returns a function that restores the original writer."""
    original = E.append_jsonl

    def tagged(path, rec):
        original(path, {**rec, "start_ram_gib": start_ram_gib, "resource_constrained_deviation": deviation})

    E.append_jsonl = tagged
    return lambda: setattr(E, "append_jsonl", original)


class LoggingGuard(ResourceGuard):
    """ResourceGuard that additionally records this process's memory at every check (same limits/logic)."""

    def check(self, stage: str) -> dict:
        mi = psutil.Process().memory_info()
        vm = psutil.virtual_memory()
        rec = {"utc": E._now(), "stage": stage, "process_rss": mi.rss, "process_peak_wset": getattr(mi, "peak_wset", None),
               "process_pagefile": getattr(mi, "pagefile", None), "process_peak_pagefile": getattr(mi, "peak_pagefile", None),
               "system_available": vm.available, "system_used_pct": vm.percent,
               "swap_used": psutil.swap_memory().used}
        E.append_jsonl(MEM_LOG, rec)
        return super().check(stage)


def verify_optimisation() -> dict:
    if C.file_digest(EVIDENCE) != EVIDENCE_SHA256:
        raise E.EvalError("validation evidence file hash differs from the approved one")
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if ev.get("A_all_identical") is not True:
        raise E.EvalError("validation evidence does not show identical outputs")
    import mem_opt
    st, rec = mem_opt.status(), ev["mem_opt"]
    for k in ("upstream_attention_forward_sha256", "upstream_add_decomposed_rel_pos_sha256", "patched_forward_sha256",
              "chunk_rows", "min_attn_bytes"):
        if st[k] != rec[k]:
            raise E.EvalError(f"mem_opt {k} differs from the validated version")
    return {"evidence_sha256": EVIDENCE_SHA256, "mem_opt_file_sha256": C.file_digest(Path(mem_opt.__file__)), **st}


def preflight(subjects: list[str]) -> dict:
    cfg, cfg_sha, wl = E.load_frozen(E.EVAL_DIR, C.PROJECT_ROOT)  # frozen config + code hashes + worklist hash
    if C.file_digest(C.PROJECT_ROOT / cfg["checkpoint"]["path"]) != cfg["checkpoint"]["sha256"]:
        raise E.EvalError("checkpoint hash differs")
    out = {"config_sha256": cfg_sha, "subjects": {}}
    for s in subjects:
        items = [w for w in wl if w["subject"] == s]
        if not items:
            raise E.EvalError(f"{s} not in worklist")
        done, todo = [], []
        for it in items:
            (done if E.slice_paths(E.EVAL_DIR, it)["json"].exists() else todo).append(it)
        for it in done:
            E.verify_slice(E.EVAL_DIR, it, cfg_sha)
        for it in todo:  # inputs of the slices still to run
            for f, h in ((C.PREPROCESSED / "images" / it["file"], it["image_sha256"]), (C.PREPROCESSED / "masks" / it["file"], it["mask_sha256"]),
                         (C.PROJECT_ROOT / it["support_image"], it["support_image_sha256"]),
                         (C.PROJECT_ROOT / it["support_mask"], it["support_mask_sha256"])):
                if C.file_digest(f) != h:
                    raise E.EvalError(f"input hash mismatch {f}")
        out["subjects"][s] = {"items": len(items), "completed_verified": [i["z"] for i in done], "to_run": [i["z"] for i in todo],
                              "done_marker": (E.EVAL_DIR / "subjects" / f"{s}.done").exists()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--i-have-explicit-approval", action="store_true")
    ap.add_argument("--start-ram-gib", type=float, default=None,
                    help=f"start RAM limit in GiB (default {DEFAULT_START_RAM_GIB}); lower values need --resource-constrained-deviation")
    ap.add_argument("--resource-constrained-deviation", action="store_true",
                    help="explicitly accept a lower start limit; recorded in every run-history entry")
    a = ap.parse_args(argv)
    start_gib, deviation = resolve_start_ram(a.start_ram_gib, a.resource_constrained_deviation)  # validated before any work
    opt = verify_optimisation()
    pf = preflight(a.subjects)
    probe = ResourceGuard(int(start_gib * GB), int(DISK_GB * GB), Path("C:/"))  # probe only, no log
    try:
        g = probe.check("launcher preflight probe")
        guard = {"ok": True, "ram_available_gb": round(g["ram_available_bytes"] / GB, 3), "disk_free_gb": round(g["disk_free_bytes"] / GB, 2)}
    except ResourceStop as e:
        guard = {"ok": False, "reason": e.reason}
    print(json.dumps({"optimisation": opt, "preflight": pf, "start_guard_probe": guard,
                      "start_ram_gib": start_gib, "resource_constrained_deviation": deviation}, indent=1))
    if a.check_only:
        return 0
    if not (a.execute and a.i_have_explicit_approval):
        print("Refusing: requires --execute and --i-have-explicit-approval", file=sys.stderr)
        return 1
    import mem_opt
    st = mem_opt.install()
    if not st["active"]:
        raise E.EvalError("mem_opt failed to install")
    restore = tag_history(start_gib, deviation)
    try:
        E.append_jsonl(E.EVAL_DIR / "logs" / "run_history.jsonl", {
            "utc": E._now(), "event": "launcher_mem_opt_installed", "launcher": "run_with_mem_opt.py",
            "launcher_sha256": C.file_digest(Path(__file__).resolve()), "mem_opt": opt, "subjects": a.subjects,
            "guards": {"start_ram_gib": start_gib, "run_ram_gib": RUN_RAM_GB, "disk_gib": DISK_GB,
                       "default_start_ram_gib": DEFAULT_START_RAM_GIB, "resource_constrained_deviation": deviation}})
        log = E.EVAL_DIR / "logs" / "resource_checks.jsonl"
        gs = LoggingGuard(int(start_gib * GB), int(DISK_GB * GB), Path("C:/"), log)
        gr = LoggingGuard(int(RUN_RAM_GB * GB), int(DISK_GB * GB), Path("C:/"), log)
        cfg, _, _ = E.load_frozen(E.EVAL_DIR, C.PROJECT_ROOT)
        P = C.PROJECT_ROOT
        return E.execute(eval_dir=E.EVAL_DIR, project_root=P, query_dir=C.PREPROCESSED, guard_start=gs, guard_run=gr,
                         methods_factory=lambda: E.real_methods(P, cfg["checkpoint"]["sha256"]), fns=E.real_fns(),
                         subjects=a.subjects)
    finally:
        restore()


if __name__ == "__main__":
    raise SystemExit(main())
