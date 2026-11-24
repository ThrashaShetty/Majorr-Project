"""finalize_eval.py on the mocked runner environment (no SAM)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize_eval as F  # noqa: E402
from test_eval_runner import do_prepare, env, no_real_sam, run_exec  # noqa: E402,F401  (fixtures reused)


def rows(p: Path):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def test_refuses_until_all_subjects_complete_then_writes_tables(env):
    do_prepare(env)
    assert F.main(eval_dir=env["eval"], project_root=env["root"]) == 2      # nothing evaluated yet
    for s in ("s0001", "s0002", "s0004"):                                  # batch-style runs: never finalize
        rc, _, _ = run_exec(env, subjects=[s])
        assert rc == 0
    assert not (env["eval"] / "raw_rows.csv").exists()
    assert F.main(eval_dir=env["eval"], project_root=env["root"]) == 0
    allr, prim, sens = (rows(env["eval"] / n) for n in ("raw_rows.csv", "raw_rows_primary.csv", "raw_rows_sensitivity.csv"))
    assert (len(allr), len(prim), len(sens)) == (6, 4, 4)
    assert all(r["in_primary"] == "True" for r in prim) and all(r["in_sensitivity"] == "True" for r in sens)
    # write-once: a second finalize is an identical no-op
    assert F.main(eval_dir=env["eval"], project_root=env["root"]) == 0
