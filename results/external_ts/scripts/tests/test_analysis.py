"""Tests for analyze_external_ts.py (synthetic data only; no external results are read)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_external_ts as A  # noqa: E402


def synth(n_pat=12, per=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        for z in range(per):
            b = rng.uniform(0, 1)
            e = np.clip(b - 0.2 + rng.normal(0, .1), 0, 1)
            box = rng.uniform(.3, 2.5)
            a5 = b if 0.5 <= box <= 2.0 else e
            r = {"query_image": f"ts_s{p:04d}_slice_{z:03d}.png", "patient": f"s{p:04d}", "a5_choice": "box" if 0.5 <= box <= 2.0 else "enhanced"}
            for key, v in (("baseline", b), ("enhanced", e), ("a5", a5), ("single", np.clip(e - .02, 0, 1))):
                for k in A.K:
                    r[f"{key}_{k}"] = float(v)
            rows.append(r)
    return pd.DataFrame(rows)


def test_deterministic_and_row_order_independent():
    df = synth()
    a = A.analyse_cohort(df)
    b = A.analyse_cohort(df.sample(frac=1, random_state=3))
    assert a == b


def test_descriptive_does_not_change_confirmatory_stream():
    df = synth().sort_values("query_image").reset_index(drop=True)
    alone = A.confirmatory(df)
    A.descriptive(df)                      # separate generator (Amendment 01)
    again = A.confirmatory(df)
    assert alone == again


def test_holm_is_monotone_and_capped():
    t = [{"wilcoxon_p": p} for p in (0.04, 0.001, 0.3, 0.02)]
    A.holm(t)
    order = np.argsort([x["wilcoxon_p"] for x in t])
    adj = [t[i]["holm_p"] for i in order]
    assert adj == sorted(adj) and all(x <= 1 for x in adj)
    assert t[1]["holm_p"] == pytest.approx(0.004)


def test_all_zero_differences_give_p_one():
    df = synth()
    df["a5_dice"] = df["baseline_dice"]
    r = A.compare(df, "a5", "baseline", "x", np.random.default_rng(42))
    assert r["wilcoxon_p"] == 1.0 and r["n_nonzero_for_wilcoxon"] == 0


def test_patient_is_the_unit():
    df = synth(n_pat=8, per=7)
    r = A.compare(df, "a5", "baseline", "x", np.random.default_rng(42))
    assert r["n_patients"] == 8 and r["n_slices"] == 56


def fake_eval_tree(root: Path, subjects, done, raw_rows=False):
    """A temporary results/external_ts tree: worklist for `subjects`, .done markers for `done`."""
    ev = root / "eval"
    (ev / "subjects").mkdir(parents=True)
    items = [{"subject": s, "z": z} for s in subjects for z in (0, 1)]
    (ev / "worklist.json").write_text(json.dumps({"items": items}))
    for s in done:
        (ev / "subjects" / f"{s}.done").write_text("{}")
    if raw_rows:
        for f in ("raw_rows_primary.csv", "raw_rows_sensitivity.csv"):
            (ev / f).write_text("patient\n")
    return root


def test_run_refuses_when_incomplete(tmp_path, monkeypatch):
    # Simulated unfinished evaluation (independent of the real project's state): 3 subjects in the
    # worklist, only 2 with a completion marker. The SAP/Amendment hash check still reads the real files.
    fake = fake_eval_tree(tmp_path / "ts", ["s0001", "s0002", "s0003"], ["s0001", "s0002"])
    monkeypatch.setattr(A, "R", fake)
    with pytest.raises(SystemExit, match=r"evaluation incomplete: 1 subjects without completion marker"):
        A.run(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_refuses_when_not_finalized(tmp_path, monkeypatch):
    # All subjects done, but finalize_eval.py has not written raw_rows_*.csv yet.
    fake = fake_eval_tree(tmp_path / "ts", ["s0001", "s0002"], ["s0001", "s0002"])
    monkeypatch.setattr(A, "R", fake)
    with pytest.raises(SystemExit, match=r"raw_rows_primary\.csv missing"):
        A.run(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_refuses_existing_out_dir(tmp_path, monkeypatch):
    fake = fake_eval_tree(tmp_path / "ts", ["s0001"], ["s0001"], raw_rows=True)
    monkeypatch.setattr(A, "R", fake)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        A.run(tmp_path / "out")
    assert list((tmp_path / "out").iterdir()) == []


def test_v3_reproduction():
    assert A.validate_v3() == 0
