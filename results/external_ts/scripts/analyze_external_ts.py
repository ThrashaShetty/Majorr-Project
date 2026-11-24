"""Statistical analysis of the TS external evaluation — implements SAP v1 + Amendment 01 exactly.

SAP v1  : results/external_ts/analysis_plan/STATISTICAL_ANALYSIS_PLAN_v1.md (sha256 6a1fee26…)
Amend 01: docs/SAP_AMENDMENT_01_2026-11-23.md (sha256 18be6f47…)

Confirmatory (per cohort, primary family and separate sensitivity family):
  H1 A5 vs Baseline, H2 A5 vs Enhanced v2, H3 Enhanced v2 vs Baseline, H4 Single-mask vs Enhanced v2.
  Unit = patient (unweighted mean over that patient's slices in the cohort). Two-sided Wilcoxon signed-rank
  on per-patient differences after dropping |d| <= 1e-12 (scipy defaults; p = 1.0 if nothing remains).
  Holm over the 4 tests. Bootstrap: one default_rng(42) per cohort, used only for H1..H4 in order, each
  drawing a fresh (10,000 x n_patients) index matrix -> patient-mean and slice-pooled 95% percentile CIs.
  (This is the procedure of results/v3/scripts/v3_report.py; `validate-v3` reproduces v3's published numbers.)
Descriptive CIs (Amendment 01): a SEPARATE default_rng(42) per cohort, methods Baseline->Enhanced v2->A5->Single,
  metrics Dice->IoU->Precision->Recall->Boundary IoU, fresh (10,000 x n) matrix each.

Commands:
  validate-v3                 reproduce results/v3/adaptive/summary.json tests from results/v3/raw_holdout_rows.csv
  run --out-dir <new dir>     analyse the completed external evaluation (refuses if incomplete or unverified)
Outputs are written once to a NEW directory; nothing existing is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCRIPTS = Path(__file__).resolve().parent
P = SCRIPTS.parents[2]
R = P / "results" / "external_ts"
SAP = R / "analysis_plan" / "STATISTICAL_ANALYSIS_PLAN_v1.md"
SAP_SHA = "6a1fee26869a2bc8abf2a2e9d587838f1dc4d8871c29a2f84d1b02e8e306285d"
AMEND = P / "docs" / "SAP_AMENDMENT_01_2026-11-23.md"
AMEND_SHA = "18be6f479aeb9fb8242fb92f208d78017447c765c91d0bedd871c12cc2bcd05a"
K = ["dice", "iou", "precision", "recall", "boundary_iou"]
METHODS = [("baseline", "Baseline"), ("enhanced", "Enhanced v2"), ("a5", "A5"), ("single", "Single-mask")]
TESTS = [("a5", "baseline", "H1 A5 vs Baseline (PRIMARY)"), ("a5", "enhanced", "H2 A5 vs Enhanced v2"),
         ("enhanced", "baseline", "H3 Enhanced v2 vs Baseline"), ("single", "enhanced", "H4 Single-mask vs Enhanced v2")]
B = 10_000


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------ confirmatory (identical to v3_report.compare)
def compare(df: pd.DataFrame, a: str, b: str, label: str, rng: np.random.Generator) -> dict:
    pats = df.patient.unique()
    da, db = df[f"{a}_dice"], df[f"{b}_dice"]
    pa, pb = da.groupby(df.patient).mean(), db.groupby(df.patient).mean()
    diff_p = pa - pb
    d = (da - db).groupby(df.patient).agg(["sum", "count"])
    idx = rng.integers(0, len(pats), (B, len(pats)))
    slice_ci = np.percentile(d["sum"].values[idx].sum(1) / d["count"].values[idx].sum(1), [2.5, 97.5])
    pat_ci = np.percentile(diff_p.values[idx].mean(1), [2.5, 97.5])
    nz = diff_p[diff_p.abs() > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
    return {"comparison": label, "n_patients": int(len(pats)), "n_slices": int(len(df)),
            "slice_mean_diff": float((da - db).mean()), "slice_diff_CI95": [float(x) for x in slice_ci],
            "patient_mean_diff": float(diff_p.mean()), "patient_diff_CI95": [float(x) for x in pat_ci],
            "patients_better": int((diff_p > 0.005).sum()), "patients_worse": int((diff_p < -0.005).sum()),
            "patients_tied(|d|<=0.005)": int((diff_p.abs() <= 0.005).sum()), "n_nonzero_for_wilcoxon": int(len(nz)),
            "wilcoxon_p": p}


def holm(tests: list[dict]) -> None:
    order = np.argsort([t["wilcoxon_p"] for t in tests])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, tests[i]["wilcoxon_p"] * (len(tests) - rank)))
        tests[i]["holm_p"] = running


def confirmatory(df: pd.DataFrame) -> list[dict]:
    rng = np.random.default_rng(42)
    tests = [compare(df, a, b, lab, rng) for a, b, lab in TESTS]
    holm(tests)
    for t in tests:
        t["supported"] = bool(t["holm_p"] < 0.05 and not (t["patient_diff_CI95"][0] <= 0 <= t["patient_diff_CI95"][1]))
    return tests


# ------------------------------------------------------------------ descriptive (SAP §8 + Amendment 01)
def descriptive(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(42)  # separate generator (Amendment 01)
    pp = df.groupby("patient")
    n = df.patient.nunique()
    out = {"n_patients": int(n), "n_slices": int(len(df)), "methods": {}}
    for key, name in METHODS:
        m = {}
        for k in K:
            s = df[f"{key}_{k}"]
            per_pat = pp[f"{key}_{k}"].mean()
            idx = rng.integers(0, n, (B, n))
            ci = np.percentile(per_pat.values[idx].mean(1), [2.5, 97.5])
            m[k] = {"slice_mean": float(s.mean()), "slice_sd_ddof0": float(s.std(ddof=0)), "slice_median": float(s.median()),
                    "slice_iqr": [float(s.quantile(.25)), float(s.quantile(.75))], "slice_min": float(s.min()), "slice_max": float(s.max()),
                    "patient_mean": float(per_pat.mean()), "patient_median": float(per_pat.median()),
                    "patient_sd_ddof0": float(per_pat.std(ddof=0)), "patient_mean_CI95": [float(x) for x in ci]}
        out["methods"][name] = m
    return out


def exploratory(df: pd.DataFrame) -> dict:
    succ = df.baseline_dice >= 0.5
    diff = df.enhanced_dice - df.baseline_dice
    keep = df.a5_choice == "box"
    correct = np.where(diff.abs() <= 0.01, True, np.where(keep, diff < 0, diff > 0))
    fm = {}
    for key, name in METHODS:
        if key == "a5":
            continue
        px = df[f"{key}_precision"]; rc = df[f"{key}_recall"]; dc = df[f"{key}_dice"]
        fm[name] = {"dice_lt_0.1": int((dc < 0.1).sum()), "oversegmentation(prec<0.1&rec>0.9)": int(((px < 0.1) & (rc > 0.9)).sum()),
                    "undersegmentation(rec<0.1)": int((rc < 0.1).sum())}
    return {
        "baseline_success_split": {name: {"success_n": int(succ.sum()), "dice_on_success": float(df.loc[succ, f"{k}_dice"].mean()) if succ.any() else None,
                                          "failure_n": int((~succ).sum()), "dice_on_failure": float(df.loc[~succ, f"{k}_dice"].mean()) if (~succ).any() else None}
                                   for k, name in METHODS},
        "a5_selection": {"box_only": int(keep.sum()), "enhanced_activated": int((~keep).sum()), "correct": int(correct.sum()), "of": int(len(df)),
                         "missed_opportunity(kept box, enhanced better >0.01)": int((keep & (diff > 0.01)).sum()),
                         "harmful_activation(chose enhanced, baseline better >0.01)": int((~keep & (diff < -0.01)).sum()),
                         "oracle_dice": float(np.maximum(df.baseline_dice, df.enhanced_dice).mean())},
        "failure_modes": fm,
        "note": "exploratory; descriptive only; no confirmatory claims (SAP §9)",
    }


def analyse_cohort(df: pd.DataFrame) -> dict:
    df = df.sort_values("query_image").reset_index(drop=True)
    return {"confirmatory_family": confirmatory(df), "descriptive": descriptive(df), "exploratory": exploratory(df),
            "per_patient_dice": df.groupby("patient")[[f"{k}_dice" for k, _ in METHODS]].mean().round(6).reset_index().to_dict("records")}


# ------------------------------------------------------------------ commands
def validate_v3() -> int:
    df = pd.read_csv(P / "results/v3/raw_holdout_rows.csv").sort_values("query_image").reset_index(drop=True)
    ref = json.loads((P / "results/v3/adaptive/summary.json").read_text())["tests"]
    got = confirmatory(df)
    keys = ["slice_mean_diff", "slice_diff_CI95", "patient_mean_diff", "patient_diff_CI95", "patients_better",
            "patients_worse", "patients_tied(|d|<=0.005)", "wilcoxon_p", "holm_p"]
    bad = []
    for r, g in zip(ref, got):
        for k in keys:
            a, b = np.asarray(r[k], float), np.asarray(g[k], float)
            if not np.allclose(a, b, rtol=0, atol=1e-12):
                bad.append((r["comparison"], k, r[k], g[k]))
    ok_means = all(abs(json.loads((P / "results/v3/adaptive/summary.json").read_text())["mean_of_patient_means_dice"][k]
                       - df.groupby("patient")[f"{k}_dice"].mean().mean()) < 1e-12 for k in ("baseline", "enhanced", "a5", "single"))
    print(json.dumps({"v3_rows": len(df), "tests_compared": len(ref), "mismatches": bad, "patient_means_match": ok_means,
                      "REPRODUCED_EXACTLY": not bad and ok_means}, indent=1))
    return 0 if (not bad and ok_means) else 1


def run(out_dir: Path) -> int:
    if sha(SAP) != SAP_SHA or sha(AMEND) != AMEND_SHA:
        raise SystemExit("SAP or Amendment 01 hash mismatch — refusing to analyse")
    eval_dir = R / "eval"
    wl = json.loads((eval_dir / "worklist.json").read_text())["items"]
    subjects = sorted({w["subject"] for w in wl})
    done = {p.stem for p in (eval_dir / "subjects").glob("*.done")}
    missing = [s for s in subjects if s not in done]
    if missing:
        raise SystemExit(f"evaluation incomplete: {len(missing)} subjects without completion marker — refusing (SAP §3)")
    for f in ("raw_rows_primary.csv", "raw_rows_sensitivity.csv"):
        if not (eval_dir / f).exists():
            raise SystemExit(f"{f} missing — finalize the evaluation first")
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists — refusing to overwrite")
    prim = pd.read_csv(eval_dir / "raw_rows_primary.csv")
    sens = pd.read_csv(eval_dir / "raw_rows_sensitivity.csv")
    exp = {"primary": (826, 74), "sensitivity": (300, 90)}
    got = {"primary": (len(prim), prim.patient.nunique()), "sensitivity": (len(sens), sens.patient.nunique())}
    report = {"sap_sha256": SAP_SHA, "amendment_01_sha256": AMEND_SHA, "expected_n": exp, "actual_n": got,
              "n_matches_sap": exp == got, "primary": analyse_cohort(prim), "sensitivity": analyse_cohort(sens),
              "inputs_sha256": {f: sha(eval_dir / f) for f in ("raw_rows_primary.csv", "raw_rows_sensitivity.csv", "run_config.json", "worklist.json")}}
    out_dir.mkdir(parents=True)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("expected_n", "actual_n", "n_matches_sap")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate-v3")
    r = sub.add_parser("run")
    r.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    return validate_v3() if a.cmd == "validate-v3" else run(a.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
