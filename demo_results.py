"""One-command results demo (no SAM, no dataset download, no GPU needed; about 15 seconds).

    python demo_results.py

Uses only files that are in the GitHub repository. It
  1. re-runs the frozen statistics on the internal v3 held-out rows and checks they equal the published v3 numbers;
  2. re-runs the frozen external analysis (SAP v1 + Amendment 01) on the 90-subject TotalSegmentator raw rows and
     checks the result is identical to results/external_ts/analysis/phaseB_20261124T052029Z/analysis.json;
  3. prints the result tables.
Read-only: it writes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "results/external_ts/scripts"))
import pandas as pd  # noqa: E402

import analyze_external_ts as A  # noqa: E402

ANALYSIS = ROOT / "results/external_ts/analysis/phaseB_20261124T052029Z/analysis.json"
NAMES = {"baseline": "Baseline", "enhanced": "Enhanced v2", "a5": "A5", "single": "Single-mask"}


def line(c="-", n=78):
    print(c * n)


def show(title: str, cohort: dict):
    line("=")
    d = cohort["descriptive"]
    print(f"{title}: {d['n_patients']} patients, {d['n_slices']} slices")
    line()
    print(f"{'Method':<14}{'Per-patient mean Dice':>22}{'95% CI':>20}{'Precision':>11}{'Recall':>9}")
    for name in NAMES.values():
        m = d["methods"][name]
        lo, hi = m["dice"]["patient_mean_CI95"]
        print(f"{name:<14}{m['dice']['patient_mean']:>22.3f}   [{lo:.3f}, {hi:.3f}]"
              f"{m['precision']['slice_mean']:>11.3f}{m['recall']['slice_mean']:>9.3f}")
    line()
    print(f"{'Pre-registered test':<38}{'Delta Dice':>11}{'95% CI':>20}{'Holm p':>9}")
    for t in cohort["confirmatory_family"]:
        lo, hi = t["patient_diff_CI95"]
        print(f"{t['comparison'][:37]:<38}{t['patient_mean_diff']:>+11.3f}   [{lo:+.3f}, {hi:+.3f}]{t['holm_p']:>9.2g}")


def main() -> int:
    print("Background-Aware Prompting with SAM - results demo (no SAM needed)")
    line("=")
    print("Step 1: reproduce the internal v3 held-out statistics (33 patients) ...")
    v3_ok = A.validate_v3() == 0
    print("  v3 statistics reproduced exactly:", "YES" if v3_ok else "NO")

    print("Step 2: re-run the frozen external analysis from the raw per-slice rows ...")
    rep = json.loads(ANALYSIS.read_text())
    same = {}
    for cohort, f in (("primary", "raw_rows_primary.csv"), ("sensitivity", "raw_rows_sensitivity.csv")):
        got = json.loads(json.dumps(A.analyse_cohort(pd.read_csv(ROOT / "results/external_ts/eval" / f))))
        same[cohort] = got == rep[cohort]
        print(f"  {cohort:<11} identical to the saved analysis.json:", "YES" if same[cohort] else "NO")

    show("EXTERNAL PRIMARY COHORT (TotalSegmentator, Basel)", rep["primary"])
    show("EXTERNAL SENSITIVITY COHORT", rep["sensitivity"])
    line("=")
    print("Reading: H1 (A5 vs Baseline) is the pre-registered primary test. A 95% CI that contains 0 means no proven")
    print("difference. Enhanced v2 is significantly WORSE than Baseline (negative Delta, tiny Holm p).")
    ok = v3_ok and all(same.values())
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
