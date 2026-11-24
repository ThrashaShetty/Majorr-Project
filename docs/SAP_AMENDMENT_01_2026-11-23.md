# Amendment 01 to the Statistical Analysis Plan (SAP) v1 — 2026-11-23

| Item | Value |
|---|---|
| Amends | `results/external_ts/analysis_plan/STATISTICAL_ANALYSIS_PLAN_v1.md` |
| Frozen plan SHA-256 (unchanged) | `6a1fee26869a2bc8abf2a2e9d587838f1dc4d8871c29a2f84d1b02e8e306285d` |
| Type | Clarification of an ambiguity. No change to hypotheses, cohorts, tests, correction, endpoints or interpretation rules |
| When | Written before the external cohort was analysed. As stated in SAP §0, only the s0965 pilot (16 slices) had been seen |
| Reason | Found by the read-only documentation audit on 2026-11-23 |

## 1. The ambiguity

SAP §7 says:

* the confirmatory comparisons use "a single `numpy.random.default_rng(42)` per cohort", in the order H1, H2,
  H3, H4, with a fresh 10,000 × n_patients index matrix per comparison;
* "Descriptive metric CIs: the same scheme (seed 42, 10,000 resamples) for each method's patient-level mean."

It doesn't say whether the descriptive confidence intervals:

* (a) keep drawing from the **same** generator after H4; or
* (b) use a **separate** generator.

It also doesn't fix the order in which methods and metrics draw resamples. Each choice gives different
(equally valid) bootstrap draws, so exact numeric reproduction would depend on an unstated choice.

## 2. Clarification (binding for the analysis)

1. **Confirmatory comparisons (unchanged).** One `numpy.random.default_rng(42)` per cohort, used **only** for
   H1 → H2 → H3 → H4, exactly as SAP §7 and `results/v3/scripts/v3_report.py` describe. No other draw uses
   this generator.
2. **Descriptive CIs (clarified).** A **separate** `numpy.random.default_rng(42)` is created per cohort,
   after the confirmatory comparisons. It is used in this fixed order:
   * methods: Baseline → Enhanced v2 → A5 → Single-mask;
   * within each method, metrics: Dice → IoU → Precision → Recall → Boundary IoU;
   * each (method, metric) pair draws a fresh 10,000 × n_patients index matrix (patients with replacement);
   * the CI is the 2.5th–97.5th percentile of the resampled mean of per-patient means.
3. **The primary and sensitivity cohorts each get their own pair of generators**, created in the same way.

## 3. Effect

* **Confirmatory results:** none. Hypothesis tests, p-values, Holm adjustment and the confirmatory CIs
  are exactly as they would be under the original wording.
* **Descriptive CIs:** now uniquely defined and reproducible.

## 4. Status of this amendment file

* This file lives in `docs/` because the amendment was made under a documentation-only scope. The
  frozen plan file and its `.sha256` sidecar were not modified.
* The analysis script must cite both the plan and this amendment.
* **Backup:** this file is not yet covered by `make_backup.py`, which backs up only
  `results/external_ts/`. See the backup plan in the audit report.
