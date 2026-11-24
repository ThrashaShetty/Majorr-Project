# External evaluation — progress log

Append-only. Each entry records measured values only.

## 2026-11-23 03:28 (local, UTC+05:30)

**Phase A — audit (read-only): PASS**

* Unchanged: the 1,488 original files, the Phase 1 baseline, the frozen SAP (`6a1fee26…`), the checkpoint
  (`ec2df627…`), `run_config.json` (`3933ca8c…`), `worklist.json` (`726bb4ee…`) and `cohort_manifest.json`
  (`b6de738e…`).
* Subject accounting. The source of truth is `eval/subjects/*.done` plus `eval/slices/` against the worklist:

| Category | Count |
|---|---|
| Worklist subjects | 90 |
| Worklist slices | 1,078 |
| Completed | 1 (s0965, 16 slices; independently verified in `audit/s0965_verification.json`) |
| Failed | 0 |
| Partial | 0 |
| **Pending** | **89 subjects / 1,062 slices** |
| Duplicate or unknown outputs | none |

* Run history: the last event is `run_end` (2026-11-22T19:57:44Z). There is one earlier `RESOURCE_STOP` (the s0965 pilot,
  later resumed and completed).
* Git: 1 commit (`8828775`). Untracked: `docs/`, `rem.txt`, `results/external_ts/`.

**Phase B — readiness: BLOCKED**

* AC power: **offline**, battery 51 %.
* USB: none connected. OneDrive copy: sync error (state 7) and OneDrive has been quit. No independent
  backup exists. The same-disk staging backup (`backups/staging_20261122T210831Z`, 5,306,697 B) is verified 6/6.
* RAM available (5 readings over 25 s): 0.903–0.925 GiB, against the 1.6 GiB start guard. Committed 15.99 / 20.31 GiB.

**Phase C — RAM optimisation: exhausted without forced termination.** Background apps were stopped earlier
with approval; the freed memory was reabsorbed by Antigravity (about 1 GB) and the AI assistant (about 0.76 GB),
both protected.

**Phase D — lower start limit: proposed only, awaiting approval.** No launcher change has been made.

**Evaluation: NOT started.**

## 2026-11-23 03:35 — independent work while blocked

* New: `results/external_ts/scripts/analyze_external_ts.py`, which implements SAP v1 and Amendment 01.
  * `validate-v3` reproduces **every** published v3 statistic exactly from `results/v3/raw_holdout_rows.csv`:
    4 tests, CIs, Wilcoxon and Holm p-values, and better/worse/tied counts (tolerance 1e-12).
  * `run` refuses while the evaluation is incomplete (89 subjects without a completion marker).
* New: `scripts/tests/test_analysis.py` (7 tests). Full external_ts suite: **85 passed**.
* No external results were analysed. Evaluation not started.

## 2026-11-23 — Phase D implemented (user-approved)

* `scripts/run_with_mem_opt.py`:
  * new `--start-ram-gib` option (default 1.6) and `--resource-constrained-deviation` flag;
  * a value of 0.8 or more and below 1.6 is accepted only with the flag; higher values or a missing flag are refused before any work;
  * the stop limit (0.3 GiB) and disk limit (2.0 GiB) are fixed;
  * every run-history record is tagged with `start_ram_gib` and `resource_constrained_deviation`, without editing the frozen runner.
* `scripts/run_batches.py`: the same option and flag are passed through, validated first, and tagged in every driver record.
* New `scripts/tests/test_start_ram_override.py` (22 tests). Full external_ts suite: **107 passed**.
* The frozen run config's code hashes (`eval_external_ts.py`, `ts_common.py`, `ts_resources.py`) are unaffected. The SAP is unchanged.
* Pre-flight with `--start-ram-gib 1.0 --resource-constrained-deviation`: all integrity checks pass (89 subjects / 1,062 slices).
  **Guard FAIL: 0.969 GiB available < 1.0 GiB.**
* AC power offline, battery 46 %. No USB drive. Backup authorisation unresolved. **Evaluation not started.**

## 2026-11-23 — new session: state check and Phase C.1 (exploratory)

* State check: the 1,488 protected files, the frozen run config (`3933ca8c…`), the worklist (`726bb4ee…`), the SAP
  (`6a1fee26…`) and the recorded code hashes all verify. Subjects: 1/90 done (s0965), 89 pending.
* Live: AC power **Online** (57 %); RAM available **0.747 GiB** (< 1.0 GiB approved start limit); no USB;
  C: 14.85 GiB free. **Phase A not started** (awaiting the user's confirmation and backup decision; RAM also short).
* Phase C.1, exploratory and NOT part of the frozen analysis:
  * new `src/extensions/__init__.py` and `src/extensions/uncertainty.py` (multimask entropy and disagreement;
    seeded K-perturbation variance; per-slice scores over a dilated ROI; inferno heatmap);
  * new `tests/test_ext_uncertainty.py` (8 tests, synthetic).
  * Project test suite: **66 passed**. NOTE: this run did not deselect the 2 real-checkpoint tests, so SAM
    was loaded once by pytest (both passed; no project file changed).
* Not yet done for C.1: the v3 correlation study (Spearman of score vs Dice). It needs SAM on 156 v3 slices
  and RAM headroom.

## 2026-11-23 — user decisions; RUN_PHASE_A.md; finalize gap fixed; Phase C.2 and C.3 (exploratory)

* User decisions recorded:
  * the charger stays connected;
  * same-disk backup risk is accepted for the batch run, and the user uploads backups off-disk;
  * the user runs the batches themselves with Antigravity and the AI assistant closed;
  * the checkpoint tests stay excluded unless the user approves them.
* **Gap found and fixed:** batch runs pass `--subjects`, so the frozen runner never calls `finalize()`, and
  `raw_rows_primary.csv` / `raw_rows_sensitivity.csv` would never be written (Phase B would refuse to run).
  * New `scripts/finalize_eval.py` calls the frozen `finalize()` unchanged. It refuses unless all 90 `.done`
    markers exist.
  * New `scripts/tests/test_finalize_eval.py` covers it. Live check: refused with exit 2 (89 of 90 incomplete);
    nothing was written.
* New `RUN_PHASE_A.md`: exact PowerShell steps covering set-up, pre-flight at 1.6 and 1.0 GiB, dry run, run (normal
  and fallback), exit codes, resume, then finalize, verification over all 90 subjects and a backup.
  * Its read-only commands were run once as written. Pre-flight at 1.6 GiB: config `3933ca8c4837`, 89 subjects,
    1,062 slices, **guard FAIL (1.050 GiB available)** with the AI assistant open. Dry run: 9 batches, AC true.
* Phase C.2 (exploratory):
  * new `src/extensions/refinement.py` (mask_input plus click refinement; the simulated user clicks the largest
    false-negative or false-positive component at its interior point farthest from the edge; deterministic);
  * new `tests/test_ext_refinement.py` (8 tests).
  * New `experiments/exploratory/v3_uncertainty_and_clicks.py`: the C.1 Spearman study and the C.2 Dice vs 0–5
    clicks, for Baseline and A5 on the 156 v3 slices. `--check` passes (156 slices, 33 patients).
    **Not run (HELD until RAM allows).**
* Phase C.3 (exploratory, new files only):
  * `backend/extensions/{__init__,service,app,serve}.py`: the original API is reused unchanged, plus `/ext/health`,
    `/ext/segment` (methods Baseline, Enhanced v2, A5, Single-mask; prompt overlay; uncertainty heatmaps and
    scores; metrics), `/ext/refine` and `/ext/reset`. Data goes to `app_data/`.
  * New `frontend/dashboard.html` (`index.html` unchanged).
  * New `tests/test_ext_api.py` (13 tests, fakes).
* Tests: project **85 passed, 2 deselected** (checkpoint tests); external_ts scripts **108 passed**.
* Integrity after this phase:
  * the 1,488 protected files verify;
  * the run config, worklist and SAP hashes are unchanged;
  * all 16 `run_config` code hashes and the checkpoint hash match;
  * the Phase 1 list differs only in `eval/logs/resource_checks.jsonl` and `run_history.jsonl`, append-only logs
    last written 2026-11-22 19:57 UTC by the approved s0965 run. Nothing today touched them.

## 2026-11-23 — state check before Phase B: Phase A NOT run

* The user's message reported Phase A finished, but its exit-code fields were left blank (`__`).
* On-disk state:
  * 1 of 90 `.done` markers (s0965 only);
  * no `batches/` plan;
  * no `eval/raw_rows*.csv`;
  * no batch or final verification reports;
  * only `run_logs/preflight_1.6.json`, from my own read-only check;
  * `run_history.jsonl` last event is `run_end` at 2026-11-22T19:57:44Z (s0965).
* Phase B not started; it would refuse without all 90 subjects. Nothing modified.

## 2026-11-23 — Phase C.4 (exploratory): experiment history, compare, CSV/HTML/PDF export

* New `backend/extensions/history.py`: the `ext_runs` table in `<data>/history_ext.db`, separate from the original
  `experiments` table and `experiments.db`.
  * One row per `/ext/segment` run: method, method_used, A5 ratio, metrics, uncertainty scores, prompts, overlay.
  * Refine and reset update `n_clicks`, clicks and the `refined_*` metrics. No delete route.
* New `backend/extensions/report.py`:
  * CSV, with spreadsheet-formula neutralisation;
  * self-contained HTML, with escaped text, embedded overlays and image paths confined to the data dir;
  * PDF via matplotlib: summary table, initial vs after-clicks Dice chart, overlay pages.
  * Every export carries the exploratory, not-cohort-results note.
* `backend/extensions/app.py` and `frontend/dashboard.html` (both new this session; no original file touched):
  * new routes `/ext/history`, `/ext/history/compare`, `/ext/history/export.csv`, `/ext/history/report.html`,
    `/ext/history/report.pdf` and `/ext/history/{id}`;
  * a History panel with filter, compare (deltas vs the first selected run) and export buttons.
* New `tests/test_ext_history.py` (8 tests, fake SAM). Dashboard JavaScript passes `node --check`.
* Tests: project **93 passed, 2 deselected** (checkpoint tests); external_ts scripts **108 passed**.
* Integrity: the 1,488 protected files OK; the run config, worklist and SAP unchanged; all 16 code hashes match.
  Still 1/90 `.done`. SAM not loaded in this step.
* Status: Phase A still to be run by the user (`RUN_PHASE_A.md`).

## 2026-11-23 — Phase D docs (no SAM, no downloads; docs only)

* **New in `docs/`:**
  * `EXPLORATORY_FEATURES.md`: C.1–C.4 design, run guide, limitations;
  * `ARCHITECTURE_DIAGRAM.md`: a Mermaid diagram of the full pipeline;
  * `OBJECTIVES_STATUS.md`: 10 objectives;
  * `FINAL_REPORT_DRAFT.md`: all chapters;
  * `SLIDES_OUTLINE.md`: 15 slides;
  * `VIVA_QA.md`: 30 questions and answers.
* **Updated in `docs/`:**
  * `API_DOCUMENTATION.md`: every `/ext` endpoint, and the stale "not exposed" note fixed;
  * `SYSTEM_ARCHITECTURE.md`, `INSTALLATION_GUIDE.md` (dashboard run guide, test counts), `LIMITATIONS.md`,
    `FUTURE_WORK.md`, `PROJECT_OVERVIEW.md`, `EXPERIMENTAL_SETUP.md`, `README.md` (index);
  * `REFERENCES.md`: an open-items list. The FoB citation is unverified; the Efron edition, MaierHein2024
    and Holm1979 are flagged;
  * `REPORT_OUTLINE.md` and `PRESENTATION_OUTLINE.md` now point to the drafts.
* **Placeholders:** all external and exploratory results are written as
  `[PENDING: Phase A/B | v3 exploratory run | dashboard test | final audit]`. No s0965 pilot numbers are
  used as results.
* **Objectives:** the consent form is not in the project folder. `OBJECTIVES_STATUS.md` uses 10
  objectives reconstructed from the title, the README and the Phase C brief, with a banner asking the
  team to replace them with the form's exact wording.
* **Checks:**
  * every figure path in the slides and every `.md` link exists;
  * the 1,488 protected files OK; the run config, worklist and SAP unchanged; 16 code hashes match;
  * SAP Amendment 01 unchanged (`18be6f47…`);
  * still 1/90 `.done`.
* **Not changed:** no SAM load, no download, no git operation.

## 2026-11-23 — consent-form objectives (docs only)

* `docs/OBJECTIVES_STATUS.md`:
  * rewritten with the exact consent-form wording of the 10 objectives, supplied by the user;
  * the "reconstructed wording" banner removed;
  * the status mapping kept and re-aligned to the new numbering (Done: 1, 2, 9; Partial: 3–8, 10; MRI not done).
* New `docs/CONSENT_FORM_SUMMARY.md`: the title, guide, 4 students with USNs, and the 10 objectives. Abstract,
  scope and expected outcomes are left as `[TO ADD]`, because their text was not supplied.
* The objectives are quoted verbatim in `FINAL_REPORT_DRAFT.md` §2 (team names filled in) and in shortened form
  on slide 3 of `SLIDES_OUTLINE.md` (team filled in). `README.md`, `REPORT_OUTLINE.md` and
  `PRESENTATION_OUTLINE.md` point to them.
* Checks: verbatim match confirmed by script; no broken doc links. No code, data or frozen files touched.

## 2026-11-23 — consent-form abstract, scope and outcomes (docs only)

* `docs/CONSENT_FORM_SUMMARY.md`: the abstract, scope and expected outcomes (1–5) were filled in with the exact
  text supplied by the user. No `[TO ADD]` remains. The docs index entry was updated. No other file was changed.

## 2026-11-23 — BCM naming (docs only)

* "Background Context Modeling" was replaced by "Background-centric Context Modeling" in every `docs/*.md` file, to
  match the consent form. The expansion was also added at the first unexpanded mention in `FINAL_REPORT_DRAFT.md`,
  `VIVA_QA.md`, `SLIDES_OUTLINE.md` and the Mermaid diagram.
* NOT changed: the project `README.md` (line 95, `### 7. BCM: Background Context Modeling`). It is one of the 1,488
  protected original files. No code changed.
* The 1,488 protected files verify.

## 2026-11-23 — deferred, user-approved change to a protected file (NOT yet done)

* The user approved ONE change to the protected `medical_sam_project/README.md`, to be made **after Phase B is
  complete and verified**. Line 95, currently `### 7. BCM: Background Context Modeling`, becomes
  `### 7. BCM: Background-centric Context Modeling`. The `### 7.` heading prefix is kept.
* When it is done:
  * record the old and new SHA-256 here;
  * re-baseline only that entry in the protected-hash snapshot;
  * confirm that the other 1,487 files are unchanged.

## 2026-11-23 04:10 UTC — Phase A wrapper written; NOT launched (laptop on battery)

* **On-disk state:** 1 of 90 `.done` (s0965); no `batches/`; no batch or final verification reports; no `raw_rows*.csv`;
  no batch process running. This is Section A.
* **New `scripts/launch_phase_a.ps1`:**
  * waits 180 s, then checks AC power (on battery → exit 12);
  * pre-flight at 1.6 GiB, then at 1.0 GiB with the deviation flags; both failing the RAM check → exit 10;
    an integrity error → exit 11;
  * runs `run_batches --execute --i-have-explicit-approval --backup-dest backups\external_ts`, adding the
    deviation flags only if the 1.6 GiB check failed, and passes its exit code through;
  * writes a timestamped log, per-step `.out`/`.err` files and `phaseA_exitcode_<stamp>.txt`;
  * test switches `-SkipWait`, `-DryRun` and `-TestIgnoreAC`. The last is refused unless `-DryRun` is set, and
    dry-run files are prefixed `TEST_`. Parses cleanly.
* **Tests run here (Antigravity open; nothing evaluated, no SAM loaded):**
  * `phaseA_20261123T040608Z` (real path, no switches except `-SkipWait`): **exit 12**, AC not connected.
    Correct refusal.
  * `-TestIgnoreAC` without `-DryRun`: refused, exit 1.
  * `phaseA_20261123T040610Z` (dry run, written before the `TEST_` prefix was added): both pre-flights passed
    integrity (config `3933ca8c4837`, 89 subjects, 1,062 slices) but failed the RAM check (0.628 GiB < 1.6;
    0.905 GiB < 1.0) → **exit 10**. These two stamps are TEST runs, not Phase A runs.
* **Measured:** AC NOT connected (`Win32_Battery` status 1 = discharging; `PowerOnline` False; 69% → 66%);
  C: free 9.62 GiB; RAM available 0.63–1.17 GiB with Antigravity open.
* **Not launched**, because the laptop is on battery. Waiting for the charger to register.

## 2026-11-23 04:09:50 UTC — Phase A LAUNCHED (detached)

* **Pre-launch checks:** AC confirmed on (`Win32_Battery` status 2, `PowerOnline` True, `Charging` True, psutil
  True; 67%). C: free 9.51 GiB. RAM available 0.625 GiB with Antigravity open; the wrapper checks RAM itself
  after its 180 s wait. No other batch process was running.
* **Launched** `scripts/launch_phase_a.ps1` with no switches (a real run), through WMI `Win32_Process.Create`,
  so it is detached from this session. Parent `WmiPrvSE.exe`; window minimized.
  **PID 2648.** Log: `run_logs/phaseA_20261123T040950Z.log`. Exit code will be written to
  `run_logs/phaseA_exitcode_20261123T040950Z.txt`.
* **Sequence:** wait 180 s; AC check; pre-flight at 1.6 GiB (falling back to 1.0 GiB with the deviation);
  `run_batches --execute`. About 3.9 h of compute is estimated if the RAM check passes.
* **To resume after a stop,** re-launch the same wrapper: completed slices are re-verified by hash and skipped.
* **To stop it early:** `Stop-Process -Id <python PID>`. Results committed so far are kept, because they are
  write-once.

## 2026-11-24 — resume check; NOT re-launched (laptop on battery)

* **On-disk state:** 9 of 90 `.done`; no `raw_rows*.csv`; no `run_batches` or launcher process running. The latest
  exit-code file is `run_logs/phaseA_exitcode_20261123T040950Z.txt`, containing `3` (`run_batches: resource_stop`).
* **The real stop cause was DISK, not RAM.** `phaseA_20261123T040950Z_run_batches.err` reads
  `RESOURCE STOP: free disk 0.526 GB < limit 2.000 GB`, at 04:35:30Z during batch 1. C: had 11.46 GiB free at the
  start; 10.73 GiB is free now. The likely cause is Windows growing `pagefile.sys` (now 13,415 MB allocated) under
  memory pressure. This is not confirmed.
* **Measured now:** AC NOT connected (`Win32_Battery` status 1, `PowerOnline` False, 16%). Free RAM 0.878 GiB of
  7.34 GiB.
* **Wrapper hash:** `launch_phase_a.ps1` SHA-256 is `813938e11a879399e70fa8664b0efa6fe561f7f104304da23b482bb3d563cce4`.
  No earlier hash was recorded in this log, so it could not be compared. It is recorded here as the reference.
* **Not launched,** because the laptop is on battery (Section A step 1). Nothing else was changed.

## 2026-11-23 20:16:39 UTC — Phase A RE-LAUNCHED (detached)

* **Pre-launch checks:** AC on (`Win32_Battery` status 2, `PowerOnline` True, `Charging` True; 16%). Free RAM 0.883 GiB.
  C: free 10.81 GiB. The `launch_phase_a.ps1` SHA-256 matches the reference `813938e1…cce4`. No batch process was
  running. 9 of 90 subjects were done.
* **Launched** the wrapper with no switches through WMI `Win32_Process.Create`, minimized and detached.
  **PID 17276.** Log: `run_logs/phaseA_20261123T201638Z.log`. Exit code will be written to
  `run_logs/phaseA_exitcode_20261123T201638Z.txt`. Completed subjects are re-verified and skipped.
* **Watch:** the previous stop was the 2 GiB free-disk guard (probably pagefile growth). If it is exit 3 again, check
  the `.err` file for disk or RAM.

## 2026-11-24 — Phase A COMPLETE (Section C, step 1)

* **Runs not logged earlier:** `phaseA_20261123T201638Z` ended **exit 3** at 21:15:57Z (`RESOURCE STOP: available
  RAM 1.537 GB < limit 1.600 GB`; 81 → 59 subjects remaining). A further launch, `phaseA_20261124T022744Z`
  (launcher PID 5356), followed.
* **`phaseA_20261124T022744Z`:** pre-flight 1.6 failed the RAM guard (1.479 GB), and pre-flight 1.0 passed, so it ran
  with the **1.0 GiB resource-constrained deviation** (`--start-ram-gib 1.0 --resource-constrained-deviation`,
  recorded in every run-history entry). `run_batches` ran 02:31:26Z–04:59:44Z and ended `driver_complete`, last at
  batch 9. Its `.err` is empty. `phaseA_exitcode_20261124T022744Z.txt` = **0 (run_batches: ok)**.
* **On disk:** 90 of 90 worklist subjects have `.done` (the worklist has 90 subjects and 1,078 items). No
  `raw_rows_*.csv` exist yet; `finalize_eval.py` produces them.

## 2026-11-24 — finalize_eval.py (Section C, step 2): exit 0

* 90 subjects; config `3933ca8c4837…`. Written: `eval/raw_rows.csv` `58ce8cc2…6bd1`, `eval/raw_rows_primary.csv`
  `9bec7a4a…ff87`, `eval/raw_rows_sensitivity.csv` `0de084af…2c72`.

## 2026-11-24 — verify_eval_subjects.py, all 90 subjects (Section C, step 3): exit 0

* Report `audit/final_verification_20261124_050936.json` (SHA-256 `71bf3a9a…83be`): `all_passed` True, `errors` [],
  `max_abs_metric_deviation` 0.0, no other subjects with outputs, config `3933ca8c4837…`, 90 subjects.

## 2026-11-24 — post-run backup (Section C, step 4): exit 0, verified

* `backups/external_ts/external_ts_tier1_20261124T051044Z.zip`: 10,169,158 bytes, SHA-256
  `59b9f27170a2fd08ddd1ec27c3856d6d3cda0bd00cefb99568721f878c14034e`, 2,433 files, all 90 subjects done.
* `make_backup.py --verify-only`: **exit 0**. `crc_ok` True, `members_equal_manifest` True, no member hash
  mismatches; the `.zip.sha256` sidecar matches. The zip contains `raw_rows.csv`, `raw_rows_primary.csv`,
  `raw_rows_sensitivity.csv` and `audit/final_verification_20261124_050936.json`.
* **To do (user):** upload this zip plus its `.sha256` and `.manifest.sha256` to Google Drive. C: has 9.4 GiB free.
* Phase A is done. Waiting for "go" before Phase B. No git commit made.

## 2026-11-24 — Phase B: frozen analysis (SAP v1 + Amendment 01) — exit 0

* Hashes: SAP `6a1fee26…` and Amendment 01 `18be6f47…` match. `analyze_external_ts.py` is `a9569f27…97a5`.
* `validate-v3`: **exit 0**. `REPRODUCED_EXACTLY` True (156 rows, 4 tests, no mismatches).
* `run --out-dir results/external_ts/analysis/phaseB_20261124T052029Z`: **exit 0**. N matches the SAP: primary 826
  slices / 74 patients, sensitivity 300 / 90. `analysis.json` SHA-256 `762d22a7…e0b4`.
* **Primary (Holm, 4 tests). Patient-mean Dice difference [95% bootstrap CI], Holm p:**
  * H1 A5 vs Baseline: −0.0019 [−0.0210, +0.0165], p 1.0. **Not supported** (23 better / 25 worse / 26 tied).
  * H2 A5 vs Enhanced v2: +0.1613 [+0.1259, +0.1963], p 6.4e-10. Supported.
  * H3 Enhanced v2 vs Baseline: −0.1632 [−0.2069, −0.1195], p 4.3e-8. Significant **in the harmful direction**
    (Enhanced v2 is worse). The script's `supported` flag is two-sided, so its `True` here does not mean improvement.
  * H4 Single-mask vs Enhanced v2: −0.0005 [−0.0170, +0.0163], p 1.0. No difference.
  * Patient-mean Dice: Baseline 0.452, A5 0.450, Enhanced v2 0.289, Single-mask 0.288.
* **Sensitivity (separate Holm family):** the same pattern. H1 +0.0101 [−0.0105, +0.0320], Holm p 0.45, not
  supported. H2 +0.193, supported. H3 −0.183, significant and harmful. H4 −0.001, p 0.89.
* Exploratory (primary): A5 activated Enhanced on 188 of 826 slices. It picked the better method on 619 of 826;
  153 were missed opportunities and 54 were harmful activations. Oracle Dice 0.530.
* Caveat for the write-up: Phase A used the 1.0 GiB resource-constrained start-RAM deviation. This affects
  scheduling only, not the computation.
* **Post-Phase-B backup:** `backups/external_ts/external_ts_tier1_20261124T052059Z.zip`, SHA-256 `37aa85dd…c0c9`.
  `--verify-only` exit 0. It contains `analysis/phaseB_20261124T052029Z/analysis.json`. It supersedes `…051044Z.zip`
  as the copy to upload.
* **Next, pending the user's go:** the approved README line-95 edit (BCM wording), which is allowed now that Phase B
  is verified. No git commit has been made.

## 2026-11-24 — approved README edit DONE (the deferred item from 2026-11-23)

* Before the edit, all 1,488 protected files verified against `protect_before.sha256` (exit 0).
* `medical_sam_project/README.md` line 95: `### 7. BCM: Background Context Modeling` →
  `### 7. BCM: Background-centric Context Modeling`. That is the only change. The file's CRLF line endings are kept,
  and it grew from 23,061 to 23,069 bytes.
  * Old SHA-256: `a4aa89edc4e65bcb1bf9007afb1113f998a76280e1ccb34d65f1ec7a75660b5e`
  * New SHA-256: `80ba4f48d0a662d835ee3a550de5b82292f4e8666558771271cd045c94d3464d`
* Note: a first attempt with `sed -i` changed the whole file's line endings. It was restored at once from a byte copy
  (the hash matched the original `a4aa89ed…`), then redone as an exact byte replacement.
* Re-baselined snapshot: `protect_after_readme.sha256` in this session's scratchpad (SHA-256 `5ced418c…0b97`). It
  differs from `protect_before.sha256` only on the `./README.md` line.
* Afterwards: all 1,488 files verify against the new snapshot (exit 0). Against the old snapshot, only `./README.md`
  fails, so the other 1,487 files are unchanged.
* No git commit has been made.

## 2026-11-24 — protected snapshot persisted; docs updated with Phase B results (user approved both)

* The re-baselined protected-hash snapshot was copied to
  `results/external_ts/audit/protected_snapshot_after_readme_20261124.sha256` (SHA-256 `5ced418c…0b97`, 1,488 entries).
* `docs/` updated from `analysis/phaseB_20261124T052029Z/analysis.json`. Every Phase A / Phase B `[PENDING]` marker was
  replaced with measured numbers. Edited files: FINAL_REPORT_DRAFT (abstract, §7, full §8.3 tables, RQ4, conclusion,
  future work, appendix E), VIVA_QA (Q17, 27, 29, 30), SLIDES_OUTLINE (RQ4, slides 12, 13, 15), OBJECTIVES_STATUS
  (objectives 2, 3, 7, 8, 10 and the summary; 7 and 8 are now Done), EXPERIMENTAL_SETUP, FUTURE_WORK, LIMITATIONS
  (also records the 1.0 GiB run deviation), PROBLEM_STATEMENT, PROJECT_OVERVIEW, README (docs index),
  REPORT_OUTLINE, PRESENTATION_OUTLINE, REPRODUCIBILITY_GUIDE, ARCHITECTURE_DIAGRAM.
* H3 is reported as "significantly **worse**" everywhere. The analysis script's `supported` flag is not used as a
  claim of improvement.
* Still pending, not part of this update: the v3 exploratory run, the real-SAM dashboard test, the Phase B figures,
  and citation checks.
* Line endings were kept in every file, with no mixed files. All 1,488 protected files verify (exit 0); `docs/` is not
  protected. `docs/` is outside `make_backup.py`'s scope, so these edits are not in the backup zips. No git commit.

## 2026-11-24 — docs/ backup

* `backups/docs/docs_20261124T053606Z.zip`: 23 files, 67,926 bytes, SHA-256
  `ddf48e7721a9f75dfd9636d4bb92bbe8702c8703131bcfea672b675214484b38`. Sidecars: `.zip.sha256` and `.zip.manifest.sha256`.
* Verified: the zip CRC test passes; every member hash equals the manifest; the live `docs/` matches the manifest; and
  `sha256sum -c` on the sidecar is OK. Created with exclusive-create, so no earlier file was overwritten.
* To do (user): upload it to Google Drive with `external_ts_tier1_20261124T052059Z.zip`.
* Note: both sidecars were first written with CRLF, which `sha256sum -c` could not read. They were rewritten with LF; then `sha256sum -c` on the zip sidecar passed and the manifest matched the live `docs/`.

## 2026-11-24 — Phase B figures

* New script `results/external_ts/scripts/make_phaseB_figures.py` (SHA-256 prefix `b8313c7fc11f`). It reads only
  `raw_rows_primary.csv`, `raw_rows_sensitivity.csv` and `analysis.json`. It refuses if the raw rows changed since the
  analysis, and it writes once to `results/external_ts/figures/phaseB_20261124T052029Z/`.
* The per-patient means are recomputed from the raw rows and match `analysis.json`'s `patient_mean_diff` within 1e-9
  for all 8 tests (otherwise the script stops). CIs and Holm p are read from `analysis.json`; there is no resampling.
* Outputs (SHA-256 prefixes): `fig1_external_primary_dice_box.png` `b86ab22bce83`,
  `fig2_external_sensitivity_dice_box.png` `b9962c2d721a`, `fig3_external_primary_paired_diff.png` `c22d28cc80bd`,
  `fig4_external_sensitivity_paired_diff.png` `a29e65bc4230`, `figures_manifest.json` `fa3d5a7131d7`.
* The first render had overlapping x-labels and annotations covering points in the paired plots. I deleted that output
  directory (only my own files, created minutes earlier), fixed the layout and regenerated. All 4 PNGs were inspected
  visually.
* `docs/FINAL_REPORT_DRAFT.md` §8.3: the figure placeholder was replaced with tags [Figure 1–4: <filename>]. CRLF kept.
* RAM after the figures: 0.93 GiB free (psutil 0.981 GiB available) of 7.34 GiB; commit 17.91 / 20.01 GiB. That is
  below both the 1.6 GiB guard and the 1.0 GiB deviation floor. AC is on (status 2, 100%). C: has 9.4 GiB free.
* The exploratory run was NOT started and SAM was NOT loaded, as instructed.

## 2026-11-24 — citation checks (findings only; docs NOT changed)

* **FoB:** found. Y. Bo, Y. Zhu, P. Koniusz, H. Zhang. "Focus on Background: Exploring SAM's Potential in Few-shot
  Medical Image Segmentation with Background-centric Prompting." CVPR 2026, pp. 30032–30041 (CVF open-access
  BibTeX); arXiv:2603.21287; code github.com/primebo1/FoB_SAM. In docs it is still "pending / not listed" →
  a reference needs adding. Note: the paper's FoB is a *trained* prompt generator; its abstract describes support
  background-prompt generation, context modelling and a structural constraint for refinement, which correspond to
  BPPC/BCM/SPR. Docs currently call these modules "this project's own implementation", so the wording must say
  "a training-free re-implementation inspired by FoB [Bo2026]".
* **Efron1993:** ISBN 0-412-04231-2 / 978-0-412-04231-7 is the 1993 Chapman & Hall (New York) first edition,
  Monographs on Statistics and Applied Probability 57. Crossref's Springer US record (doi:10.1007/978-1-4899-4541-9)
  carries that same ISBN; the Chapman & Hall/CRC record (doi:10.1201/9780429246593, 1994-05-15) is a reissue. Only one
  edition was found. The docs say the earlier "Chapman & Hall/CRC, 1993" matched neither record and ask the team to
  choose; recommendation: cite Chapman & Hall, 1993.
* **MaierHein2024:** docs/ contains **no DOI**. The correct DOI 10.1038/s41592-023-02151-z resolves (doi.org 302 →
  nature.com). Crossref: Nature Methods 21(2), 195–212, Feb 2024, 73 authors, first author Maier-Hein. The docs
  details match, but the issue (2) and DOI are missing and the entry is wrongly marked UNVERIFIED.
* **Holm1979:** matches docs exactly: Scand. J. Statist. 6(2), 65–70, 1979; JSTOR 4615733. No DOI in Crossref (a
  bibliographic search returns only later papers). The entry is wrongly marked UNVERIFIED.

## 2026-11-24 — C6 final audit (read-only; no SAM loaded)

* Report: `audit/final_audit_20261124T054443Z.json`, created with exclusive create. **All 19 checks pass**, exit 0:
  * the 1,488 protected files match the durable snapshot;
  * run_config `3933ca8c…`; the 16 code hashes; the worklist, manifest, full-QC JSON and SAM checkpoint hashes;
  * the SAP and Amendment 01 hashes;
  * 90 of 90 `.done`;
  * the final verification passed with max deviation 0.0; 9 of 9 batch verifications passed;
  * run history has 272 entries, 180 tagged with the 1.0 GiB deviation;
  * `analysis.json` `762d22a7…`, its inputs unchanged, N matches the SAP; the 4 figures are unchanged;
  * both backup zips pass their CRC test and match their `.sha256` sidecars.
* **Tests:** project `tests/` 93 passed (2 real-SAM tests deselected). `results/external_ts/scripts/tests` 107 passed and
  **1 failed**: `test_analysis.py::test_run_refuses_when_incomplete`. The test assumes the evaluation is still
  incomplete. Now that all 90 subjects are done, `run()` succeeds (into pytest's tmp dir only; nothing was written to
  the project). This is an expected consequence of the project's state, not a defect in the analysis. The test was
  NOT changed and is waiting for a user decision.
* **Stale backups:** the docs zip predates the figure tags in `FINAL_REPORT_DRAFT.md`. The results zip `…052059Z`
  predates `figures/` and this audit report.
* Remaining `[PENDING]` markers in docs (36 in 11 files): v3 exploratory run, dashboard test, the slide figure,
  citations. Citation fixes have been found but not applied (see the citation check above).

## 2026-11-24 — citation fixes, test fix, refreshed backups (user approved all three)

* **Citations** (docs only; line endings kept):
  * `REFERENCES.md`:
    * added **[Bo2026]** FoB, CVPR 2026, pp. 30032–30041, arXiv:2603.21287, VERIFIED, with the note "training-free
      re-implementation inspired by FoB (Bo et al., CVPR 2026)";
    * **[Efron1993]** → Chapman & Hall, New York, 1993, ISBN 0-412-04231-2, Monographs 57; both duplicate DOI records
      removed;
    * **[MaierHein2024]** → 21(2), doi:10.1038/s41592-023-02151-z, VERIFIED;
    * **[Holm1979]** → VERIFIED, JSTOR 4615733, "no DOI exists";
    * the open-items and "Pending (team must supply)" sections were reduced to the PARTIAL entries plus a "resolved"
      note.
  * `FINAL_REPORT_DRAFT.md`: §3.3 now cites [Bo2026] and replaces "this project's own implementation" with the
    re-implementation wording; the "(unverified)" / "(edition to be chosen)" tags were removed from §3.5; the
    References open items were updated.
  * `VIVA_QA.md` Q28: re-implementation wording and [Bo2026].
  * No "own implementation", "citation pending" or UNVERIFIED entries remain, apart from the legend lines.
* **Test fix:** in `results/external_ts/scripts/tests/test_analysis.py` (not a protected file; SHA-256 prefix
  `e0f93002e595`), `test_run_refuses_when_incomplete` now monkeypatches `A.R` to a temp tree with 3 worklist subjects
  and 2 `.done` markers. It asserts the exact "evaluation incomplete: 1 subjects…" refusal and that no output
  directory is created. Two refusal tests were added: `test_run_refuses_when_not_finalized` (no raw_rows) and
  `test_run_refuses_existing_out_dir`. No test was deleted. external_ts tests: **110 passed**.
  `analyze_external_ts.py` itself was not changed. The real `analysis/` directory is untouched.
* **Backups** (both verified):
  * `backups/external_ts/external_ts_tier1_20261124T055116Z.zip`: 2,442 files, SHA-256
    `b485b2542d4ef2f883fe38becaf77b754071eb81440c7a31dc1f2ee0f122f3ea`. `--verify-only` exit 0; CRC ok; members
    equal the manifest; the sidecar is OK. It contains `figures/`, `final_audit_…json`, the protected snapshot and the
    fixed test.
  * `backups/docs/docs_20261124T055130Z.zip`: 23 files, SHA-256
    `454f25b35c835618b7da4fd8bf44634218a93d990fac19b68964fa0d6b2aca63`. CRC ok; member hashes equal the manifest;
    the live `docs/` matches it; `sha256sum -c` OK (sidecars written with LF this time). It contains [Bo2026] and the
    Figure 1–4 tags.
  * Note: this log entry was written after the results zip, so the zip's copy of PROGRESS_LOG.md stops at the
    citation-check entry.
* Protected files: 1,488 still verify. No git commit.

## 2026-11-24 — slide 12 figure tags; RAM check; open-items review

* `docs/SLIDES_OUTLINE.md` slide 12: the figure placeholder was replaced with [Figure 1: fig1_external_primary_dice_box.png]
  and [Figure 3: fig3_external_primary_paired_diff.png]. CRLF kept. The docs zip `…055130Z` is now one edit behind.
* RAM: 0.48 GiB free (psutil 0.589 GiB available) of 7.34 GiB; commit 17.92 / 20.01 GiB. That is below both the 1.6
  and the 1.0 GiB start guards.
* Found during the review, NOT fixed yet: `docs/LIMITATIONS.md` line 41 still says the external run "is estimated at
  about 4 hours … it has not yet been run". That is stale; the run was completed on 2026-11-24.

## 2026-11-24 — LIMITATIONS fix; docs backup refreshed

* `docs/LIMITATIONS.md` (Engineering): "it has not yet been run" was replaced with: the run completed on 2026-11-24
  over three resumable runs; two were stopped by the free-disk and free-RAM guards; the final run used the 1.0 GiB
  deviation. CRLF kept. No "not yet been run" remains in docs/.
* New `backups/docs/docs_20261124T064625Z.zip`: 23 files, SHA-256
  `c3a1db033b6040072a83862cd30587210103842ad350690932c8d1a94ffb8a2b`. CRC ok; member hashes equal the manifest; the
  live docs/ matches it; `sha256sum -c` OK. It contains the slide 12 figure tags and the LIMITATIONS fix, and
  supersedes `…055130Z`.
* No git commit.

## 2026-11-24 — user confirmed upload of the 6 backup files; appendix D, v3 slide figure, rem.txt

* The user reported uploading the 6 files (external_ts `…055116Z` and docs `…064625Z`, each with its sidecars) to
  Google Drive.
* **v3 slide figure:** new `scripts/make_v3_slide_figure.py` (SHA-256 `d600cff8…`; reuses the style of
  `make_phaseB_figures.py`). It writes once to `figures/slides_v3/`:
  * `fig5_v3_holdout_dice_box.png` (`0169138a…`) and `figures_manifest.json`;
  * input `results/v3/raw_holdout_rows.csv` (156 slices, 33 patients), read-only;
  * all 4 per-patient means equal `results/v3/adaptive/summary.json` within 1e-12 (otherwise the script stops).
  * Inspected visually. Slide 11 in `SLIDES_OUTLINE.md` now has [Figure 5: fig5_v3_holdout_dice_box.png].
* **Memory bar chart NOT made:** no file on disk records the 3.07 GiB (3.29 GB) unoptimised peak quoted in docs. Only
  the optimised run is recorded (`audit/mem_opt_validation.json`, peak working set 1,547,558,912 B = 1.44 GiB).
  Slide 13's "(to make)" is left as is. The 3.07 GiB figure in docs has no file-level source.
* **FINAL_REPORT_DRAFT appendices:** D now lists the test results (93 project tests pass with 2 real-SAM tests
  deselected; 110 external_ts tests pass). E now names the final audit and the current results backup `…055116Z`.
  CRLF kept.
* **rem.txt deleted** (user approved): it was a stray list of 89 subject IDs in the project root, written 2026-11-22,
  SHA-256 `d2202e27…`, not a protected file.
* **Backups refreshed** (both verified):
  * `backups/external_ts/external_ts_tier1_20261124T065546Z.zip`: 2,445 files, SHA-256 `3c4a6c59709f85fd5c0b1069a9b71f794936047904977459ffc0ddd2442b978c`.
    `--verify-only` exit 0; CRC ok; members equal the manifest; the sidecar is OK. It contains `figures/slides_v3/` and
    `make_v3_slide_figure.py`.
  * `backups/docs/docs_20261124T065609Z.zip`: 23 files, SHA-256 `87251d7692103f2ef6fed379f8a03af5a988004ef8558a9abc0f8dd1fd39b034`.
    CRC ok; member hashes equal the manifest; the live docs/ matches it; the sidecar is OK.
  * These supersede `…055116Z` and `…064625Z`. Protected files: 1,488 still verify. No git commit.

## 2026-11-24 — git commit (user requested)

* The user confirmed uploading the backups `…065546Z` and `…065609Z`.
* Created branch **`external-evaluation-results`** from `master` (`8828775`), following the rule not to commit on the
  default branch. Commit **`011fc51`**: 2,484 files (2,483 added, 1 modified: README.md line 95). Afterwards the working
  tree was clean, and the committed README blob equals the working file (`77f9b4ef…`). `.gitattributes` `* -text`
  keeps every file byte-for-byte. `git fsck` reports no errors; it lists 3 dangling blobs, which are harmless leftovers from staging.
* Not pushed; no remote action. `master` is unchanged.
* This log entry itself is the only uncommitted change.

## 2026-11-24 — GitHub remote; merge to main; push REFUSED (403)

* Committed the previous log entry on `external-evaluation-results` as `c465592`.
* `git remote add origin https://github.com/ThrashaShetty/Majorr-Project.git`. `git ls-remote` succeeded and showed
  the remote is empty, so nothing would be overwritten.
* `git switch master`, then `git merge --ff-only external-evaluation-results` (fast-forward to `c465592`), then
  `git branch -m master main`. After the switch and merge, the working tree was clean: 90 `.done` markers present, and
  all 1,488 protected files verify.
* `git push -u origin main` **failed**: `remote: Permission to ThrashaShetty/Majorr-Project.git denied to Anvith433.
  … error: 403`. The stored GitHub credential is the account `Anvith433`, which has no write access to that repo.
  Nothing was pushed. The branch push was not attempted, because it would fail the same way. No PAT is available to
  the AI assistant.
* This entry is uncommitted.

## 2026-11-24 — pre-push secrets check and history regroup (user requested)

* Secrets scan of all 2,804 tracked files (`ghp_`, `github_pat_`, `sk-`, `AIza`, `Bearer`, `password=`, `api_key=`,
  `token=`; files `.env`, `secrets.json`, `credentials.json`, `config.json`): **no secrets**. One false positive:
  `scripts/tests/test_ts_external.py:81` ("ma**sk-**eligible"). No `.env` file exists.
* `sam_vit_b_01ec64.pth` is **not tracked** (`*.pth` and `checkpoints/*` already ignored). No `backups/` path is tracked.
* Root `.gitignore` is a **protected file** (in the 1,488-file snapshot), so it was not edited. It already lists
  `.env`, `*.pth`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.DS_Store`, `Thumbs.db`. The missing patterns
  (`backups/`, `*.log`, `node_modules/`, `dist/`, `build/`) were added to the local-only `.git/info/exclude`.
  The 9 already-tracked evidence `.log` files stay tracked.
* Added `.env.example` (placeholders for the 4 environment variables the code reads: `MEDSAM_CHECKPOINT`,
  `MEDSAM_DEVICE`, `MEDSAM_DB`, `MEDSAM_EXT_DATA`).
* History: 3 local, never-pushed commits (`8828775` initial preservation, `011fc51`, `c465592`). The initial
  preservation commit is kept; `main` was reset (mixed, working tree untouched) to `8828775` and the later changes
  re-committed in groups. Old tip kept as local tag `pre-regroup-c465592`. `external-evaluation-results` moved to the
  new `main` tip so the pushed branches share one history. No `DEMO_GUIDE.md` exists (skipped).
