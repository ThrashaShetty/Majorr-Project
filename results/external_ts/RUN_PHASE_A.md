# Phase A — running the remaining external evaluation yourself

Remaining work: **89 subjects, 1,062 slices, 9 batches**. Estimated time: about 3.9 h of compute
(13.2 s per slice measured on s0965), plus about 1 min per batch for verification and backup.

Before you start:

* Close Antigravity, the AI assistant and browsers.
* Keep the charger connected and the **lid open**. The driver keeps Windows awake only while it runs.
* Backup decision recorded: same-disk risk accepted for this run. Upload the newest
  `backups\external_ts\external_ts_tier1_*.zip` off-disk after each session.

All commands are for **Windows PowerShell**, run from the project folder. Copy them one block at a time.

## 0. Set up and check resources

```powershell
cd C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"
New-Item -ItemType Directory -Force results\external_ts\run_logs | Out-Null

& $py -B -I -c "import psutil,shutil; b=psutil.sensors_battery(); print('RAM available GiB', round(psutil.virtual_memory().available/2**30,3), '| C: free GiB', round(shutil.disk_usage('C:/').free/2**30,2), '| AC plugged:', None if b is None else b.power_plugged)"
```

You need **AC plugged: True** and RAM available of **at least 1.6 GiB** (or at least 1.0 GiB for the fallback in 2b).

## 1. Pre-flight check (read-only; loads no model, writes no results)

```powershell
$ids = (& $py -B -I -c "import sys; sys.path.insert(0,'results/external_ts/scripts'); import run_batches as R; print(' '.join(R.remaining(R.EVAL)[0]))").Trim().Split(' ')
$ids.Count          # expect 89

# 1a. normal start limit (1.6 GiB)
& $py -B -I results\external_ts\scripts\run_with_mem_opt.py --check-only --subjects $ids | Out-File -Encoding utf8 results\external_ts\run_logs\preflight_1.6.json
# 1b. fallback start limit (1.0 GiB, approved resource-constrained deviation)
& $py -B -I results\external_ts\scripts\run_with_mem_opt.py --check-only --start-ram-gib 1.0 --resource-constrained-deviation --subjects $ids | Out-File -Encoding utf8 results\external_ts\run_logs\preflight_1.0.json

# summary of both
& $py -B -I -c "import json,sys; [print(f, '| config', d['preflight']['config_sha256'][:12], '| subjects', len(d['preflight']['subjects']), '| slices to run', sum(len(v['to_run']) for v in d['preflight']['subjects'].values()), '| start limit', d['start_ram_gib'], '| guard', d['start_guard_probe']) for f in sys.argv[1:] for d in [json.load(open(f, encoding='utf-8-sig'))]]" results\external_ts\run_logs\preflight_1.6.json results\external_ts\run_logs\preflight_1.0.json
```

**Pass** means: config `3933ca8c4837`, 89 subjects, 1062 slices to run, and `guard {'ok': True, ...}` for the start
limit you will use. Any error message (hash mismatch, missing input) means **stop and tell me**.

Optional dry run (writes nothing):

```powershell
& $py -B -I results\external_ts\scripts\run_batches.py --dry-run --backup-dest C:\Anvith\FINALYEARPROJECTTHRASHA\backups\external_ts
```

It should show `"remaining_subjects": 89`, `"remaining_slices": 1062`, `"batches": 9` and `"ac_power": true`.

## 2. Run the batches

Keep a transcript of the console:

```powershell
Start-Transcript -Path ("results\external_ts\run_logs\phase_a_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".txt")
```

**2a. Normal: use this if the 1.6 GiB guard passed.**

```powershell
& $py -B -I results\external_ts\scripts\run_batches.py --execute --i-have-explicit-approval --backup-dest C:\Anvith\FINALYEARPROJECTTHRASHA\backups\external_ts
"exit code: $LASTEXITCODE"
```

**2b. Fallback: only if 1.6 GiB fails and 1.0 GiB passes.** This is recorded as a resource-constrained deviation.

```powershell
& $py -B -I results\external_ts\scripts\run_batches.py --execute --i-have-explicit-approval --backup-dest C:\Anvith\FINALYEARPROJECTTHRASHA\backups\external_ts --start-ram-gib 1.0 --resource-constrained-deviation
"exit code: $LASTEXITCODE"
```

When it finishes or stops:

```powershell
Stop-Transcript
```

What the driver does for each of the 9 batches:

1. Checks AC power.
2. Takes a hash snapshot of existing results.
3. Evaluates up to 10 subjects.
4. Independently verifies them.
5. Writes a new verified backup zip.

It stops at the first problem. You can also press **Ctrl+C** at any time; everything finished so far is kept.

| Exit code | Meaning | What to do |
|---|---|---|
| 0 | All 9 batches complete and verified | Go to section 4 |
| 1 | Refused: approval flags missing | Use the command exactly as written |
| 2 | Runner error | Stop; tell me (the transcript has the error) |
| 3 | Resource stop: RAM below the limit, or disk below 2 GiB | Free RAM, then resume (section 3) |
| 4 | AC power lost before a batch | Reconnect the charger, then resume |
| 5 | Independent verification failed | **Stop; tell me. Do not rerun.** |
| 6 | Backup failed | Stop; tell me |
| 130 | You pressed Ctrl+C | Resume when ready |

## 3. Resuming after a stop

Run **exactly the same command again** (2a or 2b). Nothing has to be cleaned up first.

* **Batch plan:** `batches\batch_plan.json` is written once and checked by hash, so the 9 batches never change.
* **Finished batches:** a batch whose subjects all have completion markers **and** a passing verification report is
  skipped. The console prints `batch_skipped_verified`.
* **Finished subjects:** a subject with a completion marker (`eval\subjects\<id>.done`) is re-verified by hash and
  skipped.
* **Partly finished subjects:** each finished slice's result file is re-verified (config hash, worklist fields,
  prediction-file SHA-256) and skipped. Only the missing slices run.
* **Safety:** a result file is never overwritten. If a stored file has changed, the run stops with "hash mismatch"
  instead of redoing it.
* **Evidence:** this behaviour is covered by the tests `test_interrupt_then_resume` and
  `test_rerun_skips_verified_no_duplicates`. They show that only missing slices are re-run and finished files keep
  their timestamps.

To see progress or confirm what was skipped (read-only):

```powershell
& $py -B -I -c "import json,collections; from pathlib import Path; E=Path('results/external_ts/eval'); wl=json.loads((E/'worklist.json').read_text())['items']; done={p.stem for p in (E/'subjects').glob('*.done')}; n=collections.Counter(p.parent.name for p in (E/'slices').rglob('*.json')); print('subjects done', len(done), 'of', len({w['subject'] for w in wl}), '| slice results', sum(n.values()), 'of', len(wl))"
```

Run it before and after a resume. The slice count only goes up, and no existing result file changes.

## 4. After all batches finish (exit code 0)

```powershell
# 4a. write the final raw-row tables (re-verifies every subject; refuses if anything is missing)
& $py -B -I results\external_ts\scripts\finalize_eval.py

# 4b. independent verification of all 90 subjects (recomputes every metric from the saved predictions)
$all = (& $py -B -I -c "import json; print(' '.join(sorted({w['subject'] for w in json.load(open('results/external_ts/eval/worklist.json'))['items']})))").Trim().Split(' ')
& $py -B -I results\external_ts\scripts\verify_eval_subjects.py --subjects $all --out ("results\external_ts\audit\final_verification_" + (Get-Date -Format yyyyMMdd_HHmmss) + ".json")
"exit code: $LASTEXITCODE"   # 0 = all passed

# 4c. one more verified backup that includes the final tables
& $py -B -I results\external_ts\scripts\make_backup.py --dest C:\Anvith\FINALYEARPROJECTTHRASHA\backups\external_ts
```

Upload the newest `backups\external_ts\external_ts_tier1_*.zip` off-disk.

## 5. What to tell me

You don't need to send any files; I read them in place. Tell me the **exit codes** from steps 2 and 4. I will check:

* `results\external_ts\run_logs\` (the transcript and the pre-flight JSON files);
* `results\external_ts\eval\logs\run_history.jsonl`, `resource_checks.jsonl` and `launcher_memory.jsonl`;
* `results\external_ts\audit\batch_*_verification_*.json` and `final_verification_*.json`;
* `results\external_ts\eval\raw_rows_primary.csv` (expect 826 rows) and `raw_rows_sensitivity.csv` (expect 300 rows).

Then I run Phase B, the frozen analysis.
