<#
Phase A unattended launcher (Windows PowerShell 5.1).

  1. Waits 180 s (time to close Antigravity / AI assistant / browsers).
  2. Checks AC power (exit 12 if on battery; nothing is run).
  3. Read-only pre-flight at the normal 1.6 GiB start limit. If its RAM guard fails, pre-flight again at
     --start-ram-gib 1.0 --resource-constrained-deviation. If both fail -> exit 10 (nothing is run).
     A pre-flight integrity error (non-zero exit: hash mismatch, missing input) -> exit 11.
  4. Runs run_batches.py --execute --i-have-explicit-approval --backup-dest <backups\external_ts>
     (plus the two deviation flags only if the 1.6 GiB guard failed). Its exit code is passed through:
     0 ok, 2 runner_failed, 3 resource_stop, 4 ac_lost, 5 verify_failed, 6 backup_failed, 130 interrupted.
  5. Logs every step to run_logs\phaseA_<stamp>.log (+ per-step .out/.err files) and writes the final
     exit code to run_logs\phaseA_exitcode_<stamp>.txt.

Re-running this script resumes: completed batches/subjects/slices are re-verified by hash and skipped.
Test switches: -SkipWait (no 180 s wait), -DryRun (pre-flights, then run_batches --dry-run; evaluates nothing).
Launch detached:  powershell -NoProfile -ExecutionPolicy Bypass -File <this file>
#>
param([switch]$SkipWait, [switch]$DryRun, [switch]$TestIgnoreAC, [int]$WaitSeconds = 180)
if ($TestIgnoreAC -and -not $DryRun) { Write-Error "-TestIgnoreAC is only allowed with -DryRun"; exit 1 }

$ErrorActionPreference = "Continue"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path          # medical_sam_project
Set-Location $root
$env:PYTHONDONTWRITEBYTECODE = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "results\external_ts\run_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$prefix = if ($DryRun) { "TEST_phaseA" } else { "phaseA" }        # dry-run files can never be mistaken for a real run
$log = Join-Path $logDir "${prefix}_$stamp.log"
$codeFile = Join-Path $logDir "${prefix}_exitcode_$stamp.txt"
$backupDest = "C:\Anvith\FINALYEARPROJECTTHRASHA\backups\external_ts"
$scripts = "results\external_ts\scripts"

function Log([string]$msg) {
    $line = "{0}  {1}" -f (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"), $msg
    Add-Content -Path $log -Value $line -Encoding UTF8
    Write-Host $line
}
function Finish([int]$code, [string]$why) {
    Log "FINAL exit code $code ($why)"
    Set-Content -Path $codeFile -Value "$code`r`n$why`r`nlog: $log" -Encoding UTF8
    exit $code
}
function Run-Step([string]$name, [string]$arguments) {
    $out = Join-Path $logDir "${prefix}_${stamp}_$name.out"
    $err = Join-Path $logDir "${prefix}_${stamp}_$name.err"
    Log "START $name : python $arguments"
    $p = Start-Process -FilePath $py -ArgumentList $arguments -WorkingDirectory $root -NoNewWindow -PassThru `
         -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle                     # cache the handle so ExitCode is available after the wait (PS 5.1)
    $p.WaitForExit()
    $code = $p.ExitCode
    Log "END   $name : exit $code (stdout $out ; stderr $err)"
    return @{ code = $code; out = $out; err = $err }
}
function Resources {
    $r = & $py -B -I -c "import psutil,shutil; b=psutil.sensors_battery(); print(round(psutil.virtual_memory().available/2**30,3), round(shutil.disk_usage('C:/').free/2**30,2), (b is None) or bool(b.power_plugged))"
    $f = "$r".Trim().Split(" ")
    return @{ ram = $f[0]; disk = $f[1]; ac = ($f[2] -eq "True") }
}

Log "Phase A launcher started (PID $PID). root=$root DryRun=$DryRun"
if (-not $SkipWait) {
    Log "Waiting $WaitSeconds s - close Antigravity, the AI assistant and browsers now."
    Start-Sleep -Seconds $WaitSeconds
}

$res = Resources
Log ("Resources after wait: RAM available {0} GiB | C: free {1} GiB | AC plugged {2}" -f $res.ram, $res.disk, $res.ac)
if (-not $res.ac) {
    if ($TestIgnoreAC) { Log "TEST: AC not connected - ignored because -DryRun -TestIgnoreAC" }
    else { Finish 12 "AC power not connected - plug in the charger and re-launch" }
}

# remaining subjects (same function run_batches uses)
$ids = (& $py -B -I -c "import sys; sys.path.insert(0,'results/external_ts/scripts'); import run_batches as R; print(' '.join(R.remaining(R.EVAL)[0]))")
$ids = "$ids".Trim()
if (-not $ids) {
    Log "No remaining subjects: every worklist subject already has a completion marker."
} else {
    Log ("Remaining subjects: {0}" -f ($ids.Split(" ").Count))
}

$deviation = $false
if ($ids) {
    $s16 = Run-Step "preflight_1.6" "-B -I $scripts\run_with_mem_opt.py --check-only --subjects $ids"
    if ($s16.code -ne 0) { Finish 11 "pre-flight integrity error at 1.6 GiB - see $($s16.err)" }
    $j16 = Get-Content $s16.out -Raw | ConvertFrom-Json
    Log ("Pre-flight 1.6: config {0} | slices to run {1} | guard ok {2} {3}" -f $j16.preflight.config_sha256.Substring(0,12),
         (($j16.preflight.subjects.PSObject.Properties | ForEach-Object { $_.Value.to_run.Count }) | Measure-Object -Sum).Sum,
         $j16.start_guard_probe.ok, $j16.start_guard_probe.reason)
    if (-not $j16.start_guard_probe.ok) {
        $s10 = Run-Step "preflight_1.0" "-B -I $scripts\run_with_mem_opt.py --check-only --start-ram-gib 1.0 --resource-constrained-deviation --subjects $ids"
        if ($s10.code -ne 0) { Finish 11 "pre-flight integrity error at 1.0 GiB - see $($s10.err)" }
        $j10 = Get-Content $s10.out -Raw | ConvertFrom-Json
        Log ("Pre-flight 1.0: guard ok {0} {1}" -f $j10.start_guard_probe.ok, $j10.start_guard_probe.reason)
        if (-not $j10.start_guard_probe.ok) {
            Finish 10 ("both pre-flights failed the RAM guard. At 1.6: {0}. At 1.0: {1}. Close more apps and re-launch" -f
                       $j16.start_guard_probe.reason, $j10.start_guard_probe.reason)
        }
        $deviation = $true
        Log "Using the 1.0 GiB resource-constrained deviation (recorded in every run-history entry)."
    }
}

$batchArgs = "-B -I $scripts\run_batches.py --backup-dest $backupDest"
if ($DryRun) { $batchArgs += " --dry-run" } else { $batchArgs += " --execute --i-have-explicit-approval" }
if ($deviation) { $batchArgs += " --start-ram-gib 1.0 --resource-constrained-deviation" }
$rb = Run-Step $(if ($DryRun) { "run_batches_dryrun" } else { "run_batches" }) $batchArgs
$names = @{ 0 = "ok"; 1 = "refused (flags)"; 2 = "runner_failed"; 3 = "resource_stop"; 4 = "ac_lost"; 5 = "verify_failed";
            6 = "backup_failed"; 130 = "interrupted" }
$why = if ($names.ContainsKey($rb.code)) { $names[$rb.code] } else { "unexpected" }
if ($DryRun) { $why = "DRY RUN only - $why" }
Finish $rb.code "run_batches: $why"
