"""Write the final raw-row tables once ALL worklist subjects are complete and verified.

Batch runs pass an explicit subject list, so eval_external_ts.execute never reaches its finalize step.
This calls the frozen runner's own `finalize` (unchanged): it re-verifies every subject's completion
marker, CSV and slice hashes, then writes (write-once):
  eval/raw_rows.csv, eval/raw_rows_primary.csv, eval/raw_rows_sensitivity.csv
Refuses (non-zero exit, nothing written) if any subject is missing or fails verification.

Usage (from medical_sam_project):  python -B -I results/external_ts/scripts/finalize_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_external_ts as E  # noqa: E402
import ts_common as C  # noqa: E402


def main(eval_dir: Path = E.EVAL_DIR, project_root: Path = C.PROJECT_ROOT) -> int:
    cfg, cfg_sha, worklist = E.load_frozen(eval_dir, project_root)
    subjects = sorted({w["subject"] for w in worklist})
    missing = [s for s in subjects if not (eval_dir / "subjects" / f"{s}.done").exists()]
    if missing:
        print(f"REFUSED: {len(missing)} of {len(subjects)} subjects not complete: {missing[:10]}{' ...' if len(missing) > 10 else ''}",
              file=sys.stderr)
        return 2
    hashes = E.finalize(eval_dir, worklist, cfg_sha)
    print(json.dumps({"config_sha256": cfg_sha, "subjects": len(subjects), "written_sha256": hashes}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
