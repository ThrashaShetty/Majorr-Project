"""Preprocess TotalSegmentator subjects for external evaluation, one subject at a time.

* CT is streamed slice-by-slice from the verified ZIP (never extracted to disk).
* Every candidate slice (step-5 and step-17 grids) goes through the EXISTING
  ``validate_slice_pair`` unchanged (orient_slice, [-200, 300] HU window, 512x512
  resize, label/contrast checks, >= 50 px foreground).
* Accepted slices are written as ts_<subject>_slice_<z>.png into
  preprocessed/images and preprocessed/masks with write-once atomic writes.
* Per-subject JSON log records every candidate slice (accepted or excluded + reason),
  header/geometry facts, HU diagnostics, output SHA-256s, RAM and bytes written.
* A ``<subject>.done`` marker (holding the log's SHA-256) is written LAST. Re-runs
  verify completed subjects by hash and skip them; incomplete subjects are redone
  (identical bytes are kept, different bytes raise - never overwritten).
* The first failure stops the run (exit code 2). No subject is skipped silently.

Safety: refuses more than --limit subjects (default 3) unless --allow-more is given.

Usage:
  python -B -I prep_external_ts.py --subjects s1389 s0965 s0425
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as C  # noqa: E402
from build_manifest import load_manifest  # noqa: E402

CODE_HASH_KEYS = ("src/preprocessing.py", "src/__init__.py", "scripts/ts_common.py")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _mem() -> dict[str, int]:
    mi = psutil.Process().memory_info()
    return {"rss": int(mi.rss), "peak_wset": int(getattr(mi, "peak_wset", 0))}


def encode_png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise C.TSError("PNG encoding failed")
    return buf.tobytes()


def paths_for(out_root: Path, subject: str) -> dict[str, Path]:
    logs = out_root / "logs"
    return {"images": out_root / "images", "masks": out_root / "masks", "logs": logs,
            "log": logs / f"{subject}.json", "done": logs / f"{subject}.done", "history": logs / "run_history.jsonl"}


def verify_subject(out_root: Path, subject: str) -> dict[str, Any]:
    """Verify a completed subject: done marker -> log hash -> every output hash."""
    p = paths_for(out_root, subject)
    if not p["done"].exists():
        raise C.TSError(f"{subject}: no done marker")
    want = p["done"].read_text().split()[0]
    if C.file_digest(p["log"]) != want:
        raise C.TSError(f"{subject}: log hash does not match done marker")
    log = json.loads(p["log"].read_text(encoding="utf-8"))
    n = 0
    for s in log["slices"]:
        if s["valid"]:
            for kind in ("image", "mask"):
                f = p[f"{kind}s"] / s["file"]
                if not f.exists() or C.file_digest(f) != s[f"{kind}_sha256"]:
                    raise C.TSError(f"{subject}: {kind} {s['file']} missing or hash mismatch")
                n += 1
    return {"subject": subject, "verified_files": n, "log_sha256": want}


def append_history(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=C._json_default) + "\n")
        fh.flush()


def process_subject(row: dict[str, Any], *, archive: Path, extracted: Path, out_root: Path,
                    manifest_sha: str, _fail_after_slices: int | None = None) -> dict[str, Any]:
    """Process one manifest subject. ``_fail_after_slices`` is a test hook only."""
    subject = row["subject"]
    p = paths_for(out_root, subject)
    if p["done"].exists():
        v = verify_subject(out_root, subject)
        return {"subject": subject, "status": "skipped_verified", **v}

    t0 = time.perf_counter()
    mem0 = _mem()
    mpath = extracted / subject / "segmentations" / "spleen.nii.gz"
    if C.file_digest(mpath) != row["mask_sha256"]:
        raise C.TSError(f"{subject}: spleen mask hash differs from manifest")
    mask_img = nib.load(str(mpath))
    mask = np.asanyarray(mask_img.dataobj)

    cand5, cand17 = set(row["candidate_step5"]), set(row["candidate_step17"])
    me5, me17 = set(row["mask_eligible_step5"]), set(row["mask_eligible_step17"])
    wanted = cand5 | cand17
    slices: list[dict[str, Any]] = []
    bytes_written = 0
    n_done = 0
    with C.CTStream(archive, subject) as ct:
        geo = C.check_geometry(ct.header, mask_img)
        if not geo["ok"]:
            raise C.TSError(f"{subject}: geometry check failed: {geo['errors']}")
        if ct.nz != row["n_axial"]:
            raise C.TSError(f"{subject}: CT has {ct.nz} slices, manifest {row['n_axial']}")
        stats = C.HUStats(ct.slope, ct.inter)
        for z, stored, scaled in ct.slices():
            stats.add(stored)
            if z not in wanted:
                continue
            valid, reason, img, msk, _ = C.validate_slice_pair(scaled, mask[:, :, z])
            rec: dict[str, Any] = {
                "z": z, "in_step5": z in cand5, "in_step17": z in cand17,
                "mask_eligible_step5": z in me5, "mask_eligible_step17": z in me17,
                "valid": bool(valid), "reason": reason,
                "fg_px_after_resize": None if msk is None else int((msk > 0).sum()),
                "image_std": None if img is None else round(float(img.std()), 3),
            }
            if valid:
                name = C.output_name(subject, z)
                st_i, sha_i = C.write_once(p["images"] / name, ib := encode_png(img))
                st_m, sha_m = C.write_once(p["masks"] / name, mb := encode_png(msk))
                bytes_written += (len(ib) if st_i == "written" else 0) + (len(mb) if st_m == "written" else 0)
                rec.update(file=name, image_sha256=sha_i, mask_sha256=sha_m, image_status=st_i, mask_status=st_m)
            slices.append(rec)
            n_done += 1
            if _fail_after_slices is not None and n_done >= _fail_after_slices:
                raise KeyboardInterrupt("simulated interruption (test hook)")
        trailing = ct.finish()
        if trailing:
            raise C.TSError(f"{subject}: {trailing} unexpected trailing bytes after voxel data")
    hu = stats.summary()
    del mask

    # consistency with the frozen manifest (never edits the manifest)
    proc5 = sorted(s["z"] for s in slices if s["valid"] and s["in_step5"])
    proc17 = sorted(s["z"] for s in slices if s["valid"] and s["in_step17"])
    not_predicted = sorted(set(proc5) - me5 | set(proc17) - me17)
    if not_predicted:
        raise C.TSError(f"{subject}: processed slices {not_predicted} were not mask_eligible - logic inconsistency")
    warnings = list(hu["hu_warnings"])
    dropped5, dropped17 = sorted(me5 - set(proc5)), sorted(me17 - set(proc17))
    if dropped5 or dropped17:
        warnings.append(f"mask_eligible but rejected on CT: step5 {dropped5}, step17 {dropped17}")
    if row["primary"] and not proc5:
        warnings.append("primary subject has 0 processed step-5 slices")

    log = {
        "subject": subject, "manifest_sha256": manifest_sha, "finished_utc": _now(),
        "prep_script_sha256": C.file_digest(Path(__file__).resolve()),
        "cohort": {"primary": row["primary"], "sensitivity": row["sensitivity"], "boundary_touching": row["boundary_touching"]},
        "geometry": geo, "hu_diagnostics": hu, "ct_member_gz_bytes": ct.info.file_size,
        "counts": {
            "step5": {"candidate": len(cand5), "mask_eligible": len(me5), "processed": len(proc5)},
            "step17": {"candidate": len(cand17), "mask_eligible": len(me17), "processed": len(proc17)},
            "slices_evaluated": len(slices), "png_pairs_written_or_verified": sum(s["valid"] for s in slices),
        },
        "processed_step5": proc5, "processed_step17": proc17,
        "warnings": warnings,
        "resources": {"seconds": round(time.perf_counter() - t0, 2), "mem_before": mem0, "mem_after": _mem(),
                      "bytes_written": bytes_written},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "nibabel": nib.__version__,
                        "opencv": cv2.__version__},
        "slices": slices,
    }
    p["logs"].mkdir(parents=True, exist_ok=True)
    C.atomic_write_bytes(p["log"], C.json_bytes(log))  # not final until the done marker exists
    C.write_once(p["done"], f"{C.file_digest(p['log'])}  {p['log'].name}\n".encode())
    return {"subject": subject, "status": "processed", "counts": log["counts"], "warnings": warnings,
            "resources": log["resources"]}


def run(subjects: list[str], *, manifest_path: Path, archive: Path, extracted: Path, out_root: Path,
        project_root: Path, limit: int, allow_more: bool, verify_archive_md5: bool) -> int:
    if len(subjects) > limit and not allow_more:
        raise C.TSError(f"{len(subjects)} subjects requested; limit is {limit} (full runs need --allow-more and approval)")
    if len(set(subjects)) != len(subjects):
        raise C.TSError("duplicate subject ids requested")
    manifest = load_manifest(manifest_path)
    msha = C.file_digest(manifest_path)
    for key in CODE_HASH_KEYS:  # code must be exactly what the manifest froze
        rel = key.replace("scripts/", "results/external_ts/scripts/")
        f = (project_root / rel) if not key.startswith("scripts/") else (C.SCRIPTS_DIR / Path(key).name)
        want = manifest["input_hashes_sha256"].get(key)
        if want is None or C.file_digest(f) != want:
            raise C.TSError(f"{key} changed since the manifest was frozen")
    if archive.stat().st_size != manifest["dataset"]["archive_size"]:
        raise C.TSError("archive size differs from manifest")
    if verify_archive_md5 and C.file_digest(archive, "md5") != manifest["dataset"]["archive_md5"]:
        raise C.TSError("archive MD5 differs from manifest")
    rows = {r["subject"]: r for r in manifest["subjects"]}
    for s in subjects:
        if s not in rows or not rows[s]["eligible"]:
            raise C.TSError(f"{s} is not an eligible subject in the manifest")

    hist = paths_for(out_root, "_")["history"]
    disk0 = shutil.disk_usage(out_root.anchor or out_root).free if out_root.exists() else None
    for s in subjects:
        append_history(hist, {"utc": _now(), "subject": s, "event": "start"})
        try:
            res = process_subject(rows[s], archive=archive, extracted=extracted, out_root=out_root, manifest_sha=msha)
        except BaseException as e:  # includes KeyboardInterrupt: record, then stop
            append_history(hist, {"utc": _now(), "subject": s, "event": "FAILED", "error": repr(e),
                                  "traceback": traceback.format_exc(limit=5)})
            print(f"FAILED {s}: {e!r} - stopping; remaining subjects not processed", file=sys.stderr)
            return 2
        append_history(hist, {"utc": _now(), "subject": s, "event": res["status"],
                              **{k: res[k] for k in ("counts", "warnings", "resources") if k in res}})
        print(json.dumps(res, default=C._json_default))
    if disk0 is not None:
        print(json.dumps({"disk_free_before": disk0, "disk_free_after": shutil.disk_usage(out_root.anchor).free}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--allow-more", action="store_true", help="NOT approved yet: required for >limit subjects")
    ap.add_argument("--verify-archive-md5", action="store_true")
    a = ap.parse_args(argv)
    return run(a.subjects, manifest_path=C.MANIFEST, archive=C.ARCHIVE, extracted=C.EXTRACTED, out_root=C.PREPROCESSED,
               project_root=C.PROJECT_ROOT, limit=a.limit, allow_more=a.allow_more, verify_archive_md5=a.verify_archive_md5)


if __name__ == "__main__":
    raise SystemExit(main())
