"""Build the frozen TotalSegmentator cohort manifest (mask-only; no CT voxels read).

Three slice levels are kept distinct:
  candidate      - every index on the fixed grid z = 0, step, 2*step, ... (step 5 primary, 17 sensitivity)
  mask_eligible  - candidate whose spleen mask, after the EXISTING orient_slice -> label_to_binary_mask
                   -> resize_mask(512x512) path, has >= MIN_FG_PIXELS (50) foreground pixels.
                   Predicted from the mask alone.
  processed      - mask_eligible AND accepted by validate_slice_pair on the real CT slice
                   (adds the contrast check). Recorded by prep_external_ts.py in per-subject logs,
                   NOT in this frozen manifest.

Cohorts (frozen protocol):
  eligible (91)   : non-empty mask AND >= 1 mask_eligible slice on the step-5 grid
  primary  (74)   : eligible AND the mask touches neither axial end (z=0 or z=n-1); step-5 slices
  sensitivity (91): all eligible subjects; step-17 slices (a subject may contribute 0 slices)

Usage (real run):
  python -B -I build_manifest.py            # verifies archive MD5, writes results/external_ts/cohort_manifest.json
The manifest is write-once: re-running with identical inputs is a no-op; different content is refused.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as C  # noqa: E402


def classify_mask(mask: np.ndarray) -> dict[str, Any]:
    """Cohort-relevant facts for one spleen mask volume (array axis 2 = axial)."""
    n_z = int(mask.shape[2])
    fg_z = np.nonzero(mask.reshape(-1, n_z).any(axis=0))[0]
    nonempty = fg_z.size > 0
    out: dict[str, Any] = {
        "n_axial": n_z, "nonempty": bool(nonempty), "fg_voxels": int(np.count_nonzero(mask)),
        "touches_lo": bool(nonempty and fg_z[0] == 0), "touches_hi": bool(nonempty and fg_z[-1] == n_z - 1),
    }
    for step, key in ((C.PRIMARY_STEP, "step5"), (C.SENSITIVITY_STEP, "step17")):
        cand = C.grid(n_z, step)
        elig = []
        for z in cand:
            fg = C.mask_fg_after_resize(mask[:, :, z])
            if fg >= C.MIN_FG_PIXELS:
                elig.append([z, fg])
        out[f"candidate_{key}"] = cand
        out[f"mask_eligible_{key}"] = [z for z, _ in elig]
        out[f"mask_eligible_{key}_fg_px"] = [fg for _, fg in elig]
    out["eligible"] = bool(nonempty and len(out["mask_eligible_step5"]) >= 1)
    out["boundary_touching"] = bool(out["touches_lo"] or out["touches_hi"])
    out["primary"] = bool(out["eligible"] and not out["boundary_touching"])
    out["sensitivity"] = out["eligible"]
    return out


def parse_support_lists(support_list: Path, query_list: Path) -> dict[str, Any]:
    pid = lambda s: re.search(r"case_(\d+)_slice_", s).group(1)  # noqa: E731
    sup = [ln.split("\t")[0] for ln in support_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dev = [ln.split("\t")[0] for ln in query_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sp = sorted({f"case_{int(pid(s)):03d}" for s in sup})
    dp = sorted({f"case_{int(pid(s)):03d}" for s in dev})
    if set(sp) & set(dp):
        raise C.TSError(f"support/dev overlap: {set(sp) & set(dp)}")
    return {"support_patients": sp, "n_support_patients": len(sp), "n_support_slices": len(sup),
            "dev_patients_excluded": dp, "n_dev_slices_excluded": len(dev)}


def build(archive: Path, extracted: Path, *, verify_md5: bool, expected: tuple[int, int, int] | None,
          support_list: Path, query_list: Path, hash_inputs: dict[str, Path],
          expected_support: tuple[int, int] | None = (33, 156)) -> dict[str, Any]:
    if archive.stat().st_size != (C.ARCHIVE_SIZE if verify_md5 else archive.stat().st_size):
        raise C.TSError(f"archive size {archive.stat().st_size} != {C.ARCHIVE_SIZE}")
    archive_md5 = C.file_digest(archive, "md5") if verify_md5 else None
    if verify_md5 and archive_md5 != C.ARCHIVE_MD5:
        raise C.TSError(f"archive MD5 {archive_md5} != {C.ARCHIVE_MD5}")

    with zipfile.ZipFile(archive) as zf:
        infos = {i.filename: i for i in zf.infolist()}
    subjects = sorted(n.split("/")[0] for n in infos if re.fullmatch(r"[^/]+/ct\.nii\.gz", n))
    if expected and len(subjects) != expected[0]:
        raise C.TSError(f"{len(subjects)} subjects in archive, expected {expected[0]}")

    # metadata (verified against the archive CRC)
    meta_path = extracted / "meta.csv"
    meta_bytes = meta_path.read_bytes()
    if "meta.csv" in infos and zlib.crc32(meta_bytes) != infos["meta.csv"].CRC:
        raise C.TSError("extracted meta.csv does not match archive CRC")
    meta = {r["image_id"]: r for r in csv.DictReader(io.StringIO(meta_bytes.decode("utf-8-sig")), delimiter=";")}

    rows = []
    for s in subjects:
        mpath = extracted / s / "segmentations" / "spleen.nii.gz"
        if not mpath.exists():
            raise C.TSError(f"{s}: spleen mask missing at {mpath}")
        mbytes = mpath.read_bytes()
        member = f"{s}/segmentations/spleen.nii.gz"
        if member not in infos or zlib.crc32(mbytes) != infos[member].CRC:
            raise C.TSError(f"{s}: extracted spleen mask does not match archive CRC")
        img = nib.load(str(mpath))
        mask = np.asanyarray(img.dataobj)
        uniq = np.unique(mask)
        if not set(uniq.tolist()) <= {0, 1}:
            raise C.TSError(f"{s}: mask values {uniq[:10]} not binary")
        info = classify_mask(mask > 0)
        ct = infos[f"{s}/ct.nii.gz"]
        m = meta.get(s, {})
        rows.append({
            "subject": s, **info,
            "mask_sha256": C.sha256_bytes(mbytes), "mask_shape": list(map(int, img.shape)),
            "mask_zooms_mm": [float(z) for z in img.header.get_zooms()[:3]],
            "mask_orientation": C.orientation(img.affine), "mask_affine": np.round(img.affine, 5).tolist(),
            "ct_member": ct.filename, "ct_member_crc32": f"{ct.CRC:08x}", "ct_member_gz_bytes": ct.file_size,
            "meta": {k: m.get(k) for k in ("study_type", "split", "institute", "manufacturer", "scanner_model", "kvp", "age", "gender", "pathology")},
        })
        del mask

    elig = [r for r in rows if r["eligible"]]
    prim = [r for r in rows if r["primary"]]
    if expected and (len(elig), len(prim)) != expected[1:]:
        raise C.TSError(f"cohort sizes eligible={len(elig)} primary={len(prim)} != expected {expected[1:]} - "
                        "refusing to write; protocol needs review")
    sup = parse_support_lists(support_list, query_list)
    if expected_support and (sup["n_support_patients"], sup["n_support_slices"]) != expected_support:
        raise C.TSError(f"support pool {sup['n_support_patients']}/{sup['n_support_slices']} != {expected_support}")

    tot = lambda rs, k: int(sum(len(r[k]) for r in rs))  # noqa: E731
    counts = {
        "subjects_in_archive": len(rows), "nonempty_masks": sum(r["nonempty"] for r in rows),
        "eligible_subjects": len(elig), "primary_subjects": len(prim), "sensitivity_subjects": len(elig),
        "boundary_touching_eligible": sum(r["boundary_touching"] for r in elig),
        "primary_step5": {"candidate_slices": tot(prim, "candidate_step5"), "mask_eligible_slices": tot(prim, "mask_eligible_step5")},
        "sensitivity_step17": {"candidate_slices": tot(elig, "candidate_step17"), "mask_eligible_slices": tot(elig, "mask_eligible_step17"),
                               "subjects_with_zero_mask_eligible": [r["subject"] for r in elig if not r["mask_eligible_step17"]]},
        "all_eligible_step5_not_a_cohort": {"candidate_slices": tot(elig, "candidate_step5"), "mask_eligible_slices": tot(elig, "mask_eligible_step5")},
    }
    return {
        "manifest_version": 1,
        "dataset": {"zenodo_record": C.ZENODO_RECORD, "version": C.DATASET_VERSION, "licence": "CC BY 4.0",
                    "archive_name": archive.name, "archive_size": archive.stat().st_size, "archive_md5": archive_md5},
        "protocol": {
            "axial_axis": 2, "grid_start": 0, "primary_step": C.PRIMARY_STEP, "sensitivity_step": C.SENSITIVITY_STEP,
            "min_fg_pixels_after_resize": C.MIN_FG_PIXELS, "target_size": list(C.TARGET_SIZE),
            "hu_window": [C.HU_MIN, C.HU_MAX], "orientation_correction": "src.preprocessing.orient_slice (np.rot90 k=1), unchanged",
            "primary_rule": "eligible and spleen mask touches neither axial end; step-5 slices",
            "sensitivity_rule": "all eligible (incl. boundary-touching); step-17 slices",
            "no_minimum_size_threshold": True,
            "slice_levels": {"candidate": "grid index", "mask_eligible": ">=50 px after existing resize, mask only",
                             "processed": "accepted by validate_slice_pair on real CT; recorded in preprocessed/logs, not here"},
            "methods": ["baseline", "enhanced_v2", "a5", "single_mask"],
            "a5": {"lo": 0.5, "hi": 2.0, "tuned": False}, "seed": 42, "device": "cpu",
            "support_pairing": "nearest 64x64 thumbnail cosine (run_evaluation._thumb) over all 156 MSD v3 support slices",
            "hu_check": "diagnostic only (air mode in [-1100,-900] warns); no gating rule approved",
        },
        "support_pool": sup,
        "counts": counts,
        "count_reconciliation": {
            "1003": "step-5 mask_eligible slices summed over ALL 91 eligible subjects (earlier inspection figure; not a cohort)",
            "826": "step-5 mask_eligible slices summed over the 74 primary subjects (primary analysis set before CT contrast check)",
            "300": "step-17 mask_eligible slices summed over all 91 sensitivity subjects",
            "candidates": "candidate = every grid index, with or without spleen; far larger than eligible",
        },
        "input_hashes_sha256": {k: C.file_digest(p) for k, p in sorted(hash_inputs.items())},
        "subjects": rows,
    }


def default_hash_inputs() -> dict[str, Path]:
    P = C.PROJECT_ROOT
    return {
        "data/support_list.txt": P / "data/support_list.txt", "data/query_list.txt": P / "data/query_list.txt",
        "data/split_summary.json": P / "data/split_summary.json",
        "results/v3/adaptive/evaluation_config.json": P / "results/v3/adaptive/evaluation_config.json",
        "results/v3/scripts/holdout.py": P / "results/v3/scripts/holdout.py",
        "results/v3/scripts/v3_report.py": P / "results/v3/scripts/v3_report.py",
        "src/preprocessing.py": P / "src/preprocessing.py", "src/__init__.py": P / "src/__init__.py",
        "src/baseline.py": P / "src/baseline.py", "src/fob/fob_pipeline.py": P / "src/fob/fob_pipeline.py",
        "src/metrics.py": P / "src/metrics.py", "run_evaluation.py": P / "run_evaluation.py",
        "checkpoints/sam_vit_b_01ec64.pth": P / "checkpoints/sam_vit_b_01ec64.pth",
        "extracted/meta.csv": C.EXTRACTED / "meta.csv",
        "scripts/ts_common.py": C.SCRIPTS_DIR / "ts_common.py",
        "scripts/build_manifest.py": C.SCRIPTS_DIR / "build_manifest.py",
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> tuple[str, str]:
    data = C.json_bytes(manifest)
    status, digest = C.write_once(path, data)
    C.write_once(path.with_suffix(".sha256"), f"{digest}  {path.name}\n".encode())
    return status, digest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and verify against the .sha256 sidecar."""
    import json
    data = path.read_bytes()
    side = path.with_suffix(".sha256")
    if not side.exists():
        raise C.TSError(f"missing {side}")
    want = side.read_text().split()[0]
    if C.sha256_bytes(data) != want:
        raise C.TSError(f"manifest hash mismatch: {path}")
    return json.loads(data)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=C.MANIFEST)
    a = ap.parse_args(argv)
    m = build(C.ARCHIVE, C.EXTRACTED, verify_md5=True,
              expected=(C.EXPECTED_N_SUBJECTS, C.EXPECTED_N_ELIGIBLE, C.EXPECTED_N_PRIMARY),
              support_list=C.PROJECT_ROOT / "data/support_list.txt", query_list=C.PROJECT_ROOT / "data/query_list.txt",
              hash_inputs=default_hash_inputs())
    status, digest = write_manifest(m, a.out)
    print(f"manifest {status}: {a.out} sha256={digest}")
    print(C.json_bytes(m["counts"]).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
