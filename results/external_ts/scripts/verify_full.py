"""Independent cohort-wide QC of the preprocessed TotalSegmentator outputs.

Per slice  : verify_pilot.check_subject (independent recomputation; hashes; format; values; alignment).
Cohort-wide:
  * every eligible subject has a valid done marker + log bound to the CURRENT manifest; no others do
  * files on disk == union of files named in logs (no missing, no extra, no duplicates across subjects)
  * images/ and masks/ hold the same file names; no leftover .tmp_* files
  * processed totals reconciled with the manifest (primary step5, sensitivity step17)
  * HU review flags (warnings only - never exclusion)
Writes a write-once JSON report and optional contact sheets (one mid slice per subject).

Usage: python -B -I verify_full.py --out <qc.json> [--sheets <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as C  # noqa: E402
from build_manifest import load_manifest  # noqa: E402
from prep_external_ts import paths_for  # noqa: E402
from verify_pilot import check_subject  # noqa: E402


def hu_flags(log: dict, qc: dict) -> list[str]:
    h, g, f = log["hu_diagnostics"], log["geometry"], []
    p = h["scaled_percentiles"]
    if g["scl_slope"] not in (None, 1.0) or g["scl_inter"] not in (None, 0.0):
        f.append(f"non-identity scaling slope={g['scl_slope']} inter={g['scl_inter']}")
    if h["air_mode_scaled"] is None or not -1100 <= h["air_mode_scaled"] <= -900:
        f.append(f"air mode {h['air_mode_scaled']}")
    if p["1"] > -900:
        f.append(f"p1={p['1']} (little air)")
    if p["1"] < -1100:
        f.append(f"p1={p['1']} (below air)")
    if h["n_below_hist"]:
        f.append(f"{h['n_below_hist']} voxels < -4096 (padding)")
    if p["99.9"] > 3000:
        f.append(f"p99.9={p['99.9']} (very dense material/metal)")
    out = qc["summary"]["slices_label_hu_outside_[-50,300]"]
    if out:
        f.append(f"label median HU outside [-50,300] on z={out}")
    if g["max_axis_tilt_deg"] > 1:
        f.append(f"tilted {g['max_axis_tilt_deg']} deg")
    if g["non_square_inplane"]:
        f.append("non-square in-plane")
    return f


def contact_sheet(tiles: list[tuple[str, np.ndarray]], path: Path, cols: int = 5, size: int = 200) -> None:
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * size, cols * size, 3), np.uint8)
    for k, (label, t) in enumerate(tiles):
        t = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        r, c = divmod(k, cols)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sheets", type=Path, default=None)
    a = ap.parse_args(argv)
    m = load_manifest(C.MANIFEST)
    msha = C.file_digest(C.MANIFEST)
    rows = {r["subject"]: r for r in m["subjects"]}
    elig = [s for s, r in rows.items() if r["eligible"]]
    out = C.PREPROCESSED
    errors: list[str] = []
    subj_reports, logged_files, tiles = [], Counter(), []
    peak_prep = 0
    for s in elig:
        p = paths_for(out, s)
        try:
            qc = check_subject(s, rows[s], out, C.ARCHIVE, C.EXTRACTED, None)
        except Exception as e:  # noqa: BLE001 - record, keep auditing the rest
            errors.append(f"{s}: {e!r}")
            continue
        log = json.loads(p["log"].read_text(encoding="utf-8"))
        if log["manifest_sha256"] != msha:
            errors.append(f"{s}: log bound to a different manifest")
        for sl in log["slices"]:
            if sl["valid"]:
                logged_files[sl["file"]] += 1
        peak_prep = max(peak_prep, log["resources"]["mem_after"]["peak_wset"])
        errors += [f"{s}: {x}" for x in qc["errors"]]
        for k in ("step5", "step17"):
            if not qc["counts_vs_manifest"][k]["identical_indices"]:
                errors.append(f"{s}: {k} processed indices differ from manifest mask_eligible")
        flags = hu_flags(log, qc)
        subj_reports.append({"subject": s, "primary": rows[s]["primary"], "boundary_touching": rows[s]["boundary_touching"],
                             "study_type": rows[s]["meta"]["study_type"],
                             "processed_step5": len(log["processed_step5"]), "processed_step17": len(log["processed_step17"]),
                             "prep_warnings": log["warnings"], "hu_review_flags": flags,
                             "hu": {k: log["hu_diagnostics"][k] for k in ("stored_min", "stored_max", "air_mode_scaled", "frac_in_window", "scaled_percentiles")},
                             "scaling": [log["geometry"]["scl_slope"], log["geometry"]["scl_inter"]],
                             "label_median_hu_range": qc["summary"]["median_hu_in_label_range"],
                             "slices_checked": qc["summary"]["slices_checked"],
                             "masks_exact": qc["summary"]["all_masks_exact"], "images_exact": qc["summary"]["all_images_exact"]})
        if a.sheets is not None:
            zs = log["processed_step5"] or log["processed_step17"]
            if zs:
                z = zs[len(zs) // 2]
                name = C.output_name(s, z)
                img = cv2.imread(str(out / "images" / name), cv2.IMREAD_GRAYSCALE)
                msk = cv2.imread(str(out / "masks" / name), cv2.IMREAD_GRAYSCALE)
                ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(ov, cnts, -1, (0, 0, 255), 3)
                tiles.append((f"{s} z{z}{' P' if rows[s]['primary'] else ' S'}", ov))

    # cohort-wide file accounting
    imgs = {f.name for f in (out / "images").iterdir() if f.is_file()}
    msks = {f.name for f in (out / "masks").iterdir() if f.is_file()}
    tmp = [f.name for d in ("images", "masks", "logs") for f in (out / d).iterdir() if f.name.startswith(".tmp_")]
    dup = [f for f, n in logged_files.items() if n > 1]
    missing = sorted(set(logged_files) - imgs) + sorted(set(logged_files) - msks)
    extra = sorted(imgs - set(logged_files)) + sorted(msks - set(logged_files))
    if imgs != msks:
        errors.append("images/ and masks/ hold different file names")
    for label, lst in (("duplicate logged outputs", dup), ("missing outputs", missing), ("unlogged extra outputs", extra), ("leftover temp files", tmp)):
        if lst:
            errors.append(f"{label}: {lst[:10]} (n={len(lst)})")
    done = {f.stem for f in (out / "logs").glob("*.done")}
    if done != set(elig):
        errors.append(f"done markers mismatch: missing {sorted(set(elig) - done)}, unexpected {sorted(done - set(elig))}")

    by = {r["subject"]: r for r in subj_reports}
    prim = [s for s in elig if rows[s]["primary"]]
    totals = {
        "eligible_subjects": len(elig), "primary_subjects": len(prim), "sensitivity_subjects": len(elig),
        "subjects_qc_checked": len(subj_reports),
        "primary_step5": {"manifest_mask_eligible": m["counts"]["primary_step5"]["mask_eligible_slices"],
                          "processed": sum(by[s]["processed_step5"] for s in prim if s in by)},
        "sensitivity_step17": {"manifest_mask_eligible": m["counts"]["sensitivity_step17"]["mask_eligible_slices"],
                               "processed": sum(by[s]["processed_step17"] for s in elig if s in by),
                               "subjects_with_zero_processed": [s for s in elig if s in by and by[s]["processed_step17"] == 0]},
        "all_eligible_step5_processed": sum(by[s]["processed_step5"] for s in elig if s in by),
        "unique_png_pairs": len(logged_files), "image_files": len(imgs), "mask_files": len(msks),
        "slices_independently_checked": sum(r["slices_checked"] for r in subj_reports),
        "subjects_with_prep_warnings": sorted(r["subject"] for r in subj_reports if r["prep_warnings"]),
        "subjects_with_hu_review_flags": sorted(r["subject"] for r in subj_reports if r["hu_review_flags"]),
        "prep_peak_working_set_bytes": peak_prep, "qc_peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", 0)),
    }
    for k in ("primary_step5", "sensitivity_step17"):
        if totals[k]["manifest_mask_eligible"] != totals[k]["processed"]:
            errors.append(f"{k}: processed {totals[k]['processed']} != manifest {totals[k]['manifest_mask_eligible']} (see per-subject prep warnings)")
    report = {"manifest_sha256": msha, "all_passed": not errors, "errors": errors, "totals": totals, "subjects": subj_reports}
    status, digest = C.write_once(a.out, C.json_bytes(report))
    if a.sheets is not None and tiles:
        for k in range(0, len(tiles), 25):
            contact_sheet(tiles[k:k + 25], a.sheets / f"sheet_{k // 25 + 1}.png")
    print(json.dumps(totals, indent=1, default=C._json_default))
    print("ERRORS:", errors[:20])
    print("ALL PASSED" if not errors else "FAILURES PRESENT", status, a.out, digest)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
