"""Independent QC of preprocessed TS subjects (does NOT call src.preprocessing or CTStream).

For every accepted slice:
  * output hashes match the subject log (and the done marker)
  * PNG format: 512x512, uint8, single channel; mask values in {0, 255}; fg >= 50 and == logged fg
  * mask == independent np.rot90(k=1) + nearest-neighbour resize of the NIfTI mask slice (exact)
  * image == independent HU clip [-200,300] -> uint8 -> rot90 -> bilinear resize of the CT slice (exact)
  * rotation is load-bearing: the un-rotated mask does not match (unless the slice is symmetric)
  * alignment: HU under the label is soft-tissue-like and the label is brighter/darker-consistent
    with a 6-pixel ring around it (reported, not gated)
Slice counts are compared with the frozen manifest.

Usage: python -B -I verify_pilot.py --subjects s1389 s0965 s0908 --overlays <dir> --out <qc.json>
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_common as C  # noqa: E402  (paths/hash helpers only)
from build_manifest import load_manifest  # noqa: E402
from prep_external_ts import paths_for, verify_subject  # noqa: E402


def independent_ct_slices(archive: Path, subject: str, wanted: set[int]):
    with zipfile.ZipFile(archive) as zf, zf.open(f"{subject}/ct.nii.gz") as raw, gzip.GzipFile(fileobj=raw) as g:
        hdr = nib.Nifti1Header.from_fileobj(io.BytesIO(g.read(352)))
        g.read(int(hdr["vox_offset"]) - 352)
        nx, ny, nz = (int(s) for s in hdr.get_data_shape()[:3])
        dt = hdr.get_data_dtype()
        slope, inter = hdr.get_slope_inter()
        slope = 1.0 if slope is None else slope
        inter = 0.0 if inter is None else inter
        for z in range(nz):
            a = np.frombuffer(g.read(nx * ny * dt.itemsize), dt).reshape((nx, ny), order="F")
            if z in wanted:
                yield z, (a.astype(np.float64) * slope + inter).astype(np.float32)


def check_subject(subject: str, row: dict, out_root: Path, archive: Path, extracted: Path, overlay_dir: Path | None) -> dict:
    p = paths_for(out_root, subject)
    ver = verify_subject(out_root, subject)
    log = json.loads(p["log"].read_text(encoding="utf-8"))
    acc = {s["z"]: s for s in log["slices"] if s["valid"]}
    mask_vol = np.asanyarray(nib.load(str(extracted / subject / "segmentations" / "spleen.nii.gz")).dataobj)
    res = {"subject": subject, "hash_verification": ver, "slices": [], "errors": []}
    for z, hu in independent_ct_slices(archive, subject, set(acc)):
        s = acc[z]
        img = cv2.imread(str(p["images"] / s["file"]), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(str(p["masks"] / s["file"]), cv2.IMREAD_UNCHANGED)
        e = []
        if img is None or msk is None or img.shape != (512, 512) or msk.shape != (512, 512) or img.dtype != np.uint8 or msk.dtype != np.uint8:
            e.append("format")
        if not set(np.unique(msk).tolist()) <= {0, 255}:
            e.append("mask values")
        fg = int((msk > 0).sum())
        if fg < 50 or fg != s["fg_px_after_resize"]:
            e.append(f"fg {fg} vs logged {s['fg_px_after_resize']}")
        lab = mask_vol[:, :, z]
        exp_m = cv2.resize(np.where(np.rot90(lab, 1) == 1, 255, 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
        exp_i = cv2.resize(np.round((np.rot90(np.clip(hu, -200, 300), 1) + 200) / 500 * 255).astype(np.uint8), (512, 512),
                           interpolation=cv2.INTER_LINEAR)
        m_eq = bool(np.array_equal(msk, exp_m))
        i_diff = int(np.abs(img.astype(int) - exp_i.astype(int)).max())
        if not m_eq:
            e.append("mask != independent recomputation")
        if i_diff != 0:
            e.append(f"image differs from independent recomputation (max {i_diff})")
        unrot = cv2.resize(np.where(lab == 1, 255, 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
        rot_matters = not np.array_equal(unrot, msk)
        # alignment diagnostics in native HU space (label vs ring around it)
        lab_b = lab > 0
        ring = cv2.dilate(lab_b.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool) & ~lab_b
        hu_in, hu_ring = float(np.median(hu[lab_b])), float(np.median(hu[ring]))
        g_in = float(img[msk > 0].mean())
        rec = {"z": z, "file": s["file"], "fg_px": fg, "mask_exact": m_eq, "image_max_abs_diff": i_diff,
               "rotation_load_bearing": rot_matters, "median_hu_in_label": round(hu_in, 1),
               "median_hu_ring_6px": round(hu_ring, 1), "mean_gray_in_label": round(g_in, 1), "errors": e}
        res["slices"].append(rec)
        res["errors"] += [f"z={z}: {x}" for x in e]
        if overlay_dir is not None:
            ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, (0, 0, 255), 2)
            cv2.putText(ov, f"{subject} z={z} HU in={hu_in:.0f} ring={hu_ring:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_dir / f"qc_{s['file']}"), ov)
    if len(res["slices"]) != len(acc):
        res["errors"].append(f"checked {len(res['slices'])} of {len(acc)} accepted slices")
    res["counts_vs_manifest"] = {
        "step5": {"mask_eligible_manifest": len(row["mask_eligible_step5"]), "processed": len(log["processed_step5"]),
                  "identical_indices": log["processed_step5"] == row["mask_eligible_step5"]},
        "step17": {"mask_eligible_manifest": len(row["mask_eligible_step17"]), "processed": len(log["processed_step17"]),
                   "identical_indices": log["processed_step17"] == row["mask_eligible_step17"]},
    }
    hu_in = [r["median_hu_in_label"] for r in res["slices"]]
    res["summary"] = {
        "slices_checked": len(res["slices"]), "all_masks_exact": all(r["mask_exact"] for r in res["slices"]),
        "all_images_exact": all(r["image_max_abs_diff"] == 0 for r in res["slices"]),
        "rotation_load_bearing_slices": sum(r["rotation_load_bearing"] for r in res["slices"]),
        "median_hu_in_label_range": [min(hu_in), max(hu_in)] if hu_in else None,
        "slices_label_hu_outside_[-50,300]": [r["z"] for r in res["slices"] if not -50 <= r["median_hu_in_label"] <= 300],
        "warnings_from_prep": log["warnings"], "resources_from_prep": log["resources"],
    }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--overlays", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    m = load_manifest(C.MANIFEST)
    rows = {r["subject"]: r for r in m["subjects"]}
    report = {"manifest_sha256": C.file_digest(C.MANIFEST), "subjects": []}
    for s in a.subjects:
        report["subjects"].append(check_subject(s, rows[s], C.PREPROCESSED, C.ARCHIVE, C.EXTRACTED, a.overlays))
    report["all_passed"] = all(not r["errors"] and r["counts_vs_manifest"]["step5"]["identical_indices"]
                               and r["counts_vs_manifest"]["step17"]["identical_indices"] for r in report["subjects"])
    status, digest = C.write_once(a.out, C.json_bytes(report))
    for r in report["subjects"]:
        print(r["subject"], json.dumps(r["summary"], default=C._json_default), "errors:", r["errors"][:5],
              "counts:", json.dumps(r["counts_vs_manifest"]))
    print("ALL PASSED" if report["all_passed"] else "FAILURES PRESENT", status, a.out, digest)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
