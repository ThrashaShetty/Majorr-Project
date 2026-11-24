"""Synthetic-data tests for the TotalSegmentator external pipeline (no real CT used).

Run:  python -B -m pytest -p no:cacheprovider -q results/external_ts/scripts/tests
"""
from __future__ import annotations

import gzip
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import build_manifest as BM  # noqa: E402
import prep_external_ts as PR  # noqa: E402
import ts_common as C  # noqa: E402

N_XY, N_Z = 40, 30
RAS_15 = np.diag([1.5, 1.5, 1.5, 1.0])


# --------------------------------------------------------------------------- #
# synthetic data helpers
# --------------------------------------------------------------------------- #
def nifti_gz(arr: np.ndarray, affine=RAS_15, slope=None, inter=None, truncate: int = 0) -> bytes:
    """Hand-written NIfTI-1 (.nii.gz) so stored values / scaling are fully controlled."""
    hdr = nib.Nifti1Header()
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    hdr.set_sform(affine, code=2)
    hdr.set_qform(affine, code=0)
    hdr.set_zooms(tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3)))
    hdr["vox_offset"] = 352
    if slope is not None:
        hdr["scl_slope"], hdr["scl_inter"] = slope, inter
    buf = io.BytesIO()
    hdr.write_to(buf)
    raw = buf.getvalue()[:348] + b"\x00" * 4 + arr.tobytes(order="F")
    if truncate:
        raw = raw[:-truncate]
    return gzip.compress(raw, mtime=0)


def make_ct(spleen: np.ndarray, rng: np.random.Generator, flat_slices=()) -> np.ndarray:
    ct = np.full((N_XY, N_XY, N_Z), -1000, np.int16)
    yy, xx = np.mgrid[:N_XY, :N_XY]
    body = (yy - 20) ** 2 + (xx - 20) ** 2 < 17 ** 2
    for z in range(N_Z):
        ct[:, :, z][body] = (40 + rng.integers(-30, 30, body.sum())).astype(np.int16)
    ct[spleen] = 120
    for z in flat_slices:  # uniform slice -> window gives a blank image (low_contrast)
        ct[:, :, z] = -1000
    return ct


def spleen_mask(z0: int, z1: int, *, l_shape=True) -> np.ndarray:
    m = np.zeros((N_XY, N_XY, N_Z), np.uint8)
    m[8:20, 22:30, z0:z1] = 1
    if l_shape:  # asymmetric so a wrong rotation/flip is detectable
        m[8:12, 14:22, z0:z1] = 1
    return m


@pytest.fixture()
def ds(tmp_path: Path):
    rng = np.random.default_rng(0)
    subjects = {
        "s0001": dict(mask=spleen_mask(8, 21)),                       # primary
        "s0002": dict(mask=spleen_mask(0, 9)),                        # touches z=0 -> sensitivity only
        "s0003": dict(mask=np.zeros((N_XY, N_XY, N_Z), np.uint8)),    # empty -> excluded
        "s0004": dict(mask=spleen_mask(3, 4)),                        # spleen only on z=3 (off-grid) -> excluded
        "s0005": dict(mask=spleen_mask(12, 25), scaled=True),         # scaled int16 storage, primary
        "s0006": dict(mask=spleen_mask(9, 22), flat=(10,)),           # z=10 mask-eligible but blank CT
        "s0007": dict(mask=spleen_mask(8, 21), truncate=5000),        # corrupt (truncated) CT stream
    }
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        meta = "image_id;age;gender;institute;study_type;split\n"
        for s, spec in subjects.items():
            m = spec["mask"]
            ct = make_ct(m > 0, rng, spec.get("flat", ()))
            if spec.get("scaled"):  # stored = (HU + 1024) / 2 ; slope 2, intercept -1024
                stored = ((ct.astype(np.int32) + 1024) // 2).astype(np.int16)
                zf.writestr(f"{s}/ct.nii.gz", nifti_gz(stored, slope=2.0, inter=-1024.0))
            else:
                zf.writestr(f"{s}/ct.nii.gz", nifti_gz(ct, truncate=spec.get("truncate", 0)))
            zf.writestr(f"{s}/segmentations/spleen.nii.gz", nifti_gz(m))
            meta += f"{s};50;m;A;ct abdomen;train\n"
        zf.writestr("meta.csv", meta)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as zf:
        for n in zf.namelist():
            if n == "meta.csv" or n.endswith("spleen.nii.gz"):
                zf.extract(n, extracted)
    data = tmp_path / "data"
    data.mkdir()
    (data / "support_list.txt").write_text("data/images/case_002_slice_070.png\tdata/masks/case_002_slice_070.png\n"
                                           "data/images/case_008_slice_010.png\tdata/masks/case_008_slice_010.png\n")
    (data / "query_list.txt").write_text("data/images/case_003_slice_020.png\tdata/masks/case_003_slice_020.png\n")
    hash_inputs = {
        "src/preprocessing.py": C.PROJECT_ROOT / "src/preprocessing.py",
        "src/__init__.py": C.PROJECT_ROOT / "src/__init__.py",
        "scripts/ts_common.py": SCRIPTS / "ts_common.py",
    }
    manifest = BM.build(archive, extracted, verify_md5=False, expected=None, expected_support=None,
                        support_list=data / "support_list.txt", query_list=data / "query_list.txt", hash_inputs=hash_inputs)
    mpath = tmp_path / "results" / "cohort_manifest.json"
    BM.write_manifest(manifest, mpath)
    return dict(tmp=tmp_path, archive=archive, extracted=extracted, manifest=manifest, mpath=mpath,
                out=tmp_path / "preprocessed", subjects=subjects, data=data, hash_inputs=hash_inputs)


def run(ds, subjects, **kw):
    args = dict(manifest_path=ds["mpath"], archive=ds["archive"], extracted=ds["extracted"], out_root=ds["out"],
                project_root=C.PROJECT_ROOT, limit=3, allow_more=False, verify_archive_md5=False)
    args.update(kw)
    return PR.run(subjects, **args)


def row(ds, s):
    return next(r for r in ds["manifest"]["subjects"] if r["subject"] == s)


def history(ds):
    p = ds["out"] / "logs" / "run_history.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_manifest_deterministic_and_write_once(ds):
    again = BM.build(ds["archive"], ds["extracted"], verify_md5=False, expected=None, expected_support=None,
                     support_list=ds["data"] / "support_list.txt", query_list=ds["data"] / "query_list.txt",
                     hash_inputs=ds["hash_inputs"])
    assert C.json_bytes(again) == C.json_bytes(ds["manifest"])
    assert BM.write_manifest(again, ds["mpath"])[0] == "exists_identical"
    changed = dict(again, manifest_version=999)
    with pytest.raises(C.OutputConflict):
        BM.write_manifest(changed, ds["mpath"])
    assert BM.load_manifest(ds["mpath"])["manifest_version"] == 1


def test_manifest_hash_tamper_detected(ds):
    raw = ds["mpath"].read_bytes()
    ds["mpath"].write_bytes(raw.replace(b'"manifest_version": 1', b'"manifest_version": 2'))
    with pytest.raises(C.TSError, match="hash mismatch"):
        BM.load_manifest(ds["mpath"])


def test_manifest_rejects_mask_not_matching_archive(ds):
    p = ds["extracted"] / "s0001" / "segmentations" / "spleen.nii.gz"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(C.TSError, match="archive CRC"):
        BM.build(ds["archive"], ds["extracted"], verify_md5=False, expected=None, expected_support=None,
                 support_list=ds["data"] / "support_list.txt", query_list=ds["data"] / "query_list.txt",
                 hash_inputs=ds["hash_inputs"])


def test_expected_counts_enforced(ds):
    with pytest.raises(C.TSError, match="cohort sizes"):
        BM.build(ds["archive"], ds["extracted"], verify_md5=False, expected=(7, 91, 74), expected_support=None,
                 support_list=ds["data"] / "support_list.txt", query_list=ds["data"] / "query_list.txt",
                 hash_inputs=ds["hash_inputs"])


def test_cohort_membership(ds):
    flags = {r["subject"]: (r["eligible"], r["primary"], r["sensitivity"]) for r in ds["manifest"]["subjects"]}
    assert flags == {
        "s0001": (True, True, True), "s0002": (True, False, True), "s0003": (False, False, False),
        "s0004": (False, False, False), "s0005": (True, True, True), "s0006": (True, True, True),
        "s0007": (True, True, True),
    }
    c = ds["manifest"]["counts"]
    assert (c["eligible_subjects"], c["primary_subjects"], c["boundary_touching_eligible"]) == (5, 4, 1)
    assert ds["manifest"]["support_pool"]["support_patients"] == ["case_002", "case_008"]
    assert ds["manifest"]["support_pool"]["dev_patients_excluded"] == ["case_003"]


def test_slice_grids_and_mask_eligibility_match_existing_rule(ds):
    r = row(ds, "s0001")
    assert r["candidate_step5"] == list(range(0, N_Z, 5))
    assert r["candidate_step17"] == [0, 17]
    m = np.asanyarray(nib.load(str(ds["extracted"] / "s0001/segmentations/spleen.nii.gz")).dataobj)
    img = np.random.default_rng(1).normal(0, 100, (N_XY, N_XY)).astype(np.float32)  # high-contrast dummy
    brute = [z for z in range(0, N_Z, 5) if C.validate_slice_pair(img, m[:, :, z])[0]]
    assert r["mask_eligible_step5"] == brute == [10, 15, 20]
    assert r["mask_eligible_step17"] == [17]
    assert ds["manifest"]["counts"]["primary_step5"]["candidate_slices"] == 4 * 6


# --------------------------------------------------------------------------- #
# geometry / streaming / orientation
# --------------------------------------------------------------------------- #
def _hdr(aff=RAS_15, shape=(N_XY, N_XY, N_Z)):
    h = nib.Nifti1Header()
    h.set_data_shape(shape)
    h.set_data_dtype(np.int16)
    h.set_sform(aff, code=2)
    h.set_zooms(tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3)))
    return h


def test_affine_validation():
    mimg = nib.Nifti1Image(np.zeros((N_XY, N_XY, N_Z), np.uint8), RAS_15)
    assert C.check_geometry(_hdr(), mimg)["ok"]
    shifted = RAS_15.copy(); shifted[0, 3] = 0.01
    assert "affine mismatch" in C.check_geometry(_hdr(shifted), mimg)["errors"][0]
    assert any("shape mismatch" in e for e in C.check_geometry(_hdr(shape=(N_XY, N_XY, N_Z + 1)), mimg)["errors"])
    lps = np.diag([-1.5, -1.5, 1.5, 1.0])
    errs = C.check_geometry(_hdr(lps), nib.Nifti1Image(np.zeros((N_XY, N_XY, N_Z), np.uint8), lps))["errors"]
    assert any("orientation" in e for e in errs)
    big = np.diag([2.0, 2.0, 2.0, 1.0])
    errs = C.check_geometry(_hdr(big), nib.Nifti1Image(np.zeros((N_XY, N_XY, N_Z), np.uint8), big))["errors"]
    assert any("spacing" in e for e in errs)
    within_tol = RAS_15.copy(); within_tol[0, 3] = 5e-4
    assert C.check_geometry(_hdr(within_tol), mimg)["ok"]


@pytest.mark.parametrize("subject", ["s0001", "s0005"])
def test_stream_matches_nibabel(ds, subject, tmp_path):
    with zipfile.ZipFile(ds["archive"]) as zf:
        f = tmp_path / "ct.nii.gz"
        f.write_bytes(zf.read(f"{subject}/ct.nii.gz"))
    ref = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32)
    with C.CTStream(ds["archive"], subject) as ct:
        for z, _, scaled in ct.slices():
            np.testing.assert_array_equal(scaled, ref[:, :, z])
        assert ct.finish() == 0
    if subject == "s0005":
        assert ref.min() == -1000 and ref.max() == 120  # scaling recovered HU


def test_outputs_orientation_resize_and_values(ds):
    assert run(ds, ["s0001"]) == 0
    m = np.asanyarray(nib.load(str(ds["extracted"] / "s0001/segmentations/spleen.nii.gz")).dataobj)
    with zipfile.ZipFile(ds["archive"]) as zf:
        raw = gzip.decompress(zf.read("s0001/ct.nii.gz"))
    ct = np.frombuffer(raw[352:], np.int16).reshape((N_XY, N_XY, N_Z), order="F").astype(np.float32)
    for z in (10, 15, 20):
        name = f"ts_s0001_slice_{z:03d}.png"
        img = cv2.imread(str(ds["out"] / "images" / name), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(str(ds["out"] / "masks" / name), cv2.IMREAD_UNCHANGED)
        assert img.shape == msk.shape == (512, 512) and img.dtype == msk.dtype == np.uint8
        assert set(np.unique(msk)) <= {0, 255}
        # independent re-implementation (does not call src.preprocessing)
        exp_m = cv2.resize(np.where(np.rot90(m[:, :, z], 1) == 1, 255, 0).astype(np.uint8), (512, 512),
                           interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(msk, exp_m)
        hu = np.rot90(np.clip(ct[:, :, z], -200, 300), 1)
        exp_i = cv2.resize(np.round((hu + 200) / 500 * 255).astype(np.uint8), (512, 512), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(img, exp_i)
        # alignment: spleen (120 HU -> 163) is brighter under the mask than the surrounding body (~40 HU -> 122)
        body = (msk == 0) & (img > 60)
        assert img[msk > 0].mean() > img[body].mean() + 20
    # rotation direction is detectable: un-rotated mask would NOT match
    wrong = cv2.resize(np.where(m[:, :, 15] == 1, 255, 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    assert not np.array_equal(wrong, cv2.imread(str(ds["out"] / "masks/ts_s0001_slice_015.png"), cv2.IMREAD_UNCHANGED))


# --------------------------------------------------------------------------- #
# logging, resume, interruption, duplicates, hashes, failures, guards
# --------------------------------------------------------------------------- #
def test_logging_every_candidate_slice(ds):
    assert run(ds, ["s0006"]) == 0
    log = json.loads((ds["out"] / "logs/s0006.json").read_text())
    r = row(ds, "s0006")
    assert [s["z"] for s in log["slices"]] == sorted(set(r["candidate_step5"]) | set(r["candidate_step17"]))
    for s in log["slices"]:
        assert s["reason"] in {"ok", "no_foreground", "low_contrast", "invalid_labels", "shape_mismatch"}
        assert ("file" in s) == s["valid"]
    z10 = next(s for s in log["slices"] if s["z"] == 10)
    assert z10["mask_eligible_step5"] and not z10["valid"] and z10["reason"] == "low_contrast"
    assert log["counts"]["step5"]["mask_eligible"] == log["counts"]["step5"]["processed"] + 1
    assert any("rejected on CT" in w for w in log["warnings"])
    assert not (ds["out"] / "images/ts_s0006_slice_010.png").exists()


def test_resume_skips_verified_without_rewriting(ds):
    assert run(ds, ["s0001"]) == 0
    files = sorted((ds["out"] / "images").iterdir()) + sorted((ds["out"] / "masks").iterdir())
    before = {f: f.stat().st_mtime_ns for f in files}
    assert run(ds, ["s0001"]) == 0
    assert {f: f.stat().st_mtime_ns for f in files} == before
    assert history(ds)[-1]["event"] == "skipped_verified"


def test_interrupted_run_resumes_to_identical_outputs(ds, tmp_path):
    r = row(ds, "s0001")
    msha = C.file_digest(ds["mpath"])
    with pytest.raises(KeyboardInterrupt):
        PR.process_subject(r, archive=ds["archive"], extracted=ds["extracted"], out_root=ds["out"], manifest_sha=msha,
                           _fail_after_slices=4)
    assert not (ds["out"] / "logs/s0001.done").exists()
    (ds["out"] / "images" / ".tmp_ts_s0001_slice_020.png.999").write_bytes(b"partial")  # stray temp file
    assert run(ds, ["s0001"]) == 0
    clean = tmp_path / "clean"
    PR.process_subject(r, archive=ds["archive"], extracted=ds["extracted"], out_root=clean, manifest_sha=msha)
    for kind in ("images", "masks"):
        a = {f.name: C.file_digest(f) for f in (ds["out"] / kind).glob("ts_*.png")}
        b = {f.name: C.file_digest(f) for f in (clean / kind).glob("ts_*.png")}
        assert a == b and len(a) == 4
    assert PR.verify_subject(ds["out"], "s0001")["verified_files"] == 8


def test_duplicate_conflict_never_overwrites(ds):
    target = ds["out"] / "images" / "ts_s0001_slice_010.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"someone else's file")
    assert run(ds, ["s0001"]) == 2
    assert target.read_bytes() == b"someone else's file"
    assert history(ds)[-1]["event"] == "FAILED" and "OutputConflict" in history(ds)[-1]["error"]
    assert not (ds["out"] / "logs/s0001.done").exists()


def test_tampered_output_detected_on_resume(ds):
    assert run(ds, ["s0001"]) == 0
    (ds["out"] / "masks/ts_s0001_slice_015.png").write_bytes(b"tampered")
    assert run(ds, ["s0001"]) == 2
    assert "hash mismatch" in history(ds)[-1]["error"]


def test_failed_subject_stops_run_and_is_recorded(ds):
    # s0007's CT stream is truncated inside the (unchanged, manifest-matching) archive
    assert run(ds, ["s0007", "s0001"]) == 2
    h = history(ds)
    assert h[-1]["subject"] == "s0007" and h[-1]["event"] == "FAILED" and "truncated" in h[-1]["error"]
    assert not (ds["out"] / "logs/s0007.done").exists()
    assert not (ds["out"] / "logs/s0001.json").exists()  # later subject NOT processed


def test_guards(ds):
    with pytest.raises(C.TSError, match="limit"):
        run(ds, ["s0001", "s0002", "s0005", "s0006"])
    with pytest.raises(C.TSError, match="duplicate"):
        run(ds, ["s0001", "s0001"])
    with pytest.raises(C.TSError, match="not an eligible"):
        run(ds, ["s0003"])


def test_code_hash_guard(ds, tmp_path):
    m = json.loads(ds["mpath"].read_text())
    m["input_hashes_sha256"]["src/preprocessing.py"] = "0" * 64
    p = tmp_path / "m2" / "cohort_manifest.json"
    BM.write_manifest(m, p)
    with pytest.raises(C.TSError, match="changed since the manifest"):
        run(ds, ["s0001"], manifest_path=p)


def test_atomic_write_leaves_no_partial_target(tmp_path, monkeypatch):
    target = tmp_path / "x.bin"

    def boom(*a, **k):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        C.atomic_write_bytes(target, b"data")
    assert not target.exists()
