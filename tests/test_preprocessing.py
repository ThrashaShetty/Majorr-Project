"""Tests for src.preprocessing using synthetic arrays and synthetic NIfTI volumes."""

import json

import cv2
import nibabel as nib
import numpy as np
import pytest

from src.preprocessing import (
    case_id_from_sample,
    label_to_binary_mask,
    normalize_ct,
    read_list,
    resize_image,
    resize_mask,
    run_preprocessing,
    split_cases,
    validate_slice_pair,
)


def test_normalize_ct_clips_and_scales():
    hu = np.array([[-1000.0, -200.0], [50.0, 2000.0]], dtype=np.float32)
    out = normalize_ct(hu)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0 and out[0, 1] == 0
    assert out[1, 1] == 255
    assert out[1, 0] == round((50 + 200) / 500 * 255)


def test_normalize_ct_handles_nan():
    out = normalize_ct(np.array([[np.nan, 300.0]], dtype=np.float32))
    assert out.tolist() == [[0, 255]]


def test_resize_image_shape():
    img = np.random.default_rng(0).integers(0, 255, (300, 200), dtype=np.uint8)
    assert resize_image(img).shape == (512, 512)


def test_mask_resize_is_nearest_and_binary():
    lbl = np.zeros((100, 100), np.int16)
    lbl[30:70, 30:70] = 1
    lbl[0:5, 0:5] = 2  # other label -> background
    mask = resize_mask(label_to_binary_mask(lbl))
    assert mask.shape == (512, 512)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[:20, :20].max() == 0


def test_validate_slice_pair_rules():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 100, (64, 64)).astype(np.float32)
    lbl = np.zeros((64, 64), np.uint8)
    assert validate_slice_pair(img, lbl)[1] == "no_foreground"
    lbl[10:40, 10:40] = 1
    ok, reason, im, mk, ratio = validate_slice_pair(img, lbl)
    assert ok and reason == "ok" and im.shape == mk.shape == (512, 512) and 0 < ratio < 1
    assert validate_slice_pair(np.full((64, 64), -1000, np.float32), lbl)[1] == "low_contrast"
    assert validate_slice_pair(img, np.full((64, 64), 3, np.uint8))[1] == "invalid_labels"
    assert validate_slice_pair(img, np.zeros((32, 32), np.uint8))[1] == "shape_mismatch"


def test_split_cases_disjoint_and_deterministic():
    ids = list(range(1, 21))
    s1, q1 = split_cases(ids, 0.8, seed=7)
    s2, q2 = split_cases(ids, 0.8, seed=7)
    assert (s1, q1) == (s2, q2)
    assert len(s1) == 16 and len(q1) == 4
    assert not set(s1) & set(q1) and set(s1) | set(q1) == set(ids)


def _write_case(dirpath, case_id, rng):
    shape = (48, 48, 20)
    img = rng.normal(40, 120, shape).astype(np.float32)
    lbl = np.zeros(shape, np.uint8)
    lbl[15:35, 15:35, 3:17] = 1
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(img, affine), str(dirpath / "imagesTr" / f"spleen_{case_id}.nii.gz"))
    nib.save(nib.Nifti1Image(lbl, affine), str(dirpath / "labelsTr" / f"spleen_{case_id}.nii.gz"))


def test_full_pipeline_on_synthetic_nifti(tmp_path):
    raw = tmp_path / "raw" / "Task09_Spleen"
    (raw / "imagesTr").mkdir(parents=True)
    (raw / "labelsTr").mkdir(parents=True)
    (raw / "imagesTr" / "._spleen_1.nii.gz").write_bytes(b"junk")  # macOS artefact must be ignored
    rng = np.random.default_rng(0)
    for cid in (1, 2, 3, 4, 5):
        _write_case(raw, cid, rng)
    out = tmp_path / "data"
    report = run_preprocessing(
        raw_dir=tmp_path / "raw", images_dir=out / "images", masks_dir=out / "masks",
        support_list=out / "support_list.txt", query_list=out / "query_list.txt", report_dir=out, seed=1,
    )
    assert report.total_cases == 5
    assert report.total_sampled_slices == 5 * 4  # z = 0,5,10,15
    assert report.valid_slices == 5 * 3  # z = 5,10,15 contain spleen; z=0 excluded
    assert report.excluded_by_reason == {"no_foreground": 5}
    assert len(report.support_cases) == 4 and len(report.query_cases) == 1
    assert not set(report.support_cases) & set(report.query_cases)
    assert json.loads((out / "preprocessing_report.json").read_text())["valid_slices"] == 15

    # list files: absolute resolution relative to project root is tested via raw lines
    lines = (out / "query_list.txt").read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        img_rel, mask_rel = line.split("\t")
        assert img_rel.split("/")[-1] == mask_rel.split("/")[-1]
    for path in (out / "images").glob("*.png"):
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        mk = cv2.imread(str(out / "masks" / path.name), cv2.IMREAD_UNCHANGED)
        assert im.shape == mk.shape == (512, 512) and im.dtype == np.uint8
        assert set(np.unique(mk).tolist()) <= {0, 255}
        assert case_id_from_sample(path) in range(1, 6)
    # raw data untouched
    assert (raw / "imagesTr" / "._spleen_1.nii.gz").read_bytes() == b"junk"


def test_read_list_rejects_malformed(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("only_one_column\n")
    with pytest.raises(ValueError):
        read_list(p)
    with pytest.raises(FileNotFoundError):
        read_list(tmp_path / "missing.txt")
