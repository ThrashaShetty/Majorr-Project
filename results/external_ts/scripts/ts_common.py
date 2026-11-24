"""Shared helpers for the TotalSegmentator (TS) external evaluation.

This module is SEPARATE from the MSD pipeline. It never modifies or monkeypatches
anything in ``src/``; it only imports the existing, validated preprocessing
functions and calls them unchanged:

    validate_slice_pair  (orientation via orient_slice, HU window via normalize_ct,
                          resize to 512x512, label checks, contrast check,
                          >= MIN_FG_PIXELS foreground after resizing)

Everything TS-specific lives here: ZIP streaming, header/geometry checks,
write-once atomic outputs, hashing and slice-grid definitions.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterator

import nibabel as nib
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[2]  # .../medical_sam_project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402  (existing functions, used unchanged)
    HU_MAX,
    HU_MIN,
    MIN_FG_PIXELS,
    SLICE_STEP,
    TARGET_SIZE,
    label_to_binary_mask,
    orient_slice,
    resize_mask,
    validate_slice_pair,
)

# --------------------------------------------------------------------------- #
# Frozen constants (changing any of these is a protocol change, not a fix)
# --------------------------------------------------------------------------- #
DATASET_ROOT = PROJECT_ROOT.parent / "external_datasets" / "totalsegmentator_small_v201"
ARCHIVE = DATASET_ROOT / "archive" / "Totalsegmentator_dataset_small_v201.zip"
EXTRACTED = DATASET_ROOT / "extracted"
PREPROCESSED = DATASET_ROOT / "preprocessed"
RESULTS = PROJECT_ROOT / "results" / "external_ts"
MANIFEST = RESULTS / "cohort_manifest.json"

ARCHIVE_MD5 = "6b5524af4b15e6ba06ef2d700c0c73e0"
ARCHIVE_SIZE = 3_244_617_817
ZENODO_RECORD = "10047263"
DATASET_VERSION = "2.0.1"

PRIMARY_STEP = 5  # == existing SLICE_STEP (asserted below)
SENSITIVITY_STEP = 17
EXPECTED_N_SUBJECTS = 102
EXPECTED_N_ELIGIBLE = 91
EXPECTED_N_PRIMARY = 74
EXPECTED_ORIENTATION = "RAS"
EXPECTED_ZOOM_MM = 1.5
AFFINE_ATOL = 1e-3
# HU diagnostic (WARN only, not a gate; a gating rule has not been approved)
AIR_MODE_RANGE = (-1100, -900)

assert SLICE_STEP == PRIMARY_STEP, "existing SLICE_STEP changed - protocol no longer matches"
assert TARGET_SIZE == (512, 512)
assert MIN_FG_PIXELS == 50
assert (HU_MIN, HU_MAX) == (-200.0, 300.0)


class TSError(RuntimeError):
    """Any validation failure. Subjects are never skipped silently on this."""


class OutputConflict(TSError):
    """An output path already holds DIFFERENT content (duplicate/overwrite prevented)."""


# --------------------------------------------------------------------------- #
# Hashing and atomic, write-once outputs
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path, algo: str = "sha256", chunk: int = 4 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary sibling then os.replace (atomic on the same volume)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_once(path: Path, data: bytes) -> tuple[str, str]:
    """Write ``data`` unless ``path`` already holds identical bytes.

    Returns (status, sha256) with status in {"written", "exists_identical"}.
    Raises OutputConflict when the existing file differs - never overwrites.
    """
    digest = sha256_bytes(data)
    if path.exists():
        existing = file_digest(path)
        if existing != digest:
            raise OutputConflict(f"{path} exists with different content ({existing[:12]} != {digest[:12]})")
        return "exists_identical", digest
    atomic_write_bytes(path, data)
    return "written", digest


def json_bytes(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, fixed separators, trailing newline)."""
    return (json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n").encode("utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------- #
# Slice grids and mask-only eligibility (uses the existing functions verbatim)
# --------------------------------------------------------------------------- #
def grid(n_slices: int, step: int) -> list[int]:
    """Fixed grid along array axis 2 starting at z=0 (same as process_case)."""
    return list(range(0, n_slices, step))


def mask_fg_after_resize(label_slice: np.ndarray) -> int:
    """Foreground pixel count exactly as validate_slice_pair computes it."""
    mask = resize_mask(label_to_binary_mask(orient_slice(label_slice)), TARGET_SIZE)
    return int((mask > 0).sum())


def output_name(subject: str, z: int) -> str:
    return f"ts_{subject}_slice_{z:03d}.png"


# --------------------------------------------------------------------------- #
# NIfTI header / geometry
# --------------------------------------------------------------------------- #
def orientation(affine: np.ndarray) -> str:
    return "".join(nib.aff2axcodes(affine))


def check_geometry(ct_hdr: nib.Nifti1Header, mask_img: nib.Nifti1Image) -> dict[str, Any]:
    """Hard geometry/scaling checks. Returns a report; ``errors`` non-empty => fail."""
    errors: list[str] = []
    ct_shape = tuple(int(s) for s in ct_hdr.get_data_shape())
    if len(ct_shape) == 4 and ct_shape[3] == 1:
        ct_shape = ct_shape[:3]
    if len(ct_shape) != 3:
        errors.append(f"CT is not 3-D: {ct_shape}")
    ct_aff = ct_hdr.get_best_affine()
    m_aff = mask_img.affine
    m_shape = tuple(int(s) for s in mask_img.shape)
    if ct_shape != m_shape:
        errors.append(f"shape mismatch CT {ct_shape} vs mask {m_shape}")
    max_diff = float(np.abs(ct_aff - m_aff).max())
    if not np.allclose(ct_aff, m_aff, atol=AFFINE_ATOL):
        errors.append(f"affine mismatch (max abs diff {max_diff:.4g} > {AFFINE_ATOL})")
    ct_or, m_or = orientation(ct_aff), orientation(m_aff)
    if ct_or != EXPECTED_ORIENTATION or m_or != EXPECTED_ORIENTATION:
        errors.append(f"orientation CT {ct_or} / mask {m_or} != {EXPECTED_ORIENTATION}")
    zooms = [float(z) for z in ct_hdr.get_zooms()[:3]]
    if not np.allclose(zooms, EXPECTED_ZOOM_MM, atol=1e-3):
        errors.append(f"voxel spacing {zooms} != {EXPECTED_ZOOM_MM} mm isotropic")
    dt = ct_hdr.get_data_dtype()
    if dt.kind not in "iuf":
        errors.append(f"unsupported CT dtype {dt}")
    slope, inter = ct_hdr.get_slope_inter()
    for name, v in (("slope", slope), ("intercept", inter)):
        if v is not None and not np.isfinite(v):
            errors.append(f"non-finite scl_{name} {v}")
    if slope == 0:
        errors.append("scl_slope == 0")
    rot = ct_aff[:3, :3] / np.linalg.norm(ct_aff[:3, :3], axis=0)
    tilt = float(np.degrees(np.arccos(np.clip(np.abs(np.diag(rot)).min(), -1.0, 1.0))))
    return {
        "ok": not errors, "errors": errors, "ct_shape": ct_shape, "mask_shape": m_shape,
        "ct_orientation": ct_or, "mask_orientation": m_or, "zooms_mm": zooms,
        "affine_max_abs_diff": max_diff, "ct_affine": np.round(ct_aff, 5).tolist(),
        "dtype": str(dt), "scl_slope": slope, "scl_inter": inter,
        "sform_code": int(ct_hdr["sform_code"]), "qform_code": int(ct_hdr["qform_code"]),
        "max_axis_tilt_deg": round(tilt, 3), "non_square_inplane": ct_shape[0] != ct_shape[1],
    }


# --------------------------------------------------------------------------- #
# Streaming a CT straight from the ZIP (never extracted to disk)
# --------------------------------------------------------------------------- #
class CTStream:
    """Sequential slice reader for ``<subject>/ct.nii.gz`` inside the archive.

    Memory: one axial slice at a time. Values are scaled exactly as nibabel would
    (stored * slope + intercept; identity when the header is unscaled) and returned
    as float32, matching ``np.asarray(img.dataobj, dtype=np.float32)[:, :, z]``.
    """

    def __init__(self, zip_path: Path, subject: str):
        self.member = f"{subject}/ct.nii.gz"
        self._zf = zipfile.ZipFile(zip_path)
        try:
            self.info = self._zf.getinfo(self.member)
        except KeyError as e:
            self._zf.close()
            raise TSError(f"{self.member} not in archive") from e
        self._raw = self._zf.open(self.member)  # zipfile verifies CRC-32 at end of stream
        self._gz = gzip.GzipFile(fileobj=self._raw)
        head = self._gz.read(352)
        if len(head) < 352:
            raise TSError(f"{self.member}: truncated header")
        self.header = nib.Nifti1Header.from_fileobj(io.BytesIO(head))
        vox_offset = int(self.header["vox_offset"])
        if vox_offset < 352:
            raise TSError(f"{self.member}: unexpected vox_offset {vox_offset}")
        self._gz.read(vox_offset - 352)
        shape = self.header.get_data_shape()
        self.nx, self.ny, self.nz = (int(s) for s in shape[:3])
        self.dtype = self.header.get_data_dtype()
        slope, inter = self.header.get_slope_inter()
        self.slope = 1.0 if slope is None else float(slope)
        self.inter = 0.0 if inter is None else float(inter)
        self.next_z = 0

    def slices(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield (z, stored_slice, scaled_float32_slice) for z = 0..nz-1, in order."""
        n = self.nx * self.ny * self.dtype.itemsize
        for z in range(self.next_z, self.nz):
            buf = self._gz.read(n)
            if len(buf) != n:
                raise TSError(f"{self.member}: truncated data at z={z} ({len(buf)}/{n} bytes)")
            stored = np.frombuffer(buf, self.dtype).reshape((self.nx, self.ny), order="F")
            scaled = (stored.astype(np.float64) * self.slope + self.inter).astype(np.float32)
            self.next_z = z + 1
            yield z, stored, scaled

    def finish(self) -> int:
        """Read to end of stream (triggers gzip + ZIP CRC checks). Returns trailing bytes."""
        trailing = len(self._gz.read())
        return trailing

    def close(self) -> None:
        for obj in (self._gz, self._raw, self._zf):
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "CTStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class HUStats:
    """Streaming integer histogram of STORED values plus scaled-value summaries."""

    LO, HI = -4096, 8191

    def __init__(self, slope: float, inter: float):
        self.slope, self.inter = slope, inter
        self.hist = np.zeros(self.HI - self.LO + 1, np.int64)
        self.below = self.above = self.n = 0
        self.vmin: float | None = None
        self.vmax: float | None = None

    def add(self, stored: np.ndarray) -> None:
        v = np.rint(stored).astype(np.int64).ravel()
        self.n += v.size
        self.vmin = float(v.min()) if self.vmin is None else min(self.vmin, float(v.min()))
        self.vmax = float(v.max()) if self.vmax is None else max(self.vmax, float(v.max()))
        lo, hi = v < self.LO, v > self.HI
        self.below += int(lo.sum())
        self.above += int(hi.sum())
        ok = ~(lo | hi)
        self.hist += np.bincount(v[ok] - self.LO, minlength=self.hist.size)

    def summary(self) -> dict[str, Any]:
        vals = np.arange(self.LO, self.HI + 1) * self.slope + self.inter
        cdf = np.cumsum(self.hist) / max(1, self.hist.sum())
        pct = {str(p): float(vals[min(len(vals) - 1, np.searchsorted(cdf, p / 100))]) for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)}
        def mode(a: float, b: float) -> float | None:
            sel = (vals >= a) & (vals <= b)
            return float(vals[sel][np.argmax(self.hist[sel])]) if self.hist[sel].sum() else None
        air = mode(*AIR_MODE_RANGE)
        in_win = float(self.hist[(vals >= HU_MIN) & (vals <= HU_MAX)].sum() / max(1, self.n))
        warnings = []
        if air is None or self.hist[(vals >= AIR_MODE_RANGE[0]) & (vals <= AIR_MODE_RANGE[1])].sum() == 0:
            warnings.append("no voxels in air range [-1100,-900]")
        if self.below:
            warnings.append(f"{self.below} voxels stored below {self.LO} (padding/sentinel; clipped to {HU_MIN} by the window)")
        return {
            "n_voxels": self.n, "stored_min": self.vmin, "stored_max": self.vmax,
            "scaled_min": None if self.vmin is None else self.vmin * self.slope + self.inter,
            "scaled_max": None if self.vmax is None else self.vmax * self.slope + self.inter,
            "scaled_percentiles": pct, "air_mode_scaled": air,
            "frac_in_window": round(in_win, 5), "n_below_hist": self.below, "n_above_hist": self.above,
            "hu_warnings": warnings,
        }
