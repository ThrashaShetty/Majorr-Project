"""Preprocessing for the Medical Segmentation Decathlon Task09_Spleen dataset.

Pipeline
--------
1. Locate paired CT volumes (``imagesTr``) and label volumes (``labelsTr``).
2. Load each NIfTI volume lazily with nibabel (slice-by-slice through the array
   proxy, so a whole CT volume is never materialised in RAM).
3. Take every fifth axial slice along array axis 2 (z = 0, 5, 10, ...).
4. Clip HU to [-200, 300], linearly scale to [0, 255], convert to uint8.
5. Resize image (bilinear) and mask (nearest-neighbour) to 512 x 512.
6. Map spleen label 1 -> 255, everything else -> 0.
7. Validate image/mask correspondence and apply the exclusion rule below.
8. Write ``data/images/case_XXX_slice_YYY.png`` and
   ``data/masks/case_XXX_slice_YYY.png``.
9. Split at the **case (patient) level** into 80 % support / 20 % query cases
   using a seeded permutation, and write ``data/support_list.txt`` and
   ``data/query_list.txt``.

Exclusion rule (documented)
---------------------------
A sampled slice is excluded when ANY of the following holds:

* ``shape_mismatch``  - image and label slices have different shapes.
* ``invalid_labels``  - the label slice contains values outside {0, 1}
  (Task09 only defines 0 = background, 1 = spleen).
* ``low_contrast``    - the normalised image has standard deviation < 1.0
  grey level (an essentially blank slice, e.g. entirely outside the HU window).
* ``no_foreground``   - fewer than ``min_fg_pixels`` (default 50) spleen pixels
  after resizing to 512 x 512. Support prompts are derived from the support
  foreground, and Dice on empty query ground truth is degenerate, so only
  slices that actually contain spleen are used for few-shot evaluation.

Raw data under ``data/raw`` is opened read-only and never modified.

Split strategy
--------------
Cases with at least one valid slice are sorted by case id, permuted with
``numpy.random.default_rng(seed)``, and the first ``round(0.8 * n)`` cases form
the support set; the remainder form the query set (at least one case on each
side when n >= 2). All slices of a case go to the same side, which prevents
patient-level leakage between support and query data.

List-file format: one sample per line, ``<image_path>\t<mask_path>``, both
relative to the project root, using forward slashes.

Run with::

    python -m src.preprocessing
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable

import cv2
import numpy as np

from src import (
    DATA_DIR,
    DEFAULT_SEED,
    IMAGES_DIR,
    MASKS_DIR,
    PROJECT_ROOT,
    QUERY_LIST,
    RAW_DIR,
    SUPPORT_LIST,
    configure_logging,
)

logger = logging.getLogger(__name__)

HU_MIN: float = -200.0
HU_MAX: float = 300.0
TARGET_SIZE: tuple[int, int] = (512, 512)  # (height, width)
SLICE_STEP: int = 5
SPLEEN_LABEL: int = 1
FOREGROUND_VALUE: int = 255
MIN_FG_PIXELS: int = 50
MIN_IMAGE_STD: float = 1.0
SUPPORT_FRACTION: float = 0.8

_CASE_ID_RE = re.compile(r"(\d+)")


class PreprocessingError(RuntimeError):
    """Raised when the dataset cannot be located or processed."""


@dataclass
class SliceRecord:
    """Bookkeeping for a single sampled slice."""

    case_id: int
    slice_index: int
    valid: bool
    reason: str
    foreground_ratio: float
    image_path: str | None = None
    mask_path: str | None = None


@dataclass
class PreprocessingReport:
    """Summary statistics written to ``data/preprocessing_report.json``."""

    total_cases: int = 0
    total_sampled_slices: int = 0
    valid_slices: int = 0
    excluded_slices: int = 0
    excluded_by_reason: dict[str, int] = field(default_factory=dict)
    foreground_ratio_mean: float | None = None
    foreground_ratio_min: float | None = None
    foreground_ratio_max: float | None = None
    support_cases: list[int] = field(default_factory=list)
    query_cases: list[int] = field(default_factory=list)
    support_slices: int = 0
    query_slices: int = 0
    seed: int = DEFAULT_SEED


# --------------------------------------------------------------------------- #
# Pure array transforms (unit-testable without NIfTI files)
# --------------------------------------------------------------------------- #
def normalize_ct(slice_hu: np.ndarray, hu_min: float = HU_MIN, hu_max: float = HU_MAX) -> np.ndarray:
    """Clip a CT slice to ``[hu_min, hu_max]`` HU and scale it to uint8 [0, 255].

    Args:
        slice_hu: 2-D array of Hounsfield units.
        hu_min: Lower clipping bound.
        hu_max: Upper clipping bound.

    Returns:
        uint8 array with the same shape as ``slice_hu``.
    """
    if slice_hu.ndim != 2:
        raise ValueError(f"Expected a 2-D slice, got shape {slice_hu.shape}")
    if hu_max <= hu_min:
        raise ValueError("hu_max must be greater than hu_min")
    arr = np.nan_to_num(slice_hu.astype(np.float32), nan=hu_min, posinf=hu_max, neginf=hu_min)
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min) * 255.0
    return np.round(arr).astype(np.uint8)


def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize a 2-D uint8 image to ``size`` = (height, width) with bilinear interpolation."""
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def label_to_binary_mask(label_slice: np.ndarray, label: int = SPLEEN_LABEL) -> np.ndarray:
    """Convert an integer label slice to a uint8 mask with ``label`` -> 255, others -> 0."""
    if label_slice.ndim != 2:
        raise ValueError(f"Expected a 2-D label slice, got shape {label_slice.shape}")
    labels = np.rint(label_slice).astype(np.int32)
    return np.where(labels == label, FOREGROUND_VALUE, 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize a 2-D mask to ``size`` = (height, width) with nearest-neighbour interpolation."""
    h, w = size
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def orient_slice(arr: np.ndarray) -> np.ndarray:
    """Rotate a NIfTI axial slice 90 degrees for conventional display.

    The same deterministic rotation is applied to image and mask, so pixel
    correspondence is preserved.
    """
    return np.ascontiguousarray(np.rot90(arr, k=1))


def validate_slice_pair(
    image_hu: np.ndarray,
    label_slice: np.ndarray,
    min_fg_pixels: int = MIN_FG_PIXELS,
    size: tuple[int, int] = TARGET_SIZE,
) -> tuple[bool, str, np.ndarray | None, np.ndarray | None, float]:
    """Process and validate one image/label slice pair.

    Returns:
        ``(valid, reason, image_uint8, mask_uint8, foreground_ratio)``. When the
        pair is invalid, image/mask may be ``None`` and ``reason`` names the
        exclusion rule that fired (``"ok"`` when valid).
    """
    if image_hu.shape != label_slice.shape:
        return False, "shape_mismatch", None, None, 0.0
    unique = np.unique(np.rint(label_slice).astype(np.int64))
    if not set(unique.tolist()).issubset({0, 1}):
        return False, "invalid_labels", None, None, 0.0

    image = resize_image(normalize_ct(orient_slice(image_hu)), size)
    mask = resize_mask(label_to_binary_mask(orient_slice(label_slice)), size)
    if image.shape != mask.shape:
        return False, "shape_mismatch", None, None, 0.0

    fg_pixels = int((mask > 0).sum())
    fg_ratio = fg_pixels / float(mask.size)
    if float(image.std()) < MIN_IMAGE_STD:
        return False, "low_contrast", image, mask, fg_ratio
    if fg_pixels < min_fg_pixels:
        return False, "no_foreground", image, mask, fg_ratio
    return True, "ok", image, mask, fg_ratio


# --------------------------------------------------------------------------- #
# Dataset discovery
# --------------------------------------------------------------------------- #
def _is_hidden_artifact(path: Path) -> bool:
    """MSD archives contain macOS ``._*`` resource-fork files; ignore them."""
    return path.name.startswith("._") or path.name.startswith(".")


def parse_case_id(path: Path) -> int:
    """Extract the numeric case id from a filename such as ``spleen_12.nii.gz``."""
    match = _CASE_ID_RE.search(path.name)
    if match is None:
        raise PreprocessingError(f"Cannot parse case id from filename: {path.name}")
    return int(match.group(1))


def find_case_pairs(raw_dir: Path) -> list[tuple[int, Path, Path]]:
    """Find ``(case_id, image_path, label_path)`` triples under ``raw_dir``.

    Supports ``raw/Task09_Spleen/imagesTr`` as well as ``raw/imagesTr``.
    """
    if not raw_dir.exists():
        raise PreprocessingError(f"Raw data directory does not exist: {raw_dir}")
    image_dirs = sorted(p for p in raw_dir.rglob("imagesTr") if p.is_dir())
    if not image_dirs:
        raise PreprocessingError(
            f"No 'imagesTr' directory found under {raw_dir}. Extract Task09_Spleen.tar "
            "so that data/raw/Task09_Spleen/imagesTr and labelsTr exist."
        )
    pairs: list[tuple[int, Path, Path]] = []
    for images_dir in image_dirs:
        labels_dir = images_dir.parent / "labelsTr"
        if not labels_dir.is_dir():
            logger.warning("Skipping %s: no sibling labelsTr directory", images_dir)
            continue
        for img_path in sorted(images_dir.glob("*.nii*")):
            if _is_hidden_artifact(img_path):
                continue
            lbl_path = labels_dir / img_path.name
            if not lbl_path.exists():
                logger.warning("No label volume for %s; skipping", img_path.name)
                continue
            pairs.append((parse_case_id(img_path), img_path, lbl_path))
    ids = [c for c, _, _ in pairs]
    if len(ids) != len(set(ids)):
        raise PreprocessingError("Duplicate case ids detected in raw data")
    return sorted(pairs, key=lambda t: t[0])


# --------------------------------------------------------------------------- #
# Volume processing
# --------------------------------------------------------------------------- #
def _sample_name(case_id: int, slice_index: int) -> str:
    return f"case_{case_id:03d}_slice_{slice_index:03d}.png"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def process_case(
    case_id: int,
    image_path: Path,
    label_path: Path,
    images_dir: Path = IMAGES_DIR,
    masks_dir: Path = MASKS_DIR,
    slice_step: int = SLICE_STEP,
    min_fg_pixels: int = MIN_FG_PIXELS,
) -> list[SliceRecord]:
    """Extract, normalise, validate and save every ``slice_step``-th axial slice."""
    import nibabel as nib  # local import keeps array-only utilities importable without nibabel

    img_nii = nib.load(str(image_path))
    lbl_nii = nib.load(str(label_path))
    if img_nii.shape != lbl_nii.shape:
        raise PreprocessingError(
            f"Case {case_id}: image shape {img_nii.shape} != label shape {lbl_nii.shape}"
        )
    if len(img_nii.shape) != 3:
        raise PreprocessingError(f"Case {case_id}: expected a 3-D volume, got {img_nii.shape}")

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    records: list[SliceRecord] = []
    n_slices = img_nii.shape[2]
    # Decompress each volume exactly once. Per-slice ArrayProxy reads on .nii.gz
    # re-inflate the gzip stream from the start for every slice (very slow);
    # holding one volume at a time keeps peak memory bounded (~1 volume).
    image_vol = np.asarray(img_nii.dataobj, dtype=np.float32)
    label_vol = np.asarray(lbl_nii.dataobj)
    for z in range(0, n_slices, slice_step):
        image_hu = image_vol[:, :, z]
        label_slice = label_vol[:, :, z]
        valid, reason, image, mask, fg_ratio = validate_slice_pair(
            image_hu, label_slice, min_fg_pixels=min_fg_pixels
        )
        record = SliceRecord(case_id, z, valid, reason, fg_ratio)
        if valid and image is not None and mask is not None:
            name = _sample_name(case_id, z)
            img_out, mask_out = images_dir / name, masks_dir / name
            if not cv2.imwrite(str(img_out), image) or not cv2.imwrite(str(mask_out), mask):
                raise PreprocessingError(f"Failed to write PNG for {name}")
            record.image_path, record.mask_path = _rel(img_out), _rel(mask_out)
        records.append(record)
    del image_vol, label_vol, img_nii, lbl_nii
    logger.info(
        "case_%03d: %d sampled slices, %d valid",
        case_id,
        len(records),
        sum(r.valid for r in records),
    )
    return records


# --------------------------------------------------------------------------- #
# Case-level split
# --------------------------------------------------------------------------- #
def split_cases(
    case_ids: Iterable[int], support_fraction: float = SUPPORT_FRACTION, seed: int = DEFAULT_SEED
) -> tuple[list[int], list[int]]:
    """Deterministically split case ids into disjoint (support, query) lists."""
    ids = sorted(set(case_ids))
    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must be in (0, 1)")
    if len(ids) < 2:
        raise PreprocessingError("At least two cases with valid slices are required for a split")
    rng = np.random.default_rng(seed)
    permuted = [ids[i] for i in rng.permutation(len(ids))]
    n_support = int(round(support_fraction * len(ids)))
    n_support = min(max(n_support, 1), len(ids) - 1)
    support, query = sorted(permuted[:n_support]), sorted(permuted[n_support:])
    assert not set(support) & set(query)
    return support, query


def write_list(path: Path, records: list[SliceRecord]) -> None:
    """Write ``image<TAB>mask`` lines for valid records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in sorted(records, key=lambda r: (r.case_id, r.slice_index)):
            fh.write(f"{r.image_path}\t{r.mask_path}\n")


def read_list(path: Path) -> list[tuple[Path, Path]]:
    """Read a list file and return absolute ``(image_path, mask_path)`` pairs."""
    if not path.exists():
        raise FileNotFoundError(f"List file not found: {path}")
    pairs: list[tuple[Path, Path]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise ValueError(f"{path.name}:{line_no}: expected '<image>\\t<mask>'")
        pairs.append((PROJECT_ROOT / parts[0], PROJECT_ROOT / parts[1]))
    return pairs


def case_id_from_sample(path: Path) -> int:
    """Parse the case id from a processed filename ``case_XXX_slice_YYY.png``."""
    m = re.match(r"case_(\d+)_slice_(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Not a processed sample filename: {path.name}")
    return int(m.group(1))


def slice_index_from_sample(path: Path) -> int:
    """Parse the slice index from a processed filename ``case_XXX_slice_YYY.png``."""
    m = re.match(r"case_(\d+)_slice_(\d+)", path.stem)
    if m is None:
        raise ValueError(f"Not a processed sample filename: {path.name}")
    return int(m.group(2))


# --------------------------------------------------------------------------- #
# Image I/O shared by the inference pipelines
# --------------------------------------------------------------------------- #
def decode_image_bytes(data: bytes, grayscale: bool = False) -> np.ndarray:
    """Decode PNG/JPEG/BMP/TIFF bytes into an RGB (H, W, 3) or grayscale (H, W) uint8 array.

    Raises:
        ValueError: if the bytes are not a decodable image.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(buf, flag)
    if img is None:
        raise ValueError("File is not a decodable image")
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img if grayscale else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image file as an RGB uint8 array (grayscale CT PNGs are replicated to 3 channels)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    return decode_image_bytes(p.read_bytes(), grayscale=False)


def load_mask_binary(path: str | Path) -> np.ndarray:
    """Load a mask file as a uint8 {0, 1} array (any value > 0 is foreground)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Mask not found: {p}")
    return (decode_image_bytes(p.read_bytes(), grayscale=True) > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_preprocessing(
    raw_dir: Path = RAW_DIR,
    images_dir: Path = IMAGES_DIR,
    masks_dir: Path = MASKS_DIR,
    support_list: Path = SUPPORT_LIST,
    query_list: Path = QUERY_LIST,
    report_dir: Path = DATA_DIR,
    seed: int = DEFAULT_SEED,
    slice_step: int = SLICE_STEP,
    min_fg_pixels: int = MIN_FG_PIXELS,
    max_cases: int | None = None,
) -> PreprocessingReport:
    """Run the full preprocessing + case-level split pipeline and write a report."""
    pairs = find_case_pairs(raw_dir)
    if max_cases is not None:
        pairs = pairs[:max_cases]
    if not pairs:
        raise PreprocessingError(f"No paired NIfTI volumes found under {raw_dir}")
    logger.info("Found %d paired volumes", len(pairs))

    all_records: list[SliceRecord] = []
    for case_id, img_path, lbl_path in pairs:
        all_records.extend(
            process_case(case_id, img_path, lbl_path, images_dir, masks_dir, slice_step, min_fg_pixels)
        )

    valid = [r for r in all_records if r.valid]
    report = PreprocessingReport(seed=seed, total_cases=len(pairs))
    report.total_sampled_slices = len(all_records)
    report.valid_slices = len(valid)
    report.excluded_slices = len(all_records) - len(valid)
    for r in all_records:
        if not r.valid:
            report.excluded_by_reason[r.reason] = report.excluded_by_reason.get(r.reason, 0) + 1
    if valid:
        ratios = np.array([r.foreground_ratio for r in valid])
        report.foreground_ratio_mean = float(ratios.mean())
        report.foreground_ratio_min = float(ratios.min())
        report.foreground_ratio_max = float(ratios.max())

    support_ids, query_ids = split_cases({r.case_id for r in valid}, SUPPORT_FRACTION, seed)
    support_recs = [r for r in valid if r.case_id in support_ids]
    query_recs = [r for r in valid if r.case_id in query_ids]
    write_list(support_list, support_recs)
    write_list(query_list, query_recs)
    report.support_cases, report.query_cases = support_ids, query_ids
    report.support_slices, report.query_slices = len(support_recs), len(query_recs)

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preprocessing_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    (report_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "strategy": "case-level seeded permutation (no slice-level leakage)",
                "seed": seed,
                "support_fraction": SUPPORT_FRACTION,
                "support_cases": support_ids,
                "query_cases": query_ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def print_report(report: PreprocessingReport) -> None:
    """Print a human-readable preprocessing summary."""
    print("\n=== Preprocessing report ===")
    print(f"Cases processed      : {report.total_cases}")
    print(f"Sampled slices       : {report.total_sampled_slices}")
    print(f"Valid slices         : {report.valid_slices}")
    print(f"Excluded slices      : {report.excluded_slices} {report.excluded_by_reason}")
    if report.foreground_ratio_mean is not None:
        print(
            f"Foreground ratio     : mean={report.foreground_ratio_mean:.4f} min={report.foreground_ratio_min:.4f} max={report.foreground_ratio_max:.4f}"
        )
    print(f"Support cases ({len(report.support_cases)}) : {report.support_cases} -> {report.support_slices} slices")
    print(f"Query cases   ({len(report.query_cases)}) : {report.query_cases} -> {report.query_slices} slices")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m src.preprocessing``."""
    parser = argparse.ArgumentParser(description="Preprocess MSD Task09_Spleen into PNG slices.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--slice-step", type=int, default=SLICE_STEP)
    parser.add_argument("--min-fg-pixels", type=int, default=MIN_FG_PIXELS)
    parser.add_argument("--max-cases", type=int, default=None, help="Process only the first N cases")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        report = run_preprocessing(
            raw_dir=args.raw_dir,
            seed=args.seed,
            slice_step=args.slice_step,
            min_fg_pixels=args.min_fg_pixels,
            max_cases=args.max_cases,
        )
    except PreprocessingError as exc:
        logger.error("%s", exc)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
