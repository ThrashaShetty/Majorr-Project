"""Background-Aware Prompting for Enhanced Medical Image Segmentation Using SAM.

Research-prototype package. All filesystem locations are resolved relative to
the project root (the parent directory of this package), never relative to the
terminal's current working directory.

This software is an academic research prototype. It is NOT clinically
validated and must not be used for medical diagnosis or treatment decisions.
"""

from __future__ import annotations

import logging
from pathlib import Path

__version__ = "0.1.0"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
IMAGES_DIR: Path = DATA_DIR / "images"
MASKS_DIR: Path = DATA_DIR / "masks"
SUPPORT_LIST: Path = DATA_DIR / "support_list.txt"
QUERY_LIST: Path = DATA_DIR / "query_list.txt"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHARTS_DIR: Path = RESULTS_DIR / "charts"
OVERLAYS_DIR: Path = RESULTS_DIR / "overlays"
CHECKPOINTS_DIR: Path = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT: Path = CHECKPOINTS_DIR / "sam_vit_b_01ec64.pth"
DEFAULT_DB_PATH: Path = PROJECT_ROOT / "experiments.db"
TMP_UPLOADS_DIR: Path = PROJECT_ROOT / "tmp_uploads"

DEFAULT_SEED: int = 42

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured, timestamped logging once for CLI entry points."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    else:
        root.setLevel(level)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
