"""Visualisation utilities: overlays, comparison panels, uncertainty maps, charts.

Images handled here are NumPy arrays in **RGB** order (H, W, 3) uint8, or
single-channel grayscale (H, W). Conversion to BGR happens only when writing
with OpenCV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend: safe on servers and in tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import METRIC_KEYS, to_binary  # noqa: E402

logger = logging.getLogger(__name__)

METHOD_COLORS: dict[str, str] = {
    "Baseline SAM": "#8c96a8",
    "FoB SAM": "#ff9f43",
    "Enhanced FoB SAM": "#00d4aa",
}
METRIC_LABELS: dict[str, str] = {
    "dice": "Dice",
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "boundary_iou": "Boundary IoU",
}


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB copy of a grayscale or RGB image."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    raise ValueError(f"Unsupported image shape {arr.shape}")


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend ``color`` over foreground pixels of ``mask`` and draw its contour.

    Args:
        image: Grayscale or RGB image.
        mask: Binary mask with the same height/width.
        color: RGB overlay colour.
        alpha: Overlay opacity in [0, 1].

    Returns:
        RGB uint8 overlay image.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    rgb = to_rgb(image)
    m = to_binary(mask)
    if m.shape != rgb.shape[:2]:
        raise ValueError(f"Mask shape {m.shape} does not match image shape {rgb.shape[:2]}")
    out = rgb.astype(np.float32)
    col = np.array(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * col
    out = np.clip(out, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 1)
    return out


def create_uncertainty_heatmap(uncertainty_map: np.ndarray) -> np.ndarray:
    """Convert a float uncertainty map (e.g. per-pixel std) to an RGB heatmap.

    Values are min-max scaled for display; a constant map renders as all-cold.
    """
    u = np.asarray(uncertainty_map, dtype=np.float32)
    if u.ndim != 2:
        raise ValueError(f"Uncertainty map must be 2-D, got {u.shape}")
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(u.min()), float(u.max())
    scaled = np.zeros_like(u) if hi - lo < 1e-12 else (u - lo) / (hi - lo)
    u8 = (scaled * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_comparison_grid(
    image: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    overlay: np.ndarray,
    uncertainty_map: np.ndarray | None = None,
) -> plt.Figure:
    """Build a 4-panel (or 5-panel with uncertainty) Matplotlib comparison figure.

    Panels: input | ground truth | prediction | overlay [| uncertainty].
    A missing ground truth is rendered as an explicit "not available" panel.
    """
    panels = 5 if uncertainty_map is not None else 4
    fig, axes = plt.subplots(1, panels, figsize=(4 * panels, 4.4), facecolor="#0a0e1a")
    img_rgb = to_rgb(image)
    items: list[tuple[str, np.ndarray | None, str | None]] = [
        ("Input image", img_rgb, None),
        ("Ground truth", None if gt_mask is None else to_binary(gt_mask).astype(np.uint8) * 255, "gray"),
        ("Prediction", to_binary(pred_mask).astype(np.uint8) * 255, "gray"),
        ("Overlay", to_rgb(overlay), None),
    ]
    for ax, (title, data, cmap) in zip(axes, items, strict=False):
        ax.set_title(title, color="white", fontsize=12)
        ax.axis("off")
        if data is None:
            ax.set_facecolor("#12182b")
            ax.text(0.5, 0.5, "not available", color="#8c96a8", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
    if uncertainty_map is not None:
        ax = axes[4]
        u = np.asarray(uncertainty_map, dtype=np.float32)
        im = ax.imshow(u, cmap="inferno")
        ax.set_title("Uncertainty (pixel std)", color="white", fontsize=12)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
    # Legend describing overlay colours
    from matplotlib.patches import Patch

    fig.legend(
        handles=[Patch(color="#00ff00", label="Predicted foreground (overlay)")],
        loc="lower center", frameon=False, labelcolor="white",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def save_comparison_grid(grid: plt.Figure | np.ndarray, output_path: str | Path) -> Path:
    """Save a comparison figure (or RGB array) as PNG and close the figure."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(grid, np.ndarray):
        if not cv2.imwrite(str(out), cv2.cvtColor(to_rgb(grid), cv2.COLOR_RGB2BGR)):
            raise OSError(f"Failed to write {out}")
    else:
        grid.savefig(out, dpi=110, facecolor=grid.get_facecolor())
        plt.close(grid)
    return out


def _named_frames(
    baseline_df: pd.DataFrame | None, fob_df: pd.DataFrame | None, enhanced_df: pd.DataFrame | None
) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for name, df in (("Baseline SAM", baseline_df), ("FoB SAM", fob_df), ("Enhanced FoB SAM", enhanced_df)):
        if df is None or df.empty:
            continue
        if "status" in df.columns:
            df = df[df["status"] == "success"]
        if not df.empty:
            frames.append((name, df))
    if not frames:
        raise ValueError("No successful evaluation rows available to plot")
    return frames


def plot_metrics_comparison(
    baseline_df: pd.DataFrame | None,
    fob_df: pd.DataFrame | None,
    enhanced_df: pd.DataFrame | None,
    output_path: str | Path,
) -> Path:
    """Grouped bar chart (mean +/- std) of the five metrics for each method."""
    frames = _named_frames(baseline_df, fob_df, enhanced_df)
    metrics: Sequence[str] = METRIC_KEYS
    x = np.arange(len(metrics))
    width = 0.8 / len(frames)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (name, df) in enumerate(frames):
        means = [float(df[m].mean()) for m in metrics]
        stds = [float(df[m].std(ddof=0)) for m in metrics]
        bars = ax.bar(x + (i - (len(frames) - 1) / 2) * width, means, width, yerr=stds, capsize=3,
                      label=f"{name} (n={len(df)})", color=METHOD_COLORS[name], edgecolor="black", linewidth=0.5)
        for b, v in zip(bars, means, strict=False):
            ax.text(b.get_x() + b.get_width() / 2, min(v + 0.02, 1.02), f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score (mean ± std)")
    ax.set_title("Segmentation metrics by method (Task09 Spleen query set)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_boxplots(
    baseline_df: pd.DataFrame | None,
    fob_df: pd.DataFrame | None,
    enhanced_df: pd.DataFrame | None,
    output_path: str | Path,
) -> Path:
    """One box plot per metric showing per-sample distributions for each method."""
    frames = _named_frames(baseline_df, fob_df, enhanced_df)
    fig, axes = plt.subplots(1, len(METRIC_KEYS), figsize=(4 * len(METRIC_KEYS), 4.8), sharey=True)
    for ax, metric in zip(axes, METRIC_KEYS, strict=False):
        data = [df[metric].astype(float).to_numpy() for _, df in frames]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        for patch, (name, _) in zip(bp["boxes"], frames, strict=False):
            patch.set_facecolor(METHOD_COLORS[name])
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(frames) + 1), [n.replace(" SAM", "") for n, _ in frames], rotation=20)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=METHOD_COLORS[n], label=n) for n, _ in frames],
               loc="lower center", ncol=len(frames), frameon=False)
    fig.suptitle("Per-sample metric distributions")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
