"""Evaluate Baseline SAM, FoB SAM and Enhanced FoB SAM on the query split.

Usage (from any directory)::

    python run_evaluation.py                       # full query set
    python run_evaluation.py --max-queries 20      # quick CPU run
    python run_evaluation.py --device cuda

Protocol
--------
* Support / query lists come from ``python -m src.preprocessing`` (case-level split).
* Case-level disjointness is re-verified here; the run aborts on any overlap.
* Each query slice is paired with ONE support slice, used by all three methods:
  - ``nearest`` (default): the support slice whose downsampled (64x64,
    zero-mean, unit-norm) image is most similar (cosine) to the query image.
    Only the query *image* is used, never its ground truth, so no label leaks.
  - ``random``: a seeded random support slice.
* All methods share one SAM model instance and are run on identical samples.
* Every sample yields one CSV row per method; failures are kept with
  ``status=failed`` and an error message.
* ``inference_ms`` = support prototype time (FoB methods) + query encoding time
  + prompting/decoding time. Cached encodings contribute their originally
  measured time so methods remain comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import (  # noqa: E402
    CHARTS_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_SEED,
    OVERLAYS_DIR,
    QUERY_LIST,
    RESULTS_DIR,
    SUPPORT_LIST,
    configure_logging,
)
from src.metrics import METRIC_KEYS  # noqa: E402
from src.preprocessing import case_id_from_sample, read_list  # noqa: E402

logger = logging.getLogger("run_evaluation")

METHOD_FILES = {"baseline": "baseline_results.csv", "fob": "fob_results.csv", "enhanced": "enhanced_results.csv"}


def validate_split(support: list[tuple[Path, Path]], query: list[tuple[Path, Path]]) -> None:
    """Check files exist, pairs match by filename, and cases are disjoint."""
    if not support or not query:
        raise SystemExit("Support or query list is empty; run `python -m src.preprocessing` first.")
    for img, msk in support + query:
        if not img.is_file() or not msk.is_file():
            raise SystemExit(f"Missing file referenced by list: {img if not img.is_file() else msk}")
        if img.name != msk.name:
            raise SystemExit(f"Image/mask pairing mismatch: {img.name} vs {msk.name}")
    s_cases = {case_id_from_sample(i) for i, _ in support}
    q_cases = {case_id_from_sample(i) for i, _ in query}
    overlap = s_cases & q_cases
    if overlap:
        raise SystemExit(f"Data leakage: cases in both support and query: {sorted(overlap)}")
    logger.info("Split OK: %d support slices / %d cases, %d query slices / %d cases, no case overlap",
                len(support), len(s_cases), len(query), len(q_cases))


def _thumb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read image {path}")
    v = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


def pair_supports(support, query, strategy: str, seed: int):
    """Return a list of EvalSample pairing each query with one support slice."""
    from src.baseline import EvalSample

    if strategy == "nearest":
        bank = np.stack([_thumb(i) for i, _ in support])
        idx = [int(np.argmax(bank @ _thumb(q))) for q, _ in query]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(support), size=len(query)).tolist()
    return [EvalSample(q_img, q_msk, support[j][0], support[j][1]) for (q_img, q_msk), j in zip(query, idx, strict=False)]


def summarise(rows_by_method: dict[str, list[dict]]) -> pd.DataFrame:
    recs = []
    for m, rows in rows_by_method.items():
        df = pd.DataFrame(rows)
        ok = df[df["status"] == "success"]
        rec = {"method": m, "evaluated": len(df), "successful": len(ok), "failed": int((df["status"] == "failed").sum())}
        for k in (*METRIC_KEYS, "inference_ms"):
            rec[f"{k}_mean"] = float(ok[k].astype(float).mean()) if len(ok) else float("nan")
            rec[f"{k}_std"] = float(ok[k].astype(float).std(ddof=0)) if len(ok) else float("nan")
        recs.append(rec)
    return pd.DataFrame(recs)


def print_table(summary: pd.DataFrame) -> None:
    head = f"{'Method':<10} {'n':>4} {'ok':>4} {'fail':>4} " + " ".join(f"{k:>17}" for k in METRIC_KEYS) + f" {'ms':>16}"
    print("\n" + head)
    print("-" * len(head))
    for _, r in summary.iterrows():
        cells = " ".join(f"{r[k + '_mean']:.4f} ± {r[k + '_std']:.4f}".rjust(17) for k in METRIC_KEYS)
        print(f"{r['method']:<10} {r['evaluated']:>4} {r['successful']:>4} {r['failed']:>4} {cells} "
              f"{r['inference_ms_mean']:>8.1f} ± {r['inference_ms_std']:<6.1f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    ap.add_argument("--support-list", type=Path, default=SUPPORT_LIST)
    ap.add_argument("--query-list", type=Path, default=QUERY_LIST)
    ap.add_argument("--max-queries", type=int, default=None, help="Evaluate only the first N query slices")
    ap.add_argument("--support-strategy", choices=("nearest", "random"), default="nearest")
    ap.add_argument("--methods", nargs="+", choices=tuple(METHOD_FILES), default=list(METHOD_FILES))
    ap.add_argument("--no-adaptive", action="store_true", help="Use fixed n_points=8 for FoB methods")
    ap.add_argument("--save-overlays", action="store_true", help="Write per-sample overlays to results/overlays/")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help="Output directory for CSVs, summary, charts/ and overlays/ (default: results/)")
    args = ap.parse_args(argv)
    configure_logging()
    results_dir = args.results_dir if args.results_dir.is_absolute() else PROJECT_ROOT / args.results_dir
    charts_dir = CHARTS_DIR if results_dir == RESULTS_DIR else results_dir / "charts"
    overlays_dir = OVERLAYS_DIR if results_dir == RESULTS_DIR else results_dir / "overlays"

    import torch

    from src.baseline import BaselineSegmentor, CheckpointNotFoundError, SamEngine, evaluate_samples, write_results_csv
    from src.fob.fob_pipeline import FoBPipeline
    from src.visualizer import plot_boxplots, plot_metrics_comparison

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        support, query = read_list(args.support_list), read_list(args.query_list)
    except FileNotFoundError as exc:
        logger.error("%s - run `python -m src.preprocessing` first.", exc)
        return 1
    validate_split(support, query)
    if args.max_queries:
        query = query[: args.max_queries]
    samples = pair_supports(support, query, args.support_strategy, args.seed)

    try:
        engine = SamEngine(args.checkpoint, args.device)
    except CheckpointNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    runners = {
        "baseline": BaselineSegmentor(engine=engine),
        "fob": FoBPipeline(engine=engine, mode="fob", use_adaptive=not args.no_adaptive, uncertainty="none", seed=args.seed),
        "enhanced": FoBPipeline(engine=engine, mode="enhanced", use_adaptive=not args.no_adaptive, uncertainty="none", seed=args.seed),
    }
    overlay_dir = overlays_dir if args.save_overlays else None
    rows: dict[str, list[dict]] = {m: [] for m in args.methods}
    logger.info("Evaluating %d samples x %d methods on %s", len(samples), len(args.methods), engine.device)
    for i, s in enumerate(samples, 1):
        for m in args.methods:  # same sample through every method -> query embedding reused
            r = runners[m]
            rows[m].extend(evaluate_samples(
                lambda x, r=r: r.predict(x.query_image, x.support_image, x.support_mask, x.query_mask),
                [s], m, overlay_dir))
        if i % 10 == 0 or i == len(samples):
            logger.info("Progress %d/%d samples", i, len(samples))

    results_dir.mkdir(parents=True, exist_ok=True)
    for m, r in rows.items():
        out = write_results_csv(r, results_dir / METHOD_FILES[m])
        logger.info("Wrote %s", out)

    summary = summarise(rows)
    print_table(summary)
    meta = {"samples": len(samples), "support_strategy": args.support_strategy, "seed": args.seed,
            "device": str(engine.device), "adaptive": not args.no_adaptive,
            "summary": json.loads(summary.to_json(orient="records"))}
    (results_dir / "evaluation_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    frames = {m: pd.DataFrame(r) for m, r in rows.items()}
    if any((df["status"] == "success").any() for df in frames.values()):
        args_df = [frames.get(m) for m in ("baseline", "fob", "enhanced")]
        plot_metrics_comparison(*args_df, charts_dir / "metrics_comparison.png")
        plot_boxplots(*args_df, charts_dir / "metrics_boxplots.png")
        logger.info("Charts written to %s", charts_dir)
    else:
        logger.error("No successful predictions for any method; charts not generated")
        return 2
    for _, r in summary.iterrows():
        if r["failed"]:
            print(f"NOTE: {r['method']}: {int(r['failed'])} of {int(r['evaluated'])} samples FAILED (see CSV 'error').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
