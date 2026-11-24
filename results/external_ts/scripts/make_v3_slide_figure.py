"""Slide 11 figure: v3 held-out per-patient mean Dice (descriptive; read-only inputs; write-once output).

Input : results/v3/raw_holdout_rows.csv (156 slices, 33 patients) — per-patient means are checked against
        results/v3/adaptive/summary.json (mean_of_patient_means_dice) to 1e-12, else the script stops.
Output: results/external_ts/figures/slides_v3/fig5_v3_holdout_dice_box.png + figures_manifest.json (refuses if exists)
Same visual encoding as make_phaseB_figures.py (palette, every patient as a dot, mean diamond, y 0–1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
P = SCRIPTS.parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(SCRIPTS))  # python -I drops the script directory from sys.path
from make_phaseB_figures import INK, INK2, METHODS, MUTED, SEED, style  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    raw, summ = P / "results/v3/raw_holdout_rows.csv", P / "results/v3/adaptive/summary.json"
    out = P / "results/external_ts/figures/slides_v3"
    if out.exists():
        raise SystemExit(f"{out} exists — refusing to overwrite")
    df = pd.read_csv(raw)
    pp = df.groupby("patient")[[f"{k}_dice" for k, _, _ in METHODS]].mean()
    ref = json.loads(summ.read_text())["mean_of_patient_means_dice"]
    for k, _, _ in METHODS:
        if abs(pp[f"{k}_dice"].mean() - ref[k]) > 1e-12:
            raise SystemExit(f"{k}: per-patient mean {pp[f'{k}_dice'].mean()} != summary.json {ref[k]}")
    out.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=300)
    data = [pp[f"{k}_dice"].to_numpy() for k, _, _ in METHODS]
    bp = ax.boxplot(data, positions=range(4), widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=INK2, linewidth=1),
                    capprops=dict(color=INK2, linewidth=1), boxprops=dict(linewidth=1, edgecolor=INK2))
    for patch, (_, _, c) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(c); patch.set_alpha(0.28)
    for i, ((_, _, c), y) in enumerate(zip(METHODS, data)):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(y)), y, s=14, color=c, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter([i], [y.mean()], marker="D", s=34, color="white", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.set_xticks(range(4), [m for _, m, _ in METHODS], fontsize=10, color=INK)
    ax.set_ylim(0, 1); ax.set_ylabel("Per-patient mean Dice", color=INK2, fontsize=10)
    ax.set_title(f"v3 held-out MSD Task09 ({len(pp)} patients, {len(df)} slices)", fontsize=10.5, color=INK, loc="left")
    ax.text(0, -0.17, "Box: median and IQR; whiskers: 1.5×IQR. Dots: every patient (none hidden). ◇ = mean.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    style(ax); fig.tight_layout()
    f = out / "fig5_v3_holdout_dice_box.png"
    fig.savefig(f, facecolor="white"); plt.close(fig)
    man = {"inputs_sha256": {"results/v3/raw_holdout_rows.csv": sha(raw), "results/v3/adaptive/summary.json": sha(summ)},
           "script_sha256": sha(Path(__file__)), "outputs_sha256": {f.name: sha(f)},
           "patient_means": {k: float(pp[f"{k}_dice"].mean()) for k, _, _ in METHODS},
           "n_patients": len(pp), "n_slices": len(df)}
    (out / "figures_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
