"""Phase B figures (descriptive; read-only on inputs; write-once output directory).

Inputs : eval/raw_rows_primary.csv, eval/raw_rows_sensitivity.csv, analysis/<phaseB>/analysis.json
Output : figures/<phaseB>/  (refuses if it exists)
  fig1_external_primary_dice_box.png        per-patient mean Dice, 4 methods (74 patients)
  fig2_external_sensitivity_dice_box.png    same, sensitivity cohort (90 patients)
  fig3_external_primary_paired_diff.png     per-patient ΔDice for H1..H4, mean + 95 % bootstrap CI, zero line
  fig4_external_sensitivity_paired_diff.png same, sensitivity cohort
  figures_manifest.json                     input + output SHA-256, settings
Per-patient values are recomputed from raw rows (unweighted mean over the patient's slices, as SAP v1) and must
match analysis.json's patient_mean_diff to 1e-9, else the script stops. CIs and Holm p are read from analysis.json
(no resampling here).
Usage: python -B -I make_phaseB_figures.py --analysis-dir results/external_ts/analysis/phaseB_20261124T052029Z
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
R = SCRIPTS.parent
METHODS = [("baseline", "Baseline", "#2a78d6"), ("enhanced", "Enhanced v2", "#eb6834"),
           ("a5", "A5", "#1baf7a"), ("single", "Single-mask", "#eda100")]
COMPS = [("H1", "a5", "baseline", "A5\n− Baseline"), ("H2", "a5", "enhanced", "A5\n− Enhanced v2"),
         ("H3", "enhanced", "baseline", "Enhanced v2\n− Baseline"), ("H4", "single", "enhanced", "Single-mask\n− Enhanced v2")]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEED = 0  # jitter only (visual); not a statistical stream


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fmt_p(p: float) -> str:
    if p >= 0.01:
        return f"{p:.2f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$"


def fmt_d(x: float) -> str:
    s = f"{x:+.3f}" if abs(x) >= 0.0005 else f"{x:+.4f}"
    return s.replace("-", "−")


def box(pp: pd.DataFrame, cohort: str, n_slices: int, out: Path):
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=300)
    data = [pp[f"{k}_dice"].to_numpy() for k, _, _ in METHODS]
    bp = ax.boxplot(data, positions=range(4), widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=INK2, linewidth=1),
                    capprops=dict(color=INK2, linewidth=1), boxprops=dict(linewidth=1, edgecolor=INK2))
    for patch, (_, _, c) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(c); patch.set_alpha(0.28)
    for i, ((_, _, c), y) in enumerate(zip(METHODS, data)):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(y)), y, s=11, color=c, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter([i], [y.mean()], marker="D", s=34, color="white", edgecolor=INK, linewidth=1.2, zorder=4)
    ax.set_xticks(range(4), [m for _, m, _ in METHODS], fontsize=10, color=INK)
    ax.set_ylim(0, 1); ax.set_ylabel("Per-patient mean Dice", color=INK2, fontsize=10)
    ax.set_title(f"External TotalSegmentator — {cohort} cohort ({len(pp)} patients, {n_slices} slices)",
                 fontsize=10.5, color=INK, loc="left")
    ax.text(0, -0.17, "Box: median and IQR; whiskers: 1.5×IQR. Dots: every patient (none hidden). ◇ = mean.",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    style(ax); fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)


def paired(pp: pd.DataFrame, tests: dict, cohort: str, out: Path) -> dict:
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    diffs = [pp[f"{a}_dice"].to_numpy() - pp[f"{b}_dice"].to_numpy() for _, a, b, _ in COMPS]
    lim = float(np.ceil(max(np.abs(d).max() for d in diffs) * 10) / 10)
    ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    check = {}
    for i, ((h, _, _, _), d) in enumerate(zip(COMPS, diffs)):
        t = tests[h]
        if abs(d.mean() - t["patient_mean_diff"]) > 1e-9:
            raise SystemExit(f"{cohort} {h}: recomputed mean {d.mean()} != analysis.json {t['patient_mean_diff']}")
        check[h] = float(d.mean())
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(d)), d, s=10, color=MUTED, alpha=0.55, edgecolor="none", zorder=2)
        lo, hi = t["patient_diff_CI95"]
        ax.errorbar([i], [t["patient_mean_diff"]], yerr=[[t["patient_mean_diff"] - lo], [hi - t["patient_mean_diff"]]],
                    fmt="D", color=INK, markersize=6, markerfacecolor="white", markeredgewidth=1.4, capsize=6,
                    elinewidth=1.6, zorder=4)
    labels = [f"{h}: {lab}\nΔ {fmt_d(tests[h]['patient_mean_diff'])}\nHolm p {fmt_p(tests[h]['holm_p'])}"
              for h, _, _, lab in COMPS]
    ax.set_xticks(range(4), labels, fontsize=8, color=INK)
    ax.set_ylim(-lim, lim); ax.set_ylabel("Per-patient Δ mean Dice", color=INK2, fontsize=10)
    ax.set_title(f"Paired differences — {cohort} cohort ({len(pp)} patients)", fontsize=10.5, color=INK, loc="left")
    fig.text(0.01, 0.01, "Grey dots: every patient's difference. ◇ with bars: mean and 95 % patient-level bootstrap CI\n"
             "(from analysis.json). Dashed line: no difference. H1 is the pre-registered primary test.",
             fontsize=7, color=MUTED, va="bottom")
    style(ax); fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, facecolor="white"); plt.close(fig)
    return {"y_limit": lim, "recomputed_patient_mean_diff": check}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    adir = a.analysis_dir.resolve()
    out = R / "figures" / adir.name
    if out.exists():
        raise SystemExit(f"{out} exists — refusing to overwrite")
    rep = json.loads((adir / "analysis.json").read_text())
    inputs = {"analysis.json": adir / "analysis.json", "raw_rows_primary.csv": R / "eval/raw_rows_primary.csv",
              "raw_rows_sensitivity.csv": R / "eval/raw_rows_sensitivity.csv"}
    if rep["inputs_sha256"]["raw_rows_primary.csv"] != sha(inputs["raw_rows_primary.csv"]) or \
       rep["inputs_sha256"]["raw_rows_sensitivity.csv"] != sha(inputs["raw_rows_sensitivity.csv"]):
        raise SystemExit("raw rows changed since the analysis — refusing")
    out.mkdir(parents=True)
    info = {}
    for n, (cohort, key, f_box, f_pair) in enumerate([
            ("primary", "raw_rows_primary.csv", "fig1_external_primary_dice_box.png", "fig3_external_primary_paired_diff.png"),
            ("sensitivity", "raw_rows_sensitivity.csv", "fig2_external_sensitivity_dice_box.png", "fig4_external_sensitivity_paired_diff.png")]):
        df = pd.read_csv(inputs[key])
        pp = df.groupby("patient")[[f"{k}_dice" for k, _, _ in METHODS]].mean().reset_index()
        tests = {t["comparison"].split()[0]: t for t in rep[cohort]["confirmatory_family"]}
        box(pp, cohort, len(df), out / f_box)
        info[cohort] = {"n_patients": len(pp), "n_slices": len(df), **paired(pp, tests, cohort, out / f_pair)}
    files = sorted(p.name for p in out.glob("*.png"))
    man = {"source_analysis": str(adir.relative_to(R.parents[1])), "inputs_sha256": {k: sha(v) for k, v in inputs.items()},
           "script_sha256": sha(Path(__file__)), "outputs_sha256": {f: sha(out / f) for f in files},
           "settings": {"dpi": 300, "box_ylim": [0, 1], "box_fliers": "not drawn separately; every patient plotted as a dot",
                        "jitter_seed": SEED, "palette": {m: c for _, m, c in METHODS}}, "cohorts": info}
    (out / "figures_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"out_dir": str(out), "files": files + ["figures_manifest.json"], "cohorts": info}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
