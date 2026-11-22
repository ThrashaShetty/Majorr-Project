"""Adaptive-Enhanced experiment, step 2: GT-free selection strategies + patient-level stats.

Thresholds are fixed a priori (SAM defaults / geometric prior). Ground truth is used
only to SCORE outcomes and decision correctness, never to choose a threshold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project")
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_mask_binary  # noqa: E402

OUT = Path(__file__).resolve().parent
K = ["dice", "iou", "precision", "recall", "boundary_iou"]
f = pd.read_csv(OUT / "features.csv")
f["support_px"] = [int((load_mask_binary(ROOT / "data/masks" / f"{s}.png") > 0).sum()) for s in f.support]
f["box_area_vs_support"] = f.box_area / f.support_px

PIOU, STAB = 0.88, 0.95  # SamAutomaticMaskGenerator defaults
keep_box = {
    "A1 piou>=.88 & stab>=.95": (f.box_pred_iou >= PIOU) & (f.box_stability >= STAB),
    "A2 piou>=.88": f.box_pred_iou >= PIOU,
    "A3 stab>=.95": f.box_stability >= STAB,
    "A4 higher pred IoU": f.box_pred_iou >= f.enh_pred_iou,
    "A5 area within 0.5-2x support": f.box_area_vs_support.between(0.5, 2.0),
    "A6 A1 & A5": (f.box_pred_iou >= PIOU) & (f.box_stability >= STAB) & f.box_area_vs_support.between(0.5, 2.0),
}

methods = {"Baseline": pd.DataFrame({m: f[f"box_{m}"] for m in K}),
           "Enhanced v2": pd.DataFrame({m: f[f"enh_{m}"] for m in K})}
time_ms = {"Baseline": f.t_encode_ms + f.t_box_ms, "Enhanced v2": f.t_encode_ms + f.t_enh_extra_ms}
for name, kb in keep_box.items():
    methods[name] = pd.DataFrame({m: np.where(kb, f[f"box_{m}"], f[f"enh_{m}"]) for m in K})
    extra = f.t_enh_extra_ms if name.startswith("A4") else np.where(kb, 0.0, f.t_enh_extra_ms)
    time_ms[name] = f.t_encode_ms + f.t_box_ms + extra
methods["Oracle (GT, upper bound)"] = pd.DataFrame({m: np.where(f.box_dice >= f.enh_dice, f[f"box_{m}"], f[f"enh_{m}"]) for m in K})
time_ms["Oracle (GT, upper bound)"] = f.t_encode_ms + f.t_box_ms + f.t_enh_extra_ms

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 40)
print("parity with recorded v2 CSVs: box", f.parity_box.all(), "enh", f.parity_enh.all())

# ---------------- indicator diagnostics (GT used only to measure, not to set thresholds)
succ = (f.box_dice >= 0.5).astype(int)
print("\n== Box-only confidence indicators: AUC for Baseline success (Dice>=0.5); 0.5 = useless ==")
for c in ["box_pred_iou", "box_stability", "box_mm_agree", "box_area_ratio", "box_in_box_frac", "box_score_spread"]:
    auc = roc_auc_score(succ, f[c])
    print(f"  {c:<18} AUC={auc:.3f}  success median={f[c][succ == 1].median():.3f}  failure median={f[c][succ == 0].median():.3f}")
auc = roc_auc_score(succ, -np.abs(np.log(f.box_area_vs_support)))
print(f"  {'|log area/support|':<18} AUC={auc:.3f} (closer to support area = success)")
enh_better = (f.enh_dice > f.box_dice).astype(int)
print("  AUC of (enh_piou - box_piou) for 'Enhanced better':", round(roc_auc_score(enh_better, f.enh_pred_iou - f.box_pred_iou), 3))

# ---------------- slice-level summary
print("\n== Slice-level (53 slices) mean ± std ==")
rows = []
for name, df in methods.items():
    r = {"method": name}
    for m in K:
        r[m] = f"{df[m].mean():.3f} ± {df[m].std(ddof=0):.3f}"
    r["time_ms"] = f"{time_ms[name].mean():.0f} ± {time_ms[name].std(ddof=0):.0f}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

# ---------------- patient-level
print("\n== Patient-level mean Dice (each patient = mean of its slices) ==")
pat = pd.DataFrame({name: df.dice.groupby(f.patient).mean() for name, df in methods.items()})
pat.insert(0, "n_slices", f.groupby("patient").size())
print(pat.round(3).to_string())
print("\nMean of patient means:")
print(pat.drop(columns="n_slices").mean().round(3).to_string())

rng = np.random.default_rng(42)
pats = f.patient.unique()
def cluster_boot(a, b, n=10000):
    d = (a - b).groupby(f.patient).agg(["sum", "count"])
    idx = rng.integers(0, len(pats), (n, len(pats)))
    s, c = d["sum"].values[idx].sum(1), d["count"].values[idx].sum(1)
    return np.percentile(s / c, [2.5, 97.5])

print("\n== Paired comparisons: patient-level Wilcoxon (n=8 patients; min attainable p=0.0078) + patient cluster-bootstrap 95% CI of slice-mean Dice diff ==")
for name in methods:
    if name in ("Baseline",):
        continue
    for ref in ("Baseline", "Enhanced v2"):
        if name == ref:
            continue
        a, b = methods[name].dice, methods[ref].dice
        pa, pb = a.groupby(f.patient).mean(), b.groupby(f.patient).mean()
        try:
            p = wilcoxon(pa, pb).pvalue
        except ValueError:
            p = float("nan")
        lo, hi = cluster_boot(a, b)
        print(f"  {name:<30} vs {ref:<12} diff={a.mean() - b.mean():+.3f} CI[{lo:+.3f},{hi:+.3f}]  patients better/worse={int((pa > pb + 1e-9).sum())}/{int((pa < pb - 1e-9).sum())}  Wilcoxon p={p:.3f}")

# ---------------- selection behaviour and correctness
print("\n== Selection behaviour (correct = picked the output with higher GT Dice; |diff|<=0.01 counts as correct) ==")
diff = f.enh_dice - f.box_dice
for name, kb in keep_box.items():
    correct = np.where(diff.abs() <= 0.01, True, np.where(kb, diff < 0, diff > 0))
    succ_mask, fail_mask = f.box_dice >= 0.5, f.box_dice < 0.5
    d = methods[name].dice
    print(f"  {name:<30} box-only={int(kb.sum()):2d} enhanced={int((~kb).sum()):2d} correct={int(correct.sum())}/53 "
          f"| kept box on {int(kb[succ_mask].sum())}/22 baseline-success, activated Enh on {int((~kb)[fail_mask].sum())}/31 baseline-failure "
          f"| Dice on success slices={d[succ_mask].mean():.3f} failure slices={d[fail_mask].mean():.3f}")
for name in ("Baseline", "Enhanced v2", "Oracle (GT, upper bound)"):
    d = methods[name].dice
    print(f"  {name:<30} Dice on success slices={d[f.box_dice >= 0.5].mean():.3f} failure slices={d[f.box_dice < 0.5].mean():.3f}")

# ---------------- per-sample
print("\n== Per-sample ==")
ps = f[["sample", "box_dice", "enh_dice", "box_pred_iou", "box_stability", "enh_pred_iou", "box_area_vs_support"]].copy()
for name, kb in keep_box.items():
    ps[name.split()[0]] = np.where(kb, "B", "E")
print(ps.round(3).to_string(index=False))
ps.to_csv(OUT / "per_sample_decisions.csv", index=False)
pat.to_csv(OUT / "per_patient_dice.csv")
