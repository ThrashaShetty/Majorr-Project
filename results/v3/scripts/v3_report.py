"""Build results/v3/{adaptive,single_mask} deliverables from the held-out raw rows.

Pre-registered (fixed before looking at held-out results):
  primary   : A5 vs Baseline, patient-level paired Wilcoxon on per-patient mean Dice
  secondary : A5 vs Enhanced v2, Enhanced v2 vs Baseline, Single-mask vs Enhanced v2 (multimask)
  correction: Holm over all 4 tests
  CIs       : patient cluster bootstrap (10,000 resamples, seed 42)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project")
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_image_rgb  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW = HERE / "holdout" / "raw_rows.csv"
V3 = ROOT / "results" / "v3"
K = ["dice", "iou", "precision", "recall", "boundary_iou"]
for p in (ROOT / "results" / "v3",):
    assert ROOT / "results" != p and ROOT / "results" / "v2" != p

df = pd.read_csv(RAW).sort_values("query_image").reset_index(drop=True)
assert len(df) == 156, len(df)
assert (df.patient != df.support_patient).all(), "same-patient pairing"
dev = {f"case_{c:03d}" for c in json.load(open(ROOT / "data/split_summary.json"))["query_cases"]}
assert not (set(df.patient) | set(df.support_patient)) & dev, "dev patient leaked into held-out"
rng = np.random.default_rng(42)
pats = df.patient.unique()


def summary(prefix, ms_col):
    r = {}
    for m in K:
        r[m] = {"mean": float(df[f"{prefix}_{m}"].mean()), "std": float(df[f"{prefix}_{m}"].std(ddof=0))}
    r["time_ms"] = {"mean": float(df[ms_col].mean()), "median": float(df[ms_col].median()), "std": float(df[ms_col].std(ddof=0))}
    return r


def per_patient(prefixes):
    out = pd.DataFrame({"n_slices": df.groupby("patient").size()})
    for p in prefixes:
        for m in K:
            out[f"{p}_{m}"] = df.groupby("patient")[f"{p}_{m}"].mean()
    return out


def compare(a, b, label):
    da, db = df[f"{a}_dice"], df[f"{b}_dice"]
    pa, pb = da.groupby(df.patient).mean(), db.groupby(df.patient).mean()
    diff_p = pa - pb
    d = (da - db).groupby(df.patient).agg(["sum", "count"])
    idx = rng.integers(0, len(pats), (10000, len(pats)))
    slice_ci = np.percentile(d["sum"].values[idx].sum(1) / d["count"].values[idx].sum(1), [2.5, 97.5])
    pat_ci = np.percentile(diff_p.values[idx].mean(1), [2.5, 97.5])
    nz = diff_p[diff_p.abs() > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
    return {"comparison": label, "slice_mean_diff": float((da - db).mean()), "slice_diff_CI95": [float(x) for x in slice_ci],
            "patient_mean_diff": float(diff_p.mean()), "patient_diff_CI95": [float(x) for x in pat_ci],
            "patients_better": int((diff_p > 0.005).sum()), "patients_worse": int((diff_p < -0.005).sum()),
            "patients_tied(|d|<=0.005)": int((diff_p.abs() <= 0.005).sum()), "wilcoxon_p": p}


tests = [compare("a5", "baseline", "A5 vs Baseline (PRIMARY)"), compare("a5", "enhanced", "A5 vs Enhanced v2"),
         compare("enhanced", "baseline", "Enhanced v2 vs Baseline"), compare("single", "enhanced", "Single-mask vs Enhanced v2 (multimask)")]
order = np.argsort([t["wilcoxon_p"] for t in tests])
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, tests[i]["wilcoxon_p"] * (len(tests) - rank)))
    tests[i]["holm_p"] = running

succ = df.baseline_dice >= 0.5
subsets = {}
for p in ("baseline", "enhanced", "a5", "single"):
    subsets[p] = {"baseline_success_n": int(succ.sum()), "dice_on_baseline_success": float(df.loc[succ, f"{p}_dice"].mean()),
                  "baseline_failure_n": int((~succ).sum()), "dice_on_baseline_failure": float(df.loc[~succ, f"{p}_dice"].mean())}
diff = df.enhanced_dice - df.baseline_dice
keep = df.a5_choice == "box"
correct = np.where(diff.abs() <= 0.01, True, np.where(keep, diff < 0, diff > 0))
a5_sel = {"box_only": int(keep.sum()), "enhanced_activated": int((~keep).sum()), "correct": int(correct.sum()), "of": len(df),
          "wrong_kept_box_when_enhanced_better": int((keep & (diff > 0.01)).sum()),
          "wrong_activated_enhanced_when_box_better": int((~keep & (diff < -0.01)).sum()),
          "kept_box_on_baseline_success": int((keep & succ).sum()), "activated_on_baseline_failure": int((~keep & ~succ).sum()),
          "oracle_dice": float(np.maximum(df.baseline_dice, df.enhanced_dice).mean())}

config = {"test_set": "33 support patients / 156 slices (never scored before)",
          "dev_set_excluded": sorted(dev), "support_pairing": "leave-one-patient-out nearest (64x64 thumbnail cosine, run_evaluation._thumb) among the other 32 held-out patients",
          "checkpoint": "checkpoints/sam_vit_b_01ec64.pth", "device": "cpu", "seed": 42, "adaptive_points": True,
          "enhanced": "production src/fob/fob_pipeline.py (v2, multimask_output=True)",
          "a5_rule": "keep box-only if 0.5 <= box_mask_area / support_mask_area <= 2.0, else Enhanced v2 (fixed, not tuned)",
          "single_mask": "identical Enhanced v2 prompts, SamPredictor.predict(multimask_output=False)",
          "statistics": "primary A5 vs Baseline; patient-level Wilcoxon signed-rank on per-patient mean Dice; Holm over 4 tests; patient cluster bootstrap 10k (seed 42)",
          "main_source_modified": False}


def draw(img, mask, gt, title, color):
    v = img.copy()
    v[mask] = (0.55 * v[mask] + 0.45 * np.array(color)).astype(np.uint8)
    cv2.drawContours(v, cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 0), 1)
    cv2.putText(v, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return v


for sub in ("adaptive", "single_mask"):
    (V3 / sub / "overlays").mkdir(parents=True, exist_ok=True)
for _, r in df.iterrows():
    z = np.load(HERE / "holdout" / "masks" / f"{Path(r.query_image).stem}.npz")
    img = load_image_rgb(ROOT / "data/images" / r.query_image)
    x0, y0, x1, y1 = z["box"].astype(int)
    img_b = img.copy(); cv2.rectangle(img_b, (x0, y0), (x1, y1), (255, 255, 0), 1)
    a5m = z["baseline"] if r.a5_choice == "box" else z["enhanced"]
    panels = [draw(img_b, z["baseline"], z["gt"], f"Baseline {r.baseline_dice:.2f}", (255, 0, 0)),
              draw(img_b, z["enhanced"], z["gt"], f"Enhanced v2 {r.enhanced_dice:.2f}", (0, 0, 255)),
              draw(img_b, a5m, z["gt"], f"A5 [{r.a5_choice}] {r.a5_dice:.2f}", (255, 0, 255))]
    cv2.imwrite(str(V3 / "adaptive" / "overlays" / f"{Path(r.query_image).stem}.png"), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
    img_p = img_b.copy()
    for (x, y), l in zip(z["points"].reshape(-1, 2), z["labels"]):
        cv2.circle(img_p, (int(x), int(y)), 4 if l == 0 else 5, (255, 0, 0) if l == 0 else (0, 255, 0), -1)
    panels = [draw(img_p, z["enhanced"], z["gt"], f"3-mask {r.enhanced_dice:.2f}", (0, 0, 255)),
              draw(img_p, z["single"], z["gt"], f"1-mask {r.single_dice:.2f}", (255, 128, 0))]
    cv2.imwrite(str(V3 / "single_mask" / "overlays" / f"{Path(r.query_image).stem}.png"), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))

base_cols = ["query_image", "patient", "support_image", "support_patient", "gt_px", "support_px"]
ad_cols = base_cols + ["box_px", "box_area_vs_support", "a5_choice", "box_pred_iou", "enh_pred_iou"] + \
    [f"{p}_{m}" for p in ("baseline", "enhanced", "a5") for m in K] + ["baseline_ms", "enhanced_ms", "a5_ms"]
df[ad_cols].to_csv(V3 / "adaptive" / "per_sample.csv", index=False)
per_patient(["baseline", "enhanced", "a5"]).to_csv(V3 / "adaptive" / "per_patient.csv")
sm_cols = base_cols + ["n_negative", "positive_point_used", "adaptive_strategy", "enh_pred_iou", "single_pred_iou"] + \
    [f"{p}_{m}" for p in ("enhanced", "single") for m in K] + ["enhanced_ms", "single_ms", "single_decode_ms"]
df[sm_cols].to_csv(V3 / "single_mask" / "per_sample.csv", index=False)
per_patient(["enhanced", "single"]).to_csv(V3 / "single_mask" / "per_patient.csv")

# Phase 3a: the four dev problem cases (recorded by collect.py with production prompts)
rows = []
for p in sorted((HERE / "wholebody").glob("*.json")):
    d = json.loads(p.read_text())
    a = d["ablation"]
    rows.append({"sample": d["sample"], "set": "development", "multimask_dice": a["box+pos+neg|mm=1"]["dice"], "multimask_px": a["box+pos+neg|mm=1"]["px"],
                 "single_dice": a["box+pos+neg|mm=0"]["dice"], "single_px": a["box+pos+neg|mm=0"]["px"], "gt_px": d["gt_px"]})
pd.DataFrame(rows).to_csv(V3 / "single_mask" / "dev_problem_cases.csv", index=False)

wb = lambda p: int(((df[f"{p}_precision"] < 0.1) & (df[f"{p}_recall"] > 0.9)).sum())  # noqa: E731
summ = {"config": config,
        "slice_level": {"baseline": summary("baseline", "baseline_ms"), "enhanced_v2": summary("enhanced", "enhanced_ms"),
                        "a5": summary("a5", "a5_ms"), "single_mask": summary("single", "single_ms")},
        "mean_of_patient_means_dice": {p: float(df.groupby("patient")[f"{p}_dice"].mean().mean()) for p in ("baseline", "enhanced", "a5", "single")},
        "tests": tests, "subsets": subsets, "a5_selection": a5_sel,
        "oversized_masks(precision<0.1&recall>0.9)": {p: wb(p) for p in ("baseline", "enhanced", "single", "a5")},
        "single_decode_ms_median": float(df.single_decode_ms.median())}
for sub in ("adaptive", "single_mask"):
    (V3 / sub / "evaluation_config.json").write_text(json.dumps(config, indent=2))
(V3 / "adaptive" / "summary.json").write_text(json.dumps({k: summ[k] for k in summ if k != "single_decode_ms_median"}, indent=2))
(V3 / "single_mask" / "summary.json").write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=1))
pp = per_patient(["baseline", "enhanced", "a5", "single"])[["n_slices", "baseline_dice", "enhanced_dice", "a5_dice", "single_dice"]]
print(pp.round(3).to_string())
print(pd.DataFrame(rows).to_string(index=False))
