"""EXPLORATORY (not part of the frozen confirmatory analysis): uncertainty vs Dice, and Dice vs simulated clicks,
on the v3 held-out MSD set (33 patients, 156 slices; supports = leave-one-patient-out nearest neighbour, exactly as
results/v3/scripts/holdout.py).

For each query slice:
  Baseline : support-mask box -> SAM (multimask, return_logits) -> best mask. Checked against the frozen
             BaselineSegmentor mask on the same embedding (any mismatch is counted and reported).
  A5       : Baseline if 0.5 <= box_px / support_px <= 2.0, else the frozen Enhanced v2 (FoBPipeline) prompt.
  Uncertainty (src/extensions/uncertainty.py): multimask entropy + disagreement, K=8 seeded perturbation
             variance, for the Baseline prompt and for the A5 prompt.
  Clicks   (src/extensions/refinement.py): Dice after 0..5 simulated clicks for Baseline and A5.
Summary: Spearman rho (scipy) of each uncertainty score vs Dice, at slice level (n = 156) and patient level
(n = 33, patient means), reported whatever the sign or p-value; mean Dice by click count.

Writes ONLY to a new folder results/exploratory/<UTC stamp>_v3_unc_clicks/ (refuses to overwrite).
Usage: python -B -I experiments/exploratory/v3_uncertainty_and_clicks.py --check          (no model, read-only)
       python -B -I experiments/exploratory/v3_uncertainty_and_clicks.py [--max-slices N]  (loads SAM; needs RAM)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from run_evaluation import _thumb  # noqa: E402  (v3 pairing function, unchanged)
from src.extensions import refinement as R  # noqa: E402
from src.extensions import uncertainty as U  # noqa: E402
from src.metrics import compute_all_metrics, dice_score  # noqa: E402
from src.preprocessing import load_image_rgb, load_mask_binary, read_list  # noqa: E402

A5_LO, A5_HI, SEED, K, N_CLICKS = 0.5, 2.0, 42, 8, 5
pid = lambda p: int(re.search(r"case_(\d+)", Path(p).name).group(1))  # noqa: E731


def pairs_v3():
    test = read_list(ROOT / "data/support_list.txt")
    dev = {pid(i) for i, _ in read_list(ROOT / "data/query_list.txt")}
    bank = np.stack([_thumb(i) for i, _ in test])
    pat = np.array([pid(i) for i, _ in test])
    out = []
    for k, (qi, qm) in enumerate(test):
        sim = bank @ bank[k]
        sim[pat == pat[k]] = -np.inf
        j = int(np.argmax(sim))
        assert pat[j] != pat[k] and pat[j] not in dev
        out.append((qi, qm, test[j][0], test[j][1]))
    return out


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(x))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="read-only input check; no model")
    ap.add_argument("--max-slices", type=int, default=None)
    a = ap.parse_args(argv)
    pairs = pairs_v3()
    print(f"v3 pairs: {len(pairs)} slices, {len({pid(q) for q, *_ in pairs})} patients")
    if a.check:
        return 0 if len(pairs) == 156 else 1
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / "exploratory" / f"{stamp}_v3_unc_clicks"
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from src.baseline import BaselineSegmentor, SamEngine, mask_bounding_box
    from src.fob.fob_pipeline import FoBPipeline
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    eng = SamEngine(ROOT / "checkpoints/sam_vit_b_01ec64.pth", "cpu")
    base = BaselineSegmentor(engine=eng)
    enh = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none", seed=SEED)
    predict = eng.predictor.predict
    rows, mismatches = [], 0
    for n, (qi, qm, si, sm) in enumerate(pairs[: a.max_slices], 1):
        t0 = time.perf_counter()
        q, gt, s, smask = load_image_rgb(qi), load_mask_binary(qm), load_image_rgb(si), load_mask_binary(sm)
        gtb = gt > 0
        hw = q.shape[:2]
        eng.invalidate()
        rb = base.predict_arrays(q, smask, gt)                       # frozen Baseline (sets the embedding)
        box = mask_bounding_box((smask > 0).astype(np.uint8))
        logits, scores, low = predict(box=box, multimask_output=True, return_logits=True)
        bi = int(np.argmax(scores))
        b_mask = logits[bi] > 0
        mismatches += int(not np.array_equal(b_mask, rb["mask"]))
        ratio = int(rb["mask"].sum()) / max(1, int((smask > 0).sum()))
        keep_box = A5_LO <= ratio <= A5_HI
        b_unc = U.slice_uncertainty(predict, {"box": box}, hw, k=K, seed=SEED)
        b_clicks = R.simulate(predict, b_mask, gtb, N_CLICKS, box=box, initial_low_res=low[bi:bi + 1])
        if keep_box:
            a_prompt, a_mask, a_low = {"box": box}, b_mask, low[bi:bi + 1]
        else:
            re_ = enh.predict_arrays(q, s, smask, gt)                # frozen Enhanced v2 (re-sets query embedding)
            c = re_["config"]
            a_prompt = {"box": np.array(c["box_xyxy"], np.float32)}
            if c["points_xy"]:
                a_prompt.update(point_coords=np.array(c["points_xy"], np.float32), point_labels=np.array(c["point_labels"], np.int32))
            lg, sc, lw = predict(multimask_output=True, return_logits=True, **a_prompt)
            ai = int(np.argmax(sc))
            a_mask, a_low = lg[ai] > 0, lw[ai:ai + 1]
        a_unc = b_unc if keep_box else U.slice_uncertainty(predict, a_prompt, hw, k=K, seed=SEED)
        a_clicks = R.simulate(predict, a_mask, gtb, N_CLICKS, box=a_prompt.get("box"), initial_low_res=a_low)
        row = {"query_image": Path(qi).name, "patient": f"case_{pid(qi):03d}", "a5_choice": "box" if keep_box else "enhanced",
               "baseline_dice": float(dice_score(b_mask, gtb)), "a5_dice": float(dice_score(a_mask, gtb)),
               "frozen_baseline_dice": float(compute_all_metrics(rb["mask"], gt, 0.0)["dice"])}
        for tag, u in (("baseline", b_unc), ("a5", a_unc)):
            for k_ in ("entropy_score", "disagreement_score", "variance_score", "roi_px"):
                row[f"{tag}_{k_}"] = u[k_]
        for tag, cl in (("baseline", b_clicks), ("a5", a_clicks)):
            for i, d in enumerate(cl["dice_by_clicks"]):
                row[f"{tag}_dice_click{i}"] = d
        row["seconds"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
        print(f"{n:3d}/{len(pairs[: a.max_slices])} {row['query_image']} base={row['baseline_dice']:.3f} a5={row['a5_dice']:.3f} "
              f"H={row['baseline_entropy_score']:.3f} V={row['baseline_variance_score']:.3f} t={row['seconds']}s", flush=True)
    with open(out / "per_slice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    pp = df.groupby("patient").mean(numeric_only=True)
    summ = {"label": "EXPLORATORY - not part of the frozen confirmatory analysis", "n_slices": len(df), "n_patients": int(df.patient.nunique()),
            "baseline_mask_mismatches_vs_frozen": mismatches, "k_perturbations": K, "seed": SEED, "correlations": {}, "clicks": {}}
    for tag in ("baseline", "a5"):
        for sc in ("entropy_score", "disagreement_score", "variance_score"):
            summ["correlations"][f"{tag}_{sc}_vs_dice"] = {"slice": spearman(df[f"{tag}_{sc}"], df[f"{tag}_dice"]),
                                                          "patient": spearman(pp[f"{tag}_{sc}"], pp[f"{tag}_dice"])}
        summ["clicks"][tag] = {f"click{i}": {"mean_dice": float(df[f"{tag}_dice_click{i}"].mean()),
                                             "patient_mean_dice": float(pp[f"{tag}_dice_click{i}"].mean())} for i in range(N_CLICKS + 1)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
