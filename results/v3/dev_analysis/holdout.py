"""Held-out validation of Adaptive A5 and the single-mask variant (prototype, not production).

Test set   : the 33 support patients (156 slices) - never scored before.
Supports   : leave-one-patient-out nearest neighbour (same 64x64 thumbnail cosine as
             run_evaluation.py) among the OTHER 32 test patients. Dev patients (8 query
             patients) are excluded entirely.
Methods    : Baseline (production BaselineSegmentor), Enhanced v2 (production FoBPipeline,
             multimask), A5 (fixed rule, decided from GT-free quantities), Enhanced single-mask
             (same prompts as Enhanced v2, multimask_output=False).
Writes     : one resumable raw CSV + per-sample mask npz in this scratchpad folder only.
"""
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project")
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "holdout"
(OUT / "masks").mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_evaluation import _thumb  # noqa: E402
from src.baseline import BaselineSegmentor, SamEngine, select_best_mask  # noqa: E402
from src.fob.fob_pipeline import FoBPipeline  # noqa: E402
from src.metrics import METRIC_KEYS, compute_all_metrics  # noqa: E402
from src.preprocessing import load_image_rgb, load_mask_binary, read_list  # noqa: E402

A5_LO, A5_HI = 0.5, 2.0  # fixed before this experiment; never tuned
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

pid = lambda p: int(re.search(r"case_(\d+)", Path(p).name).group(1))  # noqa: E731
test = read_list(ROOT / "data/support_list.txt")
dev_patients = {pid(i) for i, _ in read_list(ROOT / "data/query_list.txt")}
assert not dev_patients & {pid(i) for i, _ in test}

bank = np.stack([_thumb(i) for i, _ in test])
pat = np.array([pid(i) for i, _ in test])
pairs = []
for k, (qi, qm) in enumerate(test):
    sim = bank @ bank[k]
    sim[pat == pat[k]] = -np.inf  # leave-one-patient-out: never the same patient (incl. itself)
    j = int(np.argmax(sim))
    assert pat[j] != pat[k] and pat[j] not in dev_patients
    pairs.append((qi, qm, test[j][0], test[j][1]))

raw = OUT / "raw_rows.csv"
done = set()
if raw.exists():
    with raw.open() as fh:
        done = {r["query_image"] for r in csv.DictReader(fh)}

eng = SamEngine(ROOT / "checkpoints/sam_vit_b_01ec64.pth", "cpu")
base = BaselineSegmentor(engine=eng)
enh = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none", seed=SEED)

fields = None
for n, (qi, qm, si, sm) in enumerate(pairs, 1):
    if qi.name in done:
        continue
    q = load_image_rgb(qi); gt = load_mask_binary(qm)
    s = load_image_rgb(si); smask = load_mask_binary(sm)
    support_px = int((smask > 0).sum())

    eng.invalidate()  # every query is encoded fresh -> honest Baseline timing
    rb = base.predict_arrays(q, smask, gt)
    re_ = enh.predict_arrays(q, s, smask, gt)
    c = re_["config"]
    box = np.array(c["box_xyxy"], np.float32)
    prompt = {"box": box}
    if c["points_xy"]:
        prompt.update(point_coords=np.array(c["points_xy"], np.float32), point_labels=np.array(c["point_labels"], np.int32))
    t0 = time.perf_counter()
    ms, sc, _ = eng.predictor.predict(multimask_output=False, **prompt)
    single_decode_ms = (time.perf_counter() - t0) * 1000
    m_single, s_single, _ = select_best_mask(ms, sc)
    rs = compute_all_metrics(m_single, gt, 0.0)

    box_px = int(rb["mask"].sum())
    ratio = box_px / support_px
    keep_box = A5_LO <= ratio <= A5_HI
    tm = c["timing_ms"]
    enh_extra = tm["support_prototype"] + tm["prompting_and_decode"]
    row = dict(query_image=qi.name, patient=f"case_{pid(qi):03d}", support_image=si.name, support_patient=f"case_{pid(si):03d}",
               support_px=support_px, gt_px=int((gt > 0).sum()), box_px=box_px, box_area_vs_support=ratio,
               a5_choice="box" if keep_box else "enhanced", box_pred_iou=rb["score"], enh_pred_iou=re_["score"],
               single_pred_iou=s_single, n_negative=c["n_negative"], positive_point_used=c["positive_point_used"],
               adaptive_strategy=c["adaptive"]["strategy"], n_points_requested=c["n_points_requested"])
    for m in METRIC_KEYS:
        row[f"baseline_{m}"] = rb[m]
        row[f"enhanced_{m}"] = re_[m]
        row[f"single_{m}"] = rs[m]
        row[f"a5_{m}"] = rb[m] if keep_box else re_[m]
    row["baseline_ms"] = rb["inference_ms"]
    row["enhanced_ms"] = re_["inference_ms"]
    # Single-mask uses the identical prompt-generation path; only the decoder call differs
    # (measured separately below). Its end-to-end time is therefore Enhanced time with the
    # multimask decode swapped for the single-mask decode; both decodes are ~0.1 s on CPU.
    row["single_decode_ms"] = single_decode_ms
    row["single_ms"] = re_["inference_ms"]
    row["a5_ms"] = rb["inference_ms"] + (0.0 if keep_box else enh_extra)
    np.savez_compressed(OUT / "masks" / f"{qi.stem}.npz", gt=gt > 0, baseline=rb["mask"], enhanced=re_["mask"], single=m_single,
                        box=box, points=np.array(c["points_xy"]), labels=np.array(c["point_labels"]))
    new = not raw.exists()
    with raw.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"{n:3d}/{len(pairs)} {qi.stem} sup={si.stem} base={rb['dice']:.3f} enh={re_['dice']:.3f} single={rs['dice']:.3f} "
          f"A5={row['a5_choice']} ratio={ratio:.2f}", flush=True)
print("DONE", len(pairs))
