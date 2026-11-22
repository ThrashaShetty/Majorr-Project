"""Adaptive-Enhanced experiment, step 1: collect GT-free confidence features + metrics.

Read-only w.r.t. the project: imports project code, reads data/results, writes only
into this scratchpad folder. Same 53 queries and support pairs as results/v2.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project")
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from segment_anything.utils.amg import calculate_stability_score  # noqa: E402

from src.baseline import SamEngine, mask_bounding_box  # noqa: E402
from src.fob.fob_pipeline import FoBPipeline, body_mask  # noqa: E402
from src.metrics import METRIC_KEYS, compute_all_metrics  # noqa: E402
from src.preprocessing import load_image_rgb, load_mask_binary  # noqa: E402

WHOLE_BODY = {"case_020_slice_070", "case_052_slice_075", "case_052_slice_080", "case_052_slice_095"}

torch.manual_seed(42)
np.random.seed(42)
v2 = {m: pd.read_csv(ROOT / f"results/v2/{m}_results.csv") for m in ("baseline", "enhanced")}
eng = SamEngine(ROOT / "checkpoints/sam_vit_b_01ec64.pth", "cpu")
enh = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none", seed=42)


def conf(masks, logits, scores, box):
    i = int(np.argmax(scores))
    m = masks[i]
    stab = float(calculate_stability_score(torch.from_numpy(logits[i:i + 1]), 0.0, 1.0)[0])
    x0, y0, x1, y1 = box.astype(int)
    box_area = (x1 - x0 + 1) * (y1 - y0 + 1)
    inbox = np.zeros_like(m)
    inbox[y0:y1 + 1, x0:x1 + 1] = True
    area = int(m.sum())
    ious = [(masks[a] & masks[b]).sum() / max(1, (masks[a] | masks[b]).sum()) for a, b in ((0, 1), (0, 2), (1, 2))]
    return dict(pred_iou=float(scores[i]), stability=stab, area=area, area_ratio=area / box_area,
                in_box_frac=float((m & inbox).sum() / max(1, area)), mm_agree=float(np.mean(ious)),
                score_spread=float(scores.max() - scores.min()), idx=i), m


rows = []
for k, r in v2["baseline"].iterrows():
    q = r.query_image[:-4]
    qi = load_image_rgb(ROOT / "data/images" / r.query_image)
    gt = load_mask_binary(ROOT / "data/masks" / r.query_image)
    si = load_image_rgb(ROOT / "data/images" / r.support_image)
    sm = load_mask_binary(ROOT / "data/masks" / r.support_image)
    box = mask_bounding_box(sm)  # support and query are both 512x512 -> no resize needed
    assert sm.shape == qi.shape[:2]

    eng.invalidate()
    t0 = time.perf_counter(); eng.set_image(qi); t_enc = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    logits, scores, _ = eng.predictor.predict(box=box, multimask_output=True, return_logits=True)
    t_box = (time.perf_counter() - t0) * 1000
    masks = logits > eng.model.mask_threshold
    cb, mbox = conf(masks, logits, scores, box)
    mb = compute_all_metrics(mbox, gt, 0.0)

    t0 = time.perf_counter()
    res = enh.predict_arrays(qi, si, sm, gt)  # query embedding cached -> measures enhanced-only extra work
    t_enh = (time.perf_counter() - t0) * 1000
    c = res["config"]
    elog, escores, _ = eng.predictor.predict(
        multimask_output=True, return_logits=True, box=box,
        **({"point_coords": np.array(c["points_xy"], np.float32), "point_labels": np.array(c["point_labels"], np.int32)}
           if c["points_xy"] else {}))
    ce, menh = conf(elog > eng.model.mask_threshold, elog, escores, box)
    assert np.array_equal(menh, res["mask"]), f"{q}: enhanced replay mismatch"

    row = dict(sample=q, patient=q[:8], support=r.support_image[:-4], t_encode_ms=t_enc, t_box_ms=t_box, t_enh_extra_ms=t_enh,
               n_neg=int(c["n_negative"]), pos_used=bool(c["positive_point_used"]), gt_px=int((gt > 0).sum()))
    row.update({f"box_{m}": float(mb[m]) for m in METRIC_KEYS})
    row.update({f"enh_{m}": float(res[m]) for m in METRIC_KEYS})
    row.update({f"box_{kk}": v for kk, v in cb.items()})
    row.update({f"enh_{kk}": v for kk, v in ce.items()})
    # parity with the recorded v2 evaluation
    row["parity_box"] = abs(mb["dice"] - r.dice) < 1e-9
    row["parity_enh"] = abs(res["dice"] - v2["enhanced"].dice.iloc[k]) < 1e-9
    rows.append(row)
    np.savez_compressed(OUT / "masks" / f"{q}.npz", box=mbox, enh=menh, gt=gt > 0)

    if q in WHOLE_BODY:  # Phase 4 ablation
        pts = np.array(c["points_xy"], np.float32); lab = np.array(c["point_labels"], np.int32)
        abl = {}
        variants = {"box": {}, "box+pos": {"sel": lab == 1}, "box+neg": {"sel": lab == 0}, "box+pos+neg": {"sel": lab >= 0}}
        for name, v in variants.items():
            p = {"box": box}
            if "sel" in v and v["sel"].any():
                p.update(point_coords=pts[v["sel"]], point_labels=lab[v["sel"]])
            for mm in (True, False):
                ms, sc, _ = eng.predictor.predict(multimask_output=mm, **p)
                i = int(np.argmax(sc))
                abl[f"{name}|mm={int(mm)}"] = dict(dice=round(compute_all_metrics(ms[i], gt, 0)["dice"], 3), px=int(ms[i].sum()),
                                                   idx=i, areas=[int(x.sum()) for x in ms], scores=[round(float(s), 3) for s in sc])
        qg = cv2.cvtColor(qi, cv2.COLOR_RGB2GRAY)
        body = body_mask(qg)
        g = gt > 0
        ys, xs = np.nonzero(g)
        pos = [p for p, l in zip(c["points_xy"], c["point_labels"]) if l == 1]
        neg = [p for p, l in zip(c["points_xy"], c["point_labels"]) if l == 0]
        info = dict(sample=q, support=r.support_image, box=box.astype(int).tolist(),
                    gt_box=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    gt_in_box=float(g[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1].sum() / g.sum()),
                    gt_px=int(g.sum()), body_px=int(body.sum()), enh_px=int(menh.sum()), enh_mask_in_body=float((menh & body).sum() / max(1, menh.sum())),
                    pos=pos, pos_in_gt=[bool(g[y, x]) for x, y in pos], pos_grey=[int(qg[y, x]) for x, y in pos],
                    n_neg=len(neg), neg_in_gt=int(sum(g[y, x] for x, y in neg)), spr_candidates=c["spr_candidates"],
                    adaptive=c["adaptive"], ablation=abl)
        (OUT / "wholebody").mkdir(exist_ok=True)
        (OUT / "wholebody" / f"{q}.json").write_text(json.dumps(info, indent=1, default=int))
        vis = qi.copy()
        vis[menh] = (0.5 * vis[menh] + [0, 0, 127]).astype(np.uint8)
        cv2.drawContours(vis, cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 0), 1)
        cv2.rectangle(vis, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (255, 255, 0), 1)
        for x, y in neg:
            cv2.circle(vis, (x, y), 4, (255, 0, 0), -1)
        for x, y in pos:
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
        cv2.imwrite(str(OUT / "wholebody" / f"{q}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"{k + 1:2d}/53 {q} box={mb['dice']:.3f} enh={res['dice']:.3f} box_piou={cb['pred_iou']:.3f} box_stab={cb['stability']:.3f}", flush=True)

pd.DataFrame(rows).to_csv(OUT / "features.csv", index=False)
print("parity box:", all(r["parity_box"] for r in rows), "parity enh:", all(r["parity_enh"] for r in rows))
