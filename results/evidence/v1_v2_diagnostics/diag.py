"""Read-only diagnostic: replay Enhanced prompts with ablations. Writes only to scratchpad."""
import sys, json
from pathlib import Path
ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project")
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
from src.baseline import SamEngine, mask_bounding_box, match_size
from src.fob.fob_pipeline import FoBPipeline
from src.preprocessing import load_image_rgb, load_mask_binary
from src.metrics import compute_all_metrics

OUT = Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
SAMPLES = sys.argv[2].split(",")
e = pd.read_csv(ROOT / "results/enhanced_results.csv")
eng = SamEngine(ROOT / "checkpoints/sam_vit_b_01ec64.pth", "cpu")
pipe = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none")
rows = []
for q in SAMPLES:
    r = e[e.query_image == f"{q}.png"].iloc[0]
    qi = load_image_rgb(ROOT / "data/images" / r.query_image); gt = load_mask_binary(ROOT / "data/masks" / r.query_image) > 0
    si = load_image_rgb(ROOT / "data/images" / r.support_image); sm = load_mask_binary(ROOT / "data/masks" / r.support_image)
    res = pipe.predict_arrays(qi, si, sm, gt.astype(np.uint8))
    cfg = res["config"]; pts = np.array(cfg["points_xy"], np.float32); lab = np.array(cfg["point_labels"], np.int32)
    box = np.array(cfg["box_xyxy"], np.float32)
    ys, xs = np.nonzero(gt); gbox = [xs.min(), ys.min(), xs.max(), ys.max()]
    pos = pts[lab == 1][0].astype(int); neg = pts[lab == 0].astype(int)
    h, w = gt.shape
    att = pipe._query_attention(pipe._prototype(*match_size(si, sm, (h, w)))[0], (h, w))
    body = qi[..., 0] > 20
    info = dict(sample=q, support=r.support_image, gt_px=int(gt.sum()), box=box.astype(int).tolist(), gt_box=gbox,
                gt_in_box_frac=float(gt[int(box[1]):int(box[3])+1, int(box[0]):int(box[2])+1].sum() / gt.sum()),
                pos_pt=pos.tolist(), pos_in_gt=bool(gt[pos[1], pos[0]]), pos_intensity=int(qi[pos[1], pos[0], 0]),
                n_neg=len(neg), neg_in_body=int(sum(body[y, x] for x, y in neg)),
                att_gt_mean=float(att[gt].mean()), att_bg_mean=float(att[~gt & body].mean()), att_air_mean=float(att[~body].mean()),
                enh_dice=round(res["dice"], 3), enh_mask_px=int(res["mask"].sum()), enh_mask_idx=cfg["mask_index"],
                neg_in_pred=int(sum(res["mask"][y, x] for x, y in neg)), body_px=int(body.sum()))
    eng.set_image(qi)
    variants = {
        "box": dict(box=box),
        "box+pos": dict(box=box, point_coords=pts[lab == 1], point_labels=lab[lab == 1]),
        "box+neg": dict(box=box, point_coords=pts[lab == 0], point_labels=lab[lab == 0]) if len(neg) else None,
        "box+pos+neg": dict(box=box, point_coords=pts, point_labels=lab),
    }
    for name, p in variants.items():
        if p is None: continue
        for mm in (True, False):
            masks, scores, _ = eng.predictor.predict(multimask_output=mm, **p)
            i = int(np.argmax(scores)); m = masks[i]
            met = compute_all_metrics(m, gt.astype(np.uint8), 0.0)
            info[f"{name}|mm={int(mm)}"] = f"dice={met['dice']:.3f} px={int(m.sum())} idx={i} areas={[int(x.sum()) for x in masks]} scores={[round(float(s),3) for s in scores]}"
    rows.append(info)
    vis = qi.copy(); vis[res["mask"]] = (0.5 * vis[res["mask"]] + [0, 0, 127]).astype(np.uint8)
    cv2.drawContours(vis, cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, (0, 255, 0), 1)
    cv2.rectangle(vis, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (255, 255, 0), 1)
    cv2.circle(vis, (int(pos[0]), int(pos[1])), 5, (0, 255, 0), -1)
    for x, y in neg: cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), -1)
    a = cv2.applyColorMap(((att - att.min()) / (np.ptp(att) + 1e-8) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT / f"{q}.png"), np.hstack([cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), a]))
    print(json.dumps(info, indent=1, default=int), flush=True)
(OUT / "diag.json").write_text(json.dumps(rows, indent=1, default=int))
