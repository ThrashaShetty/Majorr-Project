"""Scratchpad-only test of proposed Enhanced fix (no project files written)."""
import sys; from pathlib import Path
ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
from src.baseline import SamEngine, mask_bounding_box, match_size
from src.fob.fob_pipeline import FoBPipeline
from src.preprocessing import load_image_rgb, load_mask_binary
from src.metrics import compute_all_metrics
def body_mask(gray, margin=10):
    m = (gray > 20).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return m.astype(bool)
    big = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(big); cv2.drawContours(filled, cnts, -1, 1, -1)
    return cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1,)*2)) > 0
e = pd.read_csv(ROOT/"results/enhanced_results.csv"); b = pd.read_csv(ROOT/"results/baseline_results.csv")
eng = SamEngine(ROOT/"checkpoints/sam_vit_b_01ec64.pth", "cpu"); pipe = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none")
for q in sys.argv[1].split(","):
    r = e[e.query_image == f"{q}.png"].iloc[0]
    qi = load_image_rgb(ROOT/"data/images"/r.query_image); gt = load_mask_binary(ROOT/"data/masks"/r.query_image) > 0
    si, sm = match_size(load_image_rgb(ROOT/"data/images"/r.support_image), load_mask_binary(ROOT/"data/masks"/r.support_image), gt.shape)
    sm = (sm > 0).astype(np.uint8); h, w = gt.shape; box = mask_bounding_box(sm); x0, y0, x1, y1 = box.astype(int)
    proto = pipe._prototype(si, sm)[0]; eng.set_image(qi); att = pipe._query_attention(proto, (h, w))
    qg = qi[..., 0]; body = body_mask(qg)
    outside = np.ones((h, w), bool); outside[max(0,y0-10):y1+11, max(0,x0-10):x1+11] = False
    neg = pipe.spr.refine(cv2.cvtColor(si, cv2.COLOR_RGB2GRAY), att, 15, 8.0, allowed_region=outside & body)
    inbox = np.zeros((h, w), bool); inbox[y0:y1+1, x0:x1+1] = True
    tissue = inbox & body & (qg >= 90) & (qg <= 170)
    a = np.where(tissue, att, np.inf); pos = None
    if np.isfinite(a).any(): py, px = np.unravel_index(int(np.argmin(a)), a.shape); pos = (px, py)
    out = {}
    for name, pts, lab in [("negBody+pos", neg + ([pos] if pos else []), [0]*len(neg) + ([1] if pos else [])),
                           ("pos_only", [pos] if pos else [], [1] if pos else [])]:
        p = dict(box=box)
        if pts: p.update(point_coords=np.array(pts, np.float32), point_labels=np.array(lab, np.int32))
        masks, sc, _ = eng.predictor.predict(multimask_output=True, **p); m = masks[int(np.argmax(sc))]
        out[name] = round(compute_all_metrics(m, gt.astype(np.uint8), 0)["dice"], 3)
    print(f"{q}: baseline={b[b.query_image==r.query_image].dice.iloc[0]:.3f} enh_old={r.dice:.3f} "
          f"negBody+pos={out['negBody+pos']} box+fixedPos={out['pos_only']} n_neg={len(neg)} pos_in_gt={bool(pos and gt[pos[1],pos[0]])}", flush=True)
