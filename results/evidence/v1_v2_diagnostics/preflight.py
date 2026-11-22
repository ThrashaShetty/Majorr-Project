import sys; from pathlib import Path
ROOT = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project"); sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, cv2
from src.baseline import SamEngine, BaselineSegmentor
from src.fob.fob_pipeline import FoBPipeline, body_mask
from src.preprocessing import load_image_rgb
b = pd.read_csv(ROOT/"results/baseline_results.csv"); e = pd.read_csv(ROOT/"results/enhanced_results.csv")
eng = SamEngine(ROOT/"checkpoints/sam_vit_b_01ec64.pth", "cpu")
base = BaselineSegmentor(engine=eng); enh = FoBPipeline(engine=eng, mode="enhanced", uncertainty="none")
ok = True
for q in ["case_006_slice_095", "case_006_slice_110", "case_052_slice_085"]:
    r = b[b.query_image == f"{q}.png"].iloc[0]
    paths = [ROOT/"data/images"/r.query_image, ROOT/"data/images"/r.support_image, ROOT/"data/masks"/r.support_image, ROOT/"data/masks"/r.query_image]
    rb = base.predict(*paths); re_ = enh.predict(*paths)
    same = all(abs(rb[k] - r[k]) < 1e-9 for k in ("dice", "iou", "precision", "recall", "boundary_iou"))
    body = body_mask(cv2.cvtColor(load_image_rgb(paths[0]), cv2.COLOR_RGB2GRAY))
    c = re_["config"]; neg = [p for p, l in zip(c["points_xy"], c["point_labels"]) if l == 0]
    inb = all(body[y, x] for x, y in neg)
    ok &= same and inb
    print(f"{q}: baseline v1={r.dice:.4f} now={rb['dice']:.4f} identical={same} | enhanced v1={e[e.query_image==r.query_image].dice.iloc[0]:.3f} "
          f"now={re_['dice']:.3f} mask_px={int(re_['mask'].sum())} n_neg={len(neg)} all_neg_in_body={inb} pos_used={c['positive_point_used']}")
print("PREFLIGHT", "PASS" if ok else "FAIL")
