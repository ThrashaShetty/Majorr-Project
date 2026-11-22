"""Reconstruct the v1 src/fob/fob_pipeline.py by reverting the four approved v2 edits.

Reads the current (v2) file read-only; writes a NEW file (exclusive create, never overwrites).
Each v2 block must occur exactly once, otherwise the script aborts without writing.
"""
import sys
from pathlib import Path

SRC = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project\src\fob\fob_pipeline.py")
DST = Path(r"C:\Anvith\FINALYEARPROJECTTHRASHA\medical_sam_project\results\v1_code\fob_pipeline_v1.py")

# (v2 text currently in the file, v1 text it replaced) - verbatim from the approved Edit calls
REVERSE = [
    # Edit 1: added constants + body_mask + select_positive_point
    ('''VALID_MODES = ("fob", "enhanced")
VALID_UNCERTAINTY = ("auto", "mc_dropout", "prompt_perturbation", "none")
DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)

# Prompt-placement constraints (grey levels refer to the preprocessed uint8 CT
# window [-200, 300] HU). See README "Prompt placement constraints".
BODY_THRESHOLD = 20          # pixels > 20 are patient/table, <= 20 are air
BODY_EROSION_PX = 10         # keep negatives at least 10 px inside the body outline
SOFT_TISSUE_RANGE = (90, 170)  # eligible grey levels for the positive point (excludes air and bone)


# --------------------------------------------------------------------------- #
# Prompt-placement helpers
# --------------------------------------------------------------------------- #
def body_mask(gray: np.ndarray, threshold: int = BODY_THRESHOLD, erosion_px: int = BODY_EROSION_PX) -> np.ndarray:
    """Boolean HxW mask of the patient body.

    ``gray > threshold`` -> largest connected component -> holes filled (outer
    contour drawn filled) -> eroded by ``erosion_px`` (elliptical kernel).
    Returns an all-False mask if nothing exceeds the threshold.
    """
    fg = (np.asarray(gray) > threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return np.zeros(fg.shape, dtype=bool)
    largest = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    contours, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(largest)
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    if erosion_px > 0:
        k = 2 * erosion_px + 1
        filled = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return filled > 0


def select_positive_point(
    attention: np.ndarray, box: np.ndarray, query_gray: np.ndarray, body: np.ndarray,
    tissue_range: tuple[int, int] = SOFT_TISSUE_RANGE,
) -> tuple[int, int] | None:
    """Minimum-attention ``(x, y)`` among in-box body pixels with soft-tissue grey level.

    Returns ``None`` when no pixel is eligible (the positive point is then omitted).
    """
    h, w = attention.shape
    x0, y0, x1, y1 = np.asarray(box).astype(int)
    eligible = np.zeros((h, w), dtype=bool)
    eligible[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
    eligible &= body & (query_gray >= tissue_range[0]) & (query_gray <= tissue_range[1])
    if not eligible.any():
        return None
    py, px = np.unravel_index(int(np.argmin(np.where(eligible, attention, np.inf))), attention.shape)
    return int(px), int(py)
''',
     '''VALID_MODES = ("fob", "enhanced")
VALID_UNCERTAINTY = ("auto", "mc_dropout", "prompt_perturbation", "none")
DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
'''),
    # Edit 2: prompt construction
    ('''            # The attention map is already at image resolution; SPR resizes it to the gray grid (no-op).
            # Negatives are restricted to the query body: in air, SAM reads them as "everything
            # that is not air" and returns the whole body (diagnosed on the v1 evaluation).
            query_gray = cv2.cvtColor(query_image_rgb, cv2.COLOR_RGB2GRAY)
            body = body_mask(query_gray)
            pos_xy: tuple[int, int] | None = None
            if self.mode == "enhanced":
                assert box is not None
                outside = np.ones((h, w), dtype=bool)
                x0, y0, x1, y1 = box.astype(int)
                m = self.box_margin
                outside[max(0, y0 - m):min(h, y1 + m + 1), max(0, x0 - m):min(w, x1 + m + 1)] = False
                neg = self.spr.refine(support_gray, attention, n_points, min_distance,
                                      allowed_region=outside & body)
                pos_xy = select_positive_point(attention, box, query_gray, body)
                pos = [pos_xy] if pos_xy is not None else []
                prompt: dict[str, Any] = {"box": box}
                coords = np.array(neg + pos, dtype=np.float32).reshape(-1, 2)
                labels = np.array([0] * len(neg) + [1] * len(pos), dtype=np.int32)
                if len(coords):  # no negatives -> box (+ positive) fallback
                    prompt.update(point_coords=coords, point_labels=labels)
            else:
                neg = self.spr.refine(support_gray, attention, n_points, min_distance, allowed_region=body)
                if not neg:
                    raise SegmentationError(
                        "SPR found no background candidates (attention>0.6 outside edge zone, inside body); "
                        "FoB has no prompts for this sample"
                    )''',
     '''            # The attention map is already at image resolution; SPR resizes it to the gray grid (no-op).
            if self.mode == "enhanced":
                assert box is not None
                outside = np.ones((h, w), dtype=bool)
                x0, y0, x1, y1 = box.astype(int)
                m = self.box_margin
                outside[max(0, y0 - m):min(h, y1 + m + 1), max(0, x0 - m):min(w, x1 + m + 1)] = False
                neg = self.spr.refine(support_gray, attention, n_points, min_distance, allowed_region=outside)
                inside = attention[y0:y1 + 1, x0:x1 + 1]
                py, px = np.unravel_index(int(np.argmin(inside)), inside.shape)
                pos = [(int(x0 + px), int(y0 + py))]
                coords = np.array(neg + pos, dtype=np.float32)
                labels = np.array([0] * len(neg) + [1], dtype=np.int32)
                prompt: dict[str, Any] = {"point_coords": coords, "point_labels": labels, "box": box}
            else:
                neg = self.spr.refine(support_gray, attention, n_points, min_distance)
                if not neg:
                    raise SegmentationError(
                        "SPR found no background candidates (attention>0.6 outside edge zone); "
                        "FoB has no prompts for this sample"
                    )'''),
    # Edit 3: extra config fields
    ('''            "points_xy": coords.astype(int).tolist(),
            "point_labels": labels.tolist(),
            "n_negative": int((labels == 0).sum()),
            "positive_point_used": pos_xy is not None,
            "body_px": int(body.sum()),''',
     '''            "points_xy": coords.astype(int).tolist(),
            "point_labels": labels.tolist(),'''),
    # Edit 4: module docstring
    ('''      2. up to ``n_points`` SPR background points (label 0) restricted to lie
         **outside** that box dilated by ``box_margin`` pixels **and inside the
         query body mask** (``body_mask``: grey > 20, largest component, holes
         filled, eroded 10 px);
      3. one positive point (label 1) at the location of **minimum** BCM
         background attention among in-box body pixels with soft-tissue grey
         level 90-170 (``select_positive_point``); omitted if none is eligible.
    If no negative survives, the prompt falls back to box (+ positive point).
    The extra information over the baseline is therefore exactly: background
    negatives and one attention-derived positive point.

    FoB mode applies the same body restriction to its negatives.

    Why the body restriction: the background prototype is dominated by air, so
    unrestricted negatives landed in the air around the patient and SAM
    returned the whole body (recall ~1, precision ~0.03 in the v1 evaluation).''',
     '''      2. up to ``n_points`` SPR background points (label 0) restricted to lie
         **outside** that box dilated by ``box_margin`` pixels;
      3. one positive point (label 1) at the location of **minimum** BCM
         background attention inside the box (the least background-like
         query pixel within the transferred box).
    The extra information over the baseline is therefore exactly: background
    negatives and one attention-derived positive point.'''),
]

raw = SRC.read_bytes()
text = raw.decode("utf-8")
crlf = "\r\n" in text
if crlf:
    text = text.replace("\r\n", "\n")
for i, (v2, v1) in enumerate(REVERSE, 1):
    n = text.count(v2)
    if n != 1:
        sys.exit(f"ABORT: edit {i} v2 block found {n} times; nothing written")
    text = text.replace(v2, v1)
for leftover in ("body_mask", "select_positive_point", "SOFT_TISSUE_RANGE", "pos_xy", "n_negative"):
    if leftover in text:
        sys.exit(f"ABORT: v2 identifier '{leftover}' still present; nothing written")
if crlf:
    text = text.replace("\n", "\r\n")
DST.parent.mkdir(parents=False, exist_ok=False)
with open(DST, "x", encoding="utf-8", newline="") as fh:  # 'x' = never overwrite
    fh.write(text)
print("written", DST, "crlf" if crlf else "lf")
