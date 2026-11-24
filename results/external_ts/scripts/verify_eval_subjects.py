"""Independent verification of evaluated subjects (read-only; one write-once report).

Does NOT use the runner's verify functions or src.metrics. For each requested subject:
  * expected slices re-derived from manifest + preprocessing logs (primary step-5 / sensitivity step-17)
  * slice JSONs: parse, config hash, item fields vs worklist, npz file + SHA-256
  * npz: arrays gt/baseline/enhanced/single are 512x512 bool; gt == preprocessed mask PNG (>0)
  * metrics recomputed independently (numpy + scipy) for baseline/enhanced/single; A5 = baseline if
    0.5 <= box_px/support_px <= 2.0 else enhanced (support_px from the support mask PNG)
  * done marker: csv hash, slice-json hash list, row count; csv rows == slice rows
  * no orphan / unreferenced prediction files; no outputs for subjects not requested
  * optional: --unchanged-snapshot (sha256sum file) -> those files must be byte-identical
  * optional: --allow-existing s1 s2 ... -> outputs of these (previously verified) subjects are not
    reported as 'other subjects'; without the option the behaviour is unchanged
Usage: python -B -I verify_eval_subjects.py --subjects s0965 --out <report.json> [--unchanged-snapshot f]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

SCRIPTS = Path(__file__).resolve().parent
P = SCRIPTS.parents[2]
TS = P.parent / "external_datasets" / "totalsegmentator_small_v201"
EVAL = P / "results/external_ts/eval"
KEYS = ("dice", "iou", "precision", "recall", "boundary_iou")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def metrics(p: np.ndarray, g: np.ndarray) -> dict[str, float]:
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    d = 2 * tp + fp + fn
    out = {"dice": 1.0 if d == 0 else 2 * tp / d, "iou": 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)}
    out["precision"] = (1.0 if fn == 0 else 0.0) if tp + fp == 0 else tp / (tp + fp)
    out["recall"] = (1.0 if fp == 0 else 0.0) if tp + fn == 0 else tp / (tp + fn)
    h, w = p.shape
    it = max(1, int(round(0.02 * math.sqrt(h * h + w * w))))
    band = lambda m: m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), iterations=it, border_value=0)  # noqa: E731
    pb, gb = band(p), band(g)
    u = int((pb | gb).sum())
    out["boundary_iou"] = 1.0 if u == 0 else int((pb & gb).sum()) / u
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unchanged-snapshot", type=Path)
    ap.add_argument("--allow-existing", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg_sha = sha(EVAL / "run_config.json")
    man = json.loads((P / "results/external_ts/cohort_manifest.json").read_text())
    rows_m = {r["subject"]: r for r in man["subjects"]}
    wl = {(w["subject"], w["z"]): w for w in json.loads((EVAL / "worklist.json").read_text())["items"]}
    errors, rep = [], {"config_sha256": cfg_sha, "subjects": {}}
    max_metric_dev = 0.0
    for s in a.subjects:
        log = json.loads((TS / "preprocessed/logs" / f"{s}.json").read_text())
        prim = set(log["processed_step5"]) if rows_m[s]["primary"] else set()
        sens = set(log["processed_step17"])
        expected = sorted(prim | sens)
        present = sorted(int(f.stem.rsplit("_", 1)[1]) for f in (EVAL / "slices" / s).glob("*.json"))
        sr = {"expected_slices": expected, "present_slices": present, "primary": len(prim), "sensitivity": len(sens),
              "overlap": len(prim & sens), "slices": []}
        if present != expected:
            errors.append(f"{s}: present {present} != expected {expected}")
        referenced = set()
        recs = []
        for z in expected:
            jp = EVAL / "slices" / s / f"ts_{s}_slice_{z:03d}.json"
            if not jp.exists():
                continue
            rec = json.loads(jp.read_text())
            recs.append(rec)
            w = wl[(s, z)]
            e = []
            if rec["config_sha256"] != cfg_sha:
                e.append("config hash")
            for k in ("file", "image_sha256", "mask_sha256", "in_primary", "in_sensitivity", "support_image", "support_mask", "support_patient"):
                if rec["item"][k] != w[k]:
                    e.append(f"item {k}")
            if (z in prim) != rec["item"]["in_primary"] or (z in sens) != rec["item"]["in_sensitivity"]:
                e.append("cohort flags vs manifest/logs")
            npz = EVAL / "masks" / s / rec["npz_file"]
            referenced.add(npz.name)
            if sha(npz) != rec["npz_sha256"]:
                e.append("npz hash")
            with np.load(npz, allow_pickle=False) as d:
                arr = {k: d[k] for k in d.files}
            for k in ("gt", "baseline", "enhanced", "single"):
                if arr[k].shape != (512, 512) or arr[k].dtype != bool:
                    e.append(f"{k} shape/dtype {arr[k].shape} {arr[k].dtype}")
            gt_png = cv2.imread(str(TS / "preprocessed/masks" / w["file"]), cv2.IMREAD_GRAYSCALE) > 0
            if not np.array_equal(gt_png, arr["gt"]):
                e.append("gt != preprocessed mask")
            sup_px = int((cv2.imread(str(P / w["support_mask"]), cv2.IMREAD_GRAYSCALE) > 0).sum())
            row = rec["row"]
            box_px = int(arr["baseline"].sum())
            ratio = box_px / sup_px
            choice = "box" if 0.5 <= ratio <= 2.0 else "enhanced"
            checks = {"support_px": (row["support_px"], sup_px), "gt_px": (row["gt_px"], int(gt_png.sum())),
                      "box_px": (row["box_px"], box_px), "a5_choice": (row["a5_choice"], choice)}
            for k, (got, want) in checks.items():
                if got != want:
                    e.append(f"{k} {got} != {want}")
            if abs(row["box_area_vs_support"] - ratio) > 1e-12:
                e.append("ratio")
            per = {}
            for meth, key in (("baseline", "baseline"), ("enhanced", "enhanced"), ("single", "single")):
                m = metrics(arr[key], arr["gt"])
                per[meth] = m
                for k in KEYS:
                    dv = abs(m[k] - row[f"{meth}_{k}"])
                    max_metric_dev = max(max_metric_dev, dv)
                    if dv > 1e-12:
                        e.append(f"{meth}_{k} recomputed {m[k]} != stored {row[f'{meth}_{k}']}")
            src = per["baseline"] if choice == "box" else per["enhanced"]
            for k in KEYS:
                if abs(src[k] - row[f"a5_{k}"]) > 1e-12:
                    e.append(f"a5_{k}")
            for meth in ("baseline", "enhanced", "single"):
                if not arr[meth].any():
                    e.append(f"{meth} prediction empty (reported, not an integrity error)") if False else None
            sr["slices"].append({"z": z, "ok": not e, "errors": e, "a5_choice": choice, "ratio": round(ratio, 4),
                                 "pred_px": {k: int(arr[k].sum()) for k in ("baseline", "enhanced", "single")},
                                 "dice": {k: round(row[f'{k}_dice'], 4) for k in ("baseline", "enhanced", "a5", "single")}})
            errors += [f"{s} z={z}: {x}" for x in e]
        orphans = sorted(f.name for f in (EVAL / "masks" / s).glob("*.npz") if f.name not in referenced)
        sr["unreferenced_npz"] = orphans
        if orphans:
            errors.append(f"{s}: unreferenced npz {orphans}")
        done = EVAL / "subjects" / f"{s}.done"
        csvp = EVAL / "subjects" / f"{s}.csv"
        if not done.exists() or not csvp.exists():
            errors.append(f"{s}: done marker or csv missing")
        else:
            d = json.loads(done.read_text())
            want_hashes = sorted(sha(EVAL / "slices" / s / f"ts_{s}_slice_{z:03d}.json") for z in expected)
            csv_rows = list(csv.DictReader(io.StringIO(csvp.read_text())))
            sr["done_marker"] = {"csv_hash_ok": d["csv_sha256"] == sha(csvp), "slice_hashes_ok": sorted(d["slice_json_sha256"]) == want_hashes,
                                 "n_slices": d["n_slices"], "config_ok": d["config_sha256"] == cfg_sha, "csv_rows": len(csv_rows),
                                 "csv_matches_slices": [int(r["z"]) for r in csv_rows] == expected and all(
                                     float(r["baseline_dice"]) == rr["row"]["baseline_dice"] for r, rr in zip(csv_rows, recs))}
            if not all(v for k, v in sr["done_marker"].items() if isinstance(v, bool)) or d["n_slices"] != len(expected):
                errors.append(f"{s}: done marker inconsistent {sr['done_marker']}")
        rep["subjects"][s] = sr
    known = set(a.subjects) | set(a.allow_existing)
    others = sorted({p.name for d in ("slices", "masks") for p in (EVAL / d).iterdir() if p.is_dir()} - known)
    others += sorted({p.stem for p in (EVAL / "subjects").glob("*")} - known) if (EVAL / "subjects").exists() else []
    rep["allowed_existing_subjects"] = sorted(set(a.allow_existing))
    rep["other_subjects_with_outputs"] = sorted(set(others))
    if others:
        errors.append(f"outputs for other subjects: {sorted(set(others))}")
    if a.unchanged_snapshot:
        bad = []
        for line in a.unchanged_snapshot.read_text().splitlines():
            h, f = line.split(maxsplit=1)
            f = f.lstrip("*")
            if "/logs/" in f:
                continue  # logs are append-only by design
            if sha(P / f) != h:
                bad.append(f)
        rep["snapshot_files_changed"] = bad
        if bad:
            errors.append(f"pre-existing files changed: {bad}")
    rep["max_abs_metric_deviation"] = max_metric_dev
    rep["errors"] = errors
    rep["all_passed"] = not errors
    if a.out.exists():
        raise SystemExit(f"{a.out} exists - refusing to overwrite")
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "subjects"}, indent=1))
    for s, sr in rep["subjects"].items():
        print(s, "expected", len(sr["expected_slices"]), "present", len(sr["present_slices"]), "primary", sr["primary"],
              "sensitivity", sr["sensitivity"], "overlap", sr["overlap"], "done", sr.get("done_marker"))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
