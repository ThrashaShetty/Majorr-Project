"""Independent audit of the frozen TS evaluation setup (no SAM, read-only except the write-once report).

Does NOT import eval_external_ts logic. Re-derives everything from primary sources:
  A. support pool  : data/support_list.txt vs split_summary.json, v3 per_patient.csv, manifest, run_config
  B. pairing rule  : own thumbnail re-implementation; proven equal to v3 by reproducing all 156 v3
                     leave-one-patient-out pairings in results/v3/adaptive/per_sample.csv; then every
                     worklist pairing is recomputed and compared
  C. cohorts       : from manifest + preprocessing logs (826 / 300 / 74 / 90; 177 excluded; s1358)
  D. method logic  : line diff of holdout.py per-query block vs eval_external_ts.evaluate_query source
  E. configuration : A5 / seed / device / checkpoint / code hashes vs v3 evaluation_config.json + files
  F. resource guard: one real read-only reading
Usage: python -B -I audit_eval_runner.py --out ../audit/eval_runner_audit.json
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
P = SCRIPTS.parents[2]
TS = P.parent / "external_datasets" / "totalsegmentator_small_v201"
EVAL = P / "results/external_ts/eval"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def thumb(path: Path) -> np.ndarray:  # independent re-implementation of the v3 rule
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    v = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-8)


# The only accepted textual differences (reviewed): row identity fields moved to the caller,
# METRIC_KEYS passed as a parameter, arrays returned instead of np.savez inside the loop.
EXPECTED_DIFF = [
    '-row = dict(query_image=qi.name, patient=f"case_{pid(qi):03d}", support_image=si.name, support_patient=f"case_{pid(si):03d}",',
    "-support_px=support_px, gt_px=int((gt > 0).sum()), box_px=box_px, box_area_vs_support=ratio,",
    "+row = dict(support_px=support_px, gt_px=int((gt > 0).sum()), box_px=box_px, box_area_vs_support=ratio,",
    "-for m in METRIC_KEYS:",
    "+for m in metric_keys:",
    '+arrays = dict(gt=gt > 0, baseline=rb["mask"], enhanced=re_["mask"], single=m_single,',
    '+box=box, points=np.array(c["points_xy"]), labels=np.array(c["point_labels"]))',
    "+return row, arrays",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rep: dict = {"checks": {}, "differences_from_v3": []}
    ok = rep["checks"]
    cfg = json.loads((EVAL / "run_config.json").read_text())
    wl = json.loads((EVAL / "worklist.json").read_text())["items"]
    man = json.loads((P / "results/external_ts/cohort_manifest.json").read_text())

    # A. support pool
    sup = [l.split("\t") for l in (P / "data/support_list.txt").read_text().splitlines() if l.strip()]
    pid_re = re.compile(r"case_(\d+)_slice")
    pid = lambda s: "case_%03d" % int(pid_re.search(s).group(1))  # noqa: E731
    sup_pat = sorted({pid(i) for i, _ in sup})
    dev = sorted({pid(l.split(chr(9))[0]) for l in (P / "data/query_list.txt").read_text().splitlines() if l.strip()})
    split = sorted(f"case_{c:03d}" for c in json.loads((P / "data/split_summary.json").read_text())["support_cases"])
    v3pp = sorted(r["patient"] for r in csv.DictReader(open(P / "results/v3/adaptive/per_patient.csv")))
    v3dev = json.loads((P / "results/v3/adaptive/evaluation_config.json").read_text())["dev_set_excluded"]
    ok["A_support_pool"] = {
        "n_patients": len(sup_pat), "n_slices": len(sup),
        "equals_split_summary": sup_pat == split, "equals_v3_per_patient": sup_pat == v3pp,
        "equals_manifest": sup_pat == man["support_pool"]["support_patients"],
        "equals_run_config": sup_pat == cfg["support_pool"]["patients"],
        "dev_patients": dev, "dev_equals_v3_excluded": dev == sorted(v3dev), "dev_disjoint": not set(dev) & set(sup_pat),
        "worklist_supports_subset_of_pool": {w["support_image"] for w in wl} <= {i for i, _ in sup},
        "distinct_supports_used": len({w["support_image"] for w in wl}),
        "supports_used_from_patients": len({w["support_patient"] for w in wl}),
    }

    # B. pairing rule
    bank = np.stack([thumb(P / i) for i, _ in sup])
    pats = np.array([pid(i) for i, _ in sup])
    names = [Path(i).name for i, _ in sup]
    v3 = {r["query_image"]: r["support_image"] for r in csv.DictReader(open(P / "results/v3/adaptive/per_sample.csv"))}
    match = 0
    for k, (i, _) in enumerate(sup):
        s = bank @ bank[k]
        s[pats == pats[k]] = -np.inf
        match += names[int(np.argmax(s))] == v3[names[k]]
    wl_match, ties = 0, 0
    for w in wl:
        s = bank @ thumb(TS / "preprocessed/images" / w["file"])
        j = int(np.argmax(s))
        wl_match += sup[j][0] == w["support_image"]
        ties += int((s >= s[j] - 1e-7).sum() > 1)
    ok["B_pairing"] = {"v3_pairings_reproduced": f"{match}/{len(sup)}", "worklist_pairings_reproduced": f"{wl_match}/{len(wl)}",
                       "near_ties_within_1e-7": ties}

    # C. cohorts from manifest + preprocessing logs
    rows = {r["subject"]: r for r in man["subjects"]}
    prim, sens, nonco, bnd5 = set(), set(), set(), set()
    for s, r in rows.items():
        if not r["eligible"]:
            continue
        log = json.loads((TS / "preprocessed/logs" / f"{s}.json").read_text())
        p5, p17 = set(log["processed_step5"]), set(log["processed_step17"])
        if r["primary"]:
            prim |= {(s, z) for z in p5}
        else:
            bnd5 |= {(s, z) for z in p5}
            nonco |= {(s, z) for z in p5 - p17}
        sens |= {(s, z) for z in p17}
    wset = {(w["subject"], w["z"]) for w in wl}
    ok["C_cohorts"] = {
        "primary_slices": len(prim), "sensitivity_slices": len(sens), "overlap": len(prim & sens), "union": len(prim | sens),
        "worklist_items": len(wl), "worklist_equals_union": wset == prim | sens,
        "worklist_flags_correct": all(w["in_primary"] == ((w["subject"], w["z"]) in prim) and
                                      w["in_sensitivity"] == ((w["subject"], w["z"]) in sens) for w in wl),
        "primary_subjects": len({s for s, _ in prim}), "sensitivity_evaluable_subjects": len({s for s, _ in sens}),
        "boundary_subject_step5_slices_all": len(bnd5),
        "boundary_step5_also_on_step17_grid_in_sensitivity": len(bnd5 & sens),
        "boundary_step5_in_neither_analysis": len(nonco), "excluded_disjoint_from_worklist": not nonco & wset,
        "boundary_step5_in_primary": len(bnd5 & prim),
        "s1358_in_worklist": any(w["subject"] == "s1358" for w in wl),
        "boundary_subjects_in_primary": sorted({w["subject"] for w in wl if w["in_primary"] and rows[w["subject"]]["boundary_touching"]}),
        "duplicate_items": len(wl) - len(wset),
    }

    # D. method logic diff (holdout per-query block vs ported function body)
    hold = (P / "results/v3/scripts/holdout.py").read_text().splitlines()
    i0 = next(i for i, l in enumerate(hold) if "support_px = int((smask > 0).sum())" in l)
    i1 = next(i for i, l in enumerate(hold) if "np.savez_compressed" in l)
    src = (SCRIPTS / "eval_external_ts.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_query")
    body = src.splitlines()[fn.body[0].lineno - 1:fn.end_lineno]
    norm = lambda ls: [l.strip() for l in ls if l.strip() and not l.strip().startswith("#")]  # noqa: E731
    diff = [d for d in difflib.unified_diff(norm(hold[i0:i1]), norm(body), "holdout.py", "evaluate_query", lineterm="", n=0)
            if not d.startswith(("---", "+++", "@@"))]
    ok["D_method_logic_diff_lines"] = diff
    ok["D_v3_holdout_sha256"] = sha(P / "results/v3/scripts/holdout.py")

    # E. configuration
    v3cfg = json.loads((P / "results/v3/adaptive/evaluation_config.json").read_text())
    ms = cfg["method_settings"]
    code_now = {f: sha(P / f) for f in cfg["code_sha256"]}
    ok["E_config"] = {
        "checkpoint_path_equals_v3": cfg["checkpoint"]["path"] == v3cfg["checkpoint"],
        "checkpoint_sha256_matches_file": cfg["checkpoint"]["sha256"] == sha(P / cfg["checkpoint"]["path"]),
        "device_equals_v3": cfg["device"] == v3cfg["device"], "seed_equals_v3": cfg["seed"] == v3cfg["seed"],
        "a5_bounds": [ms["a5"]["lo"], ms["a5"]["hi"]], "a5_matches_v3_text": "0.5 <=" in v3cfg["a5_rule"] and "<= 2.0" in v3cfg["a5_rule"],
        "adaptive_points_equals_v3": ms["enhanced_v2"]["use_adaptive"] == v3cfg["adaptive_points"],
        "enhanced_v3_text": v3cfg["enhanced"], "single_mask_v3_text": v3cfg["single_mask"],
        "code_hashes_unchanged_since_freeze": code_now == cfg["code_sha256"],
        "manifest_sha_matches": cfg["manifest_sha256"] == sha(P / "results/external_ts/cohort_manifest.json"),
        "worklist_sha_matches": cfg["worklist_sha256"] == sha(EVAL / "worklist.json"),
        "full_qc_sha_matches": cfg["full_qc_sha256"] == sha(P / "results/external_ts/qc/full_qc.json"),
    }

    # F. guard (one real, read-only reading)
    import sys
    sys.path.insert(0, str(SCRIPTS))
    from ts_resources import GB, ResourceGuard, ResourceStop
    g = ResourceGuard(int(1.6 * GB), int(2.0 * GB), Path("C:/"))
    try:
        r = g.check("audit")
    except ResourceStop as e:
        r = e.record
    ok["F_guard_real_reading"] = {k: r[k] for k in ("ram_available_bytes", "disk_free_bytes", "ok", "stop_reasons")}

    rep["differences_from_v3"] = [
        "queries are TS slices (frozen cohorts) instead of the 33 MSD held-out patients",
        "supports: all 156 v3 support slices; no leave-one-patient-out mask (TS and MSD share no patients)",
        "slices on both grids evaluated once and tagged in_primary/in_sensitivity (results are order-independent)",
        "output rows add subject/z/cohort flags/support_similarity; masks npz stored content-addressed",
        "infrastructure only: hashing, resume, write-once outputs, resource guard, approval flags",
    ]
    a_ = ok["A_support_pool"]; b_ = ok["B_pairing"]; c_ = ok["C_cohorts"]; e_ = ok["E_config"]
    rep["all_passed"] = all([
        a_["n_patients"] == 33, a_["n_slices"] == 156, a_["equals_split_summary"], a_["equals_v3_per_patient"],
        a_["equals_manifest"], a_["equals_run_config"], a_["dev_equals_v3_excluded"], a_["dev_disjoint"],
        a_["worklist_supports_subset_of_pool"], b_["v3_pairings_reproduced"] == "156/156",
        b_["worklist_pairings_reproduced"] == f"{len(wl)}/{len(wl)}",
        (c_["primary_slices"], c_["sensitivity_slices"], c_["primary_subjects"], c_["sensitivity_evaluable_subjects"]) == (826, 300, 74, 90),
        c_["worklist_equals_union"], c_["worklist_flags_correct"], c_["excluded_disjoint_from_worklist"],
        c_["boundary_subject_step5_slices_all"] == 177, c_["boundary_step5_in_neither_analysis"] == 161,
        c_["boundary_step5_also_on_step17_grid_in_sensitivity"] == 16, c_["boundary_step5_in_primary"] == 0,
        not c_["s1358_in_worklist"], not c_["boundary_subjects_in_primary"],
        c_["duplicate_items"] == 0, ok["D_method_logic_diff_lines"] == EXPECTED_DIFF,
        all(v for k, v in e_.items() if isinstance(v, bool)),
    ])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists - refusing to overwrite")
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=1, sort_keys=True))
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
