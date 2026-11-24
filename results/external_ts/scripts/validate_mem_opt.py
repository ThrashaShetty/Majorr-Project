"""Validate the chunked-attention memory optimisation against ALREADY COMPLETED results (read-only on eval/).

A. patched end-to-end re-evaluation of the completed s0965 slices (z=204, 205) with the exact frozen
   method objects; compare every prediction array (pixel-exact) and every non-timing row field (exact).
B. encoder features for the same query images: patched vs upstream (torch.equal + max abs diff);
   also measures the upstream peak per phase.
C. (only if A differs) unpatched end-to-end re-run of z=204 to test upstream determinism.

Memory: background sampler (5 ms) of process RSS per phase + Windows peak working set / pagefile.
Guard: ResourceGuard run limits (0.3 GB RAM, 2.0 GB disk) checked between steps; start limit as passed.
Writes: --tmp (scratch) and the write-once evidence JSON --out. Never writes to results/external_ts/eval.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_external_ts as E  # noqa: E402
import ts_common as C  # noqa: E402
from ts_resources import GB, ResourceGuard, ResourceStop  # noqa: E402

NON_TIMING_SKIP = {"baseline_ms", "enhanced_ms", "single_decode_ms", "single_ms", "a5_ms"}


class Sampler:
    def __init__(self):
        self.p = psutil.Process()
        self.phase, self.peaks, self._stop = "init", {}, False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while not self._stop:
            r = self.p.memory_info().rss
            if r > self.peaks.get(self.phase, 0):
                self.peaks[self.phase] = r
            time.sleep(0.005)

    def set(self, phase):
        self.phase = phase
        self.peaks.setdefault(phase, self.p.memory_info().rss)

    def stop(self):
        self._stop = True
        self.t.join()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-ram-gb-start", type=float, default=1.6)
    ap.add_argument("--with-upstream-features", action="store_true",
                    help="step B: also run the UPSTREAM encoder (recreates the ~2.4 GB attention transient)")
    a = ap.parse_args()
    a.skip_upstream_features = not a.with_upstream_features
    P = C.PROJECT_ROOT
    out_res = a.out.resolve()
    if out_res == E.EVAL_DIR.resolve() or E.EVAL_DIR.resolve() in out_res.parents:
        raise SystemExit(f"refusing --out inside the evaluation directory: {out_res}")
    rep: dict = {"started_utc": E._now()}
    gs = ResourceGuard(int(a.min_ram_gb_start * GB), 2 * GB, Path("C:/"))
    gr = ResourceGuard(int(0.3 * GB), 2 * GB, Path("C:/"))
    sm = Sampler()
    try:
        cfg, cfg_sha, wl = E.load_frozen(E.EVAL_DIR, P)
        items = [w for w in wl if w["subject"] == "s0965" and E.slice_paths(E.EVAL_DIR, w)["json"].exists()]
        ref = {w["z"]: E.verify_slice(E.EVAL_DIR, w, cfg_sha) for w in items}
        rep["reference_slices"] = sorted(ref)
        rep["guard_start"] = gs.check("validation: before model load")
        import mem_opt
        sm.set("import+load (patched)")
        rep["mem_opt"] = mem_opt.install()
        eng, base, enh = E.real_methods(P, cfg["checkpoint"]["sha256"])
        fns = E.real_fns()
        rep["A"] = []
        for w in items:
            gr.check(f"validation A before z={w['z']}")
            sm.set(f"A patched z={w['z']}")
            q = fns["load_image_rgb"](C.PREPROCESSED / "images" / w["file"])
            gt = fns["load_mask_binary"](C.PREPROCESSED / "masks" / w["file"])
            s = fns["load_image_rgb"](P / w["support_image"])
            smask = fns["load_mask_binary"](P / w["support_mask"])
            t0 = time.perf_counter()
            row, arr = E.evaluate_query(q, gt, s, smask, eng=eng, base=base, enh=enh, select_best_mask=fns["select_best_mask"],
                                        compute_all_metrics=fns["compute_all_metrics"], metric_keys=fns["metric_keys"])
            el = time.perf_counter() - t0
            r0 = ref[w["z"]]
            with np.load(E.slice_paths(E.EVAL_DIR, w)["mask_dir"] / r0["npz_file"]) as z:
                arr_eq = {k: bool(np.array_equal(z[k], np.asarray(arr[k]))) for k in z.files}
                diff_px = {k: int((z[k] != arr[k]).sum()) for k in ("baseline", "enhanced", "single")}
            row_diff = {k: (r0["row"][k], row[k]) for k in r0["row"] if k not in NON_TIMING_SKIP and r0["row"][k] != row[k]}
            rep["A"].append({"z": w["z"], "arrays_identical": arr_eq, "differing_pixels": diff_px, "row_fields_differing": row_diff,
                             "seconds": round(el, 2), "timing_old_ms": {k: r0["row"][k] for k in NON_TIMING_SKIP},
                             "timing_new_ms": {k: row[k] for k in NON_TIMING_SKIP}})
            del q, gt, s, smask, row, arr
            gc.collect()
        A_ok = all(all(x["arrays_identical"].values()) and not x["row_fields_differing"] for x in rep["A"])
        rep["A_all_identical"] = A_ok

        if not a.skip_upstream_features:
            import torch
            rep["B"] = []
            for w in items:
                gr.check(f"validation B before z={w['z']}")
                img = fns["load_image_rgb"](C.PREPROCESSED / "images" / w["file"])
                sm.set(f"B features patched z={w['z']}")
                mem_opt.install()
                fp, _ = enh.bppc.encode(img)
                gr.check(f"validation B upstream z={w['z']}")
                sm.set(f"B features UPSTREAM z={w['z']}")
                mem_opt.uninstall()
                fo, _ = enh.bppc.encode(img)
                rep["B"].append({"z": w["z"], "torch_equal": bool(torch.equal(fp, fo)),
                                 "max_abs_diff": float((fp - fo).abs().max()), "shape": list(fp.shape)})
                del fp, fo, img
                gc.collect()
            mem_opt.install()
        if not A_ok:
            import mem_opt as M
            M.uninstall()
            w = items[0]
            gr.check("validation C (upstream determinism)")
            sm.set("C upstream e2e")
            q = fns["load_image_rgb"](C.PREPROCESSED / "images" / w["file"])
            gt = fns["load_mask_binary"](C.PREPROCESSED / "masks" / w["file"])
            s = fns["load_image_rgb"](P / w["support_image"])
            smask = fns["load_mask_binary"](P / w["support_mask"])
            row, arr = E.evaluate_query(q, gt, s, smask, eng=eng, base=base, enh=enh, select_best_mask=fns["select_best_mask"],
                                        compute_all_metrics=fns["compute_all_metrics"], metric_keys=fns["metric_keys"])
            r0 = ref[w["z"]]
            with np.load(E.slice_paths(E.EVAL_DIR, w)["mask_dir"] / r0["npz_file"]) as z:
                rep["C"] = {"z": w["z"], "arrays_identical": {k: bool(np.array_equal(z[k], np.asarray(arr[k]))) for k in z.files},
                            "row_fields_differing": {k: (r0["row"][k], row[k]) for k in r0["row"]
                                                     if k not in NON_TIMING_SKIP and r0["row"][k] != row[k]}}
    except ResourceStop as e:
        rep["RESOURCE_STOP"] = e.reason
    finally:
        sm.stop()
        mi = psutil.Process().memory_info()
        rep["memory"] = {"phase_peak_rss_bytes": sm.peaks, "process_peak_working_set_bytes": getattr(mi, "peak_wset", None),
                         "process_peak_pagefile_bytes": getattr(mi, "peak_pagefile", None)}
        rep["finished_utc"] = E._now()
        rep["guard_checks"] = gs.history + gr.history
    a.out.parent.mkdir(parents=True, exist_ok=True)
    status, digest = C.write_once(a.out, C.json_bytes(rep))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("guard_checks",)}, indent=1, default=C._json_default))
    print("evidence:", status, a.out, digest)
    return 0 if rep.get("A_all_identical") else 1


if __name__ == "__main__":
    raise SystemExit(main())
