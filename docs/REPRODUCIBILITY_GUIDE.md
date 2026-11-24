# Reproducibility Guide

## 1. Environment (as used)

| Item | Version |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.11.9 |
| PyTorch | 2.10.0+cpu (6 threads) |
| segment-anything | commit `dca509fe793f601edb92606367a655c15ac00fdf` |
| numpy / scipy / pandas | 2.2.6 / 1.16.2 / 2.3.3 |
| opencv-python / nibabel / psutil | 4.12.0 / 5.3.2 / 7.1.0 |
| fastapi / uvicorn | 0.115.12 / 0.34.2 |

## 2. Identities and checksums

| Artefact | Identity |
|---|---|
| SAM checkpoint `sam_vit_b_01ec64.pth` | SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` |
| TotalSegmentator archive (Zenodo 10047263 v2.0.1) | 3,244,617,817 bytes; MD5 `6b5524af4b15e6ba06ef2d700c0c73e0` |
| External cohort manifest | SHA-256 `b6de738e9f1e168d6e7f5e3c53997efb6b5412bb948a69261d1d0548e565ca26` |
| External preprocessing QC | SHA-256 `56629d880320e6b6196f77192f46a30b4203f4cffb77bc74cea63129d35d8f22` |
| External run config (frozen) | SHA-256 `3933ca8c48374cc9fa64bb137b1bed79b5c8551f92a267bf1adc98678dc64cca` |
| External worklist | SHA-256 `726bb4ee1c23154e6e92910ab6b059a8a72ae877991b8f332b0d75975c668041` |
| Memory-optimisation evidence | SHA-256 `f4c70af9b029480e29f97aadb9536b6b15ad32460b4a2d897981871ea86c0689` |
| Statistical analysis plan (SAP) v1 | SHA-256 `6a1fee26869a2bc8abf2a2e9d587838f1dc4d8871c29a2f84d1b02e8e306285d` |
| SAP amendment 01 (`docs/SAP_AMENDMENT_01_2026-11-23.md`) | SHA-256 `18be6f479aeb9fb8242fb92f208d78017447c765c91d0bedd871c12cc2bcd05a` |
| MSD / v1 / v2 / v3 results | Hash manifests in `results/manifests/` |

Seed 42 is used for the MSD case split, SPR (Structure-guided Prompt Refinement) sampling, perturbation jitter and bootstraps.

## 3. Internal experiments (MSD)

```powershell
python -m src.preprocessing
python run_evaluation.py --results-dir results/<new-folder>     # v1/v2-style run; never reuse an existing folder
```

The v3 held-out validation scripts are kept in `results/v3/scripts/`. They are evidence of how v3 was
run, with paths fixed to this machine.

## 4. External evaluation (TotalSegmentator)

All scripts are in `results/external_ts/scripts/`. Run them with `python -B -I` so no bytecode is
written.

| Step | Command | Output |
|---|---|---|
| Manifest (write-once) | `build_manifest.py` | `cohort_manifest.json` + `.sha256` |
| Preprocess | `prep_external_ts.py --subjects ... [--allow-more]` | `external_datasets/.../preprocessed/` |
| Preprocessing QC | `verify_full.py --out ../qc/<new>.json` | QC report |
| Freeze run config | `eval_external_ts.py prepare` | `eval/run_config.json`, `eval/worklist.json` |
| Evaluate (approval required) | `run_batches.py --execute --i-have-explicit-approval --backup-dest <dir> [--mirror-dest <dir>]` | `eval/slices`, `eval/masks`, `eval/subjects` |
| Verify | `verify_eval_subjects.py --subjects ... --out <new>.json` | Independent recomputation |
| Back up | `make_backup.py --dest <dir>` | Verified, timestamped zip + manifest |
| Finalize | `finalize_eval.py` | `eval/raw_rows.csv`, `raw_rows_primary.csv`, `raw_rows_sensitivity.csv` |
| Analysis | `analyze_external_ts.py validate-v3`, then `analyze_external_ts.py run --out-dir analysis/<new>` (SAP v1 + Amendment 01) | `analysis/phaseB_20261124T052029Z/analysis.json` |

**Expected outputs when complete:**

* 1,078 slice results, each with a content-addressed prediction file;
* 90 subject CSVs with completion markers;
* `raw_rows.csv`, `raw_rows_primary.csv` (826 rows) and `raw_rows_sensitivity.csv` (300 rows).

## 5. Verifying a copy of the results

* **Every external result:** run `verify_eval_subjects.py`. It recomputes every metric from the saved
  predictions; the s0965 check found a maximum deviation of 0.0.
* **A backup:** run `make_backup.py --verify-only <zip>`, which re-hashes every member against its
  manifest.

## 6. Hardware notes

* **Measured on the test laptop** (7.34 GiB RAM): peak working set 1.44–1.45 GiB (about 1.55 GB) with `mem_opt`; about 13.2 s
  per slice on CPU.
* **Before long runs:**
  * close browsers;
  * connect AC power;
  * the batch driver keeps the system awake only while it is running.
