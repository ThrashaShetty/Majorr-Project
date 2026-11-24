# Experimental Setup

## Common settings (every experiment)

| Item | Value |
|---|---|
| Model | SAM ViT-B, `checkpoints/sam_vit_b_01ec64.pth` (SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`); not fine-tuned |
| Device | CPU (PyTorch 2.10.0+cpu, 6 threads) |
| Seed | 42 (split, SPR (Structure-guided Prompt Refinement) sampling, pairing, bootstrap) |
| Hardware | Windows 11 laptop, 7.34 GiB RAM, NVMe SSD |
| Pairing | Nearest support slice by 64×64 thumbnail cosine similarity |

## Experiments

| ID | Data | Methods | Status | Results |
|---|---|---|---|---|
| v1 | MSD, 8 query patients / 53 slices | Baseline, FoB, Enhanced (original prompts) | Done | `results/` |
| v2 | Same 53 slices | Baseline, FoB, Enhanced v2 (body-mask and positive-point constraints) | Done | `results/v2/` |
| v3 | MSD held-out: 33 patients / 156 slices, never scored before; supports from the *other* 32 patients | Baseline, Enhanced v2, A5, Single-mask | Done | `results/v3/REPORT.md` |
| External | TotalSegmentator: 74 primary / 90 sensitivity patients; 1,078 slices; supports = all 156 MSD v3 slices | Baseline, Enhanced v2, A5, Single-mask (frozen) | Done (90 of 90, independently verified; analysed under SAP v1 + Amendment 01) | `results/external_ts/analysis/phaseB_20261124T052029Z/` |

## Measured results available now

| Experiment | Key numbers |
|---|---|
| v2 (53 slices) | Baseline Dice 0.357; Enhanced v2 0.332 (Wilcoxon p = 0.97, not significant) |
| v3 (33 patients) | Mean of per-patient Dice: Baseline 0.494 · Enhanced v2 0.329 · A5 0.518 · Single-mask 0.307. A5 vs Baseline Holm p = 0.29 |
| External, primary (74 patients) | Mean of per-patient Dice: Baseline 0.452 · Enhanced v2 0.289 · A5 0.450 · Single-mask 0.288. A5 vs Baseline −0.002 [−0.021, +0.016], Holm p = 1.0; Enhanced v2 vs Baseline −0.163, Holm p = 4e-8 (worse) |
| External, sensitivity (90 patients) | Baseline 0.478 · Enhanced v2 0.295 · A5 0.488 · Single-mask 0.294. A5 vs Baseline +0.010, Holm p = 0.45; Enhanced v2 vs Baseline −0.183, Holm p = 1e-8 (worse) |
| Exploratory (not confirmatory) | Uncertainty vs Dice (Spearman), and Dice vs 0–5 simulated clicks, for Baseline and A5 on the 156 v3 slices: **[PENDING: v3 exploratory run]** (`experiments/exploratory/v3_uncertainty_and_clicks.py`) |

## Measured performance (external pilot, s0965)

These are **engineering measurements, not accuracy results**:

* **Wall time:** about 13.2 s per slice for all four methods together.
* **Peak RAM:** working set 1.44–1.45 GiB (about 1.55 GB) with the memory optimisation; 3.07 GiB (3.29 GB) without it.

## Safeguards in the external evaluation

* **Frozen inputs:** a frozen run config (code hashes) and worklist. The analysis plan was frozen
  before results were analysed.
* **Resource guard (binary units):** 1.6 GiB of RAM to start; 0.3 GiB of RAM and 2.0 GiB of disk while running.
* **Outputs:** write-once results with resume; independent verification after every batch; a verified
  backup after every batch.
