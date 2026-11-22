# v3 held-out validation: Adaptive A5 and single-mask Enhanced

Experimental validation only. **No production source file was modified.** The prototype scripts are in
`scripts/`, and the raw per-slice rows are in `raw_holdout_rows.csv`.

## Protocol

| Item | Setting |
|---|---|
| Development set (A5 was chosen here; excluded) | 8 query patients / 53 slices: case_003, 006, 014, 020, 021, 052, 059, 062 |
| Held-out test set | 33 support patients / 156 slices, never scored before |
| Support pairing | nearest neighbour (same 64×64 thumbnail cosine as `run_evaluation.py`) among the *other 32* held-out patients; never the same patient, never a development patient |
| Model / settings | `sam_vit_b_01ec64.pth`, CPU, seed 42, adaptive points, production preprocessing, metrics, Baseline and Enhanced v2 code |
| A5 rule (fixed) | keep box-only if 0.5 ≤ box-only mask area / support spleen area ≤ 2.0, else Enhanced v2 |
| Single-mask | identical Enhanced v2 prompts with `multimask_output=False` |
| Statistics (pre-registered) | primary: A5 vs Baseline. Patient-level Wilcoxon signed-rank on per-patient mean Dice (n = 33 patients). Holm correction over 4 tests. Patient cluster bootstrap (10,000 resamples) for 95% CIs |

## Slice-level results (156 slices, mean ± std)

| Method | Dice | IoU | Precision | Recall | Boundary IoU | Median time |
|---|---|---|---|---|---|---|
| Baseline | 0.515 ± 0.385 | 0.439 ± 0.360 | 0.546 ± 0.398 | 0.580 ± 0.429 | 0.402 ± 0.336 | 9.3 s |
| Enhanced v2 | 0.335 ± 0.284 | 0.243 ± 0.248 | 0.260 ± 0.260 | 0.701 ± 0.382 | 0.214 ± 0.230 | 18.8 s |
| **Adaptive A5** | **0.529 ± 0.356** | **0.441 ± 0.341** | 0.542 ± 0.381 | 0.678 ± 0.386 | 0.400 ± 0.319 | 9.9 s |
| Single-mask Enhanced | 0.314 ± 0.292 | 0.229 ± 0.253 | 0.266 ± 0.274 | 0.538 ± 0.411 | 0.209 ± 0.236 | 18.8 s |
| *Oracle: better of Baseline/Enhanced by GT (upper bound)* | 0.591 | | | | | |

Mean of the 33 per-patient means (Dice): Baseline 0.494 · Enhanced v2 0.329 · **A5 0.518** · Single 0.307.

Enhanced time includes encoding the support image for the background prototype. Single-mask decoding
took a median of 0.10 s.

## Paired patient-level tests (n = 33 patients)

| Comparison | Slice-mean ΔDice [95% CI] | Patient-mean ΔDice [95% CI] | Patients better / worse / tied | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| **A5 vs Baseline (primary)** | +0.014 [−0.015, +0.044] | +0.024 [−0.005, +0.055] | 12 / 6 / 15 | 0.145 | **0.289** |
| A5 vs Enhanced v2 | +0.194 [+0.127, +0.269] | +0.189 [+0.125, +0.251] | 26 / 5 / 2 | 6e-6 | 3e-5 |
| Enhanced v2 vs Baseline | −0.180 [−0.270, −0.097] | −0.165 [−0.238, −0.090] | 9 / 24 / 0 | 2e-4 | 6e-4 |
| Single-mask vs Enhanced v2 | −0.021 [−0.051, +0.011] | −0.022 [−0.053, +0.010] | 11 / 21 / 1 | 0.195 | 0.289 |

"Tied" means |Δ| ≤ 0.005. With A5, 15 patients are tied because A5 kept box-only on all of their slices.

## Baseline-success vs Baseline-failure slices (Baseline Dice ≥ 0.5)

| Method | 92 success slices | 64 failure slices |
|---|---|---|
| Baseline | 0.812 | 0.088 |
| Enhanced v2 | 0.414 | 0.221 |
| A5 | 0.778 | 0.171 |
| Single-mask | 0.406 | 0.182 |

## A5 selection behaviour (correctness judged afterwards against GT)

* Box-only on 118 slices; Enhanced activated on 38.
* Correct on 126 / 156 (81%). A tie of |Δ| ≤ 0.01 counts as correct.
* 23 missed opportunities: A5 kept box-only, but Enhanced was better.
* **7 harmful activations: A5 switched to Enhanced, but Baseline was better.** This never happened on
  the development set. In 6 of the 7, the area ratio was below 0.5 (0.23–0.50). There the support
  spleen was much larger than the query's box-only mask, which is typical when the query is near the
  pole of the spleen and the nearest support slice is mid-spleen. Examples: case_018_slice_075
  0.83 → 0.09, case_033_slice_055 0.94 → 0.09.
* On the 92 Baseline-success slices, A5 kept box-only on 84 and switched on 8.
* On the 64 Baseline-failure slices, A5 switched to Enhanced on 30.

## Single-mask experiment

* On the 4 development problem cases (`dev_problem_cases.csv`), single-mask improved 3 of 4 Dice
  scores (0.173 → 0.376, 0.148 → 0.516, 0.014 → 0.041) and made 1 worse (0.150 → 0.114).
* On the held-out set it is **not better**: −0.021 Dice, 21 of 33 patients worse, Holm p = 0.29. It
  cuts oversized masks (precision < 0.1 and recall > 0.9) from 20 to 8, but loses recall
  (0.70 → 0.54).
* The development result was a small, selected sample and did not generalise.

## Development vs held-out

| | Development (8 pts / 53) | Held-out (33 pts / 156) |
|---|---|---|
| Baseline Dice | 0.356 | 0.515 |
| Enhanced v2 Dice | 0.332 | 0.335 |
| A5 Dice | 0.397 | 0.529 |
| A5 − Baseline | +0.041 (6/8 patients better, 0 worse) | +0.014 (12/33 better, 6 worse), not significant |
| A5 harmful activations | 0 | 7 |

The development set was harder for Baseline (more box-transfer failures), which favoured A5 there.

## Conclusion

* **A5 is not shown to be better than Baseline.** The primary test is not significant (Holm p = 0.29;
  95% CI includes 0). The point estimate is small and positive (+0.014 slice-level, +0.024 patient-level).
* A5 is clearly better than Enhanced v2, and it is the only variant that never falls far below Baseline.
* Enhanced v2 alone is significantly worse than Baseline on held-out data (−0.18 Dice).
* Single-mask output should not be adopted.
