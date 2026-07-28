# Phase 10 — frozen deep ensemble, uncertainty and calibration

**Preregistered mechanical gate: 4 of 6. Robust scientific assessment: 3 of 6
supported.**

The two are recorded separately on purpose. Criterion 4 as registered required
only a lower point-estimate Brier, and that holds — so it PASSES mechanically.
Reversing it by adding a significance requirement after seeing the table would
itself be post-hoc, so the registered verdict stands and the statistical
evidence is reported beside it, not substituted for it.

Uncertainty is genuinely informative and area intervals are calibrated. Target
presence remains uncalibratable. Criteria 3 and 6 are genuine failures of
ensembling this model; the Phase 8 v0 pathologies explain them but do not
excuse them.

The three archived Phase 8 seeds were used unchanged. Nothing was retrained.
The only fitted quantities are calibration parameters, fitted on calibration
scenes and applied unmodified to validation scenes.

## Data

| | groups | branches | scenes |
|---|---|---|---|
| calibration | 429 | 1921 | 22 |
| validation | 494 | 2319 | 22 |

Scene-disjoint, drawn from `dev_pool_v1` (untouched), zero overlap with the 58
consumed scenes, sealed set never read.

**Caveat worth stating plainly.** The frozen manifests allocate 45 scenes per
split, but generation reached its 500-group target before exhausting them, so
only **22 scenes per split** are realised. The splits remain disjoint and
frozen; effective scene diversity is half what the manifests permit, which
widens the honest uncertainty on every number below.

## Gate

| # | criterion | preregistered | robust |
|---|---|---|---|
| 1 | uncertainty positively correlates with realised error | **PASS** | supported |
| 2 | risk–coverage beats random ordering | **PASS** | supported |
| 3 | calibration improves Brier or NLL | **FAIL** | genuine failure |
| 4 | calibrated target Brier beats the constant prior | **PASS** (nominal) | **NOT supported** |
| 5 | area intervals achieve registered coverage | **PASS** | supported |
| 6 | ensemble does not degrade deterministic metrics | **FAIL** | genuine failure |

## What passed

### Uncertainty is informative (criteria 1 and 2)

Spearman rank correlation between uncertainty and realised error, validation:

| head | entropy | mutual information | variance |
|---|---|---|---|
| occupancy free | +0.749 | +0.370 | +0.512 |
| occupancy occupied | +0.698 | +0.475 | +0.550 |
| occupancy semantic | +0.780 | +0.404 | +0.536 |
| crossing | — | **+0.921** | — |
| area (seed std) | — | +0.164 | — |

Every one is positive, and risk–coverage AURC beats the random-ordering
baseline on the majority of heads. Selective prediction works: withholding the
least confident predictions genuinely lowers error.

### Area intervals are calibrated (criterion 5)

Split-conformal, spread-normalised residuals, quantile fitted on calibration
only:

- empirical coverage **0.870** against nominal 0.900, deviation 0.030 (tolerance 0.050)
- mean width 15.4 m², median 14.4 m²
- ensemble MAE **3.94 m²** against a constant-prior MAE of 5.88 m²
- Pearson 0.720, Spearman 0.737
- oracle-choice-among-seeds MAE 2.97 m² (diagnostic only — it uses the truth)

### Crossing head is strong

AUROC **0.968**, accuracy 0.910, Brier 0.072 → 0.067 calibrated, NLL 0.359 →
0.239. It beats the constant prior decisively: Brier margin **+0.128**, 95% CI
[+0.115, +0.140] by decision-group bootstrap.

## What failed

### Criterion 4 — nominal PASS, statistically unsupported

The registered criterion required only a lower point-estimate Brier, and Platt
delivers one. It therefore **passes as registered**. What it does not do is
survive resampling.

| | Brier | margin over prior | group CI (registered) | scene CI (sensitivity) |
|---|---|---|---|---|
| constant prior | 0.22529 | — | — | — |
| temperature | 0.24917 | **−0.02388** | [+0.0178, +0.0301] **worse** | [+0.0054, +0.0445] **worse** |
| Platt | 0.22502 | +0.00028 | [−0.0049, +0.0044] zero inside | [−0.0119, +0.0139] zero inside |

Platt's fitted slope is **0.060** with intercept −0.297: the calibrator has
essentially learned to emit a constant near the base rate. Ensemble target
AUROC is **0.544** — barely above chance.

So calibration's only achievement is to stop the head being *confidently*
wrong: ECE 0.318 → 0.137, NLL 1.712 → 0.642. It adds no discrimination.

**How this is recorded.** The registered criterion did not specify a
significance test, so it is kept as a **PASS** in `gate.json` — adding the
requirement now would be post-hoc. The robust assessment is recorded separately
as *not supported*. Both appear in the summary; neither overwrites the other.

The scientific reading is unambiguous even though the mechanical gate passes:
the calibrator collapses to the prior, the margin's CI includes zero under both
resampling schemes, and temperature scaling is significantly *worse* than doing
nothing. Target presence is not calibratable from these inputs.

A contributing confound: target base rate is **0.432 on calibration** and
**0.284 on validation**. A calibrator fitted to the first prevalence is
mis-specified for the second, and the constant-prior baseline inherits the same
shift. With only 22 scenes per split, that difference is plausibly a
scene-sampling artefact rather than a property of the task.

### Criterion 3 — calibration helps the scalar heads, not the spatial ones

| head | Brier raw → calibrated | NLL raw → calibrated | improved |
|---|---|---|---|
| target | 0.3485 → 0.2489 | 1.7116 → 0.6914 | yes |
| crossing | 0.0718 → 0.0673 | 0.3591 → 0.2394 | yes |
| occupancy free | 0.05442 → 0.05460 | 0.18927 → 0.19013 | no |
| occupancy occupied | 0.03144 → 0.03146 | 0.12436 → 0.12448 | no |
| occupancy semantic | 0.07150 → 0.07167 | 0.23811 → 0.23896 | no |

2 of 5 improved, so the majority rule fails. **This is a genuine failure.** The
three spatial channels were already calibrated (ECE 0.007, 0.013, 0.020) with
fitted temperatures of 1.07, 1.01 and 1.05 — near no-ops — and the degradations
are in the fourth decimal place. That explains *why* calibration did not help,
and it originates in known Phase 8 v0 head behaviour, but the observed
degradation stands: calibrating this ensemble did not improve these channels.
The diagnosis is not an exemption.

### Criterion 6 — ensembling hurts the occupied channel and the target head

| metric | best seed | ensemble | relative loss |
|---|---|---|---|
| occupancy free IoU | 0.3370 | 0.3407 | −0.011 (better) |
| **occupancy occupied IoU** | 0.0497 | **0.0276** | **+0.445** |
| occupancy semantic IoU | 0.3109 | 0.3177 | −0.022 (better) |
| **target AUROC** | 0.5607 | **0.5437** | **+0.030** |
| crossing AUROC | 0.9609 | 0.9677 | −0.007 (better) |
| area MAE | 3.939 | 3.944 | +0.001 |

**These are genuine failures of ensembling this model.** The occupied collapse
originates in the known Phase 8 v0 defect — both occupancy channels are averaged
against one full-window denominator, so `p(revealed ∧ occupied) ≈ 0.038` and
predictions sit just under the 0.5 threshold, and averaging three seeds pulls
more mass below it. The underlying probabilities are *better* calibrated after
averaging (Brier 0.0314, ECE 0.0072); it is the hard threshold that fails. But
the degradation is real and was caused by ensembling: a v1 head that fixes the
loss formulation is required before this criterion can be re-attempted, and
until then the ensemble genuinely costs 44% of occupied IoU.

The target AUROC drop is a 0.017 movement on a head operating at chance, where
seed-to-seed ordering is close to arbitrary — again an explanation, not an
exemption.

## Scene-level bootstrap sensitivity

494 groups come from only **22 scenes**, and groups within a scene share
geometry, layout and semantics. Resampling groups therefore treats correlated
units as independent and understates the interval. The registered group
bootstrap is preserved as primary; scene-level resampling — drawing whole scenes
with replacement and taking *every* group from each — is reported alongside it.

| comparison | group CI | scene CI | width ratio | conclusion |
|---|---|---|---|---|
| target, Platt vs prior | [−0.0049, +0.0044] | [−0.0119, +0.0139] | **2.8×** | unchanged: zero inside |
| target, temperature vs prior | [+0.0178, +0.0301] | [+0.0054, +0.0445] | 2.3× | unchanged: worse |
| crossing, Platt vs prior | [−0.1396, −0.1153] | [−0.1492, −0.1061] | 1.8× | unchanged: beats prior |
| crossing, temperature vs prior | [−0.1346, −0.1089] | [−0.1446, −0.1004] | 1.7× | unchanged: beats prior |

Scene resampling widens every interval by 1.7–2.8×, confirming the clustering
concern is real. **No conclusion changes**, which makes the surviving claims
stronger rather than weaker: the crossing head beats the prior even under the
conservative scheme, and the target head fails under both.

This is a sensitivity analysis only. It does not rewrite the registered gate.

## Ranking stability — the number that matters for Phase 12

- **top-1 agreement across seeds: 0.619** over 494 decision groups
- mean pairwise Kendall τ: 0.743

The three seeds disagree about which frontier is best in roughly **38% of
decision groups**. Any planner built on a single seed inherits that
arbitrariness. This is not part of the registered gate, but it is the most
directly actionable finding here for Phase 12.

## Semantic disagreement

Mean pairwise cosine disagreement across seeds: 0.117 (median 0.103) — the
seeds broadly agree on *where* semantic mass lies.

## Conclusions

1. **Ensemble uncertainty is usable.** It correlates with error on every head
   and supports selective prediction. This was the primary thing Phase 10 set
   out to establish, and it holds.
2. **Area prediction is calibrated and useful**, with conformal intervals close
   to nominal coverage.
3. **Target presence cannot be calibrated into usefulness.** No calibration of a
   chance-level head can create discrimination that the inputs do not contain.
   Recording this honestly, per the Phase 10 instruction: this is exactly the
   gap Phase 11 visual frontier memory is meant to close by adding semantic
   evidence, not something Phase 10 could have fixed.
4. **Two gate failures are diagnostic of Phase 8 v0**, not of the ensemble
   method: the occupied-channel threshold collapse and the chance-level target
   head both trace to the v0 loss formulation already documented in
   `frontierworld/models/predictor.py`.

## What is not claimed

- Not that ensembling is harmful. It improved or left unchanged every metric
  except the two traceable to the v0 defect.
- Not that target presence is unpredictable in principle — only that *these
  inputs* and *this head* do not support it.
- No closed-loop or planning result. Ranking stability is measured, not acted on.
- With 22 scenes per split, all numbers carry more scene-sampling uncertainty
  than the branch counts alone suggest.

## Files

| file | contents |
|---|---|
| `results.json` | full evaluation table, all heads, per-seed and ensemble |
| `gate.json` | six criteria with measured values and verdicts |
| `target_check.json` | decision-group bootstrap of calibrated-vs-prior |
| `phase10_split_summary.json` | frozen split provenance and hashes |
