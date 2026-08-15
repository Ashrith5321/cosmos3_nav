# Evaluation findings — HM3D v2 ObjectNav val

Running log of measured results. Numbers here are reproduced by the scripts
named alongside them; none are estimates.

## 1. The world model does not change success rate

Paired on common (scene, episode) pairs, v85 (world model on) against v85off
(identical stack, `world_model.enabled: false`):

| n common | v85 SR | v85off SR | W/L/T | p |
|---:|---:|---:|---|---:|
| 23 | 0.870 | 0.870 | 1/1/21 | 0.75 |
| 45 | 0.889 | 0.889 | 2/2/41 | 1.00 |
| 85 | 0.859 | 0.859 | 4/4/77 | 1.00 |
| 215 | 0.902 | 0.907 | 9/10/196 | 1.00 |
| 226 | 0.898 | 0.898 | 10/10/206 | 1.00 |

Five independent samples, every one a tie. Wins and losses cancel almost exactly
rather than both being zero, so this is not a case of the world model being
inert — it changes individual episode outcomes, just symmetrically.

## 2. Why: the relevance signal does not discriminate *within* a decision

From the `--wm-oracle` logs, which record every live frontier's `wm_mean`,
`p_obs`, hand-tuned `utility`, and `oracle_value` (negative geodesic distance
from that frontier to the nearest goal viewpoint).

Scored on the **same** 6,190 decisions where every signal is defined
(`rank_quality.py`). Restricting to a common subset matters: each signal's own
availability subset differs in difficulty, and scoring each on its own subset
inverts the conclusion.

| signal | top-1 | mean regret | spearman |
|---|---:|---:|---:|
| utility (shipped Q) | 40.8% | 1.41 m | 0.139 |
| p_obs (VLM only) | 31.7% | 1.53 m | 0.057 |
| **wm_mean** | **26.4%** | **1.73 m** | **-0.026** |
| random | 28.8% | 1.71 m | — |

`wm_mean` ranks frontiers at chance. Regret = extra geodesic metres to the goal
versus picking the oracle-best frontier.

The mechanism (`wm_spread.py`) — spread of each signal *across the candidates of
a single decision*:

| signal | within-decision std | max-min gap | effectively tied |
|---|---:|---:|---:|
| utility | 0.0869 | 0.2196 | 0.5% |
| p_obs | 0.1015 | 0.2533 | 12.8% |
| wm_mean | 0.0249 | 0.0650 | 24.1% |

`wm_mean`'s *global* std is 0.164, comparable to `p_obs`'s 0.140 — the model is
not collapsed. But between the sibling frontiers of one decision it is 4×
flatter, and a quarter of decisions are effectively ties. A term that is
near-constant across a decision's candidates adds the same amount to every
frontier's Q and cannot move the argmax. That is a mechanical account of the
tie in §1, not merely a correlate of it.

It is not flat-but-correct either (`wm_conditional.py`) — splitting decisions by
how strongly `wm_mean` separates candidates, it tracks chance in every quartile:

| wm spread quartile | n | wm top-1 | chance |
|---|---:|---:|---:|
| Q1 gap 0.000-0.008 | 867 | 38.1% | 39.2% |
| Q2 gap 0.008-0.025 | 867 | 27.6% | 32.3% |
| Q3 gap 0.025-0.079 | 867 | 32.3% | 28.4% |
| Q4 gap 0.079-0.316 | 868 | 24.7% | 23.9% |

**Correction.** An earlier analysis reported `wm_mean` as the best single
ranking feature (ρ=+0.107 vs `p_obs`'s +0.042, 72k decisions). That pooled
across decisions, so it measured between-scene variance — real, but useless for
choosing among the frontiers of one decision. Within-decision it is at chance.

**Where the flatness is NOT.** The obvious suspects were checked and cleared.

The predictor is not context-collapsed — it emits genuinely frontier-specific
predictions (`wm_identical.py`, 37,866 decisions averaging 7.83 frontiers):
2.76 distinct predicted rooms and 5.88 distinct event texts per decision, with
only 12.7% of decisions assigning every frontier the same room.

Nor is the goal evaluator laundering that diversity away. Scoring each frontier
directly by the hand-written room→object co-occurrence prior on its predicted
room — bypassing the contrast-normalised CLIP cosine entirely — does not rank
better either (`room_prior_rank.py`). On the 17,318 decisions where that prior
is not identical across all frontiers:

| signal | top-1 | mean regret |
|---|---:|---:|
| p_obs | 19.7% | 2.65 m |
| room_prior | 18.5% | 2.92 m |
| random | 17.4% | 2.81 m |
| wm_mean | 14.2% | 3.14 m |

(57.4% of decisions were unusable for this test because the prior was constant
across their frontiers, and the prior table is hand-written, so this probes the
hypothesis rather than settling it exhaustively.)

**So the predictions are diverse but not *right*.** The predicted room does not
correspond to what actually lies beyond that particular frontier in any
goal-relevant way, whether read through the learned evaluator or an explicit
prior. This reframes the one surviving positive: the 0.955 cosine to realized
futures is only 0.050 above the trivial baseline's 0.905, and indoor scenes are
broadly self-similar, so most of that score is generic scene resemblance rather
than frontier-specific foresight. A within-decision contrastive objective is
still the natural next thing to try, but it is a genuine research bet, not a
known fix — the earlier framing of "the mechanism works, only the objective is
wrong" is not supported.

## 3. Where the remaining failures are

Paired against v80 on 257 common episodes, the v85 recognition changes:

```
FIXED  by v85: 18   final_stop=11, false_positive=4, max_steps=3
BROKEN by v85: 19   false_positive=10, robot_stuck=4, exception=2, ...
```

A 1:1 trade. Timeouts were converted into successes (`final_stop` fell from 38%
to 12% of failures) and bought back as false positives (33% → **55%**).

False positives are not mostly wrong-category errors — they are mostly stops
made too far away:

| arm | n | median stop distance | 1–1.5 m | 2.5–5 m | >5 m |
|---|---:|---:|---:|---:|---:|
| v80 | 76 | 2.66 m | 17% | 32% | 22% |
| v85off | 27 | 2.96 m | 15% | 37% | 30% |
| v85 | 14 | 2.15 m | 7% | 36% | 7% |

The two ends need opposite fixes: the >5 m group is a recognition failure, the
1–1.5 m group is a stopping-distance failure where stricter recognition would
make things worse.

## 4. Sampling artifact: never read a running SR

Within the same scenes, ordered by completion time (`ep_order` analysis):

| completion rank | n | SR | false positive |
|---|---:|---:|---:|
| their first 3 | 57 | 0.842 | 14.0% |
| their 9th+ | 69 | 0.928 | 2.9% |

False positives end episodes early, so the first episodes to finish in any scene
are enriched with them by 5×. A partially-complete arm's raw SR is therefore
biased by how deep each scene has got, and two arms at different completion
depths are not comparable at all. Only paired (scene, episode) comparisons are
safe — and even those are affected while an arm is young, because conditioning
on "this arm finished the episode" selects for its fast failures.

## 5. Infrastructure defects found and fixed

These cost real episodes and are not results:

- **SAM3 fleet death (cluster 1).** 7 of 8 segmentation servers were down for
  ~2 hours; 31 of 36 workers spun on connection-refused, producing 22k error
  lines each while 5 workers carried the run. Throughput was ~14% of capacity
  and the completed episodes came from only 5 scenes. Watchdog now armed.
- **`compose_images` canvas overflow** (`ec471fc`). The detection-cadence boost
  emits image counts absent from `COMPOSITIONS` (3, 5, 7) and the fallback grid
  was too small, writing the last view past the end of the canvas. Cost 3
  episodes, all within 0.7 m of the goal.
- **Segmentation grid desync** (`7517704`). The mask splitter derives its grid
  from the `n_images` it is passed and its `image_index` indexes
  `composition_depths`, but `compose_images` laid the grid out from
  `len(composition_images)`. Once the cadence boost passed `n_images_eff=2`
  while 4–5 frames were buffered, the two grids disagreed and masks came back
  sized to the wrong cell. The first fix only moved this crash downstream.

## 6. v86

Targets the false-positive mode, flag-gated so v85 behaviour is the default.

An offline A/B on 60 real stop frames (`validate_prompt.py`) put the first
version of the design in doubt: a category-discriminative confirmation prompt
rejects 29% of the false positives the shipped prompt accepts, but also **11% of
the true positives**, which loses at the ~9:1 success:FP base rate. The intended
rescue was to gate it on distance so only far, low-confidence stops paid the
stricter bar. §7 shows that gate keys on the wrong variable and the whole
mechanism is net-negative at every threshold, so it is disabled.

What v86 actually ships is the crop-rescue tightening: `max(composite, crop)` in
the zoomed-crop second opinion was a monotone loosening of the detector that
could only ever add detections, and the paired v80 comparison charges it with 10
newly-broken episodes against 4 fixed. It is replaced with a band-limited rescue
requiring the crop to clear threshold by a margin.

## 7. Stop-time re-verification cannot fix the false positives

v86's original design was a stricter stop-time confirmation. Measurement killed
it before it cost any compute.

Recovery after a rejection is measurable from the shipped `reverify_on_stop`
gate: of 38 episodes with at least one stop-time rejection, 21 still succeeded
(55.3%) against a base rate of 84.5%. So a correct rejection is worth **+0.55**
episodes and a wrong one costs **-0.45**.

Sweeping the strict threshold over stop frames the shipped detector accepts
(32 false positives, 43 successes):

| strict threshold | FP rejected | TP rejected | ratio | net eps/1000 |
|---:|---:|---:|---:|---:|
| 0.1 | 21.9% | 4.7% | 4.7 | -7.4 |
| 0.2–0.7 | 21.9% | 7.0% | 3.1 | -16.5 |

At the ~90:870 base rate the gate must reject false positives **7.9×** more
often than true positives to break even. Its best operating point reaches 4.7×.
The sweep is flat because the VLM's probabilities are bimodal (0.0 or ~1.0), so
no threshold recovers it.

The distance gate that was supposed to rescue the trade keys on the wrong
variable: the agent believes it is 0.8–1.0 m from its object in nearly every
false positive, while the true distance to a goal ranges 1–18 m. It approaches
correctly and stops at the wrong *instance*.

**These are confident misidentifications, not low-confidence errors.** Re-asking
the same VLM — discriminative prompt, full-resolution crop, at 0.85 m — reaffirms
78% of them. Fixing this failure mode needs a different information source (a
second model, geometric or size priors, instance-level reasoning), not a better
prompt. Planner reachability is also ruled out: successes hit the near-miss
replan path *more* often (20%) than false positives (16%).

v86 therefore ships only the crop-rescue tightening, which the paired v80
comparison independently charges with 10 newly-broken episodes against 4 fixed.
