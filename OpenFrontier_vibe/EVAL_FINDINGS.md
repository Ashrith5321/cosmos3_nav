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
| 254 | 0.902 | 0.902 | 12/12/230 | 1.00 |
| 329 | 0.897 | 0.897 | 16/16/297 | 1.00 |

| 486 | 0.854 | 0.854 | 31/31/424 | 1.00 |

Eight independent samples, every one a tie. Wins and losses cancel almost exactly
rather than both being zero, so this is not a case of the world model being
inert — it changes individual episode outcomes, just symmetrically. No goal
category shows a consistent advantage either (largest swing at n=486 is toilet,
-0.07, *against* the world model).

It is null on path efficiency too. Aggregate SPL over all common episodes looks
lower with the world model on, but that number is dominated by which episodes
failed rather than by path length. Restricted to the 384 episodes where **both**
arms succeeded:

```
mean SPL   WM-on 0.4291   WM-off 0.4402   diff -0.0110
WM-on higher: 169   WM-off higher: 162   ties 53
sign test p = 0.742      paired t = -1.10
```

A coin flip. (Comparing aggregate SPL across all common episodes instead is the
same conditioning error as §4 in a different guise, and it briefly produced a
spurious "the world model costs efficiency" reading.)

Against the previous iteration both arms gain the *same* amount, which is what
you would expect if the shared recognition work owns the entire improvement:

| comparison | n | SR | Δ | W/L | p |
|---|---:|---:|---:|---|---:|
| v85 (WM-on) vs v80 | 379 | 0.900 vs 0.865 | +0.034 | 29/16 | 0.072 |
| v85off (WM-off) vs v80 | 547 | 0.861 vs 0.832 | +0.029 | 46/30 | 0.085 |

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

Where the false positives actually stop, by true geodesic distance to the
nearest annotated goal viewpoint (both v85 arms pooled, n=79, median 2.67 m):

| band | n | share | reading |
|---|---:|---:|---|
| <1.0 m | 0 | 0.0% | — (success radius; none land here) |
| 1.0–1.6 m | 16 | 20.3% | right object, just outside the radius |
| 1.6–2.5 m | 18 | 22.8% | same room, likely a second instance |
| 2.5–5 m | 31 | 39.2% | different part of the room or house |
| >5 m | 14 | 17.7% | nowhere near any annotated goal |

Only ~20% are plausibly "correct object, stopped slightly too far". **Nearly 57%
are 2.5 m or further from any goal** — genuine misidentifications, not annotation
edge cases. The agent's own belief is uniform across all of these: it thinks it
is 0.8–1.0 m from its target in nearly every case, having approached correctly
and locked onto the wrong thing.

Only the 1.0–1.6 m band (20%, worth roughly +1.5 SR points if fully converted)
is cheap to attack. §7 shows the rest resists verification.

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

## 8. A learned ranker does better *without* the world model

The strongest form of the test (`listwise.py`). Rather than judging the shipped
hand-tuned Q, train a listwise ranker directly on the navmesh oracle — softmax
over the frontiers of one decision, target = the frontier actually closest to
the goal — and then ablate the two world-model terms out of the feature set.
Split by scene: 8,000 training decisions from 27 scenes, 4,000 held-out
decisions from 9 scenes never trained on.

| ranker | top-1 | mean regret |
|---|---:|---:|
| chance | 16.2% | — |
| wm_mu alone | 20.5% | 3.59 m |
| p_obs alone | 21.8% | 3.49 m |
| hand-tuned Q (shipped) | 26.5% | 3.78 m |
| learned listwise, ALL features | 23.5% | 3.65 m |
| **learned listwise, NO world model** | **26.1%** | **3.28 m** |

**World-model contribution: -2.56 points of top-1, +0.374 m of regret.**

Given an objective built to exploit them and free to weight them arbitrarily,
the ranker does better discarding the world-model features. They also quadruple
seed variance (top-1 sd 0.012 with, 0.003 without), the signature of noise
features hurting generalisation to unseen scenes.

This closes the question the SR ablation opened. The world model is not merely
redundant with signals the planner already has, and it is not being used
suboptimally by a hand-tuned blend: it carries negative information for this
task.

Two side observations. The shipped hand-tuned Q is already competitive with a
learned ranker (26.5% vs 26.1% top-1), so there is no easy win from learning
the blend. But it has the *worst* regret of any ranker tested (3.78 m vs the
learned no-world-model ranker's 3.28 m) — it picks the single best frontier
slightly more often, and is wronger when it misses.

## 9. A second opinion from an independent model does not work either

If re-asking the detector fails because it is the detector, the natural next
move is a model with uncorrelated errors. CLIP is already a dependency, so this
was free to test (`clip_disagree.py`, 45 false positives and 45 successes,
zero-shot over the six goal categories plus 20 distractors drawn from the
confusions the failure data actually shows).

| CLIP threshold | FP rejected | TP rejected | ratio | net eps/1000 |
|---:|---:|---:|---:|---:|
| 0.02 | 48.9% | 28.9% | 1.7 | -88.9 |
| 0.10 | 71.1% | 48.9% | 1.5 | -156.2 |
| 0.30 | 84.4% | 66.7% | 1.3 | -219.2 |

CLIP separates the classes in the mean (P(goal) 0.151 on false positives vs
0.295 on successes) but its rejection ratio never exceeds **1.7:1** against the
7.9:1 required — an order of magnitude short, and worse than the detector
re-asking itself (4.7:1).

Four measured attempts at this failure mode, all net-negative:

| approach | best ratio | verdict |
|---|---:|---|
| detector strict re-ask, global | 2.7:1 | -28 eps/1000 |
| detector strict, distance-gated | — | gate never fires (agent believes 0.85 m) |
| detector strict, threshold-swept | 4.7:1 | -7 eps/1000 |
| CLIP independent second opinion | 1.7:1 | -89 eps/1000 |

Both verifiers share the bias that produced the error: one *is* the detector,
the other is trained on similar web-image distributions and makes correlated
mistakes. Closing this needs a verifier whose errors are genuinely uncorrelated
with the detector's — metric evidence from depth (object extent and size, which
would catch a small lookalike being read as a sofa) or explicit multi-instance
reasoning about which instance is the episode's goal. That is a research step,
not a threshold.

## 10. The bottleneck is semantic, not perceptual

`compose_images` tiles four 640x480 frames into a 1280x960 canvas and then
`COMPRESSION = 0.5` halves it, so every camera view reaches the detector at
**320x240** — a quarter of the sensor's pixels. A chair at 5 m occupies roughly
20x20 px there, which looked like an obvious cause of the recognition failures.

It is not (`res_test.py`, same prompt and frames, only pixels differ; condition
A degrades to the effective 320x240, condition B keeps native):

| | 320x240 (shipped) | 640x480 (native) |
|---|---:|---:|
| TP mean p | 0.900 | 0.897 |
| FP mean p | 0.550 | 0.622 |
| TP accepted @0.7 | 90.0% | 90.0% |
| FP accepted @0.7 | 55.0% | 62.5% |
| **separation TP-FP** | **35.0%** | **27.5%** |

True-positive recall is identical to the decimal, and false positives get
*worse* — more detail gives the model more to read as confirmation, including on
the wrong object. (Caveat: frames come from H.264 video, so "native" is
640x512 video-quality rather than raw sensor output; both conditions share that
compression, and the direction is unambiguous.)

Three independent attacks on the recognition failure now agree:

| intervention | effect |
|---|---|
| re-ask the same VLM (strict prompt, full-res crop) | 4.7:1, net-negative |
| ask an independent model (CLIP) | 1.7:1, net-negative |
| give it 4x the pixels | recall unchanged, separation -7.5 pts |

The agent has enough pixels and enough looks. It is standing in front of a real
object and assigning it the wrong category with confidence. That is a semantic
failure, and neither more evidence nor a second opinion from a correlated model
addresses it.

## 11. The detector is maximally confident when it is wrong

Forced-choice discrimination — asking the model to *commit* to one category out
of a confusable set rather than confirm a handed hypothesis — is the best
discriminator found (`forced_choice.py`): it names something other than the goal
on **70%** of false positives against 17.5% of successes, 4.0:1, versus 21.9%
/ 7.0% for a stricter yes/no prompt. The information needed to catch these stops
is accessible to the same model; only the question format changes.

It still loses at the ~90:870 base rate (-33.9 eps/1000). The obvious escape is
to run the verifier only where errors concentrate, so the pool it sees is
enriched. Measuring the detection probability **the agent actually acted on**
(parsed from `navigation_log.txt`, not recomputed offline):

```
false positives  mean=0.930   share at exactly 1.0: 90%
successes        mean=1.000   share at exactly 1.0: 100%
```

There is nothing to gate on. The detector saturates at 1.0 on 90% of its own
false positives.

| gate: verify when conf < | FP pool | TP pool | FP rej | TP rej | net/1000 |
|---:|---:|---:|---:|---:|---:|
| ungated | 100% | 100% | 62% | 23% | -60.8 |
| 0.999 | 10% | 0% | 83% | 0% | +4.1 |

The +4.1 rests entirely on TP pool = 0% — zero of 60 successes fell below 0.999.
By the rule of three that is consistent with a true rate up to ~5%, which turns
+4.1 into -0.4. The interval straddles zero, so this is not shippable evidence.

**Why every verification approach failed, in one line.** Each needs an axis along
which errors concentrate. Distance is uninformative (the agent believes it is
0.8-1.0 m from its target in nearly every false positive). Independent models
are correlated (CLIP 1.7:1). Resolution is irrelevant (the failure is semantic).
And confidence is degenerate (1.0 on both classes). The false-positive mode is
irreducible under this architecture without instance-level grounding — knowing
*which* chair is the episode's goal, not merely that a chair is present.

**Caveat.** An earlier note in this file cited a 0.70-vs-0.94 confidence
separation between the classes. That came from recomputing the loose prompt
offline on the final frame, not from the values the agent acted on, and it does
not survive contact with the logs.
