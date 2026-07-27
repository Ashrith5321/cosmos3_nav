# Phase 9D — Cosmos 3 Nano counterfactual rollouts

**Status: CLOSED — NEGATIVE RESULT.**

The preregistered in-distribution yaw diagnostic FAILED in both arms. See
[`PHASE9_CLOSURE.md`](PHASE9_CLOSURE.md) for the decisive experiment and the
disposition. The sections below record how that conclusion was reached.

> Cosmos 3 Nano consumes camera-pose actions, but fails directional action
> adherence **even within its reference rotational regime**. At 0.25-0.50 deg
> per frame, bracketing the reference trajectory's own 0.264 deg/frame, the
> model still does not turn the camera in the commanded direction; with zero
> translation it does not turn at all. So the failure is not a distribution-shift
> artefact of ObjectNav's 30 deg discrete turns.

Steps 8, 10 and 11 were not run. Phase 10 will use the Phase 8 structured-model
ensemble. Cosmos is retained only as a qualitative, non-action-faithful baseline.

The credential blocker is resolved: the HuggingFace token was supplied, the
Cosmos-Guardrail1 licence accepted, and **guardrails were enabled for every
generation run** (`--no-guardrails` was never used). Seven rollouts were
generated. What they show is a clear negative result, reported below without
adjusting any threshold to soften it.

## The headline finding

Cosmos 3 Nano **does consume** the action conditioning, but **does not follow
commanded turn direction** at the rotation rates the ObjectNav option requires.
The cause is quantified and specific.

### It consumes the action

Holding conditioning frame, prompt and seed fixed and varying *only* the action:

| probe | mean frame difference | mean horizontal flow |
|---|---|---|
| `still` (identity) | 2.29 | −0.06 |
| `forward` (0.25 m/frame) | 2.72 | −0.13 |
| `turn_left` (+30°/frame) | 31.60 | −17.50 |
| `turn_right` (−30°/frame) | 39.44 | −17.80 |

The action tensor is unambiguously driving the output — rotation probes produce
an order of magnitude more motion than the `still` control.

### It does not follow the commanded direction

`turn_left` and `turn_right` are opposite commands, yet both pan the camera the
**same** way (flow −17.50 vs −17.80, separation 0.30). This is not a sign
convention error: a flipped convention would still yield *opposite* results.
Repeating at a gentler 7.5°/frame gives −10.10 vs −0.33 — a 30× magnitude
difference, so the two commands are distinguished, but still not opposite.

Within the smoke rollout itself, commanded yaw correlates with apparent
horizontal flow at only **0.16** against a preregistered threshold of 0.30. That
check is recorded as FAIL. The threshold was not lowered.

### Why: rotation is 156× out of distribution

Comparing our option's action tensor against `camera_action_44.json`, the
reference camera trajectory NVIDIA ships with the model:

| | reference | our option | ratio |
|---|---|---|---|
| translation / frame | 0.884 m | 0.584 m | 0.66× |
| \|yaw\| / frame | **0.264°** (max 0.349°) | **41.25°** (max 120°) | **156×** |

The `camera_pose` embodiment is exercised on near-pure translation with
sub-degree per-frame rotation. Our translation is comfortably in range; our
rotation is not remotely. ObjectNav's 30° discrete turn is the problem.

**A missing normalizer is ruled out** as the explanation: no `camera_pose`
normalizer file exists, and the framework's own inference passes
`action_normalizer=None`, so raw metric actions are the expected input.

### What this implies

Reaching the reference's ~0.3°/frame would need roughly 100 frames per single
30° primitive turn — thousands of frames for a 66-action option. Cosmos
`camera_pose` forward dynamics, as shipped, cannot represent this option space.

Qualitatively the smoke rollout holds coherent indoor geometry for about 12 of
17 frames and then degrades into smeared texture (see `figures/`).

## Why Steps 8, 10 and 11 were not run

They were gated on action fidelity, which failed twice: first at the option's
own 30°/frame, then decisively at the model's own 0.25-0.50°/frame reference
regime. Both outcomes and their consequences were predeclared. See
[`PHASE9_CLOSURE.md`](PHASE9_CLOSURE.md).

## What is NOT claimed

- **Not** that Cosmos action conditioning is broken in general. The failure is
  specific to rotation rates far outside the shipped reference distribution.
- **No** comparison against the Phase 8 predictor was made.
- **No** revelation-quality result of any kind.
- The 7.5°/frame probe is a single pair of rollouts at one seed; it indicates
  rather than establishes the low-rate behaviour.

## What is established



### The action interface is real (Step 2)

Cosmos 3 Nano is **Class C**: it both consumes and emits actions, selected by an
explicit `model_mode`. `forward_dynamics` reads an action file; `policy` and
`inverse_dynamics` pass zeros and generate actions instead. So the
counterfactual "what would I see if I executed option ω_i" is directly
expressible, and the rule about not faking action conditioning with text never
came into play.

The right embodiment is `camera_pose` (domain 2, raw action dim **9**) — a
free-moving camera in a static scene, which is exactly a robot ego-camera
executing a frontier crossing. Its 9-D action coincides with the
`POSE_DELTA_DIM = 9` already fixed in `cosmos_contract.py`.

Full evidence in [`action_interface_report.md`](action_interface_report.md).

### The model fits one GPU (Step 4)

| quantity | value |
|---|---|
| transformer parameters | 15,173,669,120 |
| VAE parameters | 704,688,668 |
| weights in BF16 | 30,319.6 MB |
| reserved VRAM after load | 30,600 MB |
| A6000 capacity | 49,140 MB |
| headroom | 18,067 MB |

One A6000 in BF16, no sharding, no offload, no quantization. The second GPU is
free for the depth estimator or a second rollout.

Weights were loaded component-by-component rather than through
`Cosmos3OmniPipeline.from_pretrained`, because that constructor instantiates the
guardrail safety checker eagerly. This is a *measurement* path, not a generation
path; no safety component was disabled by it.

### The smoke branch was chosen before generation (Step 5)

`5biL7VEkByM_0_t12_max_info_gain`, branch 1, frontier 1 — first in lexical order
among **1497** branches satisfying a rule fixed in advance: crosses its
frontier, contains both rotation and translation, and reveals a nontrivial
amount of both free and occupied space. Its group has 7 frontiers, so it also
serves Step 10's "≥3 candidate options" requirement.

A note on the scene pool. `dev_pool_v1.json` turned out to be the wrong pool: it
is the set of scenes still *untouched*, reserved for v1 development, and it is
disjoint from every scene any dataset has been generated on. The smoke branch is
drawn instead from the `full.json` **train** split — already consumed by Phase 8
v0 training, so it carries no residual held-out value.

`sealed_final_v1.json` was never opened. Two independent guards: it is drawn
from HM3D **val-split** scenes while `full.json` is entirely HM3D **train-split**,
and every selected path is asserted to lie under `hm3d_v0.2/train/`.

### The conditioning is honest (Step 6)

The 17 camera poses are the **nominal** option trajectory — the kinematic ideal
of the 66 recorded actions (38 forward, 28 turns), integrated from the
decision-point pose. The executed trajectory is deliberately not used: it
encodes which forward steps were blocked by geometry the agent had not yet
observed, which is part of the answer the model is being asked to predict.

A useful check fell out of this. The decision-point pose is not stored in the
dataset; it is recovered by inverting the first recorded action. Re-rendering
the view at the recovered pose reproduces the stored conditioning frame to a
mean absolute difference of **0.001/255**, with zero pixels differing by more
than 8. The recovery is exact, which in turn means the re-rendered conditioning
depth is the genuine observed sensor depth at that instant — not future
information.

Sanity on the action tensor: 16 × 9, finite, per-step chords summing to 9.344 m
against a nominal path length of 9.5 m (chords must be shorter, and are), with a
straight-line displacement of 4.458 m.

### Cache keys discriminate (Step 9)

The request key covers checkpoint hash, preprocessing version, scheduler, steps,
guidance, shift, resolution, frame count, fps, seed, dtype, action mode,
embodiment, prompt, conditioning-frame hash and the action vectors themselves.
21 tests assert that identical inputs collide and that changing any one of these
does not — in particular that **changing the option changes the key**, without
which counterfactual rollouts would silently share cache entries.

The rot6d layout was cross-checked against an independent reimplementation:
agreement to 2.4e-7 on random SE(3), while the row-major variant disagrees, so
the check actually discriminates.

## What is not established

Still open, and deliberately not attempted after the Step 7 gate failed:

- whether generated video survives the frozen depth → scale → converter path
- whether Cosmos beats the Phase 8 predictor on any decision group
- whether a lower-rotation option encoding (or a different embodiment) would
  restore action fidelity

No partial or indicative result is claimed for any of these.

## Files

| file | contents |
|---|---|
| `FROZEN.json` | per-gate status, findings, constraints |
| `PHASE9_CLOSURE.md` | the decisive diagnostic and the closure decision |
| `prereg_in_distribution_yaw.md` | criteria fixed before the final rollouts |
| `yaw_diagnostic_report.json` | the 11-rollout in-distribution result |
| `action_fidelity.json` | probe results and the 156x rotation comparison |
| `rollout_verification.json` | the 9 smoke-rollout checks with measured values |
| `sensitivity_report.json` | the four action probes |
| `figures/` | contact sheets: smoke rollout, turn_left, turn_right |
| `action_interface_report.md` | Class C determination with source citations |
| `resource_profile.json` | load timings, RAM, VRAM, device map, topology |
| `smoke_selection.json` | rule, 1497 candidates, selected branch, runners-up |
| `cosmos_env_lock.json` | torch/CUDA/driver/framework commit |
| `cosmos_env_packages.txt` | 413 pinned packages |

Working outputs live in `outputs/phase9d/smoke_conditioning` (manifest,
conditioning RGB and depth, nominal poses) and `outputs/phase9d/smoke_request`
(9-D action tensor, inference input spec, cache key).

## Reproducing

    # Habitat env
    python scripts/select_smoke_branch.py
    python scripts/build_cosmos_manifest.py

    # Cosmos env
    .venv/bin/python scripts/profile_cosmos.py --checkpoint $CK --device 0 --out ...
    .venv/bin/python scripts/cosmos_worker.py \
        --conditioning outputs/phase9d/smoke_conditioning \
        --out outputs/phase9d/smoke_request \
        --checkpoint $CK --checkpoint-hash 272cb18f94eaa850

Adding `--generate` runs the inference CLI with guardrails enabled. The action
probes are reproduced with `scripts/action_sensitivity.py` and the rollout checks
with `scripts/verify_rollout.py`.
