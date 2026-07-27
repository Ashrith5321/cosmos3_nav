# Preregistration — final in-distribution yaw diagnostic

Written and committed **before** the rollouts were generated. This is the last
Cosmos experiment in Phase 9; its outcome decides whether Cosmos is closed as a
negative result or a crossing-only reformulation is attempted.

## Question

The 30°/frame and 7.5°/frame probes showed action sensitivity without
directional adherence, but both are far outside the shipped reference
trajectory's rotational regime (0.264°/frame mean, 0.349° max). Does Cosmos
follow commanded yaw *direction* when the commanded yaw is inside that regime?

## Design

Everything is held fixed except the commanded per-frame yaw:

- conditioning image: `outputs/phase9d/smoke_conditioning/conditioning_rgb.png`
- prompt: the Phase 9D default (no ObjectNav goal)
- seed 0, num_steps 30, guidance 1.0, shift 5.0, fps 30, image_size 480
- 17 frames, `action_chunk_size` 16, `camera_pose`, `forward_dynamics`

Yaw conditions, straddling the reference mean:

    Δψ ∈ {−0.50, −0.25, 0.00, +0.25, +0.50} degrees per frame

**Two arms**, because "in-distribution" is a claim about the joint action, not
yaw alone. The reference trajectory is translation-dominated, so a pure-rotation
probe is itself atypical:

- **Arm A — zero translation.** Isolates yaw, so any horizontal flow is
  attributable to rotation alone. Cleanest attribution, but pure rotation is not
  what the reference contains.
- **Arm B — reference translation, 0.884 m/frame forward** (the reference
  trajectory's mean). Jointly in-distribution in both channels. Flow here mixes
  rotation and translation, so only the *difference across yaw conditions*
  is interpretable.

A conclusion is drawn only where the two arms agree; disagreement is reported as
inconclusive rather than resolved in favour of whichever arm is convenient.

## Success criteria (all four must hold, per arm)

1. **Opposite signs.** Mean horizontal flow for Δψ < 0 and Δψ > 0 have opposite
   signs. In Arm B, evaluated on flow *relative to the Δψ = 0 control*, since
   translation contributes a common offset.
2. **Monotonicity.** Mean horizontal flow is monotonic in Δψ across all five
   conditions (Spearman |ρ| = 1.0 over the five ordered levels).
3. **Correlation.** |Pearson correlation| between commanded Δψ and mean
   horizontal flow across the five conditions exceeds **0.30**, the same
   threshold already used and failed at 30°/frame.
4. **Determinism.** Re-running one condition with an identical action tensor and
   identical seed reproduces the video. Criterion: mean absolute pixel
   difference < 1.0/255 against the first run.

## Predeclared consequences

**If it FAILS** — Phase 9 closes as a negative result:

> Cosmos 3 Nano consumes camera-pose actions but fails directional action
> adherence even within its reference rotational regime.

Steps 8, 10 and 11 are not run. Phase 10 uses the Phase 8 structured-model
ensemble. Cosmos is retained only as a qualitative, non-action-faithful
baseline.

**If it PASSES** — a crossing-only reformulation may be attempted, but only
after verifying on development data, *before* implementing it, that:

- a previously observed frontier-facing keyframe exists (no privileged
  rendering from the future);
- most revelation happens after the approach pose is reached,
  `R_cross = Σ A_i^cross / Σ A_i^total ≥ 0.90`;
- target revelation during the approach segment is negligible;
- the crossing trajectory alone can be encoded with near-reference yaw
  increments.

If those conditions do not hold, Cosmos Phase 9 closes rather than the task
being redefined post hoc.

## What this test cannot settle

It measures whether commanded yaw direction is reflected in apparent image
motion. It does not measure revelation quality, geometric accuracy, or anything
about the frozen converter path. Passing would license further work, not any
claim about frontier prediction.
