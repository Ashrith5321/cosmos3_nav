# Phase 9 closure — Cosmos 3 Nano as a negative result

The final in-distribution yaw diagnostic, preregistered in
[`prereg_in_distribution_yaw.md`](prereg_in_distribution_yaw.md) and committed
as `c61c3da` before any of its rollouts existed, **FAILED**. Both arms agree.
Per the predeclared consequence, Phase 9 closes here.

## The conclusion

> **Cosmos 3 Nano consumes camera-pose actions, but fails directional action
> adherence even within its reference rotational regime.**

## The measurements

Eleven rollouts. Conditioning image, prompt, translation, seed and all
generation settings held fixed; only the commanded per-frame yaw varied, across
levels straddling the reference trajectory's 0.264°/frame.

### Arm A — zero translation, isolating yaw

| Δψ (°/frame) | −0.50 | −0.25 | 0.00 | +0.25 | +0.50 |
|---|---|---|---|---|---|
| mean horizontal flow | −0.0564 | −0.0539 | −0.0554 | −0.0548 | −0.0557 |
| relative to zero | −0.0011 | +0.0014 | 0 | +0.0006 | −0.0004 |

No response whatsoever. The spread across the entire ±0.5° range is 0.0025 px —
indistinguishable from noise. Spearman 0.10, correlation 0.09.

At this rotation rate the commanded yaw is **below the model's effective
response threshold**: it is not merely followed inaccurately, it is not followed
at all.

### Arm B — reference translation, 0.884 m/frame

| Δψ (°/frame) | −0.50 | −0.25 | 0.00 | +0.25 | +0.50 |
|---|---|---|---|---|---|
| mean horizontal flow | −0.9506 | −1.0312 | −0.9525 | −1.0816 | −1.2817 |
| relative to zero | +0.0019 | −0.0787 | 0 | −0.1291 | −0.3292 |

Here there *is* a trend — correlation −0.83, which passes criterion 3 on its
own. But it is not a directional response:

- **Negative and positive yaw do not produce opposite flow.** Mean relative flow
  is −0.038 for Δψ < 0 and −0.229 for Δψ > 0 — both negative. Criterion 1 fails.
- **It is not monotonic**: Δψ = −0.25 gives −0.079 while Δψ = 0 gives 0.
  Spearman −0.90, not ±1. Criterion 2 fails.
- The effect is **0.33 px at most**, against the 17.5 px of flow the 30°/frame
  probes produced. Turning the commanded yaw from full left to full right moves
  the image by a third of a pixel.

What Arm B shows is a weak *magnitude* sensitivity — larger |Δψ| perturbs the
rollout slightly more — not a sign-respecting rotation.

### Criterion 4 — determinism: PASS

Re-running Δψ = +0.50°/frame with an identical action tensor and identical seed
reproduced the video **exactly**: mean absolute pixel difference 0.00000. The
pipeline is bit-reproducible, so none of the above is sampling noise.

### Scorecard

| criterion | Arm A | Arm B |
|---|---|---|
| 1. opposite signs | FAIL | FAIL |
| 2. monotonic | FAIL | FAIL |
| 3. \|correlation\| > 0.30 | FAIL (0.09) | pass (0.83) |
| 4. determinism | PASS (0.00000) | PASS |

**Verdict: FAIL.** The arms agree, so the result is not an artefact of the
translation choice.

## Why this settles it

The earlier 30°/frame failure could be dismissed as operating 156× outside the
model's demonstrated rotational distribution. This test removes that defence: at
0.25–0.50°/frame — bracketing the reference's own 0.264°/frame — the model still
does not turn the camera in the commanded direction. In Arm A it does not turn
the camera at all.

So the failure is not a distribution-shift artefact of ObjectNav's 30° discrete
turns. It is a property of the released `camera_pose` forward-dynamics
behaviour: the action channel carries translation faithfully and rotation
essentially not at all.

## Consequences, as predeclared

- **Steps 8, 10 and 11 are not run.** No structured conversion, no decision
  group, no validation pilot.
- **Phase 9 is archived as a negative result.**
- **Phase 10 will use the Phase 8 structured-model ensemble.** (Recorded as a
  decision; Phase 10 has not been started.)
- **Cosmos is retained only as a qualitative, non-action-faithful baseline.** It
  may illustrate plausible indoor continuation; it may not be used for
  counterfactual option comparison, and no Q-value or utility may be derived
  from it.

The crossing-only reformulation `Ŷ_i = W(m_i, τ_i^cross)` with analytic approach
cost is **not** attempted. Its precondition was that this test pass. Pursuing it
now would be redefining the task after seeing the result.

## What remains true and useful

The Phase 9 infrastructure is sound and is not discarded:

- the action interface determination (Class C, `camera_pose`, 9-D rot6d) is
  correct and independently cross-checked;
- the nominal-pose derivation is exact, verified by re-rendering the
  decision-point view to 0.001/255;
- the conditioning manifest, cache keys and two-environment isolation all work;
- the resource profile stands: 15.88 B parameters, 30.3 GB BF16, one A6000.

If a future action-faithful video model appears, everything except the generator
is reusable.

## What is still not claimed

- Not that Cosmos 3 Nano is a poor video model — its visual continuation is
  plausible for roughly 12 frames.
- Not that action conditioning fails for other embodiments; only `camera_pose`
  was tested.
- Not any statement about revelation quality, the frozen converter path, or a
  comparison against the Phase 8 predictor. None of those were measured.
- Optical flow is a proxy for camera rotation. It is a good one for a
  yaw-dominated pan, but the null result in Arm A rests on it, and a rotation
  that produced no image motion at all would be indistinguishable from one that
  was ignored.
