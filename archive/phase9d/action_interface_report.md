# Phase 9D Step 2 — Cosmos 3 Nano action interface: definitive determination

**Question.** Does Cosmos 3 Nano accept an *action* as conditioning input, emit
actions as output, both, or neither? The answer decides whether a counterfactual
"what would I see if I took option ω_i" is expressible at all, or whether Phase
9D reduces to unconditioned video continuation.

**Answer: Class C — the model both consumes action conditioning and emits
actions.** Which of the two is active is selected by an explicit `model_mode`
field, not inferred. Counterfactual option conditioning is therefore
expressible, and no textual substitute is needed.

This was determined by reading the shipped code and assets, not by inferring
from documentation prose or from generation behaviour.

## Evidence

### 1. The checkpoint config declares an action modality

`nvidia/Cosmos3-Nano/transformer/config.json`:

    "action_gen": true,
    "max_action_dim": 64,
    "num_embodiment_domains": 32

alongside `vision_gen: true` and `sound_gen: true`. `action_gen` is a peer of
the two generative modalities that are unambiguously outputs, so on its own
this line establishes only that actions are *a* modality — not the direction.
The direction comes from the inference code.

### 2. Three explicit modes, one of which is action-conditioned

`cosmos_framework/inference/dataset.py:36`

    ModelMode = Literal["forward_dynamics", "inverse_dynamics", "policy"]

`cosmos_framework/inference/action.py:37-51` — `_load_actions` branches on the
mode and shows the direction of each:

| mode | action role | evidence |
|---|---|---|
| `forward_dynamics` | **input** | `assert action_path is not None, "action_path is required for forward_dynamics mode"` — actions are read from file and fed in |
| `inverse_dynamics` | output | actions passed as **zeros**; the model produces them |
| `policy` | output | actions passed as **zeros** |

`forward_dynamics` is exactly the counterfactual operator this project needs:
given an observation and a proposed action sequence, predict the resulting
observations. That the other two modes pass zeros is the decisive detail — a
mode that *reads* an action file and a mode that *fabricates a zero placeholder*
cannot both be output modes.

### 3. Action conditioning demonstrably changes the output

The shipped AV examples (`inputs/omni/action_forward_dynamics_av.json` and the
cookbook forward-dynamics notebooks) drive the **same start image** with
`av_traj_forward.json` / `av_traj_left.json` / `av_traj_right.json` and produce
visibly different videos. This is behavioural confirmation that the action
tensor is not decorative. It is cited as corroboration only; the code above is
the primary evidence.

### 4. A `camera_pose` embodiment exists and matches our action space

`cosmos_framework/data/generator/action/domain_utils.py`

    "camera_pose": 2      # domain id
    "camera_pose": 9      # raw action dim

This is materially better than the AV ego-pose embodiment I had assumed would
be the closest fit. `camera_pose` is a free-moving camera in a static scene,
which is precisely a robot ego-camera executing a frontier-crossing option.
Its raw action dim of **9** coincides with `POSE_DELTA_DIM = 9` already fixed in
`frontierworld/models/cosmos_contract.py`.

The shipped reference input `inputs/omni/action_forward_dynamics_camera.json`
confirms the field set:

    "domain_name": "camera_pose", "model_mode": "forward_dynamics",
    "action_chunk_size": 60, "action_path": <9-D JSON>, "vision_path": <image>,
    "fps": 30, "view_point": "ego_view", "num_steps": 30, "guidance": 1.0,
    "shift": 5.0, "image_size": 480, "seed": 0

## The action encoding, verified by execution

Actions are **relative camera poses**, layout `[translation(3), rotation(6)]`.

`cosmos_framework/data/generator/action/pose_utils.py:429` `pose_abs_to_rel`:

- input `poses_abs`: `(T, 4, 4)` **camera-to-world** homogeneous transforms, metres
- `pose_convention="backward_framewise"`: `delta_T = T_i^{-1} @ T_{i+1}`
  (each row is the motion *from* frame i *to* frame i+1, expressed in frame i)
- `rotation_format="rot6d"` ⇒ `D = 3 + 6 = 9`
- output: `(T-1, 9)`

Executed check — four absolute poses translating +0.25 m along camera *z* with
identity rotation produce `(3, 9)` with every row

    [0, 0, 0.25,  1, 0, 0,  0, 1, 0]

i.e. translation-first, then the first two columns of the identity rotation
matrix. This matches the shipped asset format byte-for-byte.

**Decision: use the framework's own `pose_abs_to_rel(..., rotation_format="rot6d")`
rather than the hand-rolled `trajectory_to_pose_deltas` in `cosmos_contract.py`.**
Both produce 9-D deltas and agree on the identity case, but the sign and
handedness conventions at an interface boundary are exactly the kind of thing
that silently produces plausible-but-wrong video. `trajectory_to_pose_deltas`
is retained for the cache key and for tests; the tensor actually fed to the
model comes from the official utility.

## Frame-count relation

`cosmos_framework/inference/action.py:95`

    target_frames = action_chunk_size + 1

So a 17-frame rollout requires `action_chunk_size = 16`. The conditioning frame
is frame 0 and the 16 actions carry frames 1..16. `build_action_batch` will
silently pad (repeating the last frame) or truncate the video to
`action_chunk_size + 1`, so this must be set exactly rather than relied upon.

## Consequence for Phase 9D

The strict rule stated for this phase — *if the model does not consume action
conditioning, do not fake it with text such as "turn left"* — is **not
triggered**. The model consumes a genuine 9-D relative-pose action sequence
through a documented, typed interface. Option conditioning for
ω_i = (τ_approach, τ_cross, H) is encoded as the camera-pose trajectory of that
option, converted by the framework's own utility. No textual action
substitution appears anywhere in the pipeline.

## Source index

| Fact | Location |
|---|---|
| `action_gen: true`, `max_action_dim: 64` | `nvidia/Cosmos3-Nano/transformer/config.json` |
| `ModelMode` literal | `cosmos_framework/inference/dataset.py:36` |
| `FORWARD_DYNAMICS` enum | `cosmos_framework/inference/args.py:166` |
| `action_path` required for forward dynamics | `cosmos_framework/inference/action.py:39` |
| zeros for policy / inverse dynamics | `cosmos_framework/inference/action.py:47-49` |
| `target_frames = action_chunk_size + 1` | `cosmos_framework/inference/action.py:95` |
| `camera_pose` domain id 2, raw dim 9 | `cosmos_framework/data/generator/action/domain_utils.py:9,32` |
| `pose_abs_to_rel` semantics | `cosmos_framework/data/generator/action/pose_utils.py:429` |
| reference camera input spec | `inputs/omni/action_forward_dynamics_camera.json` |
| framework commit | `09f23119ea92c707207bba55565e7a09d16896a2` |
