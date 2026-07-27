# Phase 9D — Cosmos 3 Nano counterfactual rollouts

**Status: PARTIAL.** Everything up to the point of pressing "generate" is done,
verified and frozen. Generation itself is blocked on a gated-model credential
and was not attempted, because the only way past it was to disable a required
safety component.

## The blocker, precisely

`cosmos_framework.auxiliary.guardrail.common.presets` constructs `Blocklist`,
`Qwen3Guard` and `RetinaFaceFilter`, all of which download from

    nvidia/Cosmos-Guardrail1 @ d6d4bfa899a71454a700907664f3e88f503950cf

That repo is `gated=auto` and this machine has no HuggingFace token, so the
download fails with *"Access to model ... is restricted. Please log in."*
Cosmos3-Nano itself is public and is already downloaded (33 GB, hash
`272cb18f94eaa850`).

The framework offers `--no-guardrails`, and that was not used. Guardrails were
listed as a required safety component, so `scripts/cosmos_worker.py` raises if
that flag is ever passed through.

**Correction to an earlier note.** During Step 3 I recorded both gated repos as
"ACCESSIBLE". That check called `HfApi().model_info(...)`, which returns public
*metadata* for a gated repo and therefore never established file access. The
accurate status is blocked pending authentication.

**To unblock:** a HuggingFace token whose account has accepted the
Cosmos-Guardrail1 licence. Nothing else is missing — the GPUs, the checkpoint,
the environment and all conditioning inputs are in place.

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

No video has been generated, so **nothing is known about whether Cosmos
produces useful frontier revelations.** In particular these remain open:

- whether generated frames follow the commanded 9-D pose sequence at all
- whether the turn direction and forward motion are respected
- whether generated video survives the frozen depth → scale → converter path
- whether Cosmos beats the Phase 8 predictor on any decision group

No partial or indicative result is claimed for any of these.

## Files

| file | contents |
|---|---|
| `FROZEN.json` | per-gate status, blocker, constraints |
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

Adding `--generate` runs the inference CLI with guardrails enabled; that step is
what the credential unblocks.
