# ASM — annotated semantic map side-channel

A top-down semantic map, built continuously from the observations the navigation
loop is already taking, annotated with object-name labels in the style of
[MapNav](https://arxiv.org/abs/2502.13451) (ACL 2025).

**No existing file in this repository is modified.** The map is a pure
by-product: nothing in the navigation, frontier, planning or detection path
reads it, and disabling the ASM restores byte-identical behaviour.

## Running it

```bash
OF_ASM=1 python -m asm.run_with_asm benchmark \
    --config config/navigation.yaml --output-path output \
    --eval_episodes 28 --max_steps 500
```

Everything after `benchmark` is forwarded verbatim, so this is a drop-in
replacement for `python benchmark.py ...`.

Without `OF_ASM=1` the patch does not install and the driver runs untouched.

Programmatic use:

```python
from asm import ASMConfig, attach

builder = attach(agent, ASMConfig(resolution_m=0.05))
image, objects, sentence = builder.latest()
```

## Outputs

Written to `<agent.save_dir>/asm/` per episode:

| file | contents |
| --- | --- |
| `latest_asm.png` | annotated map: free/occupied/unknown, trajectory, pose arrow, object labels |
| `latest_asm.json` | `{step, objects[], prompt_text, stats}` — objects carry world XY, area and vote count |
| `asm_stats.json` | frames submitted / processed / dropped, segmentation calls and time |

Set `keep_snapshots=True` for a per-step history instead of just the latest.

## Why it cannot slow navigation down

The agent thread does exactly one thing: a non-blocking `put_nowait` of copied
buffers. Measured at **0.17 ms per frame**. A worker thread owns everything
expensive.

* The queue is bounded. When the worker falls behind, frames are **dropped**,
  never queued — backpressure can never reach the navigation loop.
* The worker drains the whole queue each cycle, integrating geometry for every
  frame, and considers only the freshest frame for segmentation. Slow
  segmentation costs semantics, not geometry.
* Segmentation is gated twice: at most every `segment_every_n_frames`, and only
  once the agent has moved `min_translation_m` or turned `min_rotation_deg`.

At a 50 ms step pace the drop rate is 0. Real navigation steps take ~1 s.

## SAM3 contention

Semantics come from the SAM3 server, one request per category per keyframe — so
cost is linear in `categories` (8 by default). That server is **shared with the
main detection path**, and requests serialise.

If you see the navigation loop slow down, run a second SAM3 instance and point
the ASM at it:

```bash
python sam3_server.py 12185 &
ASM_SAM3_PORT=12185 OF_ASM=1 python -m asm.run_with_asm benchmark ...
```

## Differences from MapNav's implementation

MapNav renders the map to a PNG and then recovers object identity by matching
RGB values back to a hardcoded palette with a ±5 tolerance
(`MapNav/huatu3.py:50-55`). That round-trip is lossy and caps the map at the 11
colours in the palette.

Here the label grid stays integer end to end. Connected components run on the
label grid, and the rendered image is an output rather than an intermediate.
Consequences: no colour collisions, unlimited categories, and per-object world
coordinates (MapNav only recovers pixel centroids).

Also kept: SAM3's per-instance `scores`, thresholded at
`sam3_score_threshold`. The main pipeline discards them at
`vlm/utils.py:176-179`.

## Knobs

`asm/config.py` holds all defaults. Three are also settable from the
environment: `ASM_CATEGORIES` (comma-separated), `ASM_RESOLUTION_M`,
`ASM_SEGMENT_EVERY`.

The ones that matter most:

| field | default | effect |
| --- | --- | --- |
| `categories` | 8 entries | **linear cost driver** — each costs one SAM3 request per keyframe |
| `segment_every_n_frames` | 8 | semantic cadence |
| `resolution_m` | 0.05 | grid cell size |
| `extent_m` | 40.0 | map side; anchored at the first observed pose, points outside are clipped |
| `max_depth_m` | 3.5 | matches the sensor clip set at `utils/config_utils.py:94` |

## Frames and conventions

World is **Z-up** — `utils/transform.py:33` maps habitat `(x, y, z)` to
`(x, -z, y)`. Height bands are taken relative to the floor under the camera,
derived as `W_T_C2[2,3] - C_T_R[2,3]`, the same expression `nav/agent.py:314`
uses for `nav_level`, so the map behaves on multi-floor scenes.

Back-projection follows the pinhole convention at
`frontier/detector.py:344-349`.

## Tests

```bash
python -m asm.selftest          # synthetic room: geometry, semantics, annotation
```

Checks that four viewpoints of a synthetic room produce free and occupied cells,
that injected masks land as semantic cells on the correct sides of the origin,
that annotation returns the right labels, and that the prompt sentence names
them.

## What this is for

The frontier utility at `frontier/manager.py:905-944` is
`p^prob_sharpness · u_gain^utility_gain_factor / d`. With the shipped config that
is `p⁴ · u_gain² / d` — but `p` is assigned `1.0` for every mask at
`frontier/manager.py:365-368`, so the semantic term is currently a constant.

`prompt_text()` and the `objects[]` list in the JSON are the cheapest available
route to a real semantic signal for a frontier scorer: an inventory of what has
been seen and where, in both text and world coordinates, with no training and no
generative model. Nothing here wires that in — that stays a separate decision.
