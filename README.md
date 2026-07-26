# FrontierWorld

Counterfactual revelation models and lineage-aware predictive memory for object
navigation. Paper draft in [`paper/main.tex`](paper/main.tex); the full plan is
in [`checklist.md`](checklist.md).

The research question: *can frontier-specific memory predict what each
exploration action will reveal, and does that prediction improve frontier
selection?*

## Status

Phase 1 (repository, environment, data, one-episode gate) is complete.
Phase 2 (occupancy mapping, frontier extraction, baseline policies) is next.

## Setup

`habitat-sim` is a conda package and cannot be pip-installed, so the
environment is conda-based.

```bash
conda env create -f environment.yml
conda activate frontierworld
```

On the lab machine the existing `habitat033` environment already satisfies
this spec:

```bash
conda activate habitat033        # habitat-sim 0.3.3, habitat-lab 0.3.3
```

### Data

`data/` holds symlinks to the shared HM3D installation:

```
data/scene_datasets -> .../spatial_training/data/scene_datasets
data/datasets       -> .../spatial_training/data/datasets
```

Currently present:

| asset | status |
| --- | --- |
| HM3D v0.2 **val** scenes (100) | present |
| HM3D v0.2 **minival** scenes (10) | present |
| HM3D **train** scenes | **not downloaded** |
| ObjectNav HM3D v2 episodes (train / val / val_mini) | present |

The train *episodes* are present but the train *scene meshes* are not, so
dataset generation (Phase 5) is limited to val scenes until they are
downloaded. See [Downloading HM3D train scenes](#downloading-hm3d-train-scenes).

Semantic annotations only load when the simulator is pointed at
`hm3d_annotated_basis.scene_dataset_config.json`; `configs/base.yaml` does
this via `data.scene_dataset_config`. Without it habitat loads the bare `.glb`
stage and every semantic observation is zeros.

## Running

The Phase 1 gate -- one command, one episode, one saved episode log:

```bash
python scripts/run_episode.py
```

Options:

```bash
python scripts/run_episode.py --episodes 3
python scripts/run_episode.py --override episode.max_steps=50 --override seed.value=7
python scripts/run_episode.py --config configs/base.yaml
```

Every setting in `configs/base.yaml` can be overridden with `--override
key.subkey=value`, and the fully resolved config is copied into the run
directory, so a run is reproducible from its own output.

### Output layout

```
outputs/<experiment>/<timestamp>_<name>_<confighash>/
  config.yaml          resolved config for this run
  provenance.json      git commit, package versions, GPU
  run.jsonl            one record per logged step
  run.csv              same, flat columns
  summary.json         run-level result and RNG fingerprint
  episodes/<scene>_<episode_id>/
    episode.json             metadata, metrics, per-step index
    pose.jsonl               agent and sensor pose per step
    semantic_categories.json instance id -> category name
    rgb/000000.png
    depth/000000.npz         float32 metres
    semantic/000000.npz      int32 instance ids
    map/occupancy.npz        ternary grid, counts, geometry
    map/occupancy.png        preview
```

## Tests

```bash
pytest tests/ -v
```

`test_config.py::test_configured_data_is_present` fails if the HM3D paths in
`configs/base.yaml` are wrong, which is the fastest way to check a new machine.

## Determinism

Runs are reproducible from `seed.value`:

- `seed_everything` seeds Python, NumPy, torch and habitat.
- Per-episode randomness comes from `episode_rng(seed, scene_id, episode_id)`,
  derived rather than drawn from a running stream. An episode therefore
  replays identically regardless of how many episodes preceded it -- required
  for the Phase 5 branch protocol, which re-enters the same decision state
  many times.
- `summary.json` records an RNG fingerprint; two runs with the same seed and
  different fingerprints mean something consumed randomness outside the
  seeding path.

## Layout

```
configs/          experiment configuration
frontierworld/
  config.py       config loading and path resolution
  seeding.py      deterministic seeding
  habitat_env.py  habitat config composition, poses, semantics
  mapping/        occupancy mapping from depth and pose
  frontiers/      frontier extraction                     (Phase 2)
  lineage/        frontier lineage graph                  (Phase 7)
  memory/         per-frontier predictive memory          (Phase 8)
  models/         revelation prediction models            (Phase 9)
  planning/       options and frontier scoring            (Phases 3, 11)
  data/           observation recording, FrontierReveal   (Phase 5)
  evaluation/     metrics and experiment tracking
scripts/          entry points
tests/
paper/
```

## Downloading HM3D train scenes

Needs a Matterport access token (the HM3D EULA form) and the habitat
downloader, run from the environment that has habitat-sim:

```bash
python -m habitat_sim.utils.datasets_download \
  --username <matterport-username> --password <matterport-password> \
  --uids hm3d_train_v0.2 \
  --data-path /home/ashed/Documents/spatial_training/data
```

The train split is roughly 800 scenes; budget disk space before starting.
