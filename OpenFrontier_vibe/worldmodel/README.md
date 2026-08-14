# worldmodel — Frontier-Conditioned Counterfactual World Modeling

Implementation of `frontier_conditioned_world_model_design.md` on top of
OpenFrontier. Each persistent frontier is treated as a counterfactual
action: a goal-agnostic world model predicts a distribution over what
exploration through that frontier would reveal, a goal evaluator matches
those predicted futures against the language goal, and an
uncertainty/cost-aware ranker replaces the myopic OpenFrontier utility

```
u_i = (p_i^sharp * gain_i)^f / ||p_r - p_i||          (baseline)
```

with

```
Q_i = l_obs*P_obs + l_wm*mu_i + l_ig*IG_i + l_nov*N_i
      - l_cost*C_i - l_unc*sigma_i - l_visit*V_i - l_risk*R_i
```

## Module map (design-doc section in parentheses)

| file | role |
|---|---|
| `records.py` | `FrontierWMRecord` / `WMPrediction` / `FutureHypothesis`, lifecycle statuses (§10) |
| `memory.py` | persistent frontier memory, prediction cache + staleness, residual store (§10–13) |
| `encoder.py` | frozen CLIP (or dummy) vision/text encoder — the shared semantic space |
| `vocab.py` | rooms / objects / affordances, room→object priors, language events (§19) |
| `predictor.py` | goal-agnostic predictors: `ZeroShotClipPredictor` (no training), `LearnedPredictor` (§6, §7.2, §22) |
| `net.py` | `FrontierWorldModelNet`: K hypothesis queries, multi-head outputs (§6.4) |
| `evaluator.py` | goal evaluation: contrast-normalized cosine + object channel + optional VLM refinement (§8) |
| `ranker.py` | Q_i combination, normalization, argmax / UCB / risk-averse policies (§14–16, §49) |
| `cost.py` | geodesic path cost with caching, Euclidean fallback (§15) |
| `calibration.py` | prediction residuals + episodic reliability calibration (§17–18, §30) |
| `pipeline.py` | orchestration: context attachment → coarse-to-fine cascade → closed-loop verification (§20–21, §28–29) |
| `dataset.py` | self-supervised beyond-frontier dataset recorder + torch dataset (§23–25) |
| `train.py` | multi-task winner-takes-all training of the learned predictor (§26) |
| `oracle.py` | oracle frontier values from navmesh geodesics for ranking eval (§38) |

Integration points in the host repo:

- `frontier/frontier.py` — frontiers now carry a stable `uid` that survives
  manager merges (integer ids do not).
- `frontier/manager.py` — `merge_frontiers` propagates the dominant `uid`
  and fires `on_frontiers_merged`; `update_utility` routes through
  `external_utility_fn` when a pipeline is attached.
- `nav/agent.py` — hooks: `observe()` after each frame (residuals),
  `attach_context()` after frontier anchoring, `step()` before
  `update_utility`, `notify_goal_selected()` after planning,
  `log_step()` per step.
- `benchmark.py` — `--wm-oracle` attaches the oracle evaluator per episode;
  the pipeline is closed (memory snapshot + dataset flush) in `finally`.

## Running

Zero-shot world model (no training needed; frozen CLIP + room/object priors):

```bash
python benchmark.py --benchmark hm3d --nickname wm \
    --config config/navigation_worldmodel.yaml \
    --output-path output/wm_zero_shot
```

With oracle frontier-ranking logging (design §38):

```bash
python benchmark.py ... --wm-oracle
python scripts/analyze_wm_ranking.py 'output/wm_zero_shot/**/wm_state.jsonl'
```

Per-episode artifacts (next to the usual OpenFrontier outputs):

- `wm_state.jsonl` — one line per step: every frontier's Q-term breakdown
  (`p_obs`, `wm_mu`, `wm_sigma`, cost, novelty, revisit), predicted top
  room, language events, cache/call statistics, calibrator state, and
  (with `--wm-oracle`) oracle values.
- `wm_memory.json` — final persistent-memory snapshot: per-uid status,
  prediction summary, residual history.

## Training the learned predictor

1. Record beyond-frontier data during episodes (any config, exploration or
   navigation) by setting `world_model.record_dataset: true`. Each episode
   writes `<episode_dir>/wm_dataset/beyond_frontier.npz` with
   (context embeddings, geometry) → (future embedding, room pseudo-label,
   object pseudo-labels, realized gain) pairs. Labels are self-supervised:
   the "beyond region" is the set of later observations captured within /
   behind the frontier (§25).

2. Train:

```bash
python -m worldmodel.train \
    --data 'output/<run>/**/wm_dataset/beyond_frontier.npz' \
    --out model_weights/frontier_wm.pth --epochs 50
```

3. Switch the config: `backend: learned`, `checkpoint: model_weights/frontier_wm.pth`.

## Ablation ladder (design §36)

| rung | what | how |
|---|---|---|
| A0 | OpenFrontier baseline | `config/navigation.yaml` (world model absent) |
| A1 | + path cost | `config/navigation_worldmodel_a1_pathcost.yaml` (`lambdas.wm: 0`, geodesic cost on) |
| A2 | + single future latent | `config/navigation_worldmodel_a2_single_hypothesis.yaml` (`num_hypotheses: 1`) |
| A3 | + multi-hypothesis mu/sigma | full config (`num_hypotheses: 4`, `policy: risk_averse`) |
| A4 | prediction cache | on by default; disable by setting `cache.max_age_steps: 0`; compare `wm_calls` vs `cache_hits` in `wm_state.jsonl` |
| A5 | multi-head prediction | rooms/objects/affordances are always predicted; set `evaluator.object_weight: 0` to score with the embedding channel only |
| A6/A7 | residuals + calibration | `config/navigation_worldmodel_no_calibration.yaml` vs full |
| A8 | rich VLM scoring | `evaluator.use_vlm: true` |

## Tests

Dependency-light (numpy only; uses the dummy encoder and stub planner):

```bash
python tests/test_worldmodel.py
```

## Generative (pixel-space) backend — Phase 3 (§42)

`backend: cosmos_gen` routes prediction through Cosmos3 image2image: K
seeded "move the camera forward through the opening" edits of the frontier
crop become the K hypotheses, CLIP-encoded into the same shared space.
It needs the rollout server running under the cosmos venv (the benchmark
env cannot import cosmos_framework):

```bash
CUDA_VISIBLE_DEVICES=1 \
  /home/ashed/Documents/cosmos3_nav/cosmos/packages/cosmos3/.venv/bin/python \
  worldmodel/cosmos_gen_server.py --port 12186
```

then in the config:

```yaml
world_model:
  backend: cosmos_gen
  cosmos: {port: 12186, num_steps: 12, resolution: "480", advance_m: 2.5}
  top_m: 2          # generation is ~30-60s per frontier; keep M small
```

Known caveat, measured twice in this workspace (frontierworld Phase 9D and
eval/cosmos3_nav_test 20260811): Cosmos3 image2image re-renders its
conditioning image rather than genuinely traveling. This backend is here
for the design's Phase-3 comparison and qualitative figures; zero-shot /
learned latent backends are the defaults.

## Design notes

- **Goal-agnostic prediction, goal-conditioned evaluation** (§7.2): the
  predictor never sees the goal, so it cannot hallucinate goal-pleasing
  futures; language enters only in the evaluator.
- **Pixel/video backends were deliberately not used**: both the
  frontierworld Phase 9D study and the cosmos3_nav_test probe found
  image/video generation not action-faithful for this task; the shared
  latent + structured heads (§6.4) is the surviving path. The
  `LearnedPredictor` interface is backend-agnostic if that changes.
- **Zero-shot backend as the floor**: `ZeroShotClipPredictor` gives the
  whole system (multi-hypothesis, uncertainty, caching, calibration,
  ranking) without any training, so navigation-level effects can be
  measured before investing in the learned model.
