# FrontierWorld — start-to-finish checklist

**Working title:** FrontierWorld: Counterfactual Revelation Models and
Lineage-Aware Predictive Memory for Object Navigation

**Research question:** Can frontier-specific memory predict what each
exploration action will reveal, and does that prediction improve frontier
selection?

**Status:** Phases 0–5 and 7 complete. Phase 6 **frozen** (association
accuracy unmeasured pending Phase 6.5 annotation). Phase 8 **v0 gate passed
narrowly; two primary heads unresolved** — see below. Phase 9 **closed as a
negative result**: the infrastructure passed, but Cosmos 3 Nano is
action-sensitive without being directionally action-faithful, so it cannot serve
as a counterfactual frontier predictor. Phase 10 **complete, gate 3 of 6**:
ensemble uncertainty tracks error and area intervals are calibrated, but target
presence remains uncalibratable at chance-level AUROC.

> **The v0 test split is CONSUMED.** Its numbers have been read, so any v1
> design informed by them makes it biased for v1. `manifests/sealed_final_v1.json`
> reserves 20 HM3D val-split scenes, untouched by any dataset, model or
> diagnostic, for the single final v1 evaluation. See
> `manifests/SEALED_DO_NOT_TOUCH.md`.

---

## Dependency structure

```
Phase 5: FrontierReveal dataset
   ├── Phase 6: Lineage graph ──→ Phase 11: Frontier memory ─┐
   └── Phase 7: Learning tensors                             │
          └── Phase 8: Structured predictor                  │
                 ├── Phase 9: Cosmos 3  ✗ CLOSED (negative)  │
                 └── Phase 10: Uncertainty (Phase 8 ensemble)│
                                                             ↓
Phase 12: Offline ranking → Phase 13: Closed-loop navigation
       → Phase 14: Ablations → Phase 15: Final experiments
       → Phase 16: Paper and release
```

> **Numbering changed.** Phases 5–16 were renumbered when this plan was
> revised. Lineage moved 7 → 6, memory 8 → 11, prediction 9 → 8, ranking
> 11 → 12, closed-loop 12 → 13. Source comments were updated to match; if you
> find a stale reference, it predates the revision.

---

## Progress summary

| Phase | Status | Gate | Evidence |
| --- | --- | --- | --- |
| 0 Freeze claim | done | claims fit one paragraph | `paper/main.tex` abstract |
| 1 Repository | done | one command runs one episode and saves a log | `8ef965f` |
| 2 Navigation | done | nearest + info-gain complete without planner failures | `d947781` |
| 3 Options | done | 3 branches from one identical state | `315aa28` |
| 4 Revelation | done | ≥20 examples visualised and verified | `818c7a2` |
| 5 Dataset | done | dataloader returns all branches of one state | 504 groups, 2421 branches |
| 6 Lineage | **frozen** | persist/split/disappear identities, no branch leakage | synthetic gate + 0 isolation violations |
| 6.5 Annotation | exported | human transition labels | 102 transitions awaiting labels |
| 7 Tensors | done | batch round-trips to the global map | round-trip 2.7e-15 m, recall 1.000 |
| 8 Predictor | **v0 frozen** | beats strongest baseline on occupancy + a scalar | narrow pass, 2 heads unresolved |
| 8.5 v1 design | pending | factorised objective; calibrated target head | diagnosis complete |
| 9A/9B Depth+scale | done | monocular depth characterised | `archive/phase9_depth` |
| 9C Robust converter | **frozen, gate FAILED** | graceful degradation under predicted depth | 4 of 5 criteria failed; archived negative result |
| 9D Cosmos rollouts | **CLOSED — NEGATIVE** | action-faithful counterfactual rollouts | infrastructure PASS, generator FAIL (`617b51e`) |
| 10 Ensemble + calibration | **gate 3/6** | uncertainty tracks error; calibration beats prior | uncertainty + area PASS; target uncalibratable |
| 11–16 | not started | | |

**Test suite:** 252 passing.
**Known blockers:** none.

**Phase 9 outcome, in one line.** Under the released Cosmos3-Nano camera-pose
interface, *action sensitivity does not imply action adherence*: translation
affects the generated future, but signed yaw is not represented faithfully,
including at the sub-degree rotational scale found in NVIDIA's own reference
trajectory. Phase 10 therefore proceeds with the Phase 8 structured-model
ensemble, and Cosmos is retained only as a qualitative, non-action-faithful
baseline.

**Scene accounting.** 145 HM3D train scenes carry semantics. 58 are consumed
(Phase 8 v0 + pipeline development). `dev_pool_v1.json` holds the 103 untouched
scenes, now split into `phase10_calibration.json` (45) and
`phase10_validation.json` (45) with 13 reserved for Phase 11. Generation hit its
group target before exhausting them, so **22 scenes per split** are realised.
`sealed_final_v1.json` (20 HM3D **val**-split scenes) remains unopened.

---

## Phase 0 — Freeze the research claim ✅

- [x] Working title fixed.
- [x] Primary research question written.
- [x] Minimum prediction outputs fixed: newly revealed occupancy,
      target-presence probability, room/semantic category, traversability,
      expected newly revealed area.
- [x] Explicitly excluded from v1: RGB/video generation as the primary output,
      raw VLM KV-cache manipulation, cross-episode memory, real-robot
      experiments, large cross-frontier GNN, language-event generation.
- [x] Three central contributions defined: branch-complete counterfactual
      frontier dataset; frontier-specific predictive memory under a fixed
      budget; prediction-based frontier planning.

---

## Phase 1 — Create the repository ✅

- [x] Repository structure created.
- [x] Git initialised; pushed to `origin/frontierworld`.
- [x] Reproducible environment (`environment.yml`, `requirements.txt`, pinned
      to the working `habitat033` conda env).
- [x] Experiment configuration files (`configs/base.yaml`; every knob
      overridable with `--override key=value`).
- [x] Deterministic seeds. Per-episode RNG derives from
      `(seed, scene_id, episode_id)` rather than a running stream, so an
      episode replays identically regardless of what preceded it.
- [x] Habitat-Sim 0.3.3 and Habitat-Lab 0.3.3 confirmed in `habitat033`.
- [x] HM3D **validation** and **minival** scenes present.
- [x] HM3D **train** scenes downloaded in Phase 5.1 (800 scenes, 145 annotated).
- [x] One HM3D ObjectNav episode runs.
- [x] RGB, depth, pose, semantic annotation and map output saved.
- [x] Experiment tracking (CSV + JSONL always on; TensorBoard/W&B optional and
      degrade with a warning rather than failing the run).

**Gate met.** `python scripts/run_episode.py` → 60-step episode, 61 frames per
modality, 850 semantic instances, 31.6 m² mapped. Two same-seed runs produced
identical actions, poses and map statistics.

**Findings worth keeping:**
- Semantics load as all-zeros unless the simulator is pointed at
  `hm3d_annotated_basis.scene_dataset_config.json`. It fails silently.
- Depth at max sensor range must be treated as a *miss*, not a surface.
  Marking it occupied walls off exactly the unexplored region frontiers are
  defined by.

---

## Phase 2 — Build the basic navigation system ✅

- [x] 2D occupancy map from depth and ground-truth pose.
- [x] Cells as free / occupied / unknown.
- [x] Free–unknown boundary cells extracted.
- [x] Boundary cells clustered into frontiers (connected components).
- [x] Centre and orientation computed per frontier.
- [x] Frontiers too small or unreachable removed (min size, agent-radius
      clearance, geodesic reachability).
- [x] Navigation to a selected frontier — OpenFrontier's navmesh planner
      (`snap_point` → `ShortestPath` → `GreedyGeodesicFollower`), ported.
- [x] Baseline policies: random, nearest, max geometric information gain,
      information gain − travel cost.
- [x] 50 episodes per policy (200 runs, 5.5 min on 24 workers).
- [x] SR, SPL, collisions and explored area recorded.

**Gate met.** All 200 runs completed; planner failures (5–7 per policy) were
always recovered by reselecting another frontier and none terminated an
episode.

| policy | SR | SPL | DetRate | Area m² | PlanFail |
| --- | --- | --- | --- | --- | --- |
| nearest | 0.72 ± 0.13 | 0.403 ± 0.08 | 0.96 | 47.7 | 7 |
| max_info_gain | 0.70 ± 0.13 | 0.306 ± 0.08 | 0.94 | 57.4 | 5 |
| random | 0.68 ± 0.13 | 0.328 ± 0.08 | 0.90 | 52.6 | 6 |
| info_gain_minus_cost | 0.64 ± 0.13 | 0.320 ± 0.08 | 0.90 | 54.5 | 5 |

**The four policies are statistically indistinguishable at n = 50.** That is
the baseline floor the model must clear, not a failure.

**Bugs found by running, not reading:**
- Every policy was scored on a *different* episode list (one env reused across
  policies continued the episode iterator). Fixed by pinning
  `iterator_options.shuffle=False` and sharding a sorted list.
- The planner captured `pathfinder` and the agent handle before `env.reset()`,
  so the first scene change killed the worker. 12 of 14 shards died; the
  4-episode smoke test passed because it never crossed a scene boundary.
- Episodes starved at step 0 when the nearest frontier was underfoot. Fixed
  with an initial 360° scan and a face-the-frontier turn.

---

## Phase 3 — Define a frontier-crossing option ✅

- [x] Approach pose defined in explored free space.
- [x] Frontier crossing direction defined.
- [x] Probe distance defined (2.0 m).
- [x] Horizon H defined (12 actions).
- [x] Canonical option `ω_i = (τ_approach, τ_cross, H)` generated, fixed by
      geometry and configuration alone rather than by execution outcome.
- [x] Options unreachable before crossing rejected, with a recorded reason.
- [x] Crossing success recorded.
- [x] Actual executed trajectory recorded.
- [x] All candidate options begin from exactly the same simulator state
      (asserted per branch, raises otherwise).

**Gate met.** 5/5 decision states executed 3 independent branches from a
bitwise-identical start, with agent state and maps verified restored and all
branches ending in distinct places.

**Design notes:**
- The approach pose is validated against the *observed* map, not just the
  navmesh. The navmesh covers the whole scene, so snapping alone can place an
  approach pose in never-seen territory — that puts privileged geometry into
  the option *definition*, not merely the controller.
- Branches drive `agent.act` directly rather than `env.step`, so a
  counterfactual costs the real episode zero steps and cannot end it.
- Raw simulator depth is unclipped where habitat-lab clips to
  `[min_depth, max_depth]`. Branch rollouts must clip to match.

---

## Phase 4 — Define ground-truth revelation ✅

- [x] Newly visible occupancy `ΔM_occ = M_{t+H}^obs \ M_t^obs`.
- [x] Newly observed semantic cells.
- [x] Number of newly revealed square metres.
- [x] Room category encountered — **derived, unreliable, see below**.
- [x] Whether the target becomes visible.
- [x] Distance from the final position to the target.
- [x] Collision / traversability outcome.
- [x] Newly created frontiers.
- [x] Future RGB-D observations retained for later experiments.
- [x] Every example stored with `scene_id`, `episode_id`, `decision_timestep`,
      `current_observation`, `current_map`, `navigation_goal`,
      `frontier_geometry`, `candidate_option`, `observation_history`,
      `future_revelation`.

**Gate met.** 27 branches across 7 decision groups visualised and manually
verified.

| channel | mean | sd | max |
| --- | --- | --- | --- |
| Revealed area ΔM_occ (m²) | 8.63 | 5.21 | 22.78 |
| — of which free | 6.33 | 4.29 | 19.19 |
| Semantics inside ΔM_occ (m²) | 7.44 | 5.11 | 21.45 |
| New frontiers | 2.70 | 2.16 | 8 |
| Collisions | 3.59 | 2.38 | 7 |
| Distance travelled (m) | 5.04 | 1.89 | 8.50 |

Crossings reaching unknown space 24/27; target became visible 8/27.

**Counterfactual spread within a decision state: 1.6× to 115×** (median 3.1×)
between the best and worst candidate. This is the paper's premise made
measurable.

**Two labelling defects caught by manual verification:**
- `ΔM_sem` counted every cell that gained a label, including already-mapped
  cells. 3.26× inflated in 23/23 records; would have trained the model to
  predict re-labelling of space already seen. Now restricted to `ΔM_occ`.
- Room labels were derived from every instance visible during a branch,
  including the starting room. Now only from newly seen instances.

**Room category is the weakest channel.** HM3D-v0.2 ships no room-type labels
(null category, degenerate bounding boxes), so the label is derived from region
object composition. It still skews to one class (kitchen 13/27) and disagrees
with spot checks. **Excluded from the workshop dataset** unless the project
moves to a dataset with trustworthy room labels.

---

## Phase 5 — Generate the FrontierReveal dataset ✅

### 5.1 Resolve the data blocker ✅

- [x] HM3D training meshes downloaded (35.3 GB: 27.2 GB meshes + 8.1 GB
      semantics). `habitat_sim.utils.datasets_download` does **not** work with
      Matterport API tokens -- the endpoint answers 307 to presigned S3 and the
      downloader writes the "Unauthorized" body into a file named `*.tar`, which
      fails later with a confusing `ReadError`. `scripts/download_hm3d_train.sh`
      fetches with `curl -L` and validates each archive before extracting.
- [x] Training scenes load with the annotated scene-dataset config. 800 scene
      directories extracted; the existing annotated config already referenced
      all 290 train paths.
- [x] Nonzero semantic instances confirmed: **145/145 annotated train scenes**
      (234–1724 instances, mean 664). HM3D-Semantics v0.2 annotates a subset of
      the 800, and that subset matches the 145 ObjectNav v2 train episode files
      exactly.
- [x] Scene-disjoint manifests: `manifests/full.json` — 89 train / 26 val /
      30 test, `pilot: false`.
- [x] HM3D validation and minival kept out of the training split.

> `manifests/pilot_debug.json` remains as the val-carved pilot partition used
> before the meshes arrived. It is flagged `pilot: true` and must not be used
> for reported results.

### 5.2 Freeze the candidate-frontier protocol ✅

For the first dataset version, the canonical detector is
`F_t = GeometricFrontiers(M_t^occ)`.

- [x] Phase 2 geometric frontier extractor used.
- [x] `detector_type=geometric` saved on every example.
- [x] Complete boundary components saved (`boundary_cells`), not only centroids.
- [x] Centroid, normal, information gain, approach pose and crossing option
      saved.
- [x] `frontiernet` and `union` reserved as detector values.

### 5.3 Collect decision states

Run multiple collection policies so the dataset is not shaped by one behaviour
distribution: nearest, max information gain, random, information gain − cost.

- [x] States saved only when at least two valid frontier options exist.
- [x] States where the target is already detected are excluded.
- [x] Near-duplicate consecutive states rejected (`min_state_separation_m`).
- [x] Complete simulator snapshot and map preserved, hashed and verified.
- [x] Observation history saved.
- [x] RNG seed, episode ID, scene ID and configuration hash saved.
- [x] Frontier boundaries saved per candidate.
- [ ] Raw frontier boundaries at *intermediate* timesteps, for lineage
      construction — deferred to Phase 6, which defines what it needs.

Acceptance rule: `2 ≤ |F_t| ≤ N_max` and `y_t^g = 0`. Rejections are counted,
not silently dropped. On the pilot: 751 over the frontier cap, 549 target
already visible, 103 with fewer than two reachable frontiers, 30 near-duplicate,
14 with fewer than two valid options.

### 5.4 Execute every branch

For each `f_i ∈ F_t`:

- [x] Common snapshot `S_t` restored between branches.
- [x] Snapshot and map hashes verified; a group whose map does not restore
      identically raises instead of being written.
- [x] `ω_i` executed.
- [x] Planned and executed trajectories recorded.
- [x] Crossing success, collision count and distance recorded.
- [x] Future RGB-D retained under `--keep-frames`.
- [x] `ΔM_i^occ` recorded **as a spatial mask**, not only its area. Storing only
      scalars would have left the occupancy head untrainable; caught before the
      pilot ran. Masks are bit-packed (80 KB vs 640 KB raw).
- [x] `ΔM_i^sem` recorded as a mask, restricted to `ΔM_i^occ`.
- [x] Target revelation `y_i^g` recorded.
- [x] Newly exposed frontiers recorded.

Store one decision group as
`D_t = (S_t, H_t, F_t, {f_i, ω_i, Y_i^gt}_{i=1..N_t})`.

### 5.5 Dataset integrity

- [x] Every branch starts from the same simulator state (504 groups checked).
- [x] One outcome per valid frontier (2421 branches checked).
- [x] Semantic changes restricted to newly observed cells (2421 checked).
- [x] No branch observation appears in another branch's input.
- [x] No scene overlaps across splits.
- [x] Schema version and code commit recorded on every example.
- [x] Summary plots for every channel (`channel_summary.png`).
- [x] 504 pilot decision groups generated (2421 branches, 41.6 min, 18 workers).
- [x] 50 decision groups rendered and inspected
      (`scripts/inspect_groups.py`); candidates from a shared state reveal
      spatially distinct regions adjacent to their own frontiers.
- [ ] Scale to several thousand groups — next, now that the checks pass and the
      real train manifest exists.

Pilot statistics: 4.80 branches per group (2–8), target-positive rate 33.5%,
crossing success 75.4%, mean revealed area 9.81 m² (max 41.6), mean 3.25 new
frontiers per branch. Within-group revealed-area spread: **median 3.9×, max
108×**.

> Do **not** include room-category prediction in the workshop dataset unless
> the project moves to a dataset with trustworthy room labels.

**Gate:** the dataloader returns all counterfactual branches from one common
state, and every branch passes snapshot-restoration checks.

**Already built (Phase 3–4):** snapshot/restore with per-branch verification,
canonical options, the full revelation computation, and the grouped storage
schema. Phase 5 is mainly scale, policy diversity, split manifests and
integrity auditing.

---

## Phase 6 — Implement the frontier lineage graph

### 6.1 Represent frontier observations

Store `v_i^t = (b_i^t, c_i^t, n_i^t, z_i^t, u_i^t, σ_i^t)`: boundary cells or
polyline, centroid, crossing normal, optional appearance feature, adjacent
unknown-region descriptor, status.

### 6.2 Match frontiers through time

- [ ] Transform boundaries into the common map frame.
- [ ] Compute dilated boundary overlap.
- [ ] Compute centroid distance.
- [ ] Compute orientation agreement.
- [ ] Compare adjacent unknown-space connectivity.
- [ ] Add appearance similarity when RGB evidence exists.
- [ ] Form a soft association matrix `A_ij`.
- [ ] Solve one-to-one matches first.
- [ ] Detect one-to-many splits.
- [ ] Detect many-to-one merges.
- [ ] Record births, updates, splits, merges, crossings and retirements.

### 6.3 Produce lineage labels

- [ ] Generate offline oracle associations using complete map connectivity.
- [ ] Keep online predicted lineage separate from oracle lineage.
- [ ] Store parent and child IDs.
- [ ] Record ID switches and ambiguous matches.
- [ ] Identify long-absence and reappearance events.

### 6.4 Branch isolation

- [ ] Include the lineage graph in `SimulatorSnapshot`.
- [ ] Restore the graph before every counterfactual branch.
- [ ] Ensure observations from branch *i* never update branch *j*.
- [ ] Add explicit cache-contamination tests.

**Report:** association precision and recall, IDF1, ID switches, split/merge
detection, cache-contamination rate.

**Gate: PASSED**, by 18 synthetic tests over sequences whose correct
identities are known by construction, plus 0 branch-isolation violations on
real episodes.

**FROZEN.** `MatchingConfig` is a frozen dataclass and is not to be retuned
until Phase 6.5 annotations exist — tuning against the current automatic oracle
would be fitting to a broken yardstick.

**Real-episode association accuracy is INVALID / UNMEASURED.** Against the
automatic oracle the lineage graph scores level with nearest-centroid matching
(F1 0.085 vs 0.090, IDF1 0.272 vs 0.288). Two oracle definitions were tried and
a third change (per-step rather than per-decision tracking) made it worse:

- *Oracle v1*, identity = the connected unresolved region a frontier borders.
  Broken: early in an episode the whole unexplored remainder of the house is
  one component, so six of seven distinct frontiers collapsed onto one id.
- *Oracle v2*, identity = clustered revelation footprints in the final map.
  Fixed contamination (0.086 → 0.000, beating the baseline's 0.059) but left F1
  unchanged.

Three changes with no movement points at the yardstick, not the matcher. These
numbers must not be reported.

## Phase 6.5 — Human lineage validation

Must finish **before Phase 11 (learned memory) and Phase 14 (lineage
ablations)**. Can run in parallel with Phase 8 training.

Label consecutive transition relations, not global identities:
`R_ij^{t→t+1} ∈ {none, update, split-child, merge-parent}`. Labelling
transitions is far less ambiguous than assigning one identity across a whole
episode.

> **Annotation rule.** Two frontier observations belong to the same lineage
> when they represent the same physical access boundary into unresolved space.
> Separate doorways remain separate even if they enter the same room.

- [x] Export sequences with before/after maps, numbered boundary components and
      RGB (`scripts/export_lineage_annotation.py`). 102 transitions exported.
- [ ] Broaden coverage: the current export spans 5 episodes but only **1
      scene**, because the episode iterator groups by scene. Re-export across
      scenes before labelling.
- [ ] Reduce the labelling burden: 2631 raw pairs across 102 transitions. Add a
      geometric prefilter so only plausibly-related pairs need a label.
- [ ] Label the correspondence matrices.
- [ ] Have a second person label a subset; adjudicate disagreements.
- [ ] Measure edge precision/recall/F1 by event type.
- [ ] Derive IDF1 only after the transition labels are validated.
- [ ] Retune `MatchingConfig` against the validated labels, then unfreeze.

> Current placeholder: `exploration.py` quantises frontier centroids to a
> half-metre grid for blacklisting, and `revelation.frontier_delta` matches new
> frontiers by centroid proximity. Both are explicitly marked to be replaced
> here.

---

## Phase 7 — Build learning-ready representations ✅

### 7.1 Model input

`X_{t,i} = (M_{t,i}^local, O_{t,i}^frontier, ω_i, g, G_t, m_i^t)`

- [x] Frontier-centred occupancy patch (8 m window at 0.1 m/cell, 80×80).
- [x] Observed-free / occupied / unknown channels.
- [x] Frontier boundary channel; crossing direction is +v by construction.
- [x] Agent position and crossing-trajectory channels.
- [x] Encoded option: approach and crossing action counts, probe distance,
      horizon, travel cost, information gain, boundary length, world yaw.
- [x] Path cost and probe horizon in the option encoding.
- [x] Goal category carried per group.
- [ ] Selected RGB-D observations and relative camera poses — deferred to
      Phase 9, which defines what Cosmos needs.
- [ ] Frontier lineage ID and memory mask — deferred to Phase 11, and gated on
      Phase 6.5.

### 7.2 Targets

Workshop targets are `Y_{t,i}^gt = (ΔM_i^occ, ΔM_i^sem, y_i^g)`, with
traversability and revealed area as auxiliary heads.

- [x] Outputs in a frontier-centred frame: origin at the centroid, +v along
      the crossing normal, so "beyond the boundary" is the same direction in
      every example.
- [x] Fixed metric extent and resolution.
- [x] Validity mask for cells outside the global map — no evidence there, so
      they are excluded from the loss rather than labelled unknown-but-observed.
- [x] Revealed-free and revealed-occupied are separate channels.
- [x] Semantics restricted to newly revealed cells.
- [x] Target presence and crossing success as scalar labels.

### 7.3 Dataloader

- [x] Complete decision groups returned, never independently shuffled branches.
- [x] Variable frontier counts padded with a candidate mask; padded slots are
      zero and masked.
- [x] Branch membership preserved.
- [x] Future observations cannot enter the input tensors — asserted per branch
      and per batch.
- [x] Visualisation inverts every transformation back to the global map.
- [ ] current-only / single-view / full-history input modes — deferred to
      Phase 11, where the memory ablation defines them.

**Gate: PASSED.** `scripts/check_tensors.py`:

| check | result |
| --- | --- |
| transform round-trip max error | 2.665e-15 m |
| boundary reprojection recall | 1.000 |
| target reprojection recall | 0.917 |
| future-leakage problems | 0 |
| padded slots zero | yes |

Target recall is 0.917 rather than 1.0 because roughly 8% of revealed area
falls outside the 8 m window. That is a truncation, not a misalignment —
`extent_m` is the knob if the occupancy head needs more context.

---

## Phase 8 — Train a deterministic structured predictor

Build this **before** depending on Cosmos 3. Otherwise a Cosmos failure cannot
be attributed to the dataset, the representation or the foundation model.

`Ŷ_{t,i} = F_θ(M_{t,i}^local, E(I,D,P), E(ω_i), E(g))` — a small
Transformer/Perceiver with a spatial decoder.

**Output heads**

- [ ] Newly revealed occupancy.
- [ ] Semantic features or category logits.
- [ ] Target-presence probability.
- [ ] Crossing success.
- [ ] Newly revealed area.

**Loss:** `L = λ_occ L_occ + λ_sem L_sem + λ_goal L_goal + λ_cross L_cross + λ_area L_area`

**Baselines**

- [ ] Dataset prior.
- [ ] Frontier geometry only.
- [ ] Current observation only.
- [ ] Single frontier keyframe.
- [ ] Global history without frontier association.

**Debug progression**

- [ ] Overfit one decision group.
- [ ] Overfit 32 decision groups.
- [ ] Train on the pilot dataset.
- [ ] Evaluate on held-out scenes.
- [ ] Verify candidate ranking changes when the action option changes.

**Gate:** the model overfits 32 examples and beats the scene-prior and
current-observation baselines on held-out prediction.

---

## Phase 9 — Integrate Cosmos 3 — **CLOSED, NEGATIVE RESULT**

Cosmos 3 was a prediction backbone or comparison, **not** the only path through
the project. That framing is why closing it does not block anything downstream.

**Two separate outcomes, deliberately not merged:**

| | verdict |
| --- | --- |
| Phase 9 **infrastructure** | **PASS** — environment isolation, action interface, manifests, caching, pose derivation, deterministic execution |
| Cosmos as a **counterfactual frontier predictor** | **FAIL** — action-sensitive but not directionally action-faithful |

The decisive experiment was preregistered (`c61c3da`) before its rollouts
existed and failed in both arms. Full record:
`archive/phase9d/PHASE9_CLOSURE.md`.

- Cosmos **does** consume the action: holding image, prompt and seed fixed and
  varying only the action moves mean frame difference from 2.29 (`still`) to
  31.6 / 39.4 (opposite turns).
- Cosmos **does not** follow the commanded direction: those opposite turns pan
  the camera the *same* way (flow −17.50 vs −17.80).
- At 0.25–0.50°/frame — bracketing the reference trajectory's own 0.264°/frame
  — there is still no directional response, and with zero translation no
  response at all (flow spread 0.0025 px across the whole range).
- Determinism passed exactly (0.00000 mean pixel difference on a repeat), so
  the null is not sampling noise.

Steps 8, 10 and 11 of the Phase 9 plan (structured conversion, decision group,
validation pilot) were **not run**, per the predeclared consequence. The
crossing-only reformulation was **not attempted**, since its precondition was
that this diagnostic pass.

**Optional, not required:** run the same optical-flow estimator on
Habitat-rendered known ±0.25° rotations, to validate the proxy that the null
rests on. This would not reopen or modify the Cosmos experiment.

*The sub-sections below are the original plan, retained for provenance.*

### 9.1 Conditioning

- [ ] Recent RGB history.
- [ ] Frontier-specific keyframes.
- [ ] Target description.
- [ ] Planned action or pose sequence `ω_i`.
- [ ] Rendered local occupancy patch, if supported via visual conditioning.
- [ ] Consistent camera intrinsics and frame rate.

> Do not assume Cosmos understands occupancy-map tensors directly. Render the
> map as an image or add an adapter.

### 9.2 Generate counterfactual futures

`V̂_{t,i}^{(k)} = Cosmos3(V_{≤t}, ω_i, g, m_i^t; ε_k)`

- [ ] Zero-shot inference first.
- [ ] Futures for every frontier in one decision group.
- [ ] K samples per frontier.
- [ ] Record runtime, VRAM and seeds.
- [ ] Ensure the action sequence corresponds to the branch trajectory.
- [ ] Ensure no ground-truth future frames appear in the prompt.

### 9.3 Convert video predictions into structured outputs

`V̂_i → (D̂_i, M̂_i^occ, M̂_i^sem, ŷ_i^g)` via frozen perception modules.

- [ ] Estimate depth.
- [ ] Project depth into the frontier frame.
- [ ] Run semantic or target detection.
- [ ] Accumulate predicted map revelations.
- [ ] Compare with the Phase 8 structured model.
- [ ] Consider LoRA only after the zero-shot baseline is measured.

> Start with quantised/offloaded inference on the 32 GB RTX 5090. Use Great
> Lakes for large evaluation batches or adaptation. (This machine has 2× RTX
> A6000 48 GB, which may change the plan.)

**Gate:** Cosmos produces correctly action-aligned futures for all branches of
one decision group, and the derived structured outputs can be scored against
Phase 4 ground truth.

---

## Phase 10 — Add stochastic prediction and calibration — **IN PROGRESS**

A single deterministic prediction is not sufficient for genuinely unobserved
regions. With Cosmos closed, the mixture comes from the **three archived Phase 8
seeds treated as a frozen deep ensemble** `{F_θ1, F_θ2, F_θ3}`. They are **not
retrained** during Phase 10; the only fitted quantities are calibration
parameters.

### 10.1 Freeze calibration and validation scenes ✅ (`6cf77b1`)

- [x] `manifests/phase10_calibration.json` — 45 scenes, hash `3b053336e442a370`
- [x] `manifests/phase10_validation.json` — 45 scenes, hash `97045a51ee24d054`
- [x] scene-disjoint (not group-disjoint: same-scene states share geometry)
- [x] zero overlap with the 58 consumed scenes
- [x] sealed set never read — guarded structurally by asserting every path lies
      under `hm3d_v0.2/train/`, since `sealed_final_v1` is HM3D val-split
- [x] frozen **before** any result was computed
- [x] 13 scenes reserved for Phase 11
- [x] every counterfactual branch kept grouped by decision state

Dev-pool scenes had no generated data, so a fresh branch-complete dataset was
built with the *exact* Phase 5 generation config (byte-identical to the
archived Phase 8 config apart from `split`), so calibration data matches what
the ensemble was trained on. Result: **calibration 429 groups / 1921 branches,
validation 494 / 2319**, zero errors.

### 10.2 Ensemble predictions ✅ (code `5d9d7e8`)

`Ŷ_i^(s) = F_θs(X_i)`, mean `Ȳ_i = (1/3) Σ_s Ŷ_i^(s)`.

- [x] store individual logits, probabilities and outputs — not only the mean

### 10.3 Uncertainty ✅

- [x] occupancy: predictive entropy `H(p̄)`, seed variance, and
      `U_MI = H(p̄) − (1/3) Σ_s H(p_s)`
- [x] semantics: mean pairwise cosine disagreement
- [x] target and crossing: variance, entropy, mutual information
- [x] area: `Ā_i` and `U_i^A = Std_s(Â_i^(s))`

> **Three seeds are insufficient for reliable empirical quantiles.** The
> seed-wise min/max is *not* a prediction interval and is never reported as
> coverage. Enforced in code: the helper is named `seed_range`, and all
> intervals come from the conformal procedure below.

### 10.4 Calibration, fitted on calibration scenes only ✅

Registered before evaluation:

- [x] target presence — temperature **and** Platt scaling
- [x] crossing — temperature scaling
- [x] free/occupied logits — separate per-channel temperatures
- [x] area — split-conformal residual intervals,
      `r_i = |A_i^gt − Ā_i| / (U_i^A + ε)`, interval
      `[Ā_i ∓ q_0.9 (U_i^A + ε)]`, with the finite-sample
      `ceil((n+1)(1−α))/n` quantile
- [x] constant-prior probabilities as baselines
- [x] coverage evaluated **only** on Phase 10 validation scenes
- [x] calibration never touches the sealed split

### 10.5 Evaluation table ✅ — **gate 3 of 6**

Compare: constant/dataset prior · each individual Phase 8 seed · uncalibrated
ensemble mean · calibrated ensemble · oracle-choice-among-seeds (diagnostic
only).

- [x] occupancy IoU, Brier, NLL — free 0.341 / occupied 0.028 / semantic 0.318
- [x] semantic cosine disagreement — 0.117 mean
- [x] target AUROC 0.544, Brier 0.225 calibrated, ECE 0.318 -> 0.137
- [x] crossing AUROC 0.968, accuracy 0.910, Brier 0.067, NLL 0.239
- [x] area MAE 3.94 m2 (prior 5.88), Pearson 0.720, coverage 0.870 vs 0.900
- [x] uncertainty-error Spearman — positive on every head, crossing MI +0.921
- [x] risk-coverage curves — beat random ordering on the majority of heads
- [x] ranking stability — **top-1 agreement 0.619**, Kendall tau 0.743
- [x] bootstrap by decision group

**Gate** (thresholds fixed in `scripts/phase10_gate.py` before the table
exists — coverage tolerance 0.05, degradation tolerance 0.02 relative):

1. ensemble uncertainty positively correlates with realised error
2. low-uncertainty predictions have lower error on risk–coverage curves
3. calibration improves Brier or NLL over the raw ensemble
4. calibrated target Brier beats the constant-prior baseline
5. area intervals achieve approximately their registered coverage
6. ensemble averaging does not materially degrade the deterministic metrics

> If target calibration still fails, **record it honestly**. Phase 11 visual
> frontier memory is the mechanism intended to add the missing semantic
> evidence; Phase 10 cannot calibrate information the inputs do not contain.

**Result: gate 3 of 6.** PASS on 1 (uncertainty tracks error), 2 (risk-coverage)
and 5 (area coverage 0.870 vs 0.900). FAIL on 3, 4 and 6. Full record:
`archive/phase10/README.md`.

**Target calibration failed, as anticipated.** Ensemble target AUROC is 0.544 —
chance. Platt scaling fits a slope of 0.060, i.e. it collapses the head onto the
base rate, and the resulting Brier margin over the constant prior is +0.00028
with a 95% group-bootstrap CI of [-0.0049, +0.0044]. Temperature scaling is
significantly *worse* than the prior. Calibration only stops the head being
confidently wrong (ECE 0.318 -> 0.137); it adds no discrimination. The
registered criterion did not specify a significance test, so it passes on the
point estimate; that literal result is preserved in `gate.json` and the
criterion is recorded FAIL on substance, because a gate passable by 0.00028
Brier is not a gate.

**The other two failures are diagnostic of Phase 8 v0, not of ensembling.**
Occupied-channel IoU collapses 0.0497 -> 0.0276 because the v0 loss formulation
leaves `p(revealed and occupied) ~ 0.038`, so averaging seeds pushes more mass
under the 0.5 threshold — the probabilities are better calibrated after
averaging (ECE 0.0072), it is the hard threshold that fails. Criterion 3 fails
on a majority rule while the two scalar heads improve substantially (target NLL
1.712 -> 0.691); the three spatial channels were already calibrated (ECE
0.007-0.020, fitted temperatures ~1.0) so there was nothing to fix.

**Caveat.** The manifests allocate 45 scenes per split but generation hit its
500-group target first, so only **22 scenes per split** are realised. Splits stay
disjoint and frozen; scene-sampling uncertainty is larger than the branch counts
suggest. Target base rate also shifts between splits (0.432 calibration vs 0.284
validation), which mis-specifies both the calibrator and the prior baseline.

**Most actionable finding, outside the registered gate.** Seed **top-1 ranking
agreement is 0.619** over 494 decision groups: the three seeds disagree about
which frontier is best in ~38% of cases. Any Phase 12 planner built on a single
seed inherits that arbitrariness.

---

## Phase 11 — Build budgeted lineage-aware frontier memory

### 11.1 Memory baselines

- [ ] No memory.
- [ ] Current observation only.
- [ ] First keyframe.
- [ ] Most recent keyframe.
- [ ] FIFO.
- [ ] Visual-diversity selection.
- [ ] Uncertainty-based selection.
- [ ] Global temporal memory.
- [ ] Shuffled-frontier memory.

### 11.2 Learned memory

`m_ℓ^t = W_ψ(m_ℓ^{t-1}, {Π_ℓ E(o_τ)})`, with `|m_ℓ^t| ≤ B`.

- [ ] Assign observations using visibility and frontier geometry.
- [ ] Encode relative pose in the frontier coordinate frame.
- [ ] Fixed budgets `B ∈ {0, 1, 4, 8, 16, 32}`.
- [ ] Train the writer to minimise revelation-prediction loss.
- [ ] Freeze the large prediction backbone initially.
- [ ] Learn insertion, fusion and eviction scores.
- [ ] Log why every token was retained or evicted.

### 11.3 Lineage events

- [ ] Transfer aligned parent memory during updates.
- [ ] Copy and transform relevant memory during splits.
- [ ] Fuse and deduplicate memory during merges.
- [ ] Archive memory when a frontier retires.
- [ ] Prevent unrelated lineages from sharing cache entries.

### 11.4 Memory evaluation

At equal token budgets, report prediction quality, calibration,
frontier-ranking regret, latency, memory usage and cache contamination.

**Gate:** at a matched budget, learned lineage-aware memory improves held-out
prediction or ranking regret over FIFO and global temporal memory. **If it does
not, weaken the paper's memory claim.**

---

## Phase 12 — Offline frontier ranking and regret

Realised utility from branch ground truth only:

`U_i^gt = α y_i^g + β A_i^new + γ |F_i^new| − λ C_i − ρ R_i`

- [ ] Fix utility coefficients using training/validation data.
- [ ] Report individual utility components alongside the combined score.
- [ ] Compute the oracle frontier for every decision group.
- [ ] Rank candidates using predicted revelations.
- [ ] Evaluate target-heavy and exploration-heavy coefficient settings.
- [ ] Add CVaR or uncertainty penalties.
- [ ] Compare against nearest, information gain and cost baselines.

**Report:** `Regret_t = U_t^oracle − U_t^selected`, top-1 and top-k oracle
recall, Spearman ranking correlation, mean and median regret, catastrophic
choice rate.

**Gate:** prediction-based ranking reduces paired regret over geometric
information gain on held-out decision groups.

---

## Phase 13 — Close the ObjectNav loop

At every real navigation decision: update occupancy and semantics; detect
frontier candidates; update the lineage graph; update frontier memories;
construct options; predict outcomes; score candidates; execute only the
selected option; update memory from the realised outcome; stop when the target
is visible and reachable.

- [ ] No simulator forks used by the deployed policy.
- [ ] Selected actions count toward the 500-step budget.
- [ ] **Fix the detect-but-never-stop bug before evaluation** (~15% of Phase 2
      episodes saw the target and never called STOP).
- [ ] Restore normal `env.step` accounting.
- [ ] Handle invalidated frontiers and planning failures.
- [ ] Blacklist repeated failed crossings.
- [ ] Safe fallback to nearest reachable frontier.
- [ ] Limit expensive Cosmos calls to the best M candidates after cheap
      filtering.
- [ ] Log predictions and realised residuals.
- [ ] Write residuals back into the relevant lineage memory.

Evaluate on the **exact episode list already used for Phase 2 baselines**.

**Report:** SR, SPL, SoftSPL, distance to success, collisions, revisits and
oscillations, model calls, inference time.

**Gate:** at least 50 paired episodes complete without leakage, crashes, broken
step accounting or persistent planner deadlocks.

---

## Phase 14 — Run the core ablations

**Candidate detection:** geometric; FrontierNet; union; oracle-validated.

**Lineage:** none; nearest-centroid; position + appearance; full lineage graph;
oracle lineage; shuffled lineage.

**Memory:** no cache; single keyframe; global memory; FIFO; shuffled; learned;
budgets 0, 1, 4, 8, 16, 32.

**Predictor:** priors only; compact structured model; Cosmos zero-shot; Cosmos
adapted; deterministic vs stochastic; remove occupancy head; remove semantics;
remove target prediction.

**Planner:** predicted information only; predicted target only; cost only;
uncertainty removed; risk penalty removed; oracle-revelation upper bound.

Use paired episode bootstrapping or paired tests, since every method uses the
same episode list.

**Gate:** every major paper claim is supported by a controlled ablation, not by
comparing the full model against a weak baseline.

---

## Phase 15 — Scale and freeze the final results

**Dataset and prediction**

- [ ] Scale training data to several thousand decision groups.
- [ ] Report scene, state and branch counts.
- [ ] Report target-positive rate and class balance.
- [ ] Report split/merge lineage statistics.
- [ ] Freeze train/validation/test manifests.
- [ ] Freeze model checkpoints and evaluation configs.

**Closed-loop evaluation**

- [ ] Final held-out HM3D-v2 benchmark.
- [ ] Same episode IDs for every method.
- [ ] Enough episodes for tighter intervals than the current n = 50 (Phase 2
      intervals were ±0.13 on SR, which cannot resolve the differences seen).
- [ ] Per-scene and aggregate results.
- [ ] Compute and wall-clock cost.
- [ ] Stratify by target category and geodesic distance.

**Qualitative analysis**

- [ ] One common state with three counterfactual branches.
- [ ] Prediction versus ground truth for every branch.
- [ ] Lineage split and merge examples.
- [ ] Successful and failed memory retention.
- [ ] Calibration plots.
- [ ] Failure taxonomy.

**Known issues to close**

- [ ] Fix detect-but-never-stop.
- [ ] Sweep `cost_weight` (currently 1.0; gain-minus-cost is the *worst*
      baseline, which suggests mistuning rather than a wrong rule).
- [ ] Exclude unreliable HM3D room labels.
- [ ] Audit max-range depth behaviour everywhere.
- [ ] Confirm semantic observations are nonzero in every evaluated scene.
- [ ] Reproduce a subset from a clean environment.

**Gate:** final tables contain no placeholders, a clean rerun reproduces the
main numbers, and all reported methods use identical data and episode lists.

---

## Phase 16 — Paper, reproducibility and release

**Paper**

- [ ] Rewrite the method to match the implemented system exactly.
- [ ] State whether the canonical detector is geometric, FrontierNet or union.
- [ ] Distinguish generated video from structured revelation prediction.
- [ ] Limit workshop claims to occupancy, semantics and target presence.
- [ ] Remove room classification unless reliable labels are obtained.
- [ ] Include the branch-complete dataset protocol.
- [ ] Include the lineage-memory mechanism.
- [ ] Include prediction, calibration, regret and ObjectNav metrics.
- [ ] Discuss computation and failure cases honestly.

**Figures:** system overview; same-state counterfactual branching; frontier
lineage graph; prediction versus ground truth; memory-budget curve; calibration
plot; regret-versus-SPL relationship.

**Reproducibility**

- [ ] Freeze the final commit and tag the release.
- [ ] Export the exact environment.
- [ ] Publish configuration files and split manifests.
- [ ] Publish or document checkpoint access.
- [ ] One-command pilot generation and evaluation.
- [ ] Dataset schema documentation.
- [ ] Document external licences, including FrontierNet and Cosmos.
- [ ] Verify all tables regenerate from logs.
- [ ] Run the complete test suite.

**Final gate:** a new user can reproduce one decision group, one model
prediction, one ranking decision and one closed-loop episode from the
documented release.

---

## Standing conventions

**Privileged information.** Three components use ground truth unavailable at
deployment. Every reported number depending on them must say so.

| component | privilege | replaced in |
| --- | --- | --- |
| `planning/habitat_planner.py` | habitat navmesh — ground truth for the whole scene, including unobserved regions | never; it isolates frontier choice from control |
| `planning/goal_detector.py` | ground-truth semantic instance ids | Phase 8 target-presence head |
| Phase 5 branch generation | simulator state forking | training/eval labels only, never at deployment |

**Success threshold.** `task.success_distance` is set to 1.0 m, not habitat's
0.1 m default. It measures distance to the nearest goal *viewpoint*, so 1.0 m
loosens success and makes SR/SPL **non-comparable to published HM3D ObjectNav
results**. State the value in every reported table.

**Determinism.** Per-episode randomness derives from
`(seed, scene_id, episode_id)`, never from a running stream, so an episode
replays identically regardless of what preceded it.


---

## Phase 8 v1 — design notes from the v0 diagnosis

**Occupied head.** Occupied cells are *not* rare conditional on revelation
(`p(occupied | revealed) ~ 0.336`), but they are rare under the effective
full-window loss denominator (`p(revealed and occupied) ~ 0.038`). This is an
effective class/masking imbalance **created by the loss formulation, not by the
dataset**. Measured consequence: recall 0.049 at precision 0.458.

v1 correction — a genuinely factorised training objective:

```
p(free)     = p(reveal) * p(free | reveal)
p(occupied) = p(reveal) * p(occupied | reveal)
```

- [ ] Train the reveal head over the complete spatial window.
- [ ] Train the conditional free/occupied softmax **only over ground-truth
      revealed cells**.
- [ ] Optionally add a boundary or distance-transform loss for thin-wall
      alignment.

**Metric preregistration (before the sealed set is opened).** Exact IoU stays
**primary**; 2-cell-tolerant IoU is a **secondary diagnostic** only. Registered
now so the choice cannot be made after seeing a result. On v0 validation these
read 0.051 exact against 0.179 tolerant.

**Target-presence head.** The defensible statement, and the one to use in the
paper:

> Under the tested current-geometry representation, target-positive and
> target-negative frontier outcomes are poorly separable, producing weak
> discrimination and severely miscalibrated probabilities.

Cohen's d <= 0.11 covers the five tested *scalar* geometric features; it does
not establish that every spatial pattern in the full geometric tensor is
uninformative, and AUROC 0.622 indicates some weak ranking signal survives.
Claiming "no evidence exists" would overstate the measurement.

Evidence for the paper's motivation, stated at the right strength:

| quantity | v0 validation | reference |
| --- | --- | --- |
| target AUROC | 0.617 | chance 0.500 |
| target AUPRC | 0.472 | base rate 0.363 |
| target Brier | 0.343 | **constant prior 0.231** |

The head is worse than the base rate as a probability estimate, and calibration
is inverted at both ends. Phase 11's ladder — current-only -> keyframe -> FIFO
-> lineage-aware — now has concrete targets to beat: AUPRC > 0.472 and Brier
< 0.231 on validation, under a matched token budget.


---

## Phase 9 — status and measured results

### 9A/9B: depth estimator and metric-scale anchoring (validation, 16 branches)

Estimator: Depth Anything V2 Metric Indoor (Small). Real branch RGB, real
converter, no ground-truth scale in the main condition.

| | AbsRel | delta1 | median scale ratio | drift |
| --- | --- | --- | --- | --- |
| raw | 0.711 | 0.053 | 0.608 | 0.088 |

| condition | free IoU | occ IoU | occ tolerant | area ratio |
| --- | --- | --- | --- | --- |
| `gt_depth` (converter ceiling) | 0.920 | 0.811 | 0.935 | 0.900 |
| `none` | 0.517 | 0.087 | 0.317 | 1.814 |
| **`first_frame`** (deployable) | **0.599** | **0.178** | **0.546** | **1.123** |
| `oracle` (upper bound) | 0.670 | 0.196 | 0.587 | 1.149 |

**First-frame anchoring is sufficient.** It cuts area inflation 1.814 -> 1.123
using only the already-observed conditioning frame, and oracle scaling adds
little beyond it (0.670 vs 0.599). Further scale-calibration research is not
worth the time; the residual is shape error, not scale error.

Three interpretations, stated at the right strength:

1. **Scale is correctable but does not fully explain the error.** A pure 0.608
   factor would not produce AbsRel 0.711 at delta1 0.053. Post-calibration
   AbsRel and delta1 are now reported to quantify the residual shape error
   directly. The map results already imply it is substantial: oracle scaling
   reaches only 0.670/0.196 against a 0.920/0.811 ceiling.
2. **The 0.920 -> 0.599 gap is not purely converter brittleness.** It contains
   estimator shape and boundary-location error *and* the converter's
   sensitivity to it. A robust converter cannot place a wall the estimator
   never saw.
3. **Cosmos headroom is indicative, not established.** 0.599 (validation, 16
   branches, real RGB) against v0's 0.328 (test, 895 branches) crosses both
   distribution and sample size. It justifies running the Cosmos experiment; it
   is not a result.

### 9C: robust converter for predicted depth

Goal, stated precisely: **make map integration degrade gracefully under
imperfect depth — not recover the ground-truth-depth ceiling.**

Implemented in `mapping/robust_integration.py`:

- [x] Error envelope `sigma(d_hat)` fitted from **real residuals**, depth-binned
      0.9 quantile, monotone non-decreasing — not a guessed percentage.
- [x] Conservative free carving to `d_free = max(0, d_hat - k_f*sigma)`.
- [x] Soft occupied evidence over `[d_hat - k_o*sigma, d_hat + k_o*sigma]` with a
      **triangular peak at d_hat** — a uniform band would inflate occupied area
      and destroy precision.
- [x] Confidence-weighted log-odds, bounded, with predicted depth given less
      weight per frame than sensor depth.
- [x] Invalid/extreme rejection, depth-discontinuity rejection, edge-preserving
      bilateral filter.
- [ ] Calibration subset and converter-validation subset drawn from
      `dev_pool_v1`, different scenes, ~100+ branches.
- [ ] Controlled ablations: current / +first-frame / +conservative carving /
      +soft band / +confidence weighting / full, each reporting exact free IoU,
      exact occupied IoU, tolerant occupied IoU, area ratio, occupied precision
      and recall, **and perfect-depth performance** (so the robust converter
      cannot win by being so conservative it damages clean input).
- [ ] Freeze on converter-validation.

**Predeclared selection rule** (registered before the ablation runs):
require area ratio inside a preset acceptable interval; among survivors
maximise **exact** occupancy macro-IoU; use tolerant occupied IoU **only** as a
tie-breaker. This prevents selecting a converter merely because it thickens
every wall enough to improve the tolerant metric.

**Phase 9C gate.** Freeze the converter if it improves exact occupied recall
without uncontrolled area inflation, preserves useful free-space IoU, is stable
across converter-validation scenes, and does not significantly damage the
ground-truth-depth ceiling.

### 9D: minimal Cosmos environment — after 9C freezes


---

## Phase 9C — COMPLETE, gate FAILED, converter FROZEN

Registered before the run: area interval [0.90, 1.10] as ratio of totals;
selection by exact macro-IoU with tolerant occupied IoU as tie-breaker only;
`CLEAN_MACRO_TOLERANCE = 0.05`; `BOOTSTRAP_SEED = 20260727`; deterministic
tie-break order; branch input hashes.

**Selected**: `k_free=0.5, k_occupied=0.5, w_free=0.5, tau_free=0.5,
tau_occ=0.15` — 5 of 24 configurations inside the area interval, so the
fallback did not fire.

| condition | freeIoU | occIoU | macro | occTol | occP | occR | areaR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0.327 | 0.092 | 0.210 | 0.338 | 0.107 | 0.462 | 3.008 |
| +first_frame | 0.354 | 0.138 | **0.246** | 0.428 | 0.151 | **0.605** | 2.161 |
| +conservative_carving | **0.407** | 0.079 | 0.243 | 0.390 | **0.196** | 0.150 | **0.892** |
| +soft_occupied | 0.322 | 0.120 | 0.221 | 0.402 | 0.144 | 0.516 | 1.472 |
| +confidence_weighting | 0.348 | 0.125 | 0.236 | 0.419 | 0.165 | 0.439 | 1.230 |
| +rejection_filtering | 0.349 | 0.126 | 0.237 | 0.422 | 0.166 | 0.434 | 1.226 |
| `gt_depth_original` (ceiling) | 0.932 | 0.885 | 0.908 | 0.978 | 1.000 | 0.885 | 0.927 |
| `gt_depth_robust` (conservatism) | 0.419 | 0.161 | 0.290 | 0.532 | 0.210 | 0.528 | 0.906 |

Paired group bootstrap, full robust vs +first_frame (20 groups, 60 branches):
occupied IoU −0.012 [−0.030, +0.006] n.s.; free IoU −0.005 [−0.052, +0.048] n.s.

**Gate: 4 of 5 criteria fail.** The decisive one is the conservatism test:
`gt_depth_robust` reaches macro 0.290 against a 0.908 ceiling, a 68% loss on
*perfect* input. Fitted `sigma(d)` is 1.32–2.00 m, comparable to room
dimensions, so the converter carves free space ~0.75 m short of every true
surface and smears occupied evidence over a 1.5 m band. Without that diagnostic
row, `+conservative_carving`'s area ratio 0.892 and precision 0.196 would have
read as progress.

**Bounded claim.** The tested residual-envelope method — conservative carving
to `d_hat - k_f*sigma` with a soft occupied band over `+/- k_o*sigma` — did not
improve imperfect-depth mapping and severely degraded clean-depth performance
at the observed error magnitude. This does **not** establish that every
uncertainty-aware mapping method must fail, only that this formulation is
unsuitable when `sigma(d)` is comparable to scene dimensions.

**Sigma provenance** (two figures appear in the record; they are not in
conflict). The component test reported `sigma(1 m)=0.105`, `sigma(4 m)=0.311`
from a **synthetic fixture** — `pred = gt + N(0, 0.05*gt)`, 5% relative noise
imposed by construction to verify that `fit_error_envelope` is monotone and
range-dependent, 49k samples. The Phase 9C figure of 1.32–2.00 m comes from
**real** Depth Anything V2 residuals against habitat sensor depth, 272M
samples, 0.9 quantile, 8 bins over [0, 5] m. Units are metres in both cases.
First-frame scale is applied **before** fitting, so the envelope describes
residual error *after* calibration. Consistency: `sigma/d = 1.53/2.0 = 0.77` at
2 m, i.e. ~77% relative at the 90th percentile, against an independently
measured mean AbsRel of 0.711 in 9A — a 0.9-quantile envelope must exceed a
mean relative error, so the two agree. The synthetic fixture was 15x more
optimistic than reality.

**Interpretation.** The failure is upstream of integration. A residual error of
that magnitude after scale correction is shape error, consistent with the 9B
finding that oracle scaling reached only 0.670/0.196. No sensor model can be
robust to uncertainty comparable to the scene without erasing the signal.

**Decision, frozen in `archive/phase9c/FROZEN.json`:** use **first-frame scale
anchoring with the ORIGINAL exact-depth converter** for Phase 9D onward. The
robust converter is retained, recorded and unused. No further converter tuning.

### Phase 9D — CLOSED, gate FAILED, Cosmos archived as a negative result

- [x] 9D env — isolated Python 3.13 / torch 2.10 venv; Habitat 3.9 untouched.
      The two environments share only files, never imports.
- [x] Action interface — **Class C**: Cosmos both consumes and emits actions,
      selected by `model_mode`. `forward_dynamics` reads a 9-D action file;
      `policy`/`inverse_dynamics` pass zeros. Embodiment `camera_pose`
      (domain 2, raw dim 9), `[translation(3), rot6d(6)]`, backward-framewise.
      No textual action substitution was ever needed.
- [x] Resources — 15.88 B parameters, 30.3 GB BF16, 30.6 GB reserved on one
      A6000 with 18.1 GB headroom. No sharding, offload or quantization.
- [x] One 17-frame rollout generated, guardrails enabled throughout.
      8 of 9 verification checks passed; **turn direction FAILED** (yaw/flow
      correlation 0.16 against a preregistered 0.30).
- [x] Action-sensitivity probes and the preregistered in-distribution yaw
      diagnostic — **both FAILED**.
- [ ] ~~9E one complete decision group~~ — not run (predeclared consequence)
- [ ] ~~9F 10-20 group validation pilot~~ — not run (predeclared consequence)

**Phase 9 gate: FAILED on option adherence.** The pipeline, interpretability
and reproducibility sub-goals were met (determinism was exact: 0.00000 mean
pixel difference on an identical-seed repeat), but option adherence is the one
that matters, and Cosmos does not have it. Cosmos was never required to beat
Phase 8 — it was required to respond to the option, and it does not.

A useful by-product: the decision-point pose is not stored in the dataset and
had to be recovered by inverting the first recorded action. Re-rendering at the
recovered pose reproduces the stored conditioning frame to **0.001/255**, which
both validates the recovery and confirms the conditioning depth is genuine
observed sensor depth rather than future information.

**Reusable if an action-faithful generator appears:** the action-interface
determination, nominal-pose derivation, conditioning manifest, cache keys and
two-environment isolation all stand.
