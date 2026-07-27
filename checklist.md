# FrontierWorld — start-to-finish checklist

**Working title:** FrontierWorld: Counterfactual Revelation Models and
Lineage-Aware Predictive Memory for Object Navigation

**Research question:** Can frontier-specific memory predict what each
exploration action will reveal, and does that prediction improve frontier
selection?

**Status:** Phases 0–5 complete. Phase 6 (lineage graph) and Phase 7
(learning tensors) are next and can proceed in parallel.

---

## Dependency structure

```
Phase 5: FrontierReveal dataset
   ├── Phase 6: Lineage graph ──→ Phase 11: Frontier memory ─┐
   └── Phase 7: Learning tensors                             │
          └── Phase 8: Structured predictor                  │
                 └── Phase 9: Cosmos 3                       │
                        └── Phase 10: Uncertainty            │
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
| 6–16 | not started | | |

**Test suite:** 105 passing.
**Known blockers:** none. HM3D train meshes downloaded (800 scenes, 145 with
semantics); `manifests/full.json` is the non-pilot partition (89/26/30).

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

**Gate:** a frontier that persists, splits and later disappears maintains
correct parent–child identities without leaking branch information.

> Current placeholder: `exploration.py` quantises frontier centroids to a
> half-metre grid for blacklisting, and `revelation.frontier_delta` matches new
> frontiers by centroid proximity. Both are explicitly marked to be replaced
> here.

---

## Phase 7 — Build learning-ready representations

### 7.1 Model input

`X_{t,i} = (M_{t,i}^local, O_{t,i}^frontier, ω_i, g, G_t, m_i^t)`

- [ ] Frontier-centred occupancy patch.
- [ ] Observed / free / occupied / unknown masks.
- [ ] Frontier boundary and crossing direction.
- [ ] Selected RGB-D observations.
- [ ] Relative camera poses.
- [ ] Encoded approach and crossing trajectories.
- [ ] Path cost and probe horizon.
- [ ] Goal category.
- [ ] Optional global-map context.
- [ ] Frontier lineage ID and memory mask.

### 7.2 Targets

Workshop targets are `Y_{t,i}^gt = (ΔM_i^occ, ΔM_i^sem, y_i^g)`, with
traversability and revealed area as auxiliary heads.

- [ ] Express outputs in a frontier-centred coordinate frame.
- [ ] Fixed metric extent and resolution.
- [ ] Mask cells outside the valid target region.
- [ ] Distinguish newly revealed free from occupied space.
- [ ] Define semantics only over newly revealed cells.
- [ ] Define target presence consistently from semantic instances.

### 7.3 Dataloader

- [ ] Return complete decision groups, not independently shuffled branches.
- [ ] Pad variable frontier counts with masks.
- [ ] Preserve branch membership.
- [ ] Support current-only, single-view and full-history inputs.
- [ ] Prevent future observations from entering input tensors.
- [ ] Visualisation functions that invert every transformation.

**Gate:** one batch can be transformed to model coordinates and reconstructed
into the original map frame without alignment errors.

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

## Phase 9 — Integrate Cosmos 3

Cosmos 3 is a prediction backbone or comparison, **not** the only path through
the project.

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

## Phase 10 — Add stochastic prediction and calibration

A single generated future is not sufficient for genuinely unobserved regions.

### 10.1 Multiple hypotheses

`p_θ(Y_i | X_i) = Σ_{k=1..K} π_ik p_{θ,k}(Y_i | X_i)`

via multiple Cosmos generations, a mixture decoder on the structured model, or
a small ensemble.

### 10.2 Train and evaluate uncertainty

- [ ] Predict probability of target revelation.
- [ ] Predict crossing-success probability.
- [ ] Predict distributions over revealed area.
- [ ] Compute uncertainty across spatial predictions.
- [ ] Use proper probabilistic losses.
- [ ] Calibrate on validation scenes only.
- [ ] Produce reliability diagrams.

**Report:** NLL, Brier score, expected calibration error, best-of-K coverage,
diversity versus accuracy, uncertainty versus actual error.

> Do not report only best-of-K; it rewards diversity without testing
> calibration.

**Gate:** higher predicted uncertainty corresponds to larger realised error,
and target/crossing probabilities are measurably calibrated.

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
