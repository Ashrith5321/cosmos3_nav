FrontierWorld start-to-finish checklist
Phase 0 — Freeze the research claim
 Use the working title: FrontierWorld: Counterfactual Revelation Models for Object Navigation.

 Write the primary research question:

Can frontier-specific memory predict what each exploration action will reveal, and does that prediction improve frontier selection?

 Fix the minimum prediction outputs:
 Newly revealed occupancy
 Target-presence probability
 Room or semantic category
 Traversability
 Expected newly revealed area
 Explicitly exclude from the first version:
 RGB/video generation
 Raw VLM KV-cache manipulation
 Cross-episode memory
 Real-robot experiments
 Large cross-frontier GNN
 Language-event and affordance generation
 Define the three central contributions:
Branch-complete counterfactual frontier dataset.
Frontier-specific predictive memory under a fixed budget.
Prediction-based frontier planning.

Gate: Do not start implementation until these claims fit into one paragraph.

Phase 1 — Create the repository

Recommended structure:

frontierworld/
├── configs/
├── frontierworld/
│   ├── mapping/
│   ├── frontiers/
│   ├── lineage/
│   ├── memory/
│   ├── models/
│   ├── planning/
│   ├── data/
│   └── evaluation/
├── scripts/
│   ├── generate_branches.py
│   ├── train_predictor.py
│   ├── evaluate_prediction.py
│   └── evaluate_navigation.py
├── tests/
├── paper/
├── requirements.txt
└── README.md
 Initialize Git.
 Add a reproducible Python environment.
 Add experiment configuration files.
 Add deterministic random seeds.
 Install Habitat-Sim and Habitat-Lab.
 Download HM3D train and validation scenes.
 Confirm one HM3D ObjectNav episode runs.
 Save RGB, depth, pose, semantic annotation and map output.
 Add experiment tracking using W&B, TensorBoard or CSV logs.

Gate: One command must run one episode and save an episode log.

Phase 2 — Build the basic navigation system

Start with standard map-based frontiers, not FrontierNet.

 Construct a 2D occupancy map from depth and ground-truth pose.
 Represent cells as free, occupied or unknown.
 Extract free–unknown boundary cells.
 Cluster boundary cells into frontiers.
 Compute a center and orientation for every frontier.
 Remove frontiers that are too small or unreachable.
 Implement A* or FMM navigation to a selected frontier.
 Implement basic frontier policies:
 Random frontier
 Nearest frontier
 Maximum geometric information gain
 Information gain minus travel cost
 Run at least 50 episodes.
 Record SR, SPL, collisions and explored area.

Gate: The nearest and information-gain policies must complete episodes without planner failures.

Phase 3 — Define a frontier-crossing option

Every candidate frontier needs a reproducible macro-action.

 Define the approach pose in explored free space.
 Define the frontier crossing direction.
 Define the probe distance, such as 1–3 meters.
 Define horizon H, such as 8–16 actions.
 Generate a canonical option:
ω
i
	​

=(τ
i
approach
	​

,τ
i
cross
	​

,H).
 Reject options that are unreachable before crossing.
 Record whether the crossing succeeds.
 Record the actual executed trajectory.
 Ensure all candidate options begin from exactly the same simulator state.

Gate: Given one navigation state and three frontiers, the system must execute three independent branches from the identical starting state.

Phase 4 — Define ground-truth revelation

Before training anything, define exactly what “lies beyond a frontier” means.

For each branch, record:

 Newly visible occupancy cells:
ΔM
i
occ
	​

=M
t+H
observed
	​

∖M
t
observed
	​

.
 Newly observed semantic cells.
 Number of newly revealed square meters.
 Room category encountered.
 Whether the target becomes visible.
 Distance from the final position to the target.
 Collision or traversability outcome.
 Newly created frontiers.
 Future RGB-D observations for optional later experiments.

Store every example with:

scene_id
episode_id
decision_timestep
current_observation
current_map
navigation_goal
frontier_geometry
candidate_option
observation_history
future_revelation

Gate: Visualize at least 20 examples showing the map before crossing, chosen frontier, trajectory and newly revealed region. Manually verify them.

Phase 5 — Generate the FrontierReveal dataset

For every saved decision state:

 Save or reconstruct the complete simulator state.
 Detect all valid frontiers.
 Fork the simulator state for frontier f
1
	​

.
 Execute ω
1
	​

.
 Record Y
1
gt
	​

.
 Restore the original state.
 Repeat for every remaining frontier.
 Store all branches together as one decision group.
 Prevent train/validation scene overlap.
 Generate a pilot set of approximately 500 decision groups.
 Run dataset integrity checks.
 Scale to several thousand decision groups if storage permits.
 Calculate dataset statistics:
 Number of scenes
 Number of decision states
 Frontiers per state
 Crossing success
 Target-revelation frequency
 Room-category distribution
 Newly revealed area distribution

Gate: A dataloader must return all counterfactual branches belonging to one common decision state.

Phase 6 — Establish non-learning baselines

Implement these before the proposed model:

 Predict mean training-set revelation.
 Predict revelation using frontier geometry only.
 Predict using current RGB-D observation.
 Predict using one frontier keyframe.
 Predict using a global temporal history.
 Predict using a FIFO frontier history.
 Predict using a shuffled frontier history.
 Create an oracle predictor using ground-truth branch outcomes.

Initial model:

 Frozen visual encoder.
 Small MLP or Transformer decoder.
 Occupancy head.
 Target-presence head.
 Room-category head.
 Traversability head.
 Newly revealed area head.

Gate: The current-observation model must beat the geometry-only and dataset-prior baselines.

Phase 7 — Implement simple frontier persistence

Start with a heuristic version.

For every new frontier, compare it with previous frontiers using:

 Boundary overlap
 Center distance
 Orientation similarity
 Visual-feature similarity
 Visibility-ray agreement
 Assign persistent frontier IDs.
 Detect unmatched frontier births.
 Retire disappeared frontiers.
 Detect one-to-many splits.
 Detect many-to-one merges.
 Transfer memory during split and merge.
 Save the complete lineage graph.

Evaluate:

 Association precision and recall
 IDF1
 ID switches
 Split accuracy
 Merge accuracy
 Cache-contamination rate

Gate: Persistent association must outperform nearest-center matching.

Phase 8 — Build frontier-specific memory

Implement memory variants sequentially:

8.1 Keyframe cache
 Store the first frontier observation.
 Store the most recent observation.
 Store the most visually different observation.
 Limit every frontier to the same number of tokens or frames.
8.2 Learned memory
 Encode observations into spatial visual tokens.
 Transform tokens into a frontier-centered coordinate frame.
 Add B learnable memory slots per frontier.
 Implement a cross-attention write module.
 Implement slot fusion and eviction.
 Train the memory jointly with future-revelation prediction.
 Test budgets:
B∈{0,1,4,8,16,32}.

Compare:

 Current observation only
 First keyframe
 Global history
 Frontier FIFO
 Visual-diversity cache
 Shuffled cache
 Learned frontier memory
 Oracle-associated learned memory

Gate: Learned memory must beat FIFO and shuffled memory at the same budget.

Phase 9 — Train the revelation model

Start deterministically.

 Input current map.
 Input frontier geometry.
 Input frontier memory.
 Input target category.
 Input relative trajectory or macro-action tokens.
 Predict occupancy revelation.
 Predict semantic/room information.
 Predict target presence.
 Predict traversability.
 Predict newly revealed area.

Use a loss such as:

L=λ
occ
	​

L
occ
	​

+λ
target
	​

L
target
	​

+λ
room
	​

L
room
	​

+λ
trav
	​

L
trav
	​

+λ
area
	​

L
area
	​

.
 Verify the model actually uses action conditioning by permuting trajectory inputs.
 Verify the model actually uses frontier memory by shuffling caches.
 Train with at least three random seeds.
 Save the best checkpoint using validation prediction loss.

Gate: The full model must outperform the same architecture without action conditioning and without frontier memory.

Phase 10 — Add uncertainty

Only begin after deterministic prediction works.

 Start with an ensemble or Monte Carlo dropout.
 Predict target probability rather than a binary decision.
 Predict uncertainty for occupancy and traversability.
 Calculate Brier score.
 Calculate expected calibration error.
 Plot predicted confidence versus empirical accuracy.
 Compare uncertainty for easy and highly occluded frontiers.
 Test whether incorrect frontier associations increase uncertainty.

Optional full version:

 Add K spatial-semantic future hypotheses.
 Evaluate best-of-K accuracy.
 Evaluate diversity and hypothesis coverage.
 Check for mode collapse.

Gate: Higher-confidence predictions must be measurably more accurate.

Phase 11 — Convert predictions into frontier scores

For every candidate, calculate:

Q(f
i
	​

)=E[U
i
	​

]+ηI
i
	​

−λC
i
	​

−ρR
i
	​

.

Implement each term:

 Predicted probability of finding the target
 Expected newly revealed area
 Semantic relevance to the target
 Travel cost
 Traversability risk
 Prediction uncertainty

Before closed-loop navigation, evaluate offline:

 Predicted versus oracle frontier rank correlation
 Top-1 oracle frontier accuracy
 Top-k oracle recall
 Frontier-selection regret
Regret
t
	​

=U(f
t
oracle
	​

)−U(f
t
pred
	​

).

Gate: Predicted selection must have lower regret than nearest-frontier and geometric-information-gain selection.

Phase 12 — Run closed-loop ObjectNav

Compare:

 Random frontier
 Nearest frontier
 Geometric information gain
 Semantic/VLM frontier score
 Prediction without frontier memory
 Prediction with FIFO memory
 FrontierWorld learned memory
 Oracle frontier selection

Report:

 SR
 SPL
 SoftSPL
 Distance to success
 Frontier decisions per episode
 Revisits
 Oscillations
 Collisions
 Model latency
 Peak memory
 Visual tokens processed
 Number of VLM/world-model calls

Protocol:

 Use held-out scenes.
 Use identical episode lists for every method.
 Use at least three seeds.
 Report confidence intervals.
 Save per-episode results, not only averages.
 Save videos of representative successes and failures.

Gate: FrontierWorld should improve SPL or reduce frontier-selection regret without an unreasonable computation increase.

Phase 13 — Complete the essential ablations
 No memory
 Global memory only
 Frontier memory only
 Shuffled frontier memory
 No lineage
 Position-only matching
 Full lineage matching
 Oracle lineage
 No action conditioning
 No uncertainty
 No information-gain term
 No risk term
 Different memory budgets
 Different rollout horizons
 Different probe distances

Do not run the entire ablation grid immediately. First identify the best full model, then ablate one component at a time.

Phase 14 — Analyze failure cases

Create categories:

 Incorrect frontier association
 Split/merge failure
 Wrong room prediction
 Semantically plausible but geometrically wrong prediction
 Overconfident target prediction
 Traversability failure
 Correct prediction but bad planner choice
 Local-controller failure
 Goal detector or stopping failure

For every category:

 Measure frequency.
 Show at least one qualitative example.
 Identify whether the problem comes from mapping, memory, prediction or control.
 Avoid attributing all navigation failures to the world model.
Phase 15 — Prepare the paper results

Minimum figures:

 Full system overview
 Simulator-forking/branch-complete dataset diagram
 Frontier birth/split/merge lineage diagram
 Prediction examples for multiple frontiers from one state
 Memory-budget versus prediction graph
 Predicted confidence calibration plot
 Closed-loop navigation example

Minimum tables:

 Dataset statistics
 Prediction comparison
 Memory ablation
 Lineage ablation
 Frontier-ranking regret
 Closed-loop navigation
 Runtime and memory cost
 Replace every placeholder in the LaTeX file.
 Make sure every table supports a paper claim.
 Do not claim state of the art unless the comparison protocol is identical.
 Clearly label methods using privileged simulator information.
 State that simulator forking is used for training/evaluation labels, not during deployment.
Phase 16 — Write the paper sequentially

Write in this order:

 Method
 Dataset-generation protocol
 Experimental setup
 Results
 Related work
 Introduction
 Abstract
 Conclusion

Then:

 Verify every symbol is defined.
 Verify every contribution has an experiment.
 Verify every experimental claim has a table or figure.
 Add limitations.
 Add ethical and dataset considerations if required.
 Proofread captions independently of the main text.
 Compile from a clean directory.
 Check references and hyperlinks.
 Have Behrad and Yulun review the claims.
 Incorporate feedback.
 Freeze the final PDF.
 Submit before the deadline.