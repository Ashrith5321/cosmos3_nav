# Frontier-Conditioned World Modeling for OpenFrontier
## Detailed Research and Engineering Design

**Working names:** FrontierWorld, OpenFrontier-WM, PredictiveFrontier  
**Purpose:** Extend OpenFrontier from myopic frontier scoring to predictive, frontier-conditioned reasoning over possible unseen futures.  
**Primary task:** Language-conditioned ObjectNav / open-vocabulary navigation in unknown indoor environments.  
**Starting point:** OpenFrontier: *General Navigation with Visual-Language Grounded Frontiers*.

---

# 1. Executive Summary

OpenFrontier makes frontier selection lightweight and effective by detecting **visual frontiers directly in image space**, assigning each frontier a task-conditioned semantic probability with a VLM, multiplying that score by FrontierNet's predicted information gain, and finally discounting the result by distance.

The core OpenFrontier score is:

\[
g_i = p_i \hat{g}_i
\]

with

\[
p_i = \mathrm{VLM}(I_t, F_i, L)
\]

and global selection score:

\[
u_i = \frac{g_i}{\|p_r-p_i\|}
\]

where:

- \(I_t\): current RGB observation,
- \(F_i\): candidate visual frontier,
- \(L\): language goal,
- \(p_i\): VLM-estimated probability that frontier \(i\) leads toward the goal,
- \(\hat g_i\): FrontierNet exploration/information-gain estimate,
- \(p_r\): robot position,
- \(p_i\): frontier position.

This is effective, but the semantic term is fundamentally based on **what the robot can currently see around the frontier**. It does not explicitly estimate what lies beyond the frontier.

The proposed extension treats each frontier as a **counterfactual query to a world model**:

> "If the robot explores through this frontier, what is likely to be observed next?"

For every frontier, the system predicts one or more possible future semantic/geometric outcomes, evaluates their relevance to the language goal, accounts for uncertainty and navigation cost, and selects the frontier with the highest expected long-term utility.

The central replacement is:

\[
\boxed{
Q(F_i)
\approx
\mathbb{E}
[
\text{future task utility}
\mid
\text{explore frontier }F_i
]
}
\]

rather than:

\[
Q(F_i)
\approx
\text{semantic relevance visible from the current frame}
\]

The proposed system keeps the strengths of OpenFrontier:

- sparse frontier representation,
- image-space visual reasoning,
- modular high-level / low-level navigation,
- zero-shot language goals,
- compatibility with different VLMs,
- low global-memory cost,

while adding:

- frontier-conditioned world prediction,
- multiple future hypotheses,
- persistent frontier memory,
- goal-language matching against predicted futures,
- world-model uncertainty,
- semantic/geometric/affordance prediction,
- prediction-vs-observation residuals,
- online calibration,
- cost-aware predictive planning,
- selective expensive inference for efficiency.

---

# 2. OpenFrontier Baseline

## 2.1 Visual frontier detection

OpenFrontier applies FrontierNet to a keyframe RGB observation and produces frontier clusters:

\[
F_i = \{X_i,\hat g_i\}
\]

where:

- \(X_i=\{p_i,q_i\}\) is the frontier 3D pose,
- \(\hat g_i\) is predicted exploration information gain.

FrontierNet detects visual frontier regions and predicts the volume of unknown space expected to be revealed by visiting the frontier.

The frontier initially lives in the **image plane**, then is back-projected to 3D.

---

## 2.2 VLM semantic grounding

OpenFrontier overlays Set-of-Marks labels on frontier centroids in the RGB image.

A single VLM forward pass is asked to estimate:

\[
p_i \in [0,1]
\]

for every visible frontier.

Conceptually:

```text
RGB observation
      |
      v
FrontierNet
      |
      v
F1  F2  F3  F4
      |
      v
Set-of-Marks image
      |
      v
VLM + language goal
      |
      v
p1  p2  p3  p4
```

The VLM prompt asks whether each frontier is likely to lead to the requested target during continued exploration.

---

## 2.3 OpenFrontier utility

Semantic and exploration priors are multiplied:

\[
g_i=p_i\hat g_i
\]

Then the global manager uses:

\[
u_i=\frac{g_i}{d_i}
\]

where OpenFrontier uses Euclidean robot-to-frontier distance:

\[
d_i=\|p_r-p_i\|
\]

The frontier with the maximum utility is selected.

---

## 2.4 Target verification and termination

While navigating, OpenFrontier uses an open-vocabulary segmentation model to detect candidate target objects.

When a target mask appears:

1. mask + depth estimate target 3D centroid,
2. a high-utility "viewpoint frontier" is created,
3. the robot approaches that viewpoint,
4. the VLM verifies whether the target is truly visible/reachable,
5. if confirmed, the target centroid becomes the final PointGoal.

In the OpenFrontier simulation setup, the VLM is therefore used for two different jobs:

1. **frontier relevance scoring**, and
2. **target presence verification**.

---

# 3. Main Limitation to Address

The important weakness is not simply that frontier selection uses:

\[
\arg\max_i u_i
\]

Deterministic argmax is reasonable if \(u_i\) is a good estimate of long-term return.

The deeper limitation is that:

\[
p_i=\mathrm{VLM}(I_t,F_i,L)
\]

is predominantly a **current-observation prior**.

For example:

```text
Current image
--------------------------------
| hallway     | doorway        |
|             |                |
|   F1        |       F2       |
--------------------------------

Goal: "microwave"
```

The VLM might infer:

- F1: hallway-like, probability 0.30
- F2: kitchen-like doorway, probability 0.65

but it has no explicit model of:

```text
F1 -> corridor -> kitchen -> microwave
```

versus:

```text
F2 -> dining room -> balcony
```

The proposed method introduces that missing predictive layer.

---

# 4. Core Proposal: Frontier-Conditioned Counterfactual World Modeling

Treat each active frontier \(F_i\) as an action-like hypothesis:

\[
F_i \equiv
\text{"explore through this region"}
\]

The world model predicts:

\[
p_\theta(Z_{t+1:t+H}\mid M_t,F_i)
\]

where:

- \(M_t\): current persistent memory/state,
- \(F_i\): candidate frontier,
- \(Z_{t+1:t+H}\): future latent, semantic, geometric, textual, or visual observations,
- \(H\): prediction horizon.

The decision system evaluates:

\[
Q_i
=
\mathbb{E}_{Z\sim p_\theta}
[
R(Z,L)
]
\]

where \(R\) measures how useful the predicted future is for satisfying language goal \(L\).

---

# 5. High-Level Architecture

```text
                         LANGUAGE GOAL
                              |
                              v
                        Text Encoder / VLM
                              |
                              v
                             z_L
                              |
                              |
RGB_t -----> FrontierNet -----+-----------------------------+
  |              |                                          |
  |              v                                          |
  |        visual frontiers                                 |
  |        F1 F2 ... FN                                     |
  |              |                                          |
  |              v                                          |
  |       OpenFrontier VLM                                  |
  |              |                                          |
  |       current scores p_i                                |
  |              |                                          |
  |              v                                          |
  |      cheap candidate filter                             |
  |              |                                          |
  |         Top-M frontiers                                 |
  |              |                                          |
  |      +-------+--------+                                  |
  |      |       |        |                                  |
  |      v       v        v                                  |
  |    WM(F1)  WM(F2)   WM(F3)                              |
  |      |       |        |                                  |
  |      v       v        v                                  |
  |   predicted future distributions                         |
  |      |       |        |                                  |
  |      +-------+--------+                                  |
  |              |                                          |
  |              v                                          |
  |   semantic / geometric / affordance heads                |
  |              |                                          |
  |              v                                          |
  |      goal matching against z_L <-------------------------+
  |              |
  |              v
  |    mean relevance + uncertainty
  |              |
  +---------> Frontier Memory
                 |
                 v
      expected long-term utility Q_i
                 |
                 v
              argmax
                 |
                 v
             Planner
                 |
                 v
              Robot
                 |
                 v
        Actual observation O_t+1
                 |
                 v
        predicted vs observed
                 |
                 v
        residual + calibration
                 |
                 +------> Frontier Memory / World Model State
```

---

# 6. What Should the World Model Predict?

There are several design choices.

## 6.1 Option A — Pixel / Video Future Prediction

For each frontier:

\[
\tilde I^{1:H}_{i,k}
=
WM(M_t,F_i,\epsilon_k)
\]

where \(k\) indexes sampled future hypotheses.

Example:

```text
Frontier F2
      |
      v
World Model
      |
      +--> future sample 1: kitchen
      |
      +--> future sample 2: dining room
      |
      +--> future sample 3: kitchen corridor
      |
      +--> future sample 4: living room
```

Then use a vision-language encoder or VLM to score the generated futures.

Advantages:

- visually interpretable,
- easy qualitative figures,
- can reuse pretrained video/image world models,
- makes hallucination inspectable.

Disadvantages:

- expensive,
- unnecessary visual details consume compute,
- generated RGB quality may not correlate with navigation usefulness,
- difficult to run for many frontiers at high frequency.

---

## 6.2 Option B — Semantic Future Prediction

Predict structured semantic distributions directly:

```text
Frontier #17

room:
    kitchen          0.55
    dining room      0.25
    hallway          0.15
    other            0.05

objects:
    refrigerator     0.63
    stove            0.57
    microwave        0.44
    counter          0.71

topology:
    opens_into_room  0.81
    corridor         0.18

traversability:
    traversable      0.94
```

This can be represented as:

\[
P(R\mid F_i,M_t)
\]

\[
P(O_c\mid F_i,M_t)
\]

\[
P(A\mid F_i,M_t)
\]

for:

- room category \(R\),
- object categories \(O_c\),
- affordances \(A\).

Advantages:

- substantially cheaper,
- easy to supervise,
- interpretable,
- directly task relevant.

---

## 6.3 Option C — Latent Future Prediction

Predict a shared semantic latent:

\[
z_i^{future}=WM(z_t,F_i)
\]

Then compare it with the language embedding:

\[
s_i=
\cos(z_i^{future},z_L)
\]

This removes the expensive decoding stage.

Pipeline:

```text
memory latent
     +
frontier latent
     |
     v
world model
     |
     v
future semantic latent z_i
     |
     v
cosine(z_i, z_goal)
```

Advantages:

- fastest,
- compact,
- can evaluate many frontiers,
- easy to cache.

Disadvantages:

- less interpretable,
- representation may collapse,
- harder to diagnose failures.

---

## 6.4 Recommended Design — Multi-Head Shared Latent

Use one shared frontier-conditioned latent:

\[
z_i = f_\theta(M_t,F_i)
\]

with multiple heads:

\[
h_{\text{room}}(z_i)
\]

\[
h_{\text{objects}}(z_i)
\]

\[
h_{\text{geometry}}(z_i)
\]

\[
h_{\text{affordance}}(z_i)
\]

\[
h_{\text{language}}(z_i)
\]

\[
h_{\text{utility}}(z_i)
\]

The architecture becomes:

```text
                           +--> room distribution
                           |
                           +--> object distribution
                           |
Memory + Frontier --> z_i -+--> occupancy / geometry
                           |
                           +--> affordances
                           |
                           +--> language-event embedding
                           |
                           +--> future task utility
```

This gives both:

- direct task scoring,
- interpretable auxiliary predictions.

---

# 7. Goal Conditioning: Where Should Language Enter?

There are two alternatives.

## 7.1 Goal-conditioned generator

\[
Z_i=WM(M_t,F_i,L)
\]

This sounds natural, but has a serious risk:

> The model may hallucinate futures that satisfy the requested goal.

For example:

```text
Goal = "microwave"
```

could bias a generative model to hallucinate:

```text
doorway -> kitchen -> microwave
```

even when the underlying visual evidence is weak.

This contaminates **prediction** with **desire**.

---

## 7.2 Recommended: Goal-Agnostic Prediction + Goal-Conditioned Evaluation

World model:

\[
Z_i=WM(M_t,F_i)
\]

Evaluator:

\[
S_i=R(Z_i,L)
\]

This cleanly separates:

### Prediction
"What is likely beyond this frontier?"

from:

### Evaluation
"How useful is that future for my current language goal?"

Recommended architecture:

```text
         CURRENT WORLD STATE
                 |
                 v
      goal-agnostic world model
                 |
                 v
     predicted future semantics
                 |
                 +------ language goal
                         |
                         v
                  goal evaluator
                         |
                         v
                       score
```

This also allows predicted frontier futures to be reused across different goals.

---

# 8. Language Matching

## 8.1 Cosine Similarity Baseline

Encode the target:

\[
z_L=E_{text}(L)
\]

Encode predicted future:

\[
z_i=E_{future}(Z_i)
\]

Calculate:

\[
s_i=
\frac{z_i^Tz_L}
{\|z_i\|\|z_L\|}
\]

Example:

```text
Goal: "microwave"

Frontier A predicted future:
"bedroom with bed and wardrobe"
cosine = 0.12

Frontier B predicted future:
"kitchen with counter, stove and refrigerator"
cosine = 0.78

Frontier C predicted future:
"bathroom with sink and shower"
cosine = 0.09
```

This is a strong, simple baseline.

---

## 8.2 Limitation of Pure Cosine Matching

Cosine similarity may struggle with relational goals:

- "chair next to the monitor"
- "plant in the bathroom"
- "mug on the desk"
- "fire extinguisher near the exit"
- "the couch beside the window"

For these, add a richer VLM evaluator:

\[
s_i^{VLM}
=
VLM(Z_i,L)
\]

The VLM should estimate:

\[
P(\text{exploring }F_i\text{ leads to satisfying }L)
\]

---

## 8.3 Hybrid Matching

Recommended:

\[
S_i=
\beta
S_i^{embed}
+
(1-\beta)
S_i^{VLM}
\]

Cheap embedding score for all retained frontiers.

Run the expensive VLM only when:

- top candidates are close,
- goal contains relational language,
- uncertainty is high,
- world-model samples disagree.

---

# 9. Multi-Hypothesis Prediction

A single world-model output is dangerous.

Instead sample:

\[
Z_{i,1},Z_{i,2},...,Z_{i,K}
\sim
p_\theta(Z\mid M_t,F_i)
\]

For each:

\[
s_{i,k}=R(Z_{i,k},L)
\]

Expected semantic utility:

\[
\mu_i=
\frac{1}{K}
\sum_{k=1}^{K}
s_{i,k}
\]

Uncertainty:

\[
\sigma_i^2=
\frac{1}{K}
\sum_{k=1}^{K}
(s_{i,k}-\mu_i)^2
\]

Example:

```text
FRONTIER A
sample 1 -> kitchen        score .85
sample 2 -> kitchen        score .81
sample 3 -> dining room    score .60
sample 4 -> kitchen        score .83

mean relevance:  .772
uncertainty:      low
```

versus:

```text
FRONTIER B
sample 1 -> kitchen        score .91
sample 2 -> bathroom       score .05
sample 3 -> hallway        score .12
sample 4 -> bedroom        score .09

mean relevance:  .293
uncertainty:      high
```

A deterministic rollout incorrectly makes B look excellent if sample 1 is chosen.

The distribution makes A much safer.

---

# 10. Persistent Frontier Memory

Every frontier should have a persistent state.

Example structure:

```python
FrontierRecord:
    frontier_id
    pose_3d
    orientation
    source_keyframe_id

    # OpenFrontier
    frontiernet_information_gain
    current_vlm_probability

    # prediction
    future_latent
    predicted_room_distribution
    predicted_object_distribution
    predicted_affordances
    predicted_geometry
    predicted_language_events

    # uncertainty
    prediction_variance
    confidence

    # navigation
    geodesic_cost
    visited_probability
    traversability
    status

    # calibration
    last_prediction_step
    last_observation_step
    residual_history
```

Possible statuses:

```text
ACTIVE
SELECTED
VISITED
CONSUMED
FAILED
MERGED
ARCHIVED
```

---

# 11. Frontier Identity / Re-Identification

World-model computation becomes much more efficient if the same physical frontier is recognized across observations.

For two observations:

\[
F_i^t
\]

and

\[
F_j^{t+\Delta}
\]

we need to decide whether they represent the same portal/region.

Possible merge cues:

- 3D distance,
- normal/orientation similarity,
- image embedding similarity,
- predicted topology,
- visual frontier mask overlap after projection.

Example:

\[
D_{ij}
=
\lambda_p D_{pos}
+
\lambda_q D_{orientation}
+
\lambda_e D_{embedding}
\]

Merge if:

\[
D_{ij}<\tau_{merge}
\]

Persistent IDs enable prediction caching.

---

# 12. Prediction Cache

Do **not** regenerate a future every time a frontier appears.

Cache:

```text
frontier_id = 17

prediction:
    semantic_latent
    room_dist
    object_dist
    affordance_dist
    uncertainty
    generation_context_hash
    timestamp
```

Only update if:

\[
\Delta M_i > \tau_{update}
\]

where \(\Delta M_i\) measures how much new relevant evidence has accumulated around that frontier.

Possible triggers:

- frontier moved substantially,
- viewing angle changed significantly,
- new nearby semantic evidence,
- world model confidence dropped,
- frontier was partially explored,
- new keyframe entered its local context.

---

# 13. Local Frontier Cache + Global Memory

Use two memory levels.

## 13.1 Local Frontier Cache

Each frontier stores context specifically relevant to what might lie beyond it:

```text
Frontier 17
    nearby RGB crop/keyframe
    frontier mask
    3D pose
    nearby objects
    room context
    exploration history
    predicted future
    uncertainty
```

---

## 13.2 Global Memory

Store scene-level state:

- explored free space,
- occupied space,
- global topology,
- discovered rooms,
- discovered objects,
- visited frontier IDs,
- failed paths,
- semantic episode history,
- room transitions,
- target hypotheses.

---

## 13.3 Retrieval

When reasoning about frontier \(F_i\), retrieve only relevant context:

\[
C_i =
Retrieve(F_i,M_{global},M_{local})
\]

This avoids passing an entire episode history to the VLM/world model.

---

# 14. Improved Frontier Utility

A stronger frontier score is:

\[
\boxed{
Q_i=
\lambda_{obs}P_i^{obs}
+
\lambda_{wm}\mu_i^{WM}
+
\lambda_{IG}IG_i
+
\lambda_{nov}N_i
-
\lambda_{cost}C_i
-
\lambda_{risk}R_i
-
\lambda_{unc}U_i
-
\lambda_{visit}V_i
}
\]

where:

### Current semantic prior

\[
P_i^{obs}
=
VLM(I_t,F_i,L)
\]

This preserves OpenFrontier's strong current-frame evidence.

### World-model semantic value

\[
\mu_i^{WM}
=
\mathbb{E}_{Z_i}
[R(Z_i,L)]
\]

### Information gain

\[
IG_i=\hat g_i
\]

from FrontierNet or a learned head.

### Novelty

\[
N_i
=
P(\text{frontier reveals new region})
\]

### Navigation cost

\[
C_i
=
\text{estimated path/geodesic cost}
\]

### Risk

\[
R_i
=
P(\text{frontier/path is unsafe or unreachable})
\]

### World-model uncertainty

\[
U_i=\sigma_i
\]

### Revisit penalty

\[
V_i
=
P(\text{predicted region has already been explored})
\]

---

# 15. Replace Euclidean Distance with Path Cost

OpenFrontier uses approximately:

\[
\frac{g_i}{\|p_r-p_i\|}
\]

Euclidean distance can be misleading.

Example:

```text
Frontier A:
Euclidean distance = 2m
actual path around obstacle = 9m

Frontier B:
Euclidean distance = 4m
actual free-space path = 4.5m
```

A better cost is:

\[
C_i=
d_{\text{geo}}(p_r,p_i)
\]

or planner cost:

\[
C_i=
J_{\text{planner}}(p_r,p_i)
\]

Possible normalized score:

\[
Q_i'=
\frac{Q_i}
{(C_i+\epsilon)^\alpha}
\]

or simply use cost as an additive penalty.

---

# 16. Decision Policy

The final policy can remain deterministic:

\[
F^*=\arg\max_i Q_i
\]

This is not a weakness if \(Q_i\) captures long-term value.

If explicit exploration is desired, alternatives include:

## Boltzmann

\[
P(F_i)=
\frac{\exp(Q_i/\tau)}
{\sum_j\exp(Q_j/\tau)}
\]

## Upper Confidence Bound

If uncertainty indicates unknown upside:

\[
Q_i^{UCB}
=
\mu_i+\beta\sigma_i
\]

## Risk-averse

If uncertainty is undesirable:

\[
Q_i^{risk}
=
\mu_i-\beta\sigma_i
\]

For ObjectNav, a **risk-aware deterministic argmax** is likely the cleanest starting point.

---

# 17. Closed-Loop Prediction Calibration

Prediction should not be write-only.

Before crossing a frontier:

```text
Predicted:
    room = kitchen     0.72
    microwave =        0.38
    refrigerator =     0.65
```

After entering:

```text
Observed:
    room = laundry room
    microwave = absent
    washer = present
```

Compute residual:

\[
e_i=
z_i^{obs}-z_i^{pred}
\]

Store:

\[
\mathcal E_i=
\{e_i^1,e_i^2,...\}
\]

Use residuals for:

- uncertainty calibration,
- room-transition priors,
- frontier-type reliability,
- within-episode adaptation.

---

# 18. Online Calibration

A lightweight episodic calibration module can estimate:

\[
\hat s_i
=
Calibrate(s_i,c_i,\theta_{episode})
\]

where \(c_i\) contains:

- world-model variance,
- frontier type,
- room type,
- prediction horizon,
- recent residual statistics.

Example:

```text
Model repeatedly predicts kitchens behind wide doorways,
but three predictions were wrong.

Episode calibrator:
kitchen probability × 0.72 for similar frontier type.
```

This does **not** require full online model training.

A tiny temperature/bias/residual head is sufficient.

---

# 19. World-Model Prediction Targets

A particularly useful multi-task target set is:

## 19.1 Geometry

Predict:

- free-space occupancy,
- occupied-space occupancy,
- local topology,
- opening width,
- depth continuation.

## 19.2 Room semantics

Predict:

\[
P(room\_type\mid F_i)
\]

Examples:

- kitchen,
- bathroom,
- bedroom,
- living room,
- hallway,
- office.

## 19.3 Object semantics

Predict:

\[
P(object_c\mid F_i)
\]

for either:

- benchmark categories,
- open-vocabulary CLIP embeddings,
- region-level language features.

## 19.4 Affordances

Predict:

- traversable,
- enterable,
- dead end,
- corridor,
- doorway,
- stairs,
- open room,
- cluttered,
- high collision risk.

## 19.5 Language events

Predict textual propositions such as:

```text
"likely opens into a kitchen"
"likely contains seating furniture"
"probably continues into a hallway"
"likely dead end"
"likely contains food-preparation appliances"
```

These can be embedded and compared directly to arbitrary language goals.

## 19.6 Task utility

Learn a direct head:

\[
\hat V(F_i,L)
\]

as an auxiliary prediction, while preserving the interpretable intermediate heads.

---

# 20. Efficient Inference Strategy

Running an expensive world model for every frontier at every step is unnecessary.

Use a **coarse-to-fine cascade**.

```text
N raw frontiers
      |
      v
FrontierNet filtering
      |
      v
cheap OpenFrontier VLM scoring
      |
      v
Top M frontiers
      |
      v
cached prediction available?
  /              \
yes              no
 |                |
reuse          world model
  \              /
       |
       v
cheap embedding scoring
       |
       v
ambiguous?
  /        \
no          yes
 |           |
score      rich VLM
  \         /
      |
      v
final Q_i
```

Recommended first-pass values:

```text
N active frontiers:      ~5-20
M world-model frontiers:  2-4
K hypotheses/frontier:    2-4 initially
```

---

# 21. Adaptive World-Model Invocation

Run the world model only if:

### Condition 1 — New frontier
No prediction exists.

### Condition 2 — Prediction stale
The local context has changed substantially.

### Condition 3 — Top scores ambiguous

\[
|Q_1-Q_2|<\tau_{amb}
\]

### Condition 4 — High uncertainty

\[
\sigma_i>\tau_\sigma
\]

### Condition 5 — Relational goal
Language contains spatial/contextual relations.

### Condition 6 — Replanning after failure
A selected frontier did not produce expected progress.

This dramatically reduces token/GPU cost.

---

# 22. Possible World-Model Backends

The architecture should not depend on one specific model.

## Pixel/video generator

Possible role:

```text
frontier crop + history + direction
            |
            v
video/image world model
            |
            v
future visual rollout
```

Useful for a high-quality demonstration and qualitative figures.

## Learned semantic predictor

A smaller custom model can consume:

- image features,
- frontier geometry,
- history features,
- map features,

and produce the shared future latent.

## Frozen VLM + trained prediction head

Example:

```text
frozen VLM / vision encoder
          |
       features
          |
frontier-conditioned transformer
          |
      future latent
          |
      multi-head outputs
```

This is likely much cheaper to train.

---

# 23. Training Data Generation

HM3D training trajectories can provide self-supervised supervision.

At timestep \(t\):

1. identify candidate frontier \(F_i\),
2. determine the region beyond it,
3. use simulator oracle geometry to inspect what becomes visible after crossing,
4. generate future labels.

Possible labels:

```text
future occupancy
future room category
future object categories
future CLIP embedding
future RGB keyframe
actual information gain
goal success likelihood
path cost
```

---

# 24. Counterfactual Frontier Training Examples

For every state:

```text
state M_t
frontiers:
    F1
    F2
    F3
```

generate training records:

```text
(M_t, F1) -> future outcome 1
(M_t, F2) -> future outcome 2
(M_t, F3) -> future outcome 3
```

This is important because the model learns:

\[
p(Z_{future}\mid M_t,F_i)
\]

rather than merely predicting the globally completed map.

---

# 25. Frontier-Conditioned Data Target Definition

A clean definition is:

> The "beyond-frontier region" is the newly observable connected region revealed after the robot moves from the current explored component through frontier \(F_i\).

Then supervise:

\[
Z_i^{GT}
=
Encode(
\text{beyond-region}(F_i)
)
\]

Possible spatial restriction:

- first 2 m beyond frontier,
- first connected room,
- first \(H\) navigation steps,
- first \(N\) newly observed voxels.

This target needs a precise definition for a paper.

---

# 26. Training Objectives

For shared latent \(z_i\):

\[
z_i=f_\theta(M_t,F_i)
\]

use multi-task loss:

\[
\mathcal L
=
\lambda_g\mathcal L_{geometry}
+
\lambda_r\mathcal L_{room}
+
\lambda_o\mathcal L_{objects}
+
\lambda_a\mathcal L_{affordance}
+
\lambda_e\mathcal L_{embedding}
+
\lambda_v\mathcal L_{utility}
\]

---

## 26.1 Semantic embedding loss

If target future embedding is \(z_i^{GT}\):

\[
\mathcal L_{embed}
=
1-
\cos(z_i,z_i^{GT})
\]

---

## 26.2 Room classification

\[
\mathcal L_{room}
=
CE(\hat y_{room},y_{room})
\]

---

## 26.3 Multi-label objects

\[
\mathcal L_{objects}
=
BCE(\hat y_{objects},y_{objects})
\]

---

## 26.4 Geometry

Use occupancy BCE / IoU-related losses.

---

## 26.5 Utility

If oracle target success probability is available:

\[
\mathcal L_{utility}
=
BCE(\hat V_i,V_i^{oracle})
\]

or pairwise ranking:

\[
\mathcal L_{rank}
=
-\log
\sigma(
Q_{better}-Q_{worse}
)
\]

---

# 27. Prediction Horizon

There are several levels.

## H=1: first region beyond frontier

Cheap and stable.

## H=room: first connected room

More semantically meaningful.

## Multi-step

\[
F_i
\rightarrow
Z_i^1
\rightarrow
Z_i^2
\rightarrow
...
\rightarrow
Z_i^H
\]

Then:

\[
S_i
=
\max_h
\gamma^{h-1}
R(Z_i^h,L)
\]

or:

\[
S_i
=
\sum_h
\gamma^{h-1}
R(Z_i^h,L)
\]

Start with **one frontier crossing / one room prediction** before attempting long autoregressive rollouts.

---

# 28. OpenFrontier + World Model Combined Algorithm

```text
INPUT:
    current observation I_t
    camera pose T_t
    language goal L
    persistent frontier memory M

1. Detect visual frontier proposals
       F_new <- FrontierNet(I_t)

2. Back-project frontiers to 3D

3. Merge with persistent frontier memory
       F <- MERGE(F, F_new)

4. Produce Set-of-Marks image

5. Obtain current visual semantic probabilities
       p_obs[i] <- VLM(I_t, F, L)

6. Compute cheap preliminary score
       q_pre[i] =
           lambda_obs * p_obs[i]
         + lambda_IG  * IG[i]
         - lambda_cost * cost[i]

7. Select top-M candidate frontiers

8. For each selected frontier:
       if cached world prediction is valid:
           retrieve prediction
       else:
           Z[i,1:K] <- WorldModel(M, F_i)

9. Score predictions against language
       s[i,k] <- GoalScore(Z[i,k], L)

10. Compute:
       mu[i]    <- mean_k s[i,k]
       sigma[i] <- std_k  s[i,k]

11. Compute final score
       Q[i] =
           lambda_obs  * p_obs[i]
         + lambda_wm   * mu[i]
         + lambda_IG   * IG[i]
         + lambda_nov  * novelty[i]
         - lambda_cost * path_cost[i]
         - lambda_unc  * sigma[i]
         - lambda_visit* revisit[i]
         - lambda_risk * risk[i]

12. Select:
       F* <- argmax_i Q[i]

13. Navigate toward F*

14. Observe new state

15. If frontier region becomes visible:
       compare predicted vs observed
       update residual statistics
       update frontier memory
       calibrate confidence

16. If target detector proposes object:
       create viewpoint frontier
       approach viewpoint
       VLM verifies target

17. Repeat until success or step budget exhausted
```

---

# 29. More Formal Pseudocode

```python
def predictive_frontier_step(obs, goal, memory):
    proposals = frontier_net(obs.rgb)

    proposals_3d = project_frontiers(
        proposals,
        depth=obs.depth,
        pose=obs.pose,
    )

    memory.merge_frontiers(proposals_3d)

    active = memory.active_frontiers()

    marked_image = render_set_of_marks(obs.rgb, active)

    p_obs = frontier_vlm(
        marked_image,
        goal,
    )

    for frontier in active:
        frontier.p_obs = p_obs[frontier.id]
        frontier.path_cost = estimate_path_cost(frontier.pose)
        frontier.pre_score = cheap_score(frontier)

    candidates = top_m(active, key="pre_score")

    for frontier in candidates:
        if memory.prediction_is_stale(frontier):
            samples = world_model.sample(
                context=memory.retrieve_context(frontier),
                frontier=frontier,
                num_samples=K,
            )

            memory.store_prediction(
                frontier,
                samples,
            )
        else:
            samples = memory.get_prediction(frontier)

        scores = [
            goal_relevance(sample, goal)
            for sample in samples
        ]

        frontier.wm_mean = mean(scores)
        frontier.wm_uncertainty = std(scores)

    for frontier in active:
        frontier.final_score = compute_Q(frontier)

    target_frontier = max(
        active,
        key=lambda f: f.final_score,
    )

    return target_frontier
```

---

# 30. Prediction Verification

After traversing frontier \(F_i\):

```python
pred = memory.get_prediction(F_i)

obs_future = encode_actual_observation(new_obs)

residual = obs_future - pred.mean_latent

memory.update_residual(
    frontier=F_i,
    residual=residual,
)

calibrator.update(
    frontier_type=F_i.type,
    residual=residual,
)
```

This turns the world model into a continuously evaluated component rather than an unchecked hallucination engine.

---

# 31. Important Failure Modes

## 31.1 Hallucinated target

The world model imagines the requested object behind a frontier.

Mitigation:

- goal-agnostic generation,
- multiple samples,
- uncertainty penalty,
- require semantic support across hypotheses.

---

## 31.2 Semantic prior dominates geometry

A frontier looks semantically attractive but is difficult to reach.

Mitigation:

\[
Q_i
-
\lambda_{cost} C_i
-
\lambda_{risk}R_i
\]

with actual planner cost.

---

## 31.3 World-model mode collapse

All frontiers generate similar "generic indoor room" predictions.

Mitigation:

- frontier pose/direction conditioning,
- local visual crops,
- contrastive frontier loss,
- pairwise discrimination objectives.

---

## 31.4 Duplicate frontier predictions

The same physical doorway appears repeatedly and is regenerated.

Mitigation:

- persistent frontier identity,
- 3D merge,
- prediction cache.

---

## 31.5 Overconfident future

Model predicts one future with high confidence despite ambiguity.

Mitigation:

- ensemble,
- stochastic latent sampling,
- calibrated uncertainty,
- prediction residual feedback.

---

## 31.6 Long rollout drift

Multi-step RGB rollouts become unrealistic.

Mitigation:

- short-horizon semantic predictions,
- hierarchical room/topology prediction,
- receding-horizon replanning.

---

# 32. Receding-Horizon Operation

Do not try to imagine the full path to the final target.

Instead:

```text
predict one frontier ahead
        |
        v
navigate
        |
        v
observe
        |
        v
update memory
        |
        v
predict again
```

This is essentially model-predictive control at the semantic navigation level.

At each decision point:

\[
F_t^*=
\arg\max_i Q(F_i\mid M_t,L)
\]

After moving:

\[
M_t\rightarrow M_{t+1}
\]

and solve again.

---

# 33. Novelty Warning: ForesightNav

A simple formulation:

```text
predict unseen semantics
        |
        v
CLIP embedding
        |
        v
cosine with target text
        |
        v
navigation goal
```

is close to existing predictive semantic navigation work such as ForesightNav.

Therefore the contribution should **not** be framed simply as:

> "We imagine unseen regions and use CLIP cosine similarity."

That is likely insufficient.

---

# 34. Stronger Novel Contribution

The stronger research framing is:

## Frontier-Conditioned Counterfactual World Modeling

Instead of completing an entire global map, prediction is explicitly conditioned on a **persistent frontier hypothesis**:

\[
p(Z_{future}\mid M_t,F_i)
\]

Each frontier becomes an actionable "what-if" query.

Potential contributions:

1. **Persistent frontier-conditioned world model**
2. **Multi-hypothesis future prediction**
3. **Joint semantic + geometric + affordance prediction**
4. **Language-event prediction beyond frontiers**
5. **Uncertainty-aware frontier ranking**
6. **Prediction-vs-observation residual memory**
7. **Online episodic calibration**
8. **Selective compute / prediction caching**
9. **Counterfactual evaluation over all candidate frontiers**
10. **Token/GPU accounting for predictive navigation**

---

# 35. Potential Paper Claim

A possible high-level claim:

> Existing frontier-based navigation methods rank exploration targets primarily from current observations or globally completed semantic maps. We instead treat each persistent frontier as a counterfactual action and explicitly predict a distribution over the semantic, geometric, and task-relevant outcomes that would become observable after exploration. These predicted outcomes are compared against language goals and continuously recalibrated against subsequent observations.

That is much stronger than:

> "We add a world model to OpenFrontier."

---

# 36. Ablation Plan

A strong experiment ladder:

## A0 — OpenFrontier baseline

\[
Q_i=P_i^{obs}\hat g_i/d_i
\]

## A1 — path-cost correction

Replace Euclidean distance with geodesic/planner cost.

## A2 — single future semantic latent

\[
Q_i=A1+\lambda S_i^{future}
\]

## A3 — multi-hypothesis world model

Add:

\[
\mu_i,\sigma_i
\]

## A4 — persistent frontier prediction cache

Measure:

- SR,
- SPL,
- world-model calls,
- GPU time.

## A5 — multi-head prediction

Add geometry / room / objects / affordances.

## A6 — prediction residual memory

Add actual-vs-predicted feedback.

## A7 — episodic calibration

Update confidence online.

## A8 — rich VLM scoring

Compare embedding-only versus VLM evaluator.

---

# 37. Core Evaluation Metrics

Navigation:

- Success Rate (SR)
- SPL
- SoftSPL if appropriate
- average episode length
- distance traveled

Prediction:

- room accuracy / F1
- object AUROC / AUPRC
- semantic embedding cosine
- occupancy IoU
- traversability accuracy
- expected calibration error
- Brier score

World-model usefulness:

- frontier ranking accuracy
- top-1 oracle frontier recall
- pairwise frontier ordering accuracy
- expected regret

Efficiency:

- world-model calls / episode
- VLM calls / episode
- generated tokens / episode
- generated frames / episode
- GPU milliseconds / decision
- wall-clock latency
- memory consumption

---

# 38. Frontier Ranking Oracle

For analysis, define an oracle frontier value using simulator ground truth.

Example:

\[
Q_i^{oracle}
=
-\operatorname{dist}_{future}(F_i,target)
\]

or:

\[
Q_i^{oracle}
=
P(\text{target reachable through region }F_i)
\]

Then evaluate:

\[
RankCorr(Q_i,Q_i^{oracle})
\]

and:

\[
Top1Accuracy
=
\mathbb{1}
[
\arg\max_i Q_i
=
\arg\max_i Q_i^{oracle}
]
\]

This directly tests whether the world model improves **frontier decisions**, independent of low-level navigation failures.

---

# 39. Failure Attribution

Separate failures into:

```text
1. frontier detection failure
2. frontier merge / identity failure
3. world-model semantic prediction failure
4. geometry prediction failure
5. goal matching failure
6. uncertainty calibration failure
7. target detector false positive
8. target verifier false positive/negative
9. low-level PointNav failure
10. benchmark annotation ambiguity
11. step-budget exhaustion
```

This is important because OpenFrontier itself identifies false-positive detections, step-budget exhaustion, and low-level stuck behavior as significant failure categories.

---

# 40. Recommended Minimum Viable Prototype

Do **not** start with a huge video generator.

Build this first:

```text
RGB-D
  |
  v
FrontierNet
  |
  v
persistent frontier IDs
  |
  v
frozen vision encoder
  |
  v
small frontier-conditioned transformer
  |
  +--> predicted room category
  |
  +--> predicted target/CLIP semantic latent
  |
  +--> predicted information gain
  |
  +--> uncertainty
```

Then:

\[
Q_i=
\lambda_1 p_i^{obs}
+
\lambda_2\cos(z_i^{future},z_L)
+
\lambda_3IG_i
-
\lambda_4 C_i
-
\lambda_5U_i
\]

This is enough to establish whether predictive frontier reasoning helps.

---

# 41. Phase 2: Rich World Model

Once the latent model works, add:

```text
shared z_i
   |
   +--> room
   +--> objects
   +--> occupancy
   +--> topology
   +--> affordances
   +--> language propositions
```

Then compare:

```text
latent-only
vs.
structured semantic
vs.
generated RGB/video
```

---

# 42. Phase 3: Generative Visual Rollouts

Only after the cheaper predictor demonstrates value:

```text
current image
+ local history
+ frontier crop
+ frontier direction
      |
      v
video/image world model
      |
      v
short future rollout
      |
      v
VLM semantic evaluator
```

Use this especially for:

- qualitative paper figures,
- ambiguity analysis,
- hard relational targets.

---

# 43. Suggested System Interfaces

```python
class FrontierWorldModel:
    def predict(
        self,
        frontier,
        local_context,
        global_context,
        num_hypotheses=4,
    ):
        ...
```

```python
class GoalEvaluator:
    def score(
        self,
        prediction,
        language_goal,
    ):
        ...
```

```python
class FrontierMemory:
    def merge(self, observations):
        ...

    def retrieve(self, frontier_id):
        ...

    def update_prediction(self, frontier_id, prediction):
        ...

    def update_residual(self, frontier_id, observation):
        ...
```

```python
class FrontierSelector:
    def rank(
        self,
        frontiers,
        goal,
        robot_state,
    ):
        ...
```

This keeps the stack modular.

---

# 44. Proposed Data Flow

```text
ObservationPacket
    rgb
    depth
    pose
    timestamp
       |
       v
FrontierDetector
       |
       v
FrontierProposal[]
       |
       v
FrontierTracker
       |
       v
PersistentFrontier[]
       |
       +-------------------------+
       |                         |
       v                         v
Current VLM Prior          Frontier World Model
       |                         |
       v                         v
p_obs[i]                 FutureDistribution[i]
       |                         |
       |                         v
       |                    GoalEvaluator
       |                         |
       |                         v
       |                   p_future[i]
       |                         |
       +-----------+-------------+
                   |
                   v
            FrontierRanker
                   |
                   v
             SelectedGoal
                   |
                   v
                Planner
```

---

# 45. Important Design Principle

The world model should answer:

> "What is likely to happen if I explore here?"

The VLM / language evaluator should answer:

> "Does that predicted future help satisfy my goal?"

The planner should answer:

> "How expensive and safe is it to get there?"

The memory should answer:

> "Have I already tried or observed this region?"

Keeping these four questions separate makes the system easier to train, debug, and ablate.

---

# 46. Expected Benefits

Compared with OpenFrontier:

## Better long-horizon semantic reasoning

Can prefer:

```text
hallway -> likely kitchen
```

over:

```text
visually kitchen-like doorway -> actually irrelevant room
```

## Less repeated exploration

Persistent predictions and visit memory suppress already-explored branches.

## Better handling of partial observability

Selection depends on predicted unseen space, not only visible context.

## Better efficiency than full dense world completion

Only model a sparse set of frontiers.

## Better interpretability

Each decision can be explained as:

```text
Frontier 7 chosen because:
  current semantic score:   .51
  predicted kitchen prob:   .78
  goal similarity:          .82
  information gain:         .64
  path cost:                3.1 m
  uncertainty:              .11
```

---

# 47. Main Risks

1. World-model prediction may hallucinate.
2. Generative rollout cost may erase OpenFrontier's efficiency advantage.
3. OpenFrontier + latent world model may overlap with predictive semantic-map literature.
4. Frontier identity across time may be noisy.
5. Simulator-derived future labels may not transfer to real scenes.
6. Goal matching may exploit dataset room-object priors rather than true future prediction.
7. Auxiliary heads may improve prediction metrics without improving navigation.
8. Low-level navigation failures can obscure high-level frontier improvements.

Each risk should have a targeted ablation.

---

# 48. Most Important Experiments

If only a few experiments can be run, prioritize:

### Experiment 1 — Does future prediction improve frontier ranking?

Measure oracle ranking correlation.

### Experiment 2 — Does it improve navigation?

Compare SR/SPL against OpenFrontier.

### Experiment 3 — Does multi-hypothesis prediction beat one deterministic prediction?

Measure both ranking and calibration.

### Experiment 4 — Is the gain worth the compute?

Report latency and model calls.

### Experiment 5 — Does prediction-vs-observation feedback help?

Ablate closed-loop calibration.

---

# 49. Recommended First Final Score

A practical first implementation:

\[
\boxed{
Q_i =
0.25P_i^{obs}
+
0.40S_i^{future}
+
0.20IG_i
-
0.10\tilde C_i
-
0.05U_i
}
\]

These coefficients are **only initialization values**, not claims of optimality.

All terms should be normalized before combining.

Eventually learn or tune the weights on validation data.

---

# 50. Stronger Learned Ranker

Instead of manually combining terms forever, collect:

\[
x_i=
[
P_i^{obs},
S_i^{future},
IG_i,
C_i,
U_i,
V_i,
R_i,
...
]
\]

and train:

\[
Q_i=f_\phi(x_i)
\]

with pairwise ranking supervision.

For example:

\[
\mathcal L=
-\log
\sigma(
Q_{oracle\_better}-Q_{oracle\_worse}
)
\]

The first paper version should still include the hand-designed scorer as an interpretable baseline.

---

# 51. Short Research Pitch

> OpenFrontier reasons about visual frontiers from current image context, but does not explicitly model what lies beyond each frontier. We propose treating persistent frontiers as counterfactual actions for a world model. For every candidate frontier, the model predicts a distribution over future semantic, geometric, and affordance outcomes. These predictions are matched to arbitrary language goals and combined with information gain, navigation cost, and uncertainty to estimate expected long-term frontier utility. Predictions are cached per persistent frontier and recalibrated online by comparing predicted outcomes with observations after exploration, yielding a sparse predictive memory for long-horizon navigation.

---

# 52. One-Sentence Core Idea

\[
\boxed{
\text{Don't rank frontiers only by what is visible around them; rank them by what the robot predicts it will discover after crossing them.}
}
\]

---

# 53. Relationship to OpenFrontier

The proposed system does **not** replace OpenFrontier wholesale.

Keep:

- FrontierNet visual frontier detection,
- Set-of-Marks current-frame reasoning,
- sparse global frontier management,
- target segmentation,
- target verification,
- modular low-level planner.

Replace/augment:

```text
OpenFrontier:
current VLM probability
        +
information gain
        +
distance
        |
        v
frontier choice
```

with:

```text
current VLM probability
        +
world-model future relevance
        +
information gain
        +
planner path cost
        +
uncertainty
        +
memory / revisit penalty
        |
        v
frontier choice
```

---

# 54. Source-Grounded OpenFrontier Details Used Here

From the uploaded OpenFrontier paper:

- FrontierNet provides visual frontier proposals and predicted exploration information gain.
- A Set-of-Marks VLM query assigns a probability to each visual frontier.
- OpenFrontier combines the VLM probability and information gain multiplicatively.
- The global frontier manager discounts utility by robot-to-frontier distance.
- Frontier detection/reasoning is run periodically rather than at every simulator step.
- OpenFrontier uses SAM3-style open-vocabulary target segmentation followed by VLM target verification.
- The appendix describes persistent frontier management, merge/clear/stall thresholds, emergency rotation behavior, and simulation/real-world planning configurations.
- The paper explicitly discusses history-augmented frontier reasoning as an extension.
- Failure analysis includes false-positive targets, step-budget exhaustion, low-level navigation failures, and missed useful frontiers.

This document extends those source-supported mechanisms with a proposed predictive world-model architecture; the world-model system itself is a research design, not a claim that OpenFrontier already implements it.

---

# 55. Immediate Implementation Order

```text
[1] reproduce OpenFrontier baseline
        |
[2] expose persistent frontier records
        |
[3] add real path-cost estimate
        |
[4] define beyond-frontier GT target
        |
[5] train latent frontier future predictor
        |
[6] cosine goal scoring
        |
[7] integrate into Q_i
        |
[8] ranking-only evaluation
        |
[9] full ObjectNav evaluation
        |
[10] multi-hypothesis prediction
        |
[11] uncertainty penalty
        |
[12] persistent prediction cache
        |
[13] residual / calibration loop
        |
[14] multi-head semantics + affordances
        |
[15] optional video world-model experiment
```

---

# 56. Recommended MVP Architecture

```text
                   GOAL TEXT
                       |
                       v
                 text encoder
                       |
                      z_L
                       |
                       |
RGB-D --> FrontierNet ----> persistent frontier tracker
                              |
                              v
                      top candidate frontiers
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
          current VLM prior        future predictor
                 |                          |
               p_obs                  z_future
                 |                          |
                 |                  cosine(z_future,z_L)
                 |                          |
                 +------------+-------------+
                              |
                       information gain
                              |
                       planner path cost
                              |
                         uncertainty
                              |
                              v
                           Q_i
                              |
                              v
                           argmax
                              |
                              v
                           navigate
                              |
                              v
                   actual future observation
                              |
                              v
                    prediction residual
                              |
                              v
                     frontier memory update
```

This should be the first version built before using a large generative video world model.

---

# 57. Final Recommended Research Direction

The strongest version of this idea is not:

> "Use a world model to generate an image behind each frontier and run CLIP cosine similarity."

The stronger version is:

> **Build a sparse, persistent, frontier-conditioned predictive memory in which each frontier stores a calibrated distribution over what exploration is expected to reveal. Use language to query those predicted outcomes, use geometry to estimate cost and risk, and continuously update predictions after the robot observes what was actually beyond the frontier.**

That gives a coherent navigation architecture rather than a single additional scoring trick.

