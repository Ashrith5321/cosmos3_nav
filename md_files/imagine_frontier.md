# imagine_frontier — world-model frontier selection for Cosmos3 navigation

Design doc + project summary. Written 2026-07-17.

---

## 1. Project summary so far

**Goal:** use NVIDIA Cosmos3-Nano (16B omni world model; Qwen-lineage reasoner tower
~8.5B + diffusion generator tower) as a navigation agent on HM3D-v2 ObjectNav,
establish zero-shot baselines locally, then SFT on habitat_web data on the cluster.

### Infrastructure built (all local, single shared RTX 5090 32GB)

| Piece | Path | Notes |
|---|---|---|
| SFT pipeline (ported from spatial_training) | `sft/train_sft_cosmos3.py` + `sft/utils/` | TRL SFTTrainer + LoRA r=128, ActionMaskingVLMCollator works unchanged (same Qwen chat-template token IDs: `<\|im_start\|>assistant\n` = [151644, 77091, 198]). Smoke-tested: 4 LoRA steps, loss 6.9→3.8, 20.6 GB peak. |
| Model server | `eval/cosmos3_server.py` | HTTP action server, reasoner in bf16 (~17.6 GB), ~1 s/step at 3×384×288 frames. Now also accepts `map_image` + `frontier_text`. |
| Closed-loop eval driver | `eval/run_habitat_smoke_eval.py` | habitat033 env, deterministic episode order, skip-file resume, per-episode videos, driver.lock against concurrent runs. |
| VRAM watchdog | `eval/vram_watchdog.sh` | kills the server at 20 GB (machine cannot be restarted). Never fired across all runs. |
| Global 2D map + frontiers | `src/global_map.py` | occupancy grid from posed depth (floor/obstacle height bands), frontier extraction + clustering, rendered map image, `frontier_text()` for prompts. Verified on scene 4ok3usBNeis. |
| Live teleop with map | `src/teleop_map.py` | matplotlib TkAgg, w/a/d keys, FPV + live map side by side. |

### Zero-shot results (full 1000-ep HM3D-v2 val, 36 scenes)

| Condition | SR | SPL | soft-SPL |
|---|---|---|---|
| h3 history, 100-step cap | 5.6% | 0.036 | 0.097 |
| **h16 history, 500-step cap, 256×192** | **16.6%** | **0.067** | **0.106** |

Per-goal SR (h16): plant 28.9 · chair 21.0 · bed 19.4 · sofa 14.4 · tv 8.1 · toilet 6.6.
Greedy decoding is fully deterministic (66 accidentally double-run episodes → identical
outcomes both times).

### Failure taxonomy (h16, 834 failures)

- **~38% false detections** — voluntary stop >5 m from any goal instance (192 instant,
  ≤5 steps: declared success on spawn-visible furniture). Worst: sofa, tv_monitor.
- **~31% premature stops** — stopped 2–5 m out: right object, wrong distance.
- **~17% exploration failures** — 500-step timeout with <10% geodesic progress
  (dominated by toilet & plant: room-specific goals needing directed search).
- **~8% near-misses** — ended within 2 m but failed (stopped just outside 1 m, or stood
  at the goal and never said stop).

→ **~2/3 of failures are stop-decision errors; ~1/6 are exploration.**

Floors: **null result** — multi-floor (26 scenes) vs single-floor (10) failure rates are
identical (83.6% vs 82.8%); HM3D-v2 val contains **zero cross-floor episodes**, so a 2D
map per episode is sufficient.

### Planned fixes (in order)

1. **GT-oracle detector ablation** — habitat semantic sensor = perfect detector
   (identity + exact depth distance). Run ~100 eps with `DET:` line in the prompt to
   upper-bound detector gains *before* building anything. In sim, 100% detection
   accuracy is free — do not train a detector to approximate the sim's own labels.
2. **Trained detector → tokens** — 6-class detector fine-tuned on HM3D *train* scenes
   (labels free from semantic sensor; never train on val). Inject as text tokens after
   the frames: `DET: toilet conf=0.87 bearing=+12deg dist=3.4m`. Critical: bake the
   *real detector's* outputs (not GT) into the SFT data so the model learns its noise
   profile. Optional stop-gate: only allow `stop` when conf>τ ∧ dist<1 m.
3. **Global map + frontiers in the prompt** (server support already added).
4. **SFT on habitat_web** (cluster) with the same map/DET format as eval.

---

## 2. imagine_frontier: the idea

**Use the generator tower of the same world model to *imagine* what lies beyond each
frontier, score the imaginations for goal likelihood, and pick the frontier with the
best imagined future.** The reasoner navigates; the world model dreams ahead at
decision points.

Cosmos3's generator does image-to-video with **9D camera-motion action conditioning** —
exactly the "imagine walking through that doorway" primitive.

### Pipeline

1. **Trigger sparsely** — only at decision points (reached current frontier target, no
   goal detection, or every K≈25 steps). Imagination is seconds-per-frontier; the
   navigation loop stays fast.
2. **Condition per frontier** — pick the stored keyframe whose camera bearing best
   faces the frontier (poses are logged for every frame), plus a 9D camera-motion
   trajectory translating ~2–4 m through the frontier.
3. **Imagine short & cheap** — 16–33 frames @256p, 10–15 denoising steps. The gist,
   not a pretty video.
4. **Score the imagination** — ladder:
   - **v0**: decode frames → run detector / CLIP-similarity for the goal category.
   - **v1**: decode frames → ask the *reasoner* which imagined continuation most
     likely leads to the goal (one model family end-to-end).
   - **v2 (latent-space, the contribution)**: stop the denoiser partway and score the
     *latent* with a small trained value head — no decode. Labels are free in sim:
     GT geodesics on train scenes say whether the goal actually lies beyond a
     frontier → supervised "imagined latent → P(goal beyond frontier)".
5. **Pick** the argmax frontier → hand to frontier_management → navigate on the map.

### Practical constraints (RTX 5090, 32 GB shared)

- **VRAM**: reasoner (17.6 GB) + generator tower + VAE do not co-reside. Run the
  generator as a second on-demand server (swap ≈1 min, rare thanks to sparse
  triggering), or vLLM-Omni `--enable-layerwise-offload`.
- **Latency**: ~10–30 s per imagined frontier → prune to top-2 frontiers by a cheap
  prior; run the headline experiment on 100 eps, not 1000, until pruned.
- **Fidelity**: the model card warns of geometry/permanence artifacts. The imagination
  only needs to be *usefully biased* (a plausible bathroom beyond the hallway), not
  correct. Frame it as a learned prior, not a prediction.

### Required baselines (for the ablation to mean anything)

1. Nearest-frontier (classic frontier exploration).
2. Largest-frontier.
3. **Text-imagination**: reasoner picks a frontier from the map image + frontier_text
   alone ("which frontier most likely leads to a toilet?"). Nearly free — diffusion
   imagination must beat this to justify its cost. If v2-latent > text-imagination,
   that is a real result about world-model value beyond language priors.

### Build order

1. Text-imagination baseline (reuses the running reasoner server; ~1 day).
2. Offline v0: hand-pick ~20 frontier decision states from teleop, generate imagined
   approach videos with diffusers `Cosmos3OmniPipeline`, **eyeball whether the
   imaginations are usable at all** before designing around them.
3. Wire v0/v1 into the eval loop, 100 episodes, compare against the three baselines.
4. v2 latent value head if (and only if) the decoded version shows headroom.

---

## 3. Key artifacts

- `eval/hm3dv2_val1000_cosmos3_zeroshot.jsonl` + `_aggregate.json` — h3 baseline.
- `eval/hm3dv2_val1000_cosmos3_h16.jsonl` + `_aggregate.json` — h16 long-horizon (deduped).
- `eval/habitat_*_out/videos/` — ~1.9k rollout mp4s (gitignored; local only).
- Cluster SFT data: `habitat_web_pose_v1` at `/Projects/SG_VLN_HumanData/spatial_training/data/` (cluster only).
- NVIDIA reference SFT recipes: `cosmos/cookbooks/cosmos3/reasoner/finetune/` (8×H100, TOML;
  `convert_model_to_vlm_safetensors` extracts the reasoner tower).
