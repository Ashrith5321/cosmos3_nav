#!/usr/bin/env python
"""Phase 9 step 1: what score can the conversion machinery reach at best?

A first version of this compared ground-truth depth and poses against the
Phase 4 target and scored IoU 1.000 on every branch. That was TAUTOLOGICAL:
both paths call the same OccupancyMap.integrate on the same inputs, so they
agree by construction, and the test validated nothing a generator will face.

The ceiling that matters is the one under the constraints generation imposes:

  * FRAME COUNT. Branches run ~56 simulator steps; Cosmos generates 17 frames.
    The converter therefore sees roughly a third of the viewpoints.
  * DEPTH SCALE. Monocular depth is scale-ambiguous. A systematic scale error
    moves every projected point radially, which is exactly the error occupancy
    mapping is least tolerant of.
  * DEPTH NOISE. Per-pixel error blurs surfaces and thickens walls.

So the sweep below degrades ground truth in each of those ways and reports what
the conversion still achieves. A Cosmos score at or near the degraded ceiling
means the converter is not the bottleneck; below it, generation or depth is.

TWO LIMITS ON HOW FAR THESE NUMBERS GENERALISE.

1. The noise condition applies INDEPENDENT per-pixel error, which is close to
   the worst case for ray carving: every long ray carves its own corridor
   before something stops it, so errors accumulate rather than cancel. Real
   monocular depth error is spatially CORRELATED -- whole surfaces are off
   together -- and degrades far more gracefully. These rows therefore do NOT
   establish that an estimator must reach some particular AbsRel; they bound
   the response to a specific synthetic perturbation. Phase 9A measures the
   real estimator instead.

2. An area ratio above 1 is not a unique signature of depth noise. Positive
   scale bias produces it too (+10% scale -> 1.238 here), as can pose error or
   hallucinated geometry. Treat it as a depth/geometry warning, not as proof
   of a particular cause.

Runs a real Habitat branch through the exact Phase 9 postprocessor using
GROUND-TRUTH depth and poses, and compares the result against the Phase 4
target computed by the normal path:

    ground-truth video / depth / poses -> dM_occ -> Phase 4 comparison

This establishes the ceiling. Without it, a low Cosmos score is unattributable:
it could come from video generation, depth estimation, pose alignment or
rasterisation, and there would be no way to tell which. Whatever this run
scores is the most any generator can achieve through this pipeline.

Validation scenes only; the sealed set is not touched.

    python scripts/validate_conversion.py --episodes 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.config import check_data_paths, load_config, save_config  # noqa: E402
from frontierworld.data.tensors import (  # noqa: E402
    TGT_REVEALED_FREE,
    TGT_REVEALED_OCCUPIED,
    FrameSpec,
    FrontierFrame,
    resample_grid,
)
from frontierworld.evaluation import make_run_id  # noqa: E402
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from frontierworld.models.metrics import masked_iou  # noqa: E402
from frontierworld.models.video_to_structured import accumulate_rollout  # noqa: E402
from frontierworld.planning import make_policy  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--manifest", default="manifests/full.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--decisions", type=int, default=3)
    parser.add_argument(
        "--generator-frames", type=int, default=17,
        help="frames a generator produces; branches run far more steps",
    )
    return parser.parse_args()


def subsample(captured: dict, n_frames: int) -> dict:
    """Keep n_frames evenly spaced views, as a generator would produce.

    Evenly spaced rather than the first n: a generator covers the whole option,
    not just its beginning, and truncating instead of subsampling would confound
    "fewer viewpoints" with "shorter trajectory".
    """
    total = captured["depths"].shape[0]
    if total <= n_frames:
        return captured
    index = np.linspace(0, total - 1, n_frames).round().astype(int)
    return {
        "depths": captured["depths"][index],
        "rotations": captured["rotations"][index],
        "translations": captured["translations"][index],
        "actions": [captured["actions"][i] for i in index],
        "n_forward": captured["n_forward"],
    }


def degrade_depth(depths, scale: float = 1.0, noise: float = 0.0, seed: int = 0):
    """Apply a systematic scale error and relative per-pixel noise."""
    rng = np.random.default_rng(seed)
    out = depths * scale
    if noise > 0:
        out = out * (1.0 + rng.normal(0.0, noise, size=out.shape).astype(np.float32))
    return np.clip(out, 0.0, None).astype(np.float32)


def capture_branch(sim, option, occupancy, semantic_map, cfg):
    """Execute an option, recording depth and sensor pose at every step.

    This is what a generator would have to supply: a sequence of frames with
    known camera motion. Using the simulator's own depth and poses makes the
    conversion the only thing under test.
    """
    from frontierworld.habitat_env import sensor_extrinsics
    from frontierworld.planning.branching import sim_observations
    from frontierworld.planning.options import MOVE_FORWARD

    agent = sim.get_agent(0)
    depths, rotations, translations, actions = [], [], [], []

    for action in option.actions:
        agent.act(action)
        observations = sim_observations(sim, cfg)
        rotation, translation = sensor_extrinsics(sim, "depth")
        depths.append(np.asarray(observations["depth"], dtype=np.float32))
        rotations.append(rotation)
        translations.append(translation)
        actions.append(int(action))

    return {
        "depths": np.stack(depths) if depths else np.zeros((0, 1, 1), np.float32),
        "rotations": np.stack(rotations) if rotations else np.zeros((0, 3, 3)),
        "translations": np.stack(translations) if translations else np.zeros((0, 3)),
        "actions": actions,
        "n_forward": sum(1 for a in actions if a == MOVE_FORWARD),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), torch_deterministic=False)
    run_dir = Path(cfg.experiment.output_dir) / "phase9_conversion" / make_run_id(
        "ceiling", "v1"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    from frontierworld.data.manifests import SplitManifest
    from frontierworld.habitat_env import (
        make_env_with_dataset,
        select_episodes,
        semantic_id_to_category,
    )
    from frontierworld.mapping.occupancy import OccupancyMap
    from frontierworld.mapping.semantic import SemanticMap
    from frontierworld.planning.branching import SimulatorSnapshot, execute_option
    from frontierworld.planning.exploration import EpisodeResult, FrontierExplorer
    from frontierworld.planning.options import build_options

    manifest = SplitManifest.load(args.manifest)
    scenes = manifest.scene_ids(args.split)
    print(f"manifest {manifest.name} split={args.split}: {len(scenes)} scenes")
    print(f"run_dir: {run_dir}\n")

    dataset, _ = select_episodes(
        cfg, 200, scenes=scenes, episodes_per_scene=int(cfg.dataset.episodes_per_scene)
    )
    env = make_env_with_dataset(cfg, dataset)
    policy = make_policy("nearest")
    spec = FrameSpec()
    records = []

    try:
        for episode_index in range(args.episodes):
            explorer = FrontierExplorer(env, cfg, policy)
            observations = env.reset()
            explorer._bind_to_scene()
            episode = env.current_episode
            explorer.detector.reset(episode)
            rng = episode_rng(int(cfg.seed.value), "conversion", episode_index)

            start = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
            occupancy = OccupancyMap(
                resolution=cfg.mapping.resolution, size_m=cfg.mapping.size_m,
                obstacle_height_min=cfg.mapping.obstacle_height_min,
                obstacle_height_max=cfg.mapping.obstacle_height_max,
                column_stride=cfg.mapping.column_stride,
                min_observations=cfg.mapping.min_observations,
                center=(float(start[0]), float(start[2])), floor_y=float(start[1]),
            )
            semantic_map = SemanticMap(occupancy.geometry, floor_y=float(start[1]))
            result = EpisodeResult(str(episode.scene_id), str(episode.episode_id),
                                   getattr(episode, "object_category", None))
            explorer._last_observations = observations
            explorer._target_position = None
            explorer._integrate(observations, occupancy)
            explorer._scan(occupancy, result, int(cfg.episode.max_steps))

            for _ in range(args.decisions):
                frontiers = [f for f in explorer._frontiers(occupancy) if f.reachable]
                if len(frontiers) < 2:
                    break
                options = [
                    o for o in build_options(
                        frontiers, explorer.planner, occupancy, env.sim, cfg,
                        explorer._calibrate_turn_sign(),
                    ) if o.valid
                ][:3]
                if not options:
                    break

                grid_before = occupancy.to_grid()
                snapshot = SimulatorSnapshot.capture(env.sim, occupancy, semantic_map)
                map_origin = np.asarray(
                    [occupancy.geometry.origin_x, occupancy.geometry.origin_z]
                )

                for option in options:
                    # --- path A: the normal Phase 4 pipeline ---------------
                    branch = execute_option(env.sim, option, occupancy, cfg)
                    grid_after = occupancy.to_grid()
                    reference = (grid_before == UNKNOWN) & (grid_after != UNKNOWN)
                    snapshot.restore(env.sim, occupancy, semantic_map)

                    # --- path B: capture, then the Phase 9 postprocessor ---
                    captured = capture_branch(env.sim, option, occupancy, semantic_map, cfg)
                    snapshot.restore(env.sim, occupancy, semantic_map)
                    if captured["depths"].shape[0] == 0:
                        continue

                    frame = FrontierFrame(
                        centroid_world=np.asarray(option.frontier.centroid_world)[[0, 2]],
                        normal=np.asarray(option.frontier.orientation),
                        spec=spec,
                    )
                    conditions = {
                        "ground_truth": (captured["depths"].shape[0], 1.0, 0.0),
                        f"frames_{args.generator_frames}": (args.generator_frames, 1.0, 0.0),
                        "scale_+10%": (args.generator_frames, 1.10, 0.0),
                        "scale_-10%": (args.generator_frames, 0.90, 0.0),
                        "noise_5%": (args.generator_frames, 1.0, 0.05),
                        "noise_15%": (args.generator_frames, 1.0, 0.15),
                    }
                    # --- the Phase 4 target, in the frontier frame ---------
                    local_after, in_map = resample_grid(
                        grid_after, map_origin, float(cfg.mapping.resolution), frame,
                        fill=UNKNOWN,
                    )
                    local_reference, _ = resample_grid(
                        reference.astype(np.uint8), map_origin,
                        float(cfg.mapping.resolution), frame, fill=0,
                    )
                    revealed_reference = local_reference.astype(bool) & in_map
                    target = np.stack([
                        revealed_reference & (local_after == FREE),
                        revealed_reference & (local_after == OCCUPIED),
                    ]).astype(np.float32)
                    reference_area = float(
                        reference.sum() * cfg.mapping.resolution ** 2
                    )

                    # --- sweep the degradations a generator imposes --------
                    for name, (n_frames, scale, noise) in conditions.items():
                        view = subsample(captured, n_frames)
                        rollout = accumulate_rollout(
                            depths=degrade_depth(
                                view["depths"], scale, noise,
                                seed=option.frontier_id,
                            ),
                            rotations=view["rotations"],
                            translations=view["translations"],
                            frame=frame,
                            grid_before=grid_before,
                            map_origin=map_origin,
                            map_resolution=float(cfg.mapping.resolution),
                            hfov_deg=float(cfg.simulator.hfov),
                            max_depth=float(cfg.simulator.max_depth),
                            floor_y=float(occupancy.floor_y),
                        )
                        predicted = rollout.as_target_tensor()
                        window = in_map & rollout.valid
                        records.append({
                            "condition": name,
                            "scene": Path(result.scene_id).stem.replace(".basis", ""),
                            "episode_id": result.episode_id,
                            "frontier_id": option.frontier_id,
                            "n_frames_used": int(view["depths"].shape[0]),
                            "n_frames_available": int(captured["depths"].shape[0]),
                            "free_iou": masked_iou(predicted[0], target[0], window),
                            "occupied_iou": masked_iou(predicted[1], target[1], window),
                            "area_reference_m2": reference_area,
                            "area_converted_m2": rollout.revealed_area_m2,
                            "area_ratio": (
                                rollout.revealed_area_m2 / reference_area
                                if reference_area > 1e-6 else float("nan")
                            ),
                        })

                chosen = policy.select(frontiers, rng)
                actions = explorer.planner.try_plan(
                    chosen.approach_point(float(cfg.frontiers.approach_offset_m))
                )
                if actions is None:
                    break
                explorer._execute(actions, occupancy, result, int(cfg.episode.max_steps))
                explorer._face(chosen.yaw, occupancy, result, int(cfg.episode.max_steps))

            print(f"  episode {episode_index}: {len(records)} branches converted", flush=True)
    finally:
        env.close()

    if not records:
        print("no branches converted", file=sys.stderr)
        return 1

    from collections import defaultdict

    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)

    def mean(rows, key: str) -> float:
        values = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        name: {
            "n_branches": len(rows),
            "frames_used": mean(rows, "n_frames_used"),
            "free_iou": mean(rows, "free_iou"),
            "occupied_iou": mean(rows, "occupied_iou"),
            "area_ratio": mean(rows, "area_ratio"),
        }
        for name, rows in by_condition.items()
    }
    (run_dir / "ceiling.json").write_text(
        json.dumps({"summary": summary, "branches": records}, indent=2, default=str)
    )

    print("\n" + "=" * 72)
    print("CONVERSION CEILING under generator-realistic degradations")
    print("=" * 72)
    print(f"{'condition':<20}{'frames':>8}{'free IoU':>11}{'occ IoU':>10}{'area ratio':>12}")
    print("-" * 72)
    for name in ["ground_truth"] + [n for n in summary if n != "ground_truth"]:
        if name not in summary:
            continue
        row = summary[name]
        print(f"{name:<20}{row['frames_used']:>8.0f}{row['free_iou']:>11.3f}"
              f"{row['occupied_iou']:>10.3f}{row['area_ratio']:>12.3f}")

    baseline = summary.get("ground_truth", {})
    print(f"\n  {baseline.get('n_branches', 0)} branches per condition.")
    print("  ground_truth is tautological -- same integrator, same inputs -- and")
    print("  is shown only as a reference. The row matching the generator's frame")
    print("  count is the real ceiling; the scale and noise rows say how much")
    print("  depth error the conversion tolerates before it dominates.")
    print(f"\nresults: {run_dir / 'ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
