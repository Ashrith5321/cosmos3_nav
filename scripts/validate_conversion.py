#!/usr/bin/env python
"""Phase 9 step 1: what score can the conversion machinery reach at best?

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
    return parser.parse_args()


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
                    rollout = accumulate_rollout(
                        depths=captured["depths"],
                        rotations=captured["rotations"],
                        translations=captured["translations"],
                        frame=frame,
                        grid_before=grid_before,
                        map_origin=map_origin,
                        map_resolution=float(cfg.mapping.resolution),
                        hfov_deg=float(cfg.simulator.hfov),
                        max_depth=float(cfg.simulator.max_depth),
                        floor_y=float(occupancy.floor_y),
                    )

                    # --- compare in the frontier frame ---------------------
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

                    predicted = rollout.as_target_tensor()
                    window = in_map & rollout.valid
                    record = {
                        "scene": Path(result.scene_id).stem.replace(".basis", ""),
                        "episode_id": result.episode_id,
                        "frontier_id": option.frontier_id,
                        "n_frames": int(captured["depths"].shape[0]),
                        "n_forward_actions": captured["n_forward"],
                        "free_iou": masked_iou(predicted[0], target[0], window),
                        "occupied_iou": masked_iou(predicted[1], target[1], window),
                        "area_reference_m2": float(
                            reference.sum() * cfg.mapping.resolution ** 2
                        ),
                        "area_converted_m2": rollout.revealed_area_m2,
                    }
                    record["area_ratio"] = (
                        record["area_converted_m2"] / record["area_reference_m2"]
                        if record["area_reference_m2"] > 1e-6 else float("nan")
                    )
                    records.append(record)

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

    def mean(key: str) -> float:
        values = [r[key] for r in records if np.isfinite(r[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "n_branches": len(records),
        "free_iou_ceiling": mean("free_iou"),
        "occupied_iou_ceiling": mean("occupied_iou"),
        "area_ratio": mean("area_ratio"),
        "mean_frames": mean("n_frames"),
        "inputs": "ground-truth depth and poses from the simulator",
    }
    (run_dir / "ceiling.json").write_text(
        json.dumps({"summary": summary, "branches": records}, indent=2, default=str)
    )

    print("\n" + "=" * 66)
    print("CONVERSION CEILING (ground-truth depth and poses)")
    print("=" * 66)
    print(f"  branches converted     : {summary['n_branches']}")
    print(f"  free IoU ceiling       : {summary['free_iou_ceiling']:.3f}")
    print(f"  occupied IoU ceiling   : {summary['occupied_iou_ceiling']:.3f}")
    print(f"  revealed-area ratio    : {summary['area_ratio']:.3f}  (1.0 = exact)")
    print(f"  mean frames per branch : {summary['mean_frames']:.1f}")
    print("\n  No generator can exceed these through this pipeline. A Cosmos")
    print("  score below them is attributable to generation or depth; a score")
    print("  near them means the conversion is not the bottleneck.")
    print(f"\nresults: {run_dir / 'ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
