#!/usr/bin/env python
"""Phase 9A/9B: isolate the depth estimator, then test metric-scale anchoring.

Runs GROUND-TRUTH validation branch videos through the real estimator and the
real converter:

    I_1:T (real RGB) -> depth estimator -> converter -> dM_occ

This separates estimator + pose + integration error from Cosmos generation.
If it cannot beat Phase 8 v0 on validation, Cosmos has no geometric headroom
however good its video is, because everything downstream of generation already
loses the signal.

Calibration conditions (9B):

    none         raw metric output
    first_frame  one scale per rollout from the conditioning frame against the
                 real sensor -- DEPLOYABLE, uses only what is already observed
    oracle       per-video scale against ground-truth depth -- DIAGNOSTIC ONLY

Also reports the ground-truth-depth condition as the converter ceiling, so a
poor result is attributable to the estimator rather than the converter.

Validation scenes only. The sealed set is not touched.

    python scripts/validate_depth.py --episodes 4 --override data.split=train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.config import check_data_paths, load_config, save_config  # noqa: E402
from frontierworld.data.tensors import (  # noqa: E402
    FrameSpec,
    FrontierFrame,
    resample_grid,
)
from frontierworld.evaluation import make_run_id  # noqa: E402
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from frontierworld.models.depth_estimator import (  # noqa: E402
    MetricDepthEstimator,
    apply_scale,
    depth_metrics,
    first_frame_scale,
    oracle_scale,
)
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
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--decisions", type=int, default=2)
    parser.add_argument("--generator-frames", type=int, default=17)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def tolerant_iou(predicted, target, valid, tolerance: int = 2) -> float:
    from scipy import ndimage

    p = (predicted >= 0.5) & valid
    a = (target >= 0.5) & valid
    if not (p.any() or a.any()):
        return float("nan")
    intersection = (
        (p & ndimage.binary_dilation(a, iterations=tolerance)).sum()
        + (a & ndimage.binary_dilation(p, iterations=tolerance)).sum()
    )
    union = p.sum() + a.sum()
    return float(intersection / union) if union else float("nan")


def capture_branch(sim, option, cfg, n_frames: int):
    """Execute an option, recording RGB, sensor depth and pose per step."""
    from frontierworld.habitat_env import sensor_extrinsics
    from frontierworld.planning.branching import sim_observations

    agent = sim.get_agent(0)
    rgb, depth, rotations, translations = [], [], [], []
    for action in option.actions:
        agent.act(action)
        observations = sim_observations(sim, cfg)
        rotation, translation = sensor_extrinsics(sim, "depth")
        rgb.append(np.asarray(observations["rgb"])[..., :3].astype(np.uint8))
        depth.append(np.asarray(observations["depth"], dtype=np.float32))
        rotations.append(rotation)
        translations.append(translation)

    if not rgb:
        return None
    total = len(rgb)
    index = (
        np.linspace(0, total - 1, min(n_frames, total)).round().astype(int)
    )
    return {
        "rgb": np.stack([rgb[i] for i in index]),
        "depth": np.stack([depth[i] for i in index]),
        "rotations": np.stack([rotations[i] for i in index]),
        "translations": np.stack([translations[i] for i in index]),
        "n_available": total,
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
    run_dir = Path(cfg.experiment.output_dir) / "phase9_depth" / make_run_id("depth", "9ab")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    from frontierworld.data.manifests import SplitManifest
    from frontierworld.habitat_env import make_env_with_dataset, select_episodes
    from frontierworld.mapping.occupancy import OccupancyMap
    from frontierworld.mapping.semantic import SemanticMap
    from frontierworld.planning.branching import (
        SimulatorSnapshot,
        execute_option,
        sim_observations,
    )
    from frontierworld.planning.exploration import EpisodeResult, FrontierExplorer
    from frontierworld.planning.options import build_options

    estimator = MetricDepthEstimator(
        args.model or MetricDepthEstimator.__init__.__defaults__[0],
        max_depth=float(cfg.simulator.max_depth),
    )
    print(f"estimator: {estimator.model_name} on {estimator.device}")

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
    records, depth_records = [], []

    try:
        for episode_index in range(args.episodes):
            explorer = FrontierExplorer(env, cfg, policy)
            observations = env.reset()
            explorer._bind_to_scene()
            episode = env.current_episode
            explorer.detector.reset(episode)
            rng = episode_rng(int(cfg.seed.value), "depth", episode_index)

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
                ][:2]
                if not options:
                    break

                grid_before = occupancy.to_grid()
                snapshot = SimulatorSnapshot.capture(env.sim, occupancy, semantic_map)
                map_origin = np.asarray(
                    [occupancy.geometry.origin_x, occupancy.geometry.origin_z]
                )
                # The conditioning frame: real sensor depth, observed before
                # the rollout starts. This is what first-frame calibration
                # is allowed to use.
                conditioning = sim_observations(env.sim, cfg)
                conditioning_rgb = np.asarray(conditioning["rgb"])[..., :3].astype(np.uint8)
                conditioning_depth = np.asarray(conditioning["depth"], dtype=np.float32)
                conditioning_predicted = estimator.predict(conditioning_rgb)
                scale_first_frame = first_frame_scale(
                    conditioning_predicted, conditioning_depth
                )

                for option in options:
                    execute_option(env.sim, option, occupancy, cfg)
                    grid_after = occupancy.to_grid()
                    reference = (grid_before == UNKNOWN) & (grid_after != UNKNOWN)
                    snapshot.restore(env.sim, occupancy, semantic_map)

                    captured = capture_branch(env.sim, option, cfg, args.generator_frames)
                    snapshot.restore(env.sim, occupancy, semantic_map)
                    if captured is None:
                        continue

                    predicted_depth = estimator.predict_batch(captured["rgb"])
                    metrics = depth_metrics(predicted_depth, captured["depth"])
                    # Post-calibration metrics. A pure scale factor would not
                    # by itself explain AbsRel 0.711 at delta1 0.053, so
                    # removing scale is the only way to see how much SHAPE
                    # error remains -- which is what a converter cannot fix.
                    calibrated = depth_metrics(
                        predicted_depth * scale_first_frame, captured["depth"]
                    )
                    oracle_calibrated = depth_metrics(
                        predicted_depth * oracle_scale(predicted_depth, captured["depth"]),
                        captured["depth"],
                    )
                    depth_records.append({
                        "scene": Path(result.scene_id).stem.replace(".basis", ""),
                        "frontier_id": option.frontier_id,
                        "scale_first_frame": scale_first_frame,
                        **metrics.to_dict(),
                        "abs_rel_first_frame": calibrated.abs_rel,
                        "delta1_first_frame": calibrated.delta1,
                        "abs_rel_oracle": oracle_calibrated.abs_rel,
                        "delta1_oracle": oracle_calibrated.delta1,
                    })

                    frame = FrontierFrame(
                        centroid_world=np.asarray(option.frontier.centroid_world)[[0, 2]],
                        normal=np.asarray(option.frontier.orientation),
                        spec=spec,
                    )
                    local_after, in_map = resample_grid(
                        grid_after, map_origin, float(cfg.mapping.resolution), frame,
                        fill=UNKNOWN,
                    )
                    local_reference, _ = resample_grid(
                        reference.astype(np.uint8), map_origin,
                        float(cfg.mapping.resolution), frame, fill=0,
                    )
                    revealed = local_reference.astype(bool) & in_map
                    target = np.stack([
                        revealed & (local_after == FREE),
                        revealed & (local_after == OCCUPIED),
                    ]).astype(np.float32)
                    reference_area = float(reference.sum() * cfg.mapping.resolution ** 2)

                    conditions = {
                        "gt_depth": captured["depth"],
                        "none": apply_scale(predicted_depth, 1.0, float(cfg.simulator.max_depth)),
                        "first_frame": apply_scale(
                            predicted_depth, scale_first_frame, float(cfg.simulator.max_depth)
                        ),
                        "oracle": apply_scale(
                            predicted_depth,
                            oracle_scale(predicted_depth, captured["depth"]),
                            float(cfg.simulator.max_depth),
                        ),
                    }
                    for name, depths in conditions.items():
                        rollout = accumulate_rollout(
                            depths=depths,
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
                        predicted_tensor = rollout.as_target_tensor()
                        window = in_map & rollout.valid
                        records.append({
                            "condition": name,
                            "scene": Path(result.scene_id).stem.replace(".basis", ""),
                            "frontier_id": option.frontier_id,
                            "free_iou": masked_iou(predicted_tensor[0], target[0], window),
                            "occupied_iou": masked_iou(predicted_tensor[1], target[1], window),
                            "occupied_iou_tolerant": tolerant_iou(
                                predicted_tensor[1], target[1], window
                            ),
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

            print(f"  episode {episode_index}: {len(depth_records)} branches", flush=True)
    finally:
        env.close()

    if not records:
        print("no branches processed", file=sys.stderr)
        return 1

    def mean(rows, key):
        values = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        return float(np.mean(values)) if values else float("nan")

    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)

    print("\n" + "=" * 78)
    print("9A: DEPTH ESTIMATOR (no ground-truth scale given)")
    print("=" * 78)
    print(f"  branches            : {len(depth_records)}")
    print(f"  AbsRel              : {mean(depth_records, 'abs_rel'):.3f}")
    print(f"  RMSE                : {mean(depth_records, 'rmse'):.3f} m")
    print(f"  delta1 (<1.25)      : {mean(depth_records, 'delta1'):.3f}")
    print(f"  median scale ratio  : {mean(depth_records, 'median_scale_ratio'):.3f}  "
          f"(1.0 = correctly scaled)")
    print(f"  per-frame scale drift: {mean(depth_records, 'scale_drift'):.3f}")
    print(f"  first-frame scale   : {mean(depth_records, 'scale_first_frame'):.3f}")
    print(f"\n  after removing scale (this is residual SHAPE error):")
    print(f"    AbsRel first-frame : {mean(depth_records, 'abs_rel_first_frame'):.3f}"
          f"   delta1 {mean(depth_records, 'delta1_first_frame'):.3f}")
    print(f"    AbsRel oracle      : {mean(depth_records, 'abs_rel_oracle'):.3f}"
          f"   delta1 {mean(depth_records, 'delta1_oracle'):.3f}")

    print("\n" + "=" * 78)
    print("9B: MAP RECONSTRUCTION BY CALIBRATION MODE")
    print("=" * 78)
    print(f"{'condition':<14}{'free IoU':>11}{'occ IoU':>10}{'occ tol':>10}{'area ratio':>12}")
    print("-" * 78)
    for name in ["gt_depth", "none", "first_frame", "oracle"]:
        rows = by_condition.get(name)
        if not rows:
            continue
        print(f"{name:<14}{mean(rows,'free_iou'):>11.3f}{mean(rows,'occupied_iou'):>10.3f}"
              f"{mean(rows,'occupied_iou_tolerant'):>10.3f}{mean(rows,'area_ratio'):>12.3f}")

    summary = {
        "estimator": estimator.model_name,
        "depth": {k: mean(depth_records, k) for k in
                  ["abs_rel", "rmse", "delta1", "median_scale_ratio",
                   "scale_drift", "scale_first_frame",
                   "abs_rel_first_frame", "delta1_first_frame",
                   "abs_rel_oracle", "delta1_oracle"]},
        "map": {
            name: {k: mean(rows, k) for k in
                   ["free_iou", "occupied_iou", "occupied_iou_tolerant", "area_ratio"]}
            for name, rows in by_condition.items()
        },
        "n_branches": len(depth_records),
    }
    (run_dir / "depth_validation.json").write_text(
        json.dumps({"summary": summary, "depth": depth_records, "map": records},
                   indent=2, default=str)
    )
    print("\n  gt_depth is the converter ceiling; the gap to first_frame is the")
    print("  estimator's cost. first_frame is deployable, oracle is an upper bound.")
    print(f"\nresults: {run_dir / 'depth_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
