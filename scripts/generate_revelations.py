#!/usr/bin/env python
"""Phase 4 gate: record ground-truth revelation and visualise it.

Explores to a decision state, builds a canonical option per reachable
frontier, executes each as an independent branch from the identical state, and
records what each one revealed. Writes a six-panel figure per branch plus a
contact sheet, so the labels can be checked by eye.

    python scripts/generate_revelations.py --examples 20
    python scripts/generate_revelations.py --examples 24 --episodes 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.config import (  # noqa: E402
    check_data_paths,
    config_hash,
    load_config,
    save_config,
)
from frontierworld.data.records import RevelationExample, RevelationWriter  # noqa: E402
from frontierworld.data.revelation import compute_revelation  # noqa: E402
from frontierworld.evaluation import environment_provenance, make_run_id  # noqa: E402
from frontierworld.evaluation.visualize import contact_sheet, visualise_branch  # noqa: E402
from frontierworld.mapping.semantic import RoomLabeller, SemanticMap  # noqa: E402
from frontierworld.planning import make_policy  # noqa: E402
from frontierworld.planning.branching import (  # noqa: E402
    SimulatorSnapshot,
    execute_option,
    sim_observations,
)
from frontierworld.planning.exploration import FrontierExplorer  # noqa: E402
from frontierworld.planning.options import build_options  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--examples", type=int, default=20, help="branches to record")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--decisions", type=int, default=4)
    parser.add_argument(
        "--max-branches", type=int, default=4, help="branches per decision state"
    )
    return parser.parse_args()


def build_maps(env, cfg):
    from frontierworld.mapping.occupancy import OccupancyMap

    start = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
    occupancy = OccupancyMap(
        resolution=cfg.mapping.resolution,
        size_m=cfg.mapping.size_m,
        obstacle_height_min=cfg.mapping.obstacle_height_min,
        obstacle_height_max=cfg.mapping.obstacle_height_max,
        column_stride=cfg.mapping.column_stride,
        min_observations=cfg.mapping.min_observations,
        center=(float(start[0]), float(start[2])),
        floor_y=float(start[1]),
    )
    semantic = SemanticMap(occupancy.geometry, floor_y=float(start[1]))
    return occupancy, semantic


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), bool(cfg.seed.torch_deterministic))
    run_id = make_run_id("phase4_revelations", config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / "phase4" / run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(
        json.dumps(environment_provenance(), indent=2)
    )
    print(f"run_dir: {run_dir}\n")

    from frontierworld.habitat_env import make_env, semantic_id_to_category

    env = make_env(cfg)
    policy = make_policy("nearest")
    writer = RevelationWriter(run_dir / "dataset")
    figure_paths: list[Path] = []
    summaries: list[dict] = []

    try:
        for episode_index in range(args.episodes):
            if len(figure_paths) >= args.examples:
                break

            explorer = FrontierExplorer(env, cfg, policy)
            rng = episode_rng(int(cfg.seed.value), "phase4", episode_index)

            observations = env.reset()
            explorer._bind_to_scene()
            episode = env.current_episode
            explorer.detector.reset(episode)
            occupancy, semantic_map = build_maps(env, cfg)
            instance_to_category = semantic_id_to_category(env.sim)
            room_labeller = RoomLabeller(env.sim)
            target_ids = set(explorer.detector.target_instance_ids)

            from frontierworld.planning.exploration import EpisodeResult

            result = EpisodeResult(
                scene_id=str(episode.scene_id),
                episode_id=str(episode.episode_id),
                object_category=getattr(episode, "object_category", None),
            )
            explorer._last_observations = observations
            explorer._target_position = None
            explorer._integrate(observations, occupancy)
            explorer._scan(occupancy, result, int(cfg.episode.max_steps))

            for _ in range(args.decisions):
                frontiers = explorer._frontiers(occupancy)
                candidates = [f for f in frontiers if f.reachable]
                if not candidates:
                    break
                chosen = policy.select(candidates, rng)
                actions = explorer.planner.try_plan(
                    chosen.approach_point(float(cfg.frontiers.approach_offset_m))
                )
                if actions is None:
                    continue
                explorer._execute(actions, occupancy, result, int(cfg.episode.max_steps))
                explorer._face(chosen.yaw, occupancy, result, int(cfg.episode.max_steps))

            scene = Path(result.scene_id).stem.replace(".basis", "")
            frontiers = [f for f in explorer._frontiers(occupancy) if f.reachable]
            if not frontiers:
                print(f"episode {episode_index} ({scene}): no reachable frontiers")
                continue

            options = [
                o
                for o in build_options(
                    frontiers, explorer.planner, occupancy, env.sim, cfg,
                    explorer._calibrate_turn_sign(),
                )
                if o.valid
            ][: args.max_branches]
            if not options:
                print(f"episode {episode_index} ({scene}): no valid options")
                continue

            # State everything branches from.
            grid_before = occupancy.to_grid()
            semantic_before = semantic_map.copy_counts()
            instances_before = _visible_instances(env.sim, cfg)
            rgb_before = np.asarray(sim_observations(env.sim, cfg)["rgb"])[..., :3]
            snapshot = SimulatorSnapshot.capture(env.sim, occupancy, semantic_map)
            geodesic_before = _geodesic_to_goals(explorer.planner, episode)

            print(f"episode {episode_index} ({scene}) goal={result.object_category} "
                  f"t={result.steps}: {len(options)} branches")

            examples: list[RevelationExample] = []
            group_id = f"{scene}_{result.episode_id}_t{result.steps}"

            for option in options:
                branch = execute_option(
                    env.sim, option, occupancy, cfg,
                    semantic_map=semantic_map,
                    instance_to_category=instance_to_category,
                    target_instance_ids=target_ids,
                    keep_frames=True,
                )
                grid_after = occupancy.to_grid()
                semantic_after = semantic_map.copy_counts()
                frontiers_after = explorer._frontiers(occupancy)
                geodesic_after = _geodesic_to_goals(explorer.planner, episode)

                revelation, masks = compute_revelation(
                    frontier_id=option.frontier_id,
                    grid_before=grid_before,
                    grid_after=grid_after,
                    semantic_before=semantic_before,
                    semantic_after=semantic_after,
                    semantic_map=semantic_map,
                    frontiers_before=frontiers,
                    frontiers_after=frontiers_after,
                    branch=branch,
                    resolution=float(cfg.mapping.resolution),
                    room_labeller=room_labeller,
                    target_detection={
                        "seen": branch.target_seen_at_step is not None,
                        "pixel_count": branch.target_pixel_count,
                        "step": branch.target_seen_at_step,
                    },
                    distance_to_target_before=geodesic_before,
                    geodesic_to_target=geodesic_after,
                    instances_before=instances_before,
                )

                rgb_after = branch.rgb_frames[-1] if branch.rgb_frames else None
                meta = {
                    "scene": scene,
                    "episode_id": result.episode_id,
                    "decision_timestep": result.steps,
                    "navigation_goal": result.object_category,
                }

                figure_path = figures_dir / f"{group_id}_f{option.frontier_id}.png"
                visualise_branch(
                    path=figure_path,
                    grid_before=grid_before,
                    grid_after=grid_after,
                    revealed_mask=masks["newly_observed"],
                    semantic_mask=masks["newly_semantic"],
                    geometry=occupancy.geometry,
                    chosen_frontier=option.frontier,
                    other_frontiers=[
                        f for f in frontiers if f.frontier_id != option.frontier_id
                    ],
                    trajectory=branch.trajectory,
                    rgb_before=rgb_before,
                    rgb_after=rgb_after,
                    revelation=revelation,
                    meta=meta,
                )
                figure_paths.append(figure_path)

                examples.append(
                    RevelationExample(
                        scene_id=result.scene_id,
                        episode_id=result.episode_id,
                        decision_timestep=result.steps,
                        navigation_goal=result.object_category,
                        frontier_geometry=option.frontier.to_dict(),
                        candidate_option=option.to_dict(),
                        future_revelation=revelation.to_dict(),
                        current_observation={"rgb": "rgb_before.png"},
                        current_map={"arrays": "arrays.npz", "key": "grid_before"},
                        observation_history={
                            "steps_before_decision": result.steps,
                            "trajectory": branch.trajectory,
                        },
                        privileged={
                            "planner": "habitat_navmesh (ground-truth navmesh)",
                            "goal_detector": "oracle ground-truth semantic instance ids",
                            "room_category": "DERIVED from region object composition, "
                                             "HM3D v0.2 ships no room labels",
                            "success_distance_m": float(cfg.task.success_distance),
                        },
                    )
                )

                summaries.append(
                    {
                        "group_id": group_id,
                        "frontier_id": option.frontier_id,
                        "figure": str(figure_path.relative_to(run_dir)),
                        **revelation.to_dict(),
                    }
                )
                print(
                    f"    f{option.frontier_id}: revealed={revelation.newly_observed_area_m2:6.2f} m2 "
                    f"sem={revelation.newly_semantic_in_revealed_area_m2:5.2f} "
                    f"crossed={int(revelation.crossed)} coll={revelation.collisions:2d} "
                    f"newF={revelation.n_new_frontiers} room={revelation.room_category} "
                    f"target={int(revelation.target_became_visible)}"
                )

                snapshot.restore(env.sim, occupancy, semantic_map)

            writer.write_group(
                group_id,
                examples,
                arrays={
                    "grid_before": grid_before,
                    "semantic_before": semantic_before,
                    "map_origin": np.asarray(
                        [occupancy.geometry.origin_x, occupancy.geometry.origin_z],
                        dtype=np.float32,
                    ),
                    "resolution": np.float32(occupancy.geometry.resolution),
                },
                rgb={"rgb_before": rgb_before},
            )
    finally:
        env.close()
        writer.close()

    (run_dir / "revelations.json").write_text(
        json.dumps(summaries, indent=2, default=str)
    )

    sheet = None
    if figure_paths:
        sheet = contact_sheet(figure_paths[: args.examples], run_dir / "contact_sheet.png")

    print("\n" + "=" * 70)
    print(f"branches recorded : {len(summaries)}")
    print(f"decision groups   : {writer.n_groups}")
    print(f"figures           : {figures_dir}")
    if sheet:
        print(f"contact sheet     : {sheet}")
    print(f"dataset           : {run_dir / 'dataset'}")
    print(f"GATE (>=20 visualised): "
          f"{'PASS' if len(figure_paths) >= 20 else 'FAIL'} ({len(figure_paths)})")
    return 0 if len(figure_paths) >= 20 else 1


def _visible_instances(sim, cfg) -> set:
    """Semantic instances in view at the decision state.

    Subtracted from what a branch sees so the room label reflects the room
    entered, not the one the agent was already standing in."""
    semantic = sim_observations(sim, cfg).get("semantic")
    if semantic is None:
        return set()
    return {int(v) for v in np.unique(np.squeeze(np.asarray(semantic)))}


def _geodesic_to_goals(planner, episode) -> float:
    """Shortest geodesic distance from the agent to any goal viewpoint."""
    best = float("inf")
    for goal in getattr(episode, "goals", []) or []:
        position = getattr(goal, "position", None)
        if position is None:
            continue
        distance = planner.geodesic_distance(np.asarray(position, dtype=np.float64))
        best = min(best, distance)
    return best


if __name__ == "__main__":
    raise SystemExit(main())
