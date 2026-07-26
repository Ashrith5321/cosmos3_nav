#!/usr/bin/env python
"""Phase 3 gate: execute independent frontier branches from one identical state.

Explores until a decision state offers at least N frontiers, then builds a
canonical option per frontier and runs each as a branch, restoring the
simulator and the map between them. Verifies that every branch starts from a
bitwise-identical agent state and that the live state is unchanged afterwards.

    python scripts/demo_branches.py
    python scripts/demo_branches.py --branches 3 --decisions 4 --episodes 3
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
from frontierworld.evaluation import environment_provenance, make_run_id  # noqa: E402
from frontierworld.frontiers import extract_frontiers  # noqa: E402
from frontierworld.mapping.occupancy import OccupancyMap  # noqa: E402
from frontierworld.planning import make_policy  # noqa: E402
from frontierworld.planning.branching import (  # noqa: E402
    SimulatorSnapshot,
    run_branches,
    sim_observations,
)
from frontierworld.planning.exploration import FrontierExplorer  # noqa: E402
from frontierworld.planning.options import build_options  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--branches", type=int, default=3, help="frontiers to branch on per state"
    )
    parser.add_argument(
        "--decisions",
        type=int,
        default=4,
        help="policy decisions to take before branching, to build up a map",
    )
    return parser.parse_args()


def explore_to_decision_state(explorer: FrontierExplorer, cfg, rng, decisions: int):
    """Run the policy for a few decisions, then hand back map and frontiers."""
    env = explorer.env
    observations = env.reset()
    explorer._bind_to_scene()
    explorer.detector.reset(env.current_episode)

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

    from frontierworld.planning.exploration import EpisodeResult

    result = EpisodeResult(
        scene_id=str(env.current_episode.scene_id),
        episode_id=str(env.current_episode.episode_id),
        object_category=getattr(env.current_episode, "object_category", None),
    )
    explorer._last_observations = observations
    explorer._target_position = None
    explorer._integrate(observations, occupancy)
    explorer._scan(occupancy, result, int(cfg.episode.max_steps))

    for _ in range(decisions):
        frontiers = explorer._frontiers(occupancy)
        candidates = [f for f in frontiers if f.reachable]
        if not candidates:
            break
        chosen = explorer.policy.select(candidates, rng)
        if chosen is None:
            break
        goal = chosen.approach_point(float(cfg.frontiers.approach_offset_m))
        actions = explorer.planner.try_plan(goal)
        if actions is None:
            continue
        explorer._execute(actions, occupancy, result, int(cfg.episode.max_steps))
        explorer._face(chosen.yaw, occupancy, result, int(cfg.episode.max_steps))

    frontiers = explorer._frontiers(occupancy)
    return occupancy, [f for f in frontiers if f.reachable], result


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), bool(cfg.seed.torch_deterministic))
    run_id = make_run_id("phase3_branches", config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / "phase3" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(
        json.dumps(environment_provenance(), indent=2)
    )
    print(f"run_dir: {run_dir}\n")

    from frontierworld.habitat_env import make_env

    env = make_env(cfg)
    policy = make_policy("nearest")
    records = []
    gate_passed = 0
    gate_attempted = 0

    try:
        for episode_index in range(args.episodes):
            explorer = FrontierExplorer(env, cfg, policy)
            rng = episode_rng(int(cfg.seed.value), "phase3", episode_index)
            occupancy, frontiers, result = explore_to_decision_state(
                explorer, cfg, rng, args.decisions
            )
            scene = Path(result.scene_id).stem.replace(".basis", "")

            print(f"episode {episode_index}: {scene} goal={result.object_category} "
                  f"steps={result.steps} reachable frontiers={len(frontiers)}")

            if len(frontiers) < args.branches:
                print(f"  skipped: need {args.branches} frontiers, found {len(frontiers)}\n")
                continue

            # Build options for every reachable frontier and keep the valid
            # ones. Options are rejected for good reasons -- an approach pose
            # outside observed free space, or unreachable before crossing --
            # so the branch set is chosen from what actually yields an option,
            # not from the first N frontiers regardless of validity.
            all_options = build_options(
                frontiers,
                explorer.planner,
                occupancy,
                env.sim,
                cfg,
                explorer._calibrate_turn_sign(),
            )
            rejected = [o for o in all_options if not o.valid]
            options = [o for o in all_options if o.valid][: args.branches]
            for option in rejected:
                print(f"    frontier {option.frontier_id}: rejected "
                      f"({option.reject_reason})")

            if len(options) < args.branches:
                print(f"  skipped: {len(options)} valid options, "
                      f"need {args.branches}\n")
                continue

            gate_attempted += 1

            before = SimulatorSnapshot.capture(env.sim, occupancy)
            results, snapshot = run_branches(env.sim, options, occupancy, cfg)
            state_restored = before.matches_agent(env.sim)
            map_restored = bool(
                np.array_equal(before.free_counts, occupancy.free_counts)
                and np.array_equal(before.occupied_counts, occupancy.occupied_counts)
            )

            valid = [r for r in results if r.valid]
            distinct_endpoints = len(
                {tuple(np.round(r.final_position, 3)) for r in valid}
            )
            passed = (
                state_restored
                and map_restored
                and len(valid) >= args.branches
                and distinct_endpoints == len(valid)
            )
            gate_passed += int(passed)

            print(f"  branched from position "
                  f"{np.round(np.asarray(snapshot.agent_state.position), 3).tolist()}")
            for option, branch in zip(options, results):
                if not branch.valid:
                    print(f"    frontier {branch.frontier_id}: REJECTED "
                          f"({branch.reject_reason})")
                    continue
                print(
                    f"    frontier {branch.frontier_id}: "
                    f"{len(option.approach_actions):3d}+{len(option.cross_actions):2d} actions "
                    f"crossed={int(branch.crossed)} "
                    f"revealed={branch.newly_observed_area_m2:6.2f} m2 "
                    f"moved={branch.distance_travelled_m:5.2f} m "
                    f"coll={branch.collisions}"
                )
            print(f"  identical start state: enforced during run")
            print(f"  state restored after branches: {state_restored}")
            print(f"  map restored after branches:   {map_restored}")
            print(f"  branches ended in distinct places: "
                  f"{distinct_endpoints}/{len(valid)}")
            print(f"  GATE: {'PASS' if passed else 'FAIL'}\n")

            records.append(
                {
                    "episode_index": episode_index,
                    "scene": scene,
                    "episode_id": result.episode_id,
                    "object_category": result.object_category,
                    "steps_before_branching": result.steps,
                    "n_reachable_frontiers": len(frontiers),
                    "branch_start_position": [
                        float(v) for v in snapshot.agent_state.position
                    ],
                    "options": [o.to_dict() for o in options],
                    "branches": [r.to_dict() for r in results],
                    "trajectories": [r.trajectory for r in results],
                    "state_restored": state_restored,
                    "map_restored": map_restored,
                    "distinct_endpoints": distinct_endpoints,
                    "gate_passed": passed,
                }
            )
    finally:
        env.close()

    (run_dir / "branches.json").write_text(json.dumps(records, indent=2, default=str))

    print("=" * 64)
    print(f"gate attempted on {gate_attempted} decision states, passed {gate_passed}")
    print(f"results: {run_dir / 'branches.json'}")
    return 0 if gate_attempted and gate_passed == gate_attempted else 1


if __name__ == "__main__":
    raise SystemExit(main())
