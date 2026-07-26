#!/usr/bin/env python
"""Phase 1 gate: run HM3D ObjectNav episodes and save full episode logs.

Saves RGB, depth, semantic annotation, pose and an occupancy map per episode,
plus CSV/JSONL run logs.

    python scripts/run_episode.py
    python scripts/run_episode.py --episodes 3 --override episode.max_steps=50
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
    to_dict,
)
from frontierworld.data import EpisodeWriter  # noqa: E402
from frontierworld.evaluation import (  # noqa: E402
    RunLogger,
    environment_provenance,
    make_run_id,
)
from frontierworld.mapping import OccupancyMap  # noqa: E402
from frontierworld.seeding import (  # noqa: E402
    episode_rng,
    rng_state_fingerprint,
    seed_everything,
)

# habitat's discrete ObjectNav action space
STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN = range(6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotlist config override, repeatable",
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="number of episodes to run"
    )
    return parser.parse_args()


def choose_action(rng: np.random.Generator, blocked: bool) -> int:
    """Phase 1 placeholder policy: forward-biased random walk.

    Deterministic given the episode RNG. Phase 2 replaces this with the
    nearest-frontier and information-gain policies.
    """
    if blocked:
        return int(rng.choice([TURN_LEFT, TURN_RIGHT]))
    return int(rng.choice([MOVE_FORWARD, TURN_LEFT, TURN_RIGHT], p=[0.7, 0.15, 0.15]))


def run_episode(env, cfg, run_dir: Path, logger: RunLogger, episode_index: int) -> dict:
    from frontierworld.habitat_env import (
        agent_pose,
        semantic_id_to_category,
        sensor_extrinsics,
    )

    observations = env.reset()
    episode = env.current_episode
    rng = episode_rng(cfg.seed.value, episode.scene_id, episode.episode_id)

    start_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
    occupancy = OccupancyMap(
        resolution=cfg.mapping.resolution,
        size_m=cfg.mapping.size_m,
        obstacle_height_min=cfg.mapping.obstacle_height_min,
        obstacle_height_max=cfg.mapping.obstacle_height_max,
        column_stride=cfg.mapping.column_stride,
        min_observations=cfg.mapping.min_observations,
        center=(float(start_position[0]), float(start_position[2])),
        floor_y=float(start_position[1]),
    )

    categories = semantic_id_to_category(env.sim)
    if not categories:
        print(
            "  WARNING: scene has no semantic annotations loaded; "
            "check data.scene_dataset_config",
            file=sys.stderr,
        )

    writer = EpisodeWriter(run_dir, episode.scene_id, episode.episode_id, cfg.recording)
    writer.write_semantic_categories(categories)

    def integrate(obs) -> None:
        rotation, translation = sensor_extrinsics(env.sim, "depth")
        occupancy.integrate(
            depth=obs["depth"],
            rotation=rotation,
            translation=translation,
            agent_position=np.asarray(env.sim.get_agent_state().position),
            hfov_deg=float(cfg.simulator.hfov),
            max_depth=float(cfg.simulator.max_depth),
        )

    integrate(observations)
    writer.write_step(0, observations, agent_pose(env.sim), action=None)

    step = 0
    collisions = 0
    previous_position = start_position
    while not env.episode_over and step < int(cfg.episode.max_steps):
        blocked = collisions > 0 and step > 0 and _stalled(previous_position, env)
        action = choose_action(rng, blocked)
        previous_position = np.asarray(env.sim.get_agent_state().position)

        observations = env.step(action)
        step += 1

        moved = float(
            np.linalg.norm(np.asarray(env.sim.get_agent_state().position) - previous_position)
        )
        if action == MOVE_FORWARD and moved < 1e-3:
            collisions += 1

        integrate(observations)
        metrics = env.get_metrics()
        writer.write_step(
            step, observations, agent_pose(env.sim), action=action, info=metrics
        )

        if step % int(cfg.logging.log_every) == 0:
            logger.log(
                {
                    "event": "step",
                    "episode_index": episode_index,
                    "episode_id": episode.episode_id,
                    "scene_id": Path(episode.scene_id).stem,
                    "distance_to_goal": metrics.get("distance_to_goal"),
                    "collisions": collisions,
                    **occupancy.stats(),
                },
                step=step,
            )

    metrics = env.get_metrics()
    map_stats = occupancy.stats()
    writer.write_map(occupancy)

    result = {
        "episode_index": episode_index,
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,
        "object_category": getattr(episode, "object_category", None),
        "steps": step,
        "collisions": collisions,
        "success": float(metrics.get("success", 0.0)),
        "spl": float(metrics.get("spl", 0.0)),
        "soft_spl": float(metrics.get("soft_spl", 0.0)),
        "distance_to_goal": float(metrics.get("distance_to_goal", float("nan"))),
        "semantic_instances": len(categories),
        **map_stats,
    }
    writer.write_metadata(
        {
            "object_category": result["object_category"],
            "start_position": start_position.tolist(),
            "metrics": {k: float(v) for k, v in metrics.items() if _is_number(v)},
            "map": map_stats,
            "policy": str(cfg.episode.policy),
            "seed": int(cfg.seed.value),
        }
    )
    writer.close()

    logger.log({"event": "episode_end", **result}, step=step)
    return result


def _stalled(previous_position: np.ndarray, env) -> bool:
    current = np.asarray(env.sim.get_agent_state().position)
    return bool(np.linalg.norm(current - previous_position) < 1e-3)


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    problems = check_data_paths(cfg)
    if problems:
        print("Data path problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), bool(cfg.seed.torch_deterministic))

    run_id = make_run_id(cfg.experiment.name, config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    print(f"run_dir: {run_dir}")

    from frontierworld.habitat_env import make_env

    results = []
    with RunLogger(run_dir, cfg.logging, run_id, to_dict(cfg)) as logger:
        provenance = environment_provenance()
        logger.log({"event": "run_start", **provenance})
        (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

        env = make_env(cfg)
        try:
            for episode_index in range(args.episodes):
                result = run_episode(env, cfg, run_dir, logger, episode_index)
                results.append(result)
                print(
                    f"  episode {episode_index}: {result['object_category']} "
                    f"in {Path(result['scene_id']).stem} -- "
                    f"{result['steps']} steps, success={result['success']:.0f}, "
                    f"explored={result['explored_area_m2']:.1f} m2, "
                    f"{result['semantic_instances']} semantic instances"
                )
        finally:
            env.close()

        summary = {
            "episodes": len(results),
            "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
            "spl": float(np.mean([r["spl"] for r in results])) if results else 0.0,
            "mean_explored_area_m2": (
                float(np.mean([r["explored_area_m2"] for r in results])) if results else 0.0
            ),
            "seed": int(cfg.seed.value),
            "config_hash": config_hash(cfg),
            "rng_fingerprint": rng_state_fingerprint(),
            "results": results,
        }
        logger.write_summary(summary)

    print(f"summary: {run_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
