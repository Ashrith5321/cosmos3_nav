#!/usr/bin/env python
"""Phase 6.5: export frontier sequences for human transition annotation.

Real-episode association accuracy is unmeasured, because both automatic
oracles proved untrustworthy. The replacement is human labelling of
*consecutive transition relations* rather than global identities:

    R_ij^{t->t+1} in {none, update, split-child, merge-parent}

Labelling transitions is far less ambiguous than assigning one identity across
a whole episode, and it sidesteps the connected-unknown-region collapse that
broke oracle v1.

Annotation rule: two frontier observations belong to the same lineage when they
represent the same physical access boundary into unresolved space. Separate
doorways stay separate even when they lead into the same room.

    python scripts/export_lineage_annotation.py --episodes 8 --transitions 100
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
from frontierworld.lineage import (  # noqa: E402
    LineageGraph,
    MatchingConfig,
    observations_from_frontiers,
    update_lineage,
)
from frontierworld.lineage.matching import association_matrix  # noqa: E402
from frontierworld.planning import make_policy  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402

ANNOTATION_RULE = (
    "Two frontier observations belong to the same lineage when they represent "
    "the same physical access boundary into unresolved space. Separate doorways "
    "remain separate even if they enter the same room."
)
LABELS = ["none", "update", "split-child", "merge-parent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--transitions", type=int, default=100)
    parser.add_argument("--decisions", type=int, default=6)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def render_transition(
    path: Path,
    grid_before: np.ndarray,
    grid_after: np.ndarray,
    previous,
    current,
    geometry,
    rgb_before,
    rgb_after,
    meta: dict,
) -> Path:
    """Side-by-side view of one t -> t+1 transition, with components labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from frontierworld.evaluation.visualize import crop_bounds, grid_to_rgb, world_to_cell

    bounds = crop_bounds(grid_after)
    r0, r1, c0, c1 = bounds

    figure, axes = plt.subplots(1, 4, figsize=(21, 5.6), constrained_layout=True)

    for axis, grid, observations, title in (
        (axes[0], grid_before, previous, f"t = {meta['t']}"),
        (axes[1], grid_after, current, f"t+1 = {meta['t_next']}"),
    ):
        axis.imshow(grid_to_rgb(grid)[r0:r1, c0:c1], interpolation="nearest")
        for index, observation in enumerate(observations):
            cells = observation.boundary_cells
            axis.scatter(
                cells[:, 1] - c0, cells[:, 0] - r0, s=2.0,
                c=[plt.cm.tab10(index % 10)], linewidths=0,
            )
            row, col = world_to_cell(
                geometry, observation.centroid[0], observation.centroid[1]
            )
            axis.annotate(
                str(index), (col - c0, row - r0), fontsize=11, weight="bold",
                color="black",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.6),
            )
        axis.set_title(f"{title}: {len(observations)} frontiers", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis, image, title in (
        (axes[2], rgb_before, "RGB at t"),
        (axes[3], rgb_after, "RGB at t+1"),
    ):
        if image is not None:
            axis.imshow(np.asarray(image)[..., :3].astype(np.uint8))
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"{meta['scene']}  ep={meta['episode_id']}  transition {meta['t']}->{meta['t_next']}\n"
        f"label each (t index, t+1 index) pair: {', '.join(LABELS)}",
        fontsize=11,
        fontfamily="monospace",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), bool(cfg.seed.torch_deterministic))
    out = Path(args.out) if args.out else Path(cfg.experiment.output_dir) / "phase6_annotation"
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    from frontierworld.habitat_env import make_env
    from frontierworld.mapping.occupancy import OccupancyMap
    from frontierworld.planning.branching import sim_observations
    from frontierworld.planning.exploration import EpisodeResult, FrontierExplorer

    env = make_env(cfg)
    policy = make_policy("nearest")
    config = MatchingConfig()
    tasks: list[dict] = []

    try:
        for episode_index in range(args.episodes):
            if len(tasks) >= args.transitions:
                break

            explorer = FrontierExplorer(env, cfg, policy)
            observations = env.reset()
            explorer._bind_to_scene()
            episode = env.current_episode
            explorer.detector.reset(episode)
            rng = episode_rng(int(cfg.seed.value), "annotate", episode_index)

            start = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
            occupancy = OccupancyMap(
                resolution=cfg.mapping.resolution, size_m=cfg.mapping.size_m,
                obstacle_height_min=cfg.mapping.obstacle_height_min,
                obstacle_height_max=cfg.mapping.obstacle_height_max,
                column_stride=cfg.mapping.column_stride,
                min_observations=cfg.mapping.min_observations,
                center=(float(start[0]), float(start[2])), floor_y=float(start[1]),
            )
            result = EpisodeResult(
                scene_id=str(episode.scene_id),
                episode_id=str(episode.episode_id),
                object_category=getattr(episode, "object_category", None),
            )
            explorer._last_observations = observations
            explorer._target_position = None
            explorer._integrate(observations, occupancy)
            explorer._scan(occupancy, result, int(cfg.episode.max_steps))

            scene = Path(result.scene_id).stem.replace(".basis", "")
            graph = LineageGraph()
            previous_state = None
            tick = 0

            def capture(frontiers):
                nonlocal previous_state, tick
                grid = occupancy.to_grid()
                current = observations_from_frontiers(
                    frontiers, tick, grid, float(cfg.mapping.resolution)
                )
                rgb = np.asarray(sim_observations(env.sim, cfg)["rgb"])[..., :3]
                if previous_state is not None and previous_state["obs"] and current:
                    matrix = association_matrix(previous_state["obs"], current, config)
                    name = f"{scene}_{result.episode_id}_t{previous_state['tick']}_{tick}"
                    figure = render_transition(
                        out / "figures" / f"{name}.png",
                        previous_state["grid"], grid,
                        previous_state["obs"], current,
                        occupancy.geometry,
                        previous_state["rgb"], rgb,
                        {
                            "scene": scene,
                            "episode_id": result.episode_id,
                            "t": previous_state["tick"],
                            "t_next": tick,
                        },
                    )
                    tasks.append(
                        {
                            "id": name,
                            "scene": scene,
                            "episode_id": result.episode_id,
                            "t": previous_state["tick"],
                            "t_next": tick,
                            "figure": str(figure.relative_to(out)),
                            "n_prev": len(previous_state["obs"]),
                            "n_next": len(current),
                            # The matcher's own scores, recorded so agreement
                            # can be measured. Annotators must not see them.
                            "predicted_association": matrix.round(3).tolist(),
                            "labels": [
                                [None] * len(current)
                                for _ in range(len(previous_state["obs"]))
                            ],
                        }
                    )
                update_lineage(graph, current, tick, config)
                previous_state = {
                    "obs": [o for o in current], "grid": grid, "rgb": rgb, "tick": tick
                }
                tick += 1

            for _ in range(args.decisions):
                if len(tasks) >= args.transitions:
                    break
                frontiers = [f for f in explorer._frontiers(occupancy) if f.reachable]
                if not frontiers:
                    break
                capture(frontiers)

                chosen = policy.select(frontiers, rng)
                if chosen is None:
                    break
                actions = explorer.planner.try_plan(
                    chosen.approach_point(float(cfg.frontiers.approach_offset_m))
                )
                if actions is None:
                    continue
                for index in range(0, len(actions), max(1, args.stride)):
                    chunk = actions[index : index + max(1, args.stride)]
                    if explorer._execute(chunk, occupancy, result, int(cfg.episode.max_steps)) == 0:
                        break
                    capture([f for f in explorer._frontiers(occupancy) if f.reachable])
            print(f"ep {episode_index}: {scene}  transitions so far {len(tasks)}", flush=True)
    finally:
        env.close()

    payload = {
        "annotation_rule": ANNOTATION_RULE,
        "labels": LABELS,
        "instructions": [
            "For each transition figure, fill labels[i][j] for every pair of a",
            "frontier i at t and a frontier j at t+1.",
            "  update       same physical boundary, continued",
            "  split-child  j is one of several boundaries the single boundary i became",
            "  merge-parent i is one of several boundaries that became the single j",
            "  none         unrelated",
            "Leave null if genuinely ambiguous; ambiguous pairs are excluded from",
            "scoring rather than guessed.",
            "Do NOT consult predicted_association while labelling; it is the",
            "system output being evaluated.",
        ],
        "config_hash": config_hash(cfg),
        "n_transitions": len(tasks),
        "transitions": tasks[: args.transitions],
    }
    (out / "annotation_tasks.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{len(payload['transitions'])} transitions exported to {out}")
    print(f"  figures: {out / 'figures'}")
    print(f"  tasks  : {out / 'annotation_tasks.json'}")
    print(f"\nrule: {ANNOTATION_RULE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
