#!/usr/bin/env python
"""Phase 6 gate: track frontier lineage through an episode and score it.

Explores an episode, updating the lineage graph at every step, then scores the
predicted associations against the offline oracle. Also checks branch
isolation: counterfactual branches must not modify the graph.

    python scripts/evaluate_lineage.py --episodes 10
    python scripts/evaluate_lineage.py --episodes 20 --baseline
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
from frontierworld.lineage import (  # noqa: E402
    LineageGraph,
    MatchingConfig,
    evaluate_lineage,
    long_absence_events,
    observations_from_frontiers,
    oracle_lineage,
    update_lineage,
)
from frontierworld.lineage.matching import nearest_centroid_baseline  # noqa: E402
from frontierworld.planning import make_policy  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--decisions", type=int, default=8)
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="simulator steps between lineage updates (not decisions)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="also score nearest-centroid matching for comparison",
    )
    return parser.parse_args()


def nearest_centroid_graph(observations_by_step: dict[int, list]) -> LineageGraph:
    """Run the weak baseline over the same observations, for comparison."""
    graph = LineageGraph()
    previous: list = []
    for step in sorted(observations_by_step):
        current = [
            _clone(observation) for observation in observations_by_step[step]
        ]
        assignments = nearest_centroid_baseline(previous, current)
        for index, observation in enumerate(current):
            parent_index = assignments.get(index)
            if parent_index is not None:
                observation.lineage_id = previous[parent_index].lineage_id
                graph.add_node(observation)
                graph.add_edge(
                    previous[parent_index].node_id, observation.node_id, "update"
                )
            else:
                observation.lineage_id = graph.new_lineage_id()
                graph.add_node(observation)
                graph.add_edge("", observation.node_id, "birth")
        previous = current
        graph.timestep = step
    return graph


def _clone(observation):
    from frontierworld.lineage.graph import FrontierObservation

    return FrontierObservation(
        node_id=observation.node_id,
        timestep=observation.timestep,
        lineage_id=-1,
        boundary_cells=observation.boundary_cells.copy(),
        centroid=observation.centroid.copy(),
        normal=observation.normal.copy(),
        unknown_component=observation.unknown_component,
        unknown_area_m2=observation.unknown_area_m2,
        size_cells=observation.size_cells,
        information_gain_m2=observation.information_gain_m2,
    )


def run_episode(env, cfg, policy, rng, decisions: int, stride: int = 4) -> dict | None:
    """Explore one episode, maintaining the lineage graph as the map grows."""
    from frontierworld.mapping.occupancy import OccupancyMap
    from frontierworld.planning.branching import SimulatorSnapshot, execute_option
    from frontierworld.planning.exploration import EpisodeResult, FrontierExplorer
    from frontierworld.planning.options import build_options

    explorer = FrontierExplorer(env, cfg, policy)
    observations = env.reset()
    explorer._bind_to_scene()
    episode = env.current_episode
    explorer.detector.reset(episode)

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
    result = EpisodeResult(
        scene_id=str(episode.scene_id),
        episode_id=str(episode.episode_id),
        object_category=getattr(episode, "object_category", None),
    )
    explorer._last_observations = observations
    explorer._target_position = None
    explorer._integrate(observations, occupancy)
    explorer._scan(occupancy, result, int(cfg.episode.max_steps))

    grid_initial = occupancy.to_grid()
    graph = LineageGraph()
    config = MatchingConfig()
    observations_by_step: dict[int, list] = {}
    isolation_checks = {"checked": 0, "violations": 0}

    tick = 0

    def observe(frontiers) -> None:
        """Fold the current frontiers into the lineage graph."""
        nonlocal tick
        step_observations = observations_from_frontiers(
            frontiers, tick, occupancy.to_grid(), float(cfg.mapping.resolution)
        )
        observations_by_step[tick] = [_clone(o) for o in step_observations]
        update_lineage(graph, step_observations, tick, config)
        tick += 1

    for decision in range(decisions):
        frontiers = [f for f in explorer._frontiers(occupancy) if f.reachable]
        if not frontiers:
            break

        observe(frontiers)

        # Branch isolation: run counterfactual branches and confirm the graph
        # is untouched afterwards.
        if decision == 1 and len(frontiers) >= 2:
            before = graph.fingerprint()
            snapshot = SimulatorSnapshot.capture(
                env.sim, occupancy, None, lineage=graph
            )
            options = [
                o
                for o in build_options(
                    frontiers, explorer.planner, occupancy, env.sim, cfg,
                    explorer._calibrate_turn_sign(),
                )
                if o.valid
            ][:3]
            for option in options:
                execute_option(env.sim, option, occupancy, cfg)
                snapshot.restore(env.sim, occupancy, None, lineage=graph)
            isolation_checks["checked"] += 1
            if graph.fingerprint() != before:
                isolation_checks["violations"] += 1

        chosen = policy.select(frontiers, rng)
        if chosen is None:
            break
        actions = explorer.planner.try_plan(
            chosen.approach_point(float(cfg.frontiers.approach_offset_m))
        )
        if actions is None:
            continue

        # Execute in chunks, re-observing the lineage between them. Updating
        # only at decision points lets the agent travel tens of steps between
        # observations, by which time a boundary has moved metres and no
        # matcher can reasonably associate it.
        for start_index in range(0, len(actions), max(1, stride)):
            chunk = actions[start_index : start_index + max(1, stride)]
            if explorer._execute(chunk, occupancy, result, int(cfg.episode.max_steps)) == 0:
                break
            observe([f for f in explorer._frontiers(occupancy) if f.reachable])
        explorer._face(chosen.yaw, occupancy, result, int(cfg.episode.max_steps))

    if len(observations_by_step) < 2:
        return None

    grid_final = occupancy.to_grid()
    truth, links = oracle_lineage(observations_by_step, grid_initial, grid_final)
    metrics = evaluate_lineage(graph, truth, links)
    absences = long_absence_events(observations_by_step, truth)

    return {
        "scene": Path(result.scene_id).stem.replace(".basis", ""),
        "episode_id": result.episode_id,
        "steps": len(observations_by_step),
        "graph": graph.summary(),
        "metrics": metrics.to_dict(),
        "long_absences": len(absences),
        "isolation": isolation_checks,
        "observations_by_step": observations_by_step,
        "predicted_graph": graph,
        "truth": truth,
        "links": links,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), bool(cfg.seed.torch_deterministic))
    run_id = make_run_id("phase6_lineage", config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / "phase6" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(json.dumps(environment_provenance(), indent=2))
    print(f"run_dir: {run_dir}\n")

    from frontierworld.habitat_env import make_env

    env = make_env(cfg)
    policy = make_policy("nearest")
    records = []
    baseline_records = []
    isolation = {"checked": 0, "violations": 0}

    try:
        for index in range(args.episodes):
            rng = episode_rng(int(cfg.seed.value), "phase6", index)
            outcome = run_episode(env, cfg, policy, rng, args.decisions, args.stride)
            if outcome is None:
                continue

            metrics = outcome["metrics"]
            isolation["checked"] += outcome["isolation"]["checked"]
            isolation["violations"] += outcome["isolation"]["violations"]
            print(
                f"ep {index:2d} {outcome['scene']:14s} steps={outcome['steps']:2d} "
                f"nodes={outcome['graph']['n_nodes']:3d} "
                f"lineages={outcome['graph']['n_lineages']:3d}  "
                f"F1={metrics['association_f1']:.3f} IDF1={metrics['idf1']:.3f} "
                f"switches={metrics['id_switches']:2d} "
                f"contam={metrics['cache_contamination_rate']:.3f}  "
                f"events={outcome['graph']['events']}"
            )

            if args.baseline:
                baseline_graph = nearest_centroid_graph(outcome["observations_by_step"])
                baseline_metrics = evaluate_lineage(
                    baseline_graph, outcome["truth"], outcome["links"]
                )
                baseline_records.append(baseline_metrics.to_dict())

            records.append(
                {
                    "scene": outcome["scene"],
                    "episode_id": outcome["episode_id"],
                    "steps": outcome["steps"],
                    "graph": outcome["graph"],
                    "long_absences": outcome["long_absences"],
                    **metrics,
                }
            )
    finally:
        env.close()

    if not records:
        print("no episodes produced enough timesteps", file=sys.stderr)
        return 1

    def mean(key: str, source=None) -> float:
        source = source if source is not None else records
        values = [float(r[key]) for r in source if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "episodes": len(records),
        "association_precision": mean("association_precision"),
        "association_recall": mean("association_recall"),
        "association_f1": mean("association_f1"),
        "idf1": mean("idf1"),
        "id_switches_total": int(sum(r["id_switches"] for r in records)),
        "cache_contamination_rate": mean("cache_contamination_rate"),
        "long_absences_total": int(sum(r["long_absences"] for r in records)),
        "isolation": isolation,
    }
    if baseline_records:
        summary["baseline_nearest_centroid"] = {
            "association_f1": mean("association_f1", baseline_records),
            "idf1": mean("idf1", baseline_records),
            "id_switches_total": int(sum(r["id_switches"] for r in baseline_records)),
            "cache_contamination_rate": mean(
                "cache_contamination_rate", baseline_records
            ),
        }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "episodes.json").write_text(json.dumps(records, indent=2, default=str))

    print("\n" + "=" * 72)
    print(f"episodes                : {summary['episodes']}")
    print(f"association P / R / F1  : {summary['association_precision']:.3f} / "
          f"{summary['association_recall']:.3f} / {summary['association_f1']:.3f}")
    print(f"IDF1                    : {summary['idf1']:.3f}")
    print(f"ID switches (total)     : {summary['id_switches_total']}")
    print(f"cache contamination     : {summary['cache_contamination_rate']:.3f}")
    print(f"long-absence events     : {summary['long_absences_total']}")
    if baseline_records:
        base = summary["baseline_nearest_centroid"]
        print("\nnearest-centroid baseline:")
        print(f"  F1 {base['association_f1']:.3f}  IDF1 {base['idf1']:.3f}  "
              f"switches {base['id_switches_total']}  "
              f"contamination {base['cache_contamination_rate']:.3f}")
        better = summary["association_f1"] > base["association_f1"]
        print(f"  lineage graph better than nearest-centroid: {better}")

    print(f"\nbranch isolation: {isolation['checked']} checks, "
          f"{isolation['violations']} violations")
    passed = isolation["violations"] == 0 and summary["association_f1"] > 0
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    print(f"\nsummary: {run_dir / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
