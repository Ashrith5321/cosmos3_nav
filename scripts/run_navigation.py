#!/usr/bin/env python
"""Phase 2 gate: run frontier-policy ObjectNav episodes and record metrics.

    python scripts/run_navigation.py --policy nearest --episodes 50
    python scripts/run_navigation.py --policy all --episodes 50

Records SR, SPL, SoftSPL, collisions, explored area, planner failures and
target-detection rate per episode, plus a per-decision log.
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
from frontierworld.planning import FrontierExplorer, make_policy  # noqa: E402
from frontierworld.planning.policies import POLICIES  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--policy",
        default=None,
        help=f"one of {sorted(POLICIES)}, or 'all' to run each in turn",
    )
    parser.add_argument(
        "--save-observations",
        action="store_true",
        help="save RGB/depth/semantic per step (large; off by default)",
    )
    return parser.parse_args()


def summarise(results: list[dict]) -> dict:
    if not results:
        return {}

    def mean(key: str) -> float:
        values = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    def total(key: str) -> int:
        return int(sum(r.get(key, 0) for r in results))

    detection_steps = [
        r["detection_step"] for r in results if r.get("detection_step") is not None
    ]
    return {
        "episodes": len(results),
        "success_rate": mean("success"),
        "spl": mean("spl"),
        "soft_spl": mean("soft_spl"),
        "distance_to_goal": mean("distance_to_goal"),
        "collisions_mean": mean("collisions"),
        "collisions_total": total("collisions"),
        "explored_area_m2_mean": mean("explored_area_m2"),
        "free_area_m2_mean": mean("free_area_m2"),
        "steps_mean": mean("steps"),
        "decisions_mean": mean("decisions"),
        "planner_failures_total": total("planner_failures"),
        "planner_failure_episodes": int(
            sum(1 for r in results if r.get("planner_failures", 0) > 0)
        ),
        "frontier_exhaustion_episodes": int(
            sum(1 for r in results if r.get("frontier_exhaustions", 0) > 0)
        ),
        "detection_rate": float(np.mean([bool(r["detected"]) for r in results])),
        "steps_to_detection_mean": (
            float(np.mean(detection_steps)) if detection_steps else float("nan")
        ),
        "stop_rate": float(np.mean([bool(r["called_stop"]) for r in results])),
        "revisits_mean": mean("revisits"),
        "wall_time_s_mean": mean("wall_time_s"),
    }


def run_policy(
    env, cfg, policy_name: str, episodes: int, run_dir: Path, logger: RunLogger,
    save_observations: bool,
) -> dict:
    policy = make_policy(policy_name, cost_weight=cfg.policy.cost_weight)
    policy_dir = run_dir / policy_name
    policy_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    decisions_path = policy_dir / "decisions.jsonl"
    with decisions_path.open("w") as decisions_file:
        for index in range(episodes):
            writer = None
            if save_observations:
                # The episode is only known after reset, so pre-create nothing;
                # the explorer writes into a per-episode directory below.
                pass

            explorer = FrontierExplorer(env, cfg, policy, writer=writer)
            rng = episode_rng(int(cfg.seed.value), policy_name, index)
            result = explorer.run(rng)

            record = result.to_dict()
            record["policy"] = policy_name
            record["episode_index"] = index
            results.append(record)

            decisions_file.write(
                json.dumps(
                    {
                        "policy": policy_name,
                        "episode_index": index,
                        "episode_id": result.episode_id,
                        "scene_id": Path(result.scene_id).stem,
                        "decisions": result.decision_log,
                    }
                )
                + "\n"
            )
            decisions_file.flush()

            logger.log({"event": "episode_end", **record}, step=index)
            print(
                f"  [{policy_name}] ep {index:3d} "
                f"{str(record['object_category']):<12} "
                f"steps={record['steps']:4d} dec={record['decisions']:3d} "
                f"succ={record['success']:.0f} spl={record['spl']:.2f} "
                f"det={int(record['detected'])} col={record['collisions']:3d} "
                f"area={record['explored_area_m2']:6.1f} "
                f"plan_fail={record['planner_failures']}",
                flush=True,
            )

    summary = summarise(results)
    summary["policy"] = policy_name
    (policy_dir / "results.json").write_text(
        json.dumps({"summary": summary, "episodes": results}, indent=2, default=str)
    )
    return summary


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    policies = (
        sorted(POLICIES)
        if args.policy == "all"
        else [args.policy or str(cfg.policy.name)]
    )
    for name in policies:
        if name not in POLICIES:
            print(f"unknown policy {name!r}; have {sorted(POLICIES)}", file=sys.stderr)
            return 1

    seed_everything(int(cfg.seed.value), bool(cfg.seed.torch_deterministic))

    run_id = make_run_id(cfg.experiment.name, config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    print(f"run_dir: {run_dir}")
    print(f"policies: {policies}  episodes each: {args.episodes}")

    from frontierworld.habitat_env import make_env

    summaries = {}
    with RunLogger(run_dir, cfg.logging, run_id, to_dict(cfg)) as logger:
        provenance = environment_provenance()
        (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
        logger.log({"event": "run_start", **provenance})

        env = make_env(cfg)
        try:
            for name in policies:
                summaries[name] = run_policy(
                    env, cfg, name, args.episodes, run_dir, logger,
                    args.save_observations,
                )
                logger.log({"event": "policy_summary", **summaries[name]})
        finally:
            env.close()

        logger.write_summary({"policies": summaries, "config_hash": config_hash(cfg)})

    print("\n" + _format_table(summaries))
    (run_dir / "comparison.txt").write_text(_format_table(summaries))
    print(f"\nrun_dir: {run_dir}")
    return 0


def _format_table(summaries: dict) -> str:
    columns = [
        ("policy", "policy", "{:<22}"),
        ("SR", "success_rate", "{:>6.3f}"),
        ("SPL", "spl", "{:>6.3f}"),
        ("SoftSPL", "soft_spl", "{:>8.3f}"),
        ("DetRate", "detection_rate", "{:>8.3f}"),
        ("Coll", "collisions_mean", "{:>7.1f}"),
        ("Area m2", "explored_area_m2_mean", "{:>8.1f}"),
        ("Steps", "steps_mean", "{:>7.1f}"),
        ("Dec", "decisions_mean", "{:>6.1f}"),
        ("PlanFail", "planner_failures_total", "{:>9d}"),
    ]
    header = (
        "{:<22}".format("policy")
        + "".join(f"{name:>9}" for name, _, _ in columns[1:])
    )
    lines = [header, "-" * len(header)]
    for name, summary in summaries.items():
        row = "{:<22}".format(name)
        for _, key, fmt in columns[1:]:
            value = summary.get(key, float("nan"))
            try:
                row += fmt.format(value).rjust(9)
            except (ValueError, TypeError):
                row += "{:>9}".format("-")
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
