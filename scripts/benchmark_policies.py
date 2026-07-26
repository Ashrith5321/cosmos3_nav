#!/usr/bin/env python
"""Phase 2 gate: benchmark frontier policies in parallel across GPUs and CPUs.

Episodes are sharded over worker processes; each worker runs every policy on
its own shard, so all policies are scored on an identical episode list. Every
episode result is appended to episodes.jsonl the moment it finishes, so a long
run can be watched live:

    python scripts/benchmark_policies.py --episodes 50 --workers 12
    tail -f outputs/phase2/<run_id>/episodes.jsonl

    # live table while it runs
    python scripts/benchmark_policies.py --report outputs/phase2/<run_id>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
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
from frontierworld.planning.policies import POLICIES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=50, help="episodes per policy")
    parser.add_argument(
        "--policies",
        default="all",
        help="comma-separated policy names, or 'all'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="worker processes; 0 picks a value from CPU count and GPU memory",
    )
    parser.add_argument(
        "--gpus", default=None, help="comma-separated GPU ids, e.g. 0,1"
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="RUN_DIR",
        help="print the current table for a run directory and exit",
    )
    return parser.parse_args()


# -- metrics ---------------------------------------------------------------

METRIC_COLUMNS = [
    ("SR", "success", "{:>7.3f}"),
    ("SPL", "spl", "{:>7.3f}"),
    ("SoftSPL", "soft_spl", "{:>8.3f}"),
    ("DetRate", "detected", "{:>8.3f}"),
    ("Coll", "collisions", "{:>7.2f}"),
    ("Area m2", "explored_area_m2", "{:>8.1f}"),
    ("Steps", "steps", "{:>7.1f}"),
    ("Dec", "decisions", "{:>6.1f}"),
]


def summarise(records: list[dict]) -> dict:
    if not records:
        return {}

    def mean(key: str) -> float:
        values = [float(r[key]) for r in records if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    def confidence(key: str) -> float:
        """Half-width of the 95% interval; Phase 15 wants tighter intervals."""
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if len(values) < 2:
            return float("nan")
        return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))

    detection_steps = [
        r["detection_step"] for r in records if r.get("detection_step") is not None
    ]
    summary = {
        "episodes": len(records),
        "planner_failures_total": int(sum(r.get("planner_failures", 0) for r in records)),
        "planner_failure_episodes": int(
            sum(1 for r in records if r.get("planner_failures", 0) > 0)
        ),
        "frontier_exhaustion_episodes": int(
            sum(1 for r in records if r.get("frontier_exhaustions", 0) > 0)
        ),
        "steps_to_detection_mean": (
            float(np.mean(detection_steps)) if detection_steps else float("nan")
        ),
        "stop_rate": mean("called_stop"),
        "revisits_mean": mean("revisits"),
        "wall_time_s_mean": mean("wall_time_s"),
    }
    for _, key, _ in METRIC_COLUMNS:
        summary[key] = mean(key)
        summary[f"{key}_ci95"] = confidence(key)
    return summary


def format_table(by_policy: dict[str, list[dict]]) -> str:
    header = "{:<22}{:>5}".format("policy", "n") + "".join(
        f"{name:>9}" for name, _, _ in METRIC_COLUMNS
    ) + "{:>10}".format("PlanFail")
    lines = [header, "-" * len(header)]
    for name in sorted(by_policy):
        records = by_policy[name]
        summary = summarise(records)
        row = "{:<22}{:>5d}".format(name, len(records))
        for _, key, fmt in METRIC_COLUMNS:
            row += fmt.format(summary.get(key, float("nan"))).rjust(9)
        row += "{:>10d}".format(summary.get("planner_failures_total", 0))
        lines.append(row)
    return "\n".join(lines)


def load_records(run_dir: Path) -> dict[str, list[dict]]:
    by_policy: dict[str, list[dict]] = defaultdict(list)
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        return by_policy
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially flushed final line
            by_policy[record["policy"]].append(record)
    return by_policy


# -- worker ----------------------------------------------------------------


def worker(task: dict) -> dict:
    """Run every policy over one shard of episodes in a dedicated process."""
    shard = task["shard"]
    num_shards = task["num_shards"]
    gpu = task["gpu"]

    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from frontierworld.config import load_config
    from frontierworld.habitat_env import make_env_with_dataset, select_episodes
    from frontierworld.planning import FrontierExplorer, make_policy
    from frontierworld.seeding import episode_rng, seed_everything

    cfg = load_config(task["config"], task["overrides"])
    seed_everything(int(cfg.seed.value), torch_deterministic=False)

    dataset, keys = select_episodes(
        cfg, task["episodes"], shard=shard, num_shards=num_shards
    )
    if not keys:
        return {"shard": shard, "episodes": 0, "error": None}

    run_dir = Path(task["run_dir"])
    results_path = run_dir / "shards" / f"shard_{shard:03d}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env_with_dataset(cfg, dataset, gpu_device_id=gpu)
    written = 0
    error = None
    try:
        with results_path.open("w") as handle:
            for policy_name in task["policies"]:
                policy = make_policy(policy_name, cost_weight=cfg.policy.cost_weight)
                for index in range(len(keys)):
                    explorer = FrontierExplorer(env, cfg, policy)
                    rng = episode_rng(int(cfg.seed.value), policy_name, f"{shard}:{index}")
                    result = explorer.run(rng)

                    record = result.to_dict()
                    record.update(
                        {
                            "policy": policy_name,
                            "shard": shard,
                            "episode_index": index,
                            "gpu": gpu,
                            "scene": Path(result.scene_id).stem.replace(".basis", ""),
                        }
                    )
                    handle.write(json.dumps(record, default=str) + "\n")
                    handle.flush()
                    written += 1
    except Exception as exc:  # noqa: BLE001 - a dead worker must not kill the run
        import traceback

        error = f"{exc}\n{traceback.format_exc()}"
    finally:
        env.close()

    return {"shard": shard, "episodes": written, "error": error}


# -- driver ----------------------------------------------------------------


def merge_shards(run_dir: Path) -> int:
    """Concatenate shard logs into episodes.jsonl."""
    merged = run_dir / "episodes.jsonl"
    count = 0
    with merged.open("w") as out:
        for shard_file in sorted((run_dir / "shards").glob("shard_*.jsonl")):
            with shard_file.open() as handle:
                for line in handle:
                    if line.strip():
                        out.write(line)
                        count += 1
    return count


def choose_workers(requested: int, gpus: list[int]) -> int:
    if requested > 0:
        return requested
    cpu_workers = max(1, (os.cpu_count() or 4) - 4)
    # Habitat holds roughly 2.5 GB of GPU memory per simulator instance.
    gpu_workers = 8 * len(gpus)
    return int(min(cpu_workers, gpu_workers, 16))


def detect_gpus() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001
        pass
    return [0]


def main() -> int:
    args = parse_args()

    if args.report:
        run_dir = Path(args.report)
        by_policy = load_records(run_dir)
        if not by_policy:
            merge_shards(run_dir)
            by_policy = load_records(run_dir)
        print(format_table(by_policy))
        return 0

    cfg = load_config(args.config, args.override)
    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    policies = (
        sorted(POLICIES) if args.policies == "all" else args.policies.split(",")
    )
    for name in policies:
        if name not in POLICIES:
            print(f"unknown policy {name!r}; have {sorted(POLICIES)}", file=sys.stderr)
            return 1

    gpus = (
        [int(g) for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    )
    workers = choose_workers(args.workers, gpus)
    workers = min(workers, max(1, args.episodes))

    run_id = make_run_id(cfg.experiment.name, config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name / run_id
    (run_dir / "shards").mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(
        json.dumps(environment_provenance(), indent=2)
    )

    print(f"run_dir : {run_dir}")
    print(f"policies: {policies}")
    print(f"episodes: {args.episodes} per policy")
    print(f"workers : {workers} across GPUs {gpus}")
    print(f"total   : {args.episodes * len(policies)} episode runs")
    print(f"\nwatch:  tail -f {run_dir}/episodes.jsonl")
    print(f"table:  python scripts/benchmark_policies.py --report {run_dir}\n")

    tasks = [
        {
            "shard": shard,
            "num_shards": workers,
            "gpu": gpus[shard % len(gpus)],
            "config": args.config,
            "overrides": list(args.override),
            "episodes": args.episodes,
            "policies": policies,
            "run_dir": str(run_dir),
        }
        for shard in range(workers)
    ]

    started = time.perf_counter()
    context = mp.get_context("spawn")
    completed = 0
    errors = []
    with context.Pool(processes=workers) as pool:
        for outcome in pool.imap_unordered(worker, tasks):
            completed += 1
            status = "ok" if not outcome["error"] else "FAILED"
            print(
                f"[{completed}/{workers}] shard {outcome['shard']:3d} {status} "
                f"({outcome['episodes']} episode runs)",
                flush=True,
            )
            if outcome["error"]:
                errors.append(outcome)

    elapsed = time.perf_counter() - started
    total = merge_shards(run_dir)
    by_policy = load_records(run_dir)

    table = format_table(by_policy)
    summaries = {name: summarise(records) for name, records in by_policy.items()}
    (run_dir / "comparison.txt").write_text(table)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "policies": summaries,
                "episode_runs": total,
                "wall_time_s": elapsed,
                "workers": workers,
                "gpus": gpus,
                "errors": errors,
            },
            indent=2,
            default=str,
        )
    )

    print(f"\n{table}\n")
    print(f"{total} episode runs in {elapsed / 60:.1f} min "
          f"({elapsed / max(1, total):.1f} s per run, {workers} workers)")
    if errors:
        print(f"\n{len(errors)} shard(s) failed:", file=sys.stderr)
        for outcome in errors:
            print(f"  shard {outcome['shard']}: {outcome['error']}", file=sys.stderr)
        return 1
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
