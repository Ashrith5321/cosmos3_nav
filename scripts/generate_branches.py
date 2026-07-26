#!/usr/bin/env python
"""Phase 5: generate the branch-complete FrontierReveal dataset.

Collects decision states under several exploration policies, then executes
*every* valid frontier option from each state as an independent branch,
restoring the simulator between branches. Sharded across processes and GPUs.

    python scripts/generate_branches.py --groups 500 --workers 16
    python scripts/generate_branches.py --groups 20 --workers 2

Acceptance rule for a decision state: 2 <= |F_t| <= max_frontiers, the target
is not already visible, and the state is not a near-duplicate of the previous
one.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
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
from frontierworld.evaluation.run_logger import _git  # noqa: E402

COLLECTION_POLICIES = ["nearest", "max_info_gain", "random", "info_gain_minus_cost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--groups", type=int, default=500, help="target decision groups")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--manifest", default=None, help="path to a split manifest")
    parser.add_argument("--split", default="train", help="manifest split to draw from")
    parser.add_argument("--out", default=None, help="dataset root")
    parser.add_argument(
        "--keep-frames", action="store_true", help="retain per-step RGB-D (large)"
    )
    return parser.parse_args()


# -- worker ----------------------------------------------------------------


def worker(task: dict) -> dict:
    """Collect decision states and execute all their branches, on one shard."""
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import numpy as np

    from frontierworld.config import config_hash, load_config
    from frontierworld.data.records import RevelationWriter
    from frontierworld.habitat_env import (
        make_env_with_dataset,
        select_episodes,
        semantic_id_to_category,
    )
    from frontierworld.mapping.occupancy import OccupancyMap
    from frontierworld.mapping.semantic import SemanticMap
    from frontierworld.planning import make_policy
    from frontierworld.planning.exploration import EpisodeResult, FrontierExplorer
    from frontierworld.seeding import episode_rng, seed_everything

    shard = task["shard"]
    cfg = load_config(task["config"], task["overrides"])
    seed = int(cfg.seed.value) + shard
    seed_everything(seed, torch_deterministic=False)

    allowed = set(task["scenes"]) if task["scenes"] else None
    dataset, _ = select_episodes(cfg, task["episode_pool"], shard=0, num_shards=1)
    if allowed:
        dataset.episodes = [e for e in dataset.episodes if str(e.scene_id) in allowed]
    dataset.episodes = dataset.episodes[shard :: task["num_shards"]]
    if not dataset.episodes:
        return {"shard": shard, "groups": 0, "branches": 0, "rejected": {}, "error": None}

    writer = RevelationWriter(Path(task["out"]) / "shards" / f"shard_{shard:03d}")
    env = make_env_with_dataset(cfg, dataset, gpu_device_id=task["gpu"])

    n_groups = n_branches = 0
    rejected: dict[str, int] = {}
    error = None
    target_groups = task["groups_per_shard"]
    max_frontiers = int(cfg.dataset.max_frontiers)
    cfg_hash = config_hash(cfg)

    def note(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    try:
        for episode_index in range(task["episodes_per_shard"]):
            if n_groups >= target_groups:
                break

            policy_name = COLLECTION_POLICIES[episode_index % len(COLLECTION_POLICIES)]
            policy = make_policy(policy_name, cost_weight=cfg.policy.cost_weight)
            explorer = FrontierExplorer(env, cfg, policy)
            rng = episode_rng(seed, "phase5", f"{shard}:{episode_index}")

            observations = env.reset()
            explorer._bind_to_scene()
            episode = env.current_episode
            explorer.detector.reset(episode)
            instance_to_category = semantic_id_to_category(env.sim)
            target_ids = set(explorer.detector.target_instance_ids)

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
            semantic_map = SemanticMap(occupancy.geometry, floor_y=float(start[1]))

            result = EpisodeResult(
                scene_id=str(episode.scene_id),
                episode_id=str(episode.episode_id),
                object_category=getattr(episode, "object_category", None),
            )
            explorer._last_observations = observations
            explorer._target_position = None
            explorer._integrate(observations, occupancy)
            explorer._scan(occupancy, result, int(cfg.episode.max_steps))

            previous_position = None
            for _ in range(int(cfg.dataset.decisions_per_episode)):
                if n_groups >= target_groups:
                    break

                frontiers = [f for f in explorer._frontiers(occupancy) if f.reachable]
                accepted = False

                if len(frontiers) < 2:
                    note("fewer than 2 reachable frontiers")
                elif len(frontiers) > max_frontiers:
                    note(f"more than {max_frontiers} frontiers")
                else:
                    position = np.asarray(env.sim.get_agent_state().position)
                    duplicate = (
                        previous_position is not None
                        and float(np.linalg.norm(position - previous_position))
                        < float(cfg.dataset.min_state_separation_m)
                    )
                    explorer._check_detection(
                        explorer._last_observations, occupancy, result
                    )
                    if result.detected:
                        note("target already visible")
                    elif duplicate:
                        note("near-duplicate of previous state")
                    else:
                        written = emit_group(
                            env=env, cfg=cfg, explorer=explorer, occupancy=occupancy,
                            semantic_map=semantic_map, result=result,
                            instance_to_category=instance_to_category,
                            target_ids=target_ids, episode=episode, writer=writer,
                            policy_name=policy_name, seed=seed,
                            commit=task["git_commit"], cfg_hash=cfg_hash,
                            keep_frames=task["keep_frames"],
                            max_frontiers=max_frontiers, frontiers=frontiers,
                        )
                        if written is None:
                            note("fewer than 2 valid options")
                        else:
                            n_groups += 1
                            n_branches += written
                            previous_position = position
                            accepted = True

                del accepted
                if not frontiers:
                    break
                chosen = policy.select(frontiers, rng)
                if chosen is None:
                    break
                actions = explorer.planner.try_plan(
                    chosen.approach_point(float(cfg.frontiers.approach_offset_m))
                )
                if actions is None:
                    continue
                explorer._execute(actions, occupancy, result, int(cfg.episode.max_steps))
                explorer._face(chosen.yaw, occupancy, result, int(cfg.episode.max_steps))
    except Exception as exc:  # noqa: BLE001 - one dead shard must not kill the run
        import traceback

        error = f"{exc}\n{traceback.format_exc()}"
    finally:
        env.close()
        writer.close()

    return {
        "shard": shard,
        "groups": n_groups,
        "branches": n_branches,
        "rejected": rejected,
        "error": error,
    }


def emit_group(
    *, env, cfg, explorer, occupancy, semantic_map, result, instance_to_category,
    target_ids, episode, writer, policy_name, seed, commit, cfg_hash, keep_frames,
    max_frontiers, frontiers,
) -> int | None:
    """Execute every valid option from the current state and write the group."""
    import numpy as np

    from frontierworld.data.records import RevelationExample, array_hash
    from frontierworld.data.revelation import compute_revelation
    from frontierworld.planning.branching import (
        SimulatorSnapshot,
        execute_option,
        sim_observations,
    )
    from frontierworld.planning.options import build_options

    options = [
        o
        for o in build_options(
            frontiers, explorer.planner, occupancy, env.sim, cfg,
            explorer._calibrate_turn_sign(),
        )
        if o.valid
    ][:max_frontiers]
    if len(options) < 2:
        return None

    grid_before = occupancy.to_grid()
    semantic_before = semantic_map.copy_counts()
    observations_before = sim_observations(env.sim, cfg)
    rgb_before = np.asarray(observations_before["rgb"])[..., :3]
    instances_before = {
        int(v)
        for v in np.unique(np.squeeze(np.asarray(observations_before["semantic"])))
    }
    snapshot = SimulatorSnapshot.capture(env.sim, occupancy, semantic_map)
    start_position = [float(v) for v in snapshot.agent_state.position]

    # Recorded on every example so a reader can verify after the fact that all
    # branches really did start from one state.
    snapshot_hash = array_hash(np.asarray(start_position))
    map_hash = array_hash(grid_before, semantic_before)

    geodesic_before = geodesic_to_goals(explorer.planner, episode)
    scene = Path(result.scene_id).stem.replace(".basis", "")
    group_id = f"{scene}_{result.episode_id}_t{result.steps}_{policy_name}"

    # Per-branch target arrays. The revelation record holds scalar summaries,
    # but Phase 7's targets are the spatial maps themselves -- a dataset with
    # only areas cannot train the occupancy or semantic heads. Masks are packed
    # to bits; a mostly-empty 800x800 boolean costs a few KB packed.
    target_arrays: dict[str, np.ndarray] = {}

    examples = []
    for option in options:
        branch = execute_option(
            env.sim, option, occupancy, cfg,
            semantic_map=semantic_map,
            instance_to_category=instance_to_category,
            target_instance_ids=target_ids,
            keep_frames=keep_frames,
        )
        grid_after = occupancy.to_grid()
        semantic_after = semantic_map.copy_counts()
        revelation, masks = compute_revelation(
            frontier_id=option.frontier_id,
            grid_before=grid_before,
            grid_after=grid_after,
            semantic_before=semantic_before,
            semantic_after=semantic_after,
            semantic_map=semantic_map,
            frontiers_before=frontiers,
            frontiers_after=explorer._frontiers(occupancy),
            branch=branch,
            resolution=float(cfg.mapping.resolution),
            # Room category is excluded: HM3D-v0.2 ships no room labels and the
            # derived label proved unreliable in Phase 4.
            room_labeller=None,
            target_detection={
                "seen": branch.target_seen_at_step is not None,
                "pixel_count": branch.target_pixel_count,
                "step": branch.target_seen_at_step,
            },
            distance_to_target_before=geodesic_before,
            geodesic_to_target=geodesic_to_goals(explorer.planner, episode),
            instances_before=instances_before,
        )

        fid = option.frontier_id
        target_arrays[f"revealed_mask_f{fid}"] = np.packbits(masks["newly_observed"])
        target_arrays[f"semantic_revealed_mask_f{fid}"] = np.packbits(
            masks["newly_semantic"]
        )
        # Free vs occupied within the revealed region: the occupancy head has
        # to distinguish "there is space there" from "there is a wall there".
        target_arrays[f"grid_after_f{fid}"] = grid_after
        target_arrays[f"semantic_after_f{fid}"] = semantic_after

        examples.append(
            RevelationExample(
                scene_id=result.scene_id,
                episode_id=result.episode_id,
                decision_timestep=result.steps,
                navigation_goal=result.object_category,
                frontier_geometry={
                    **option.frontier.to_dict(),
                    # Complete boundary components, not only the centroid.
                    "boundary_cells": option.frontier.cells.tolist(),
                },
                candidate_option=option.to_dict(),
                future_revelation=revelation.to_dict(),
                current_observation={"rgb": "rgb_before.png"},
                current_map={"arrays": "arrays.npz", "key": "grid_before"},
                target_arrays={
                    "revealed_mask": f"revealed_mask_f{option.frontier_id}",
                    "semantic_revealed_mask": f"semantic_revealed_mask_f{option.frontier_id}",
                    "grid_after": f"grid_after_f{option.frontier_id}",
                    "semantic_after": f"semantic_after_f{option.frontier_id}",
                    "packed_shape": list(grid_before.shape),
                },
                observation_history={
                    "steps_before_decision": result.steps,
                    "branch_start_position": start_position,
                    "target_already_visible": False,
                    "trajectory": branch.trajectory,
                },
                privileged={
                    "planner": "habitat_navmesh (ground-truth navmesh)",
                    "goal_detector": "oracle ground-truth semantic instance ids",
                    "room_category": "EXCLUDED (HM3D room labels unreliable)",
                    "success_distance_m": float(cfg.task.success_distance),
                },
                detector_type="geometric",
                git_commit=commit,
                config_hash=cfg_hash,
                collection_policy=policy_name,
                rng_seed=seed,
                snapshot_hash=snapshot_hash,
                map_hash=map_hash,
            )
        )
        snapshot.restore(env.sim, occupancy, semantic_map)

    # Verify the restore actually put us back, before writing anything.
    restored = array_hash(occupancy.to_grid(), semantic_map.copy_counts())
    if restored != map_hash:
        raise RuntimeError(
            f"{group_id}: map not restored after branching ({restored} != {map_hash})"
        )

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
            **target_arrays,
        },
        rgb={"rgb_before": rgb_before},
    )
    return len(examples)


def geodesic_to_goals(planner, episode) -> float:
    import numpy as np

    best = float("inf")
    for goal in getattr(episode, "goals", []) or []:
        position = getattr(goal, "position", None)
        if position is None:
            continue
        best = min(
            best, planner.geodesic_distance(np.asarray(position, dtype=np.float64))
        )
    return best


# -- driver ----------------------------------------------------------------


def merge_shards(out: Path) -> tuple[int, int]:
    """Concatenate shard datasets into one index at the dataset root."""
    groups = examples = 0
    with (out / "index.jsonl").open("w") as handle:
        for shard_dir in sorted((out / "shards").glob("shard_*")):
            shard_index = shard_dir / "index.jsonl"
            if not shard_index.exists():
                continue
            for line in shard_index.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                entry["path"] = str((shard_dir / entry["path"]).relative_to(out))
                handle.write(json.dumps(entry) + "\n")
                groups += 1
                examples += entry.get("n_examples", 0)
    return groups, examples


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    problems = check_data_paths(cfg)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    scenes: list[str] = []
    manifest = None
    if args.manifest:
        from frontierworld.data.manifests import SplitManifest

        manifest = SplitManifest.load(args.manifest)
        scenes = manifest.scene_ids(args.split)
        print(f"manifest: {manifest.name} split={args.split} scenes={len(scenes)}")
        if manifest.pilot:
            print(f"  PILOT DATASET -- {manifest.pilot_reason}")
        if not scenes:
            print(f"no scenes in split {args.split!r}", file=sys.stderr)
            return 1

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    workers = args.workers or min(16, max(1, (os.cpu_count() or 4) - 4))
    workers = min(workers, max(1, args.groups))

    run_id = make_run_id("frontierreveal", config_hash(cfg))
    out = (
        Path(args.out)
        if args.out
        else Path(cfg.experiment.output_dir) / "phase5" / run_id
    )
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")
    (out / "provenance.json").write_text(json.dumps(environment_provenance(), indent=2))
    if manifest:
        manifest.save(out / "manifest.json")

    print(f"out     : {out}")
    print(f"target  : {args.groups} decision groups")
    print(f"workers : {workers} across GPUs {gpus}")
    print(f"policies: {COLLECTION_POLICIES}\n")

    groups_per_shard = int(np.ceil(args.groups / workers))
    tasks = [
        {
            "shard": shard,
            "num_shards": workers,
            "gpu": gpus[shard % len(gpus)],
            "config": args.config,
            "overrides": list(args.override),
            "scenes": scenes,
            "episode_pool": int(cfg.dataset.episode_pool),
            "episodes_per_shard": int(cfg.dataset.episodes_per_shard),
            "groups_per_shard": groups_per_shard,
            "out": str(out),
            "keep_frames": bool(args.keep_frames),
            "git_commit": _git("rev-parse", "HEAD"),
        }
        for shard in range(workers)
    ]

    started = time.perf_counter()
    completed = 0
    errors = []
    rejected: dict[str, int] = {}

    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for outcome in pool.imap_unordered(worker, tasks):
            completed += 1
            for reason, count in (outcome.get("rejected") or {}).items():
                rejected[reason] = rejected.get(reason, 0) + count
            status = "ok" if not outcome["error"] else "FAILED"
            print(
                f"[{completed}/{workers}] shard {outcome['shard']:3d} {status}  "
                f"{outcome['groups']} groups, {outcome['branches']} branches",
                flush=True,
            )
            if outcome["error"]:
                errors.append(outcome)

    elapsed = time.perf_counter() - started
    groups, examples = merge_shards(out)

    (out / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "groups": groups,
                "branches": examples,
                "wall_time_s": elapsed,
                "workers": workers,
                "policies": COLLECTION_POLICIES,
                "manifest": manifest.name if manifest else None,
                "split": args.split if manifest else None,
                "pilot": bool(manifest.pilot) if manifest else None,
                "rejected_states": rejected,
                "errors": errors,
            },
            indent=2,
            default=str,
        )
    )

    print(f"\n{groups} decision groups, {examples} branches in {elapsed/60:.1f} min")
    if rejected:
        print("rejected decision states:")
        for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {count:5d}  {reason}")
    if errors:
        print(f"\n{len(errors)} shard(s) failed", file=sys.stderr)
        for outcome in errors[:2]:
            print(f"  shard {outcome['shard']}: {outcome['error'][:700]}", file=sys.stderr)
        return 1
    print(f"\ndataset: {out}")
    print(f"next: python scripts/check_dataset.py {out}")
    return 0


def detect_gpus() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001
        pass
    return [0]


if __name__ == "__main__":
    raise SystemExit(main())
