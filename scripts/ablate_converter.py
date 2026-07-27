#!/usr/bin/env python
"""Phase 9C: the registered converter ablation.

Everything below was fixed before the run. No architectural changes.

SUBSETS. Calibration and converter-validation scenes are drawn scene-disjoint
from dev_pool_v1. Coupled parameters (k_f, k_o, w_free, w_occ, tau_occ) are
tuned JOINTLY on calibration only -- they interact, so tuning them separately
would pick a threshold that suits a band width it was never paired with. Each
condition may calibrate its own threshold on calibration; nothing is adjusted
using converter-validation.

AREA CONSTRAINT (numerical, registered):

    R_A = sum_i A_hat_i / sum_i A_gt_i        in [0.90, 1.10]

Ratio of totals, not the mean of per-branch ratios: a branch revealing 0.2 m2
can produce a per-branch ratio of 10 and dominate an average. Median
per-branch ratio is reported as a secondary diagnostic.

SELECTION (registered): among configurations inside the interval, maximise
EXACT occupancy macro-IoU; tolerant occupied IoU is a tie-breaker only.
Fallback if none qualify: take the configuration closest to R_A = 1, then
maximise exact macro-IoU.

CONDITIONS are CUMULATIVE, so each row's marginal effect is interpretable.

PERFECT-DEPTH DIAGNOSTICS, both reported:
    gt_depth_original  ground-truth depth through the exact-depth converter
                       -- the ceiling
    gt_depth_robust    ground-truth depth through the robust converter with
                       sigma(d) -- the conservatism test, showing how much
                       clean geometry the uncertainty envelope sacrifices

BOOTSTRAP resamples decision groups, keeping every branch of a sampled group.

    python scripts/ablate_converter.py --calibration-scenes 6 --validation-scenes 6
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
from frontierworld.data.tensors import FrameSpec, FrontierFrame, resample_grid  # noqa: E402
from frontierworld.evaluation import make_run_id  # noqa: E402
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, MapGeometry  # noqa: E402
from frontierworld.mapping.robust_integration import (  # noqa: E402
    ErrorEnvelope,
    RobustIntegratorConfig,
    RobustOccupancyMap,
    fit_error_envelope,
)
from frontierworld.models.depth_estimator import (  # noqa: E402
    MetricDepthEstimator,
    apply_scale,
    first_frame_scale,
)
from frontierworld.models.metrics import masked_iou, paired_bootstrap  # noqa: E402
from frontierworld.models.video_to_structured import accumulate_rollout  # noqa: E402
from frontierworld.planning import make_policy  # noqa: E402
from frontierworld.seeding import episode_rng, seed_everything  # noqa: E402

AREA_LOW, AREA_HIGH = 0.90, 1.10  # registered before the run

# Registered NUMERICAL definition of "perfect-depth performance not materially
# damaged", written as a constant so the tolerance cannot become adjustable
# after the table is seen:
#
#   macroIoU(gt, robust) >= macroIoU(gt, original) - CLEAN_MACRO_TOLERANCE
#   R_A(gt, robust) in [AREA_LOW, AREA_HIGH]
#
# Absolute, not a fraction of the ceiling: a relative rule scales the allowance
# with the ceiling, so a strong ceiling would silently permit a larger loss.
CLEAN_MACRO_TOLERANCE = 0.05

BOOTSTRAP_SEED = 20260727

# Deterministic tie-breaking for the parameter search, in order:
#   1. inside the area interval beats outside
#   2. higher exact macro-IoU
#   3. higher tolerant occupied IoU  (registered tie-breaker ONLY)
#   4. lexicographic on (k_free, k_occupied, w_free, tau_free, tau_occ)
# Rule 4 makes the outcome reproducible when scores tie exactly.
TIE_BREAK_ORDER = ["inside_interval", "macro_iou", "occupied_iou_tolerant", "params"]

CONDITIONS = [
    "original",
    "+first_frame",
    "+conservative_carving",
    "+soft_occupied",
    "+confidence_weighting",
    "+rejection_filtering",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--pool", default="manifests/dev_pool_v1.json")
    parser.add_argument("--calibration-scenes", type=int, default=6)
    parser.add_argument("--validation-scenes", type=int, default=6)
    parser.add_argument("--episodes-per-scene", type=int, default=2)
    parser.add_argument("--decisions", type=int, default=2)
    parser.add_argument("--generator-frames", type=int, default=17)
    return parser.parse_args()


# -- branch collection -----------------------------------------------------


def collect_branches(cfg, scenes, episodes, decisions, n_frames, estimator, label):
    """Capture RGB, sensor depth, poses and the Phase 4 target per branch.

    Captured once and reused for every converter condition: re-simulating per
    condition would be wasteful and would let simulator nondeterminism leak
    into the comparison.
    """
    from frontierworld.habitat_env import make_env_with_dataset, select_episodes, sensor_extrinsics
    from frontierworld.mapping.occupancy import OccupancyMap
    from frontierworld.mapping.semantic import SemanticMap
    from frontierworld.planning.branching import (
        SimulatorSnapshot, execute_option, sim_observations,
    )
    from frontierworld.planning.exploration import EpisodeResult, FrontierExplorer
    from frontierworld.planning.options import build_options

    dataset, _ = select_episodes(cfg, 400, scenes=scenes, episodes_per_scene=episodes)
    env = make_env_with_dataset(cfg, dataset)
    policy = make_policy("nearest")
    spec = FrameSpec()
    branches = []

    try:
        for episode_index in range(len(dataset.episodes)):
            explorer = FrontierExplorer(env, cfg, policy)
            observations = env.reset()
            explorer._bind_to_scene()
            episode = env.current_episode
            explorer.detector.reset(episode)
            rng = episode_rng(int(cfg.seed.value), label, episode_index)

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

            for decision in range(decisions):
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
                conditioning = sim_observations(env.sim, cfg)
                scale = first_frame_scale(
                    estimator.predict(
                        np.asarray(conditioning["rgb"])[..., :3].astype(np.uint8)
                    ),
                    np.asarray(conditioning["depth"], dtype=np.float32),
                )
                group_id = f"{Path(result.scene_id).stem}_{result.episode_id}_d{decision}"

                for option in options:
                    execute_option(env.sim, option, occupancy, cfg)
                    grid_after = occupancy.to_grid()
                    reference = (grid_before == UNKNOWN) & (grid_after != UNKNOWN)
                    snapshot.restore(env.sim, occupancy, semantic_map)

                    agent = env.sim.get_agent(0)
                    rgb, depth, rotations, translations = [], [], [], []
                    for action in option.actions:
                        agent.act(action)
                        obs = sim_observations(env.sim, cfg)
                        rotation, translation = sensor_extrinsics(env.sim, "depth")
                        rgb.append(np.asarray(obs["rgb"])[..., :3].astype(np.uint8))
                        depth.append(np.asarray(obs["depth"], dtype=np.float32))
                        rotations.append(rotation)
                        translations.append(translation)
                    snapshot.restore(env.sim, occupancy, semantic_map)
                    if not rgb:
                        continue

                    index = np.linspace(0, len(rgb) - 1, min(n_frames, len(rgb))).round().astype(int)
                    frame = FrontierFrame(
                        centroid_world=np.asarray(option.frontier.centroid_world)[[0, 2]],
                        normal=np.asarray(option.frontier.orientation), spec=spec,
                    )
                    local_after, in_map = resample_grid(
                        grid_after, map_origin, float(cfg.mapping.resolution), frame, fill=UNKNOWN
                    )
                    local_reference, _ = resample_grid(
                        reference.astype(np.uint8), map_origin,
                        float(cfg.mapping.resolution), frame, fill=0
                    )
                    revealed = local_reference.astype(bool) & in_map

                    branches.append({
                        "group_id": group_id,
                        "scene": Path(result.scene_id).stem.replace(".basis", ""),
                        "frontier_id": option.frontier_id,
                        "rgb": np.stack([rgb[i] for i in index]),
                        "sensor_depth": np.stack([depth[i] for i in index]),
                        "rotations": np.stack([rotations[i] for i in index]),
                        "translations": np.stack([translations[i] for i in index]),
                        "scale_first_frame": scale,
                        "grid_before": grid_before,
                        "map_origin": map_origin,
                        "geometry": occupancy.geometry,
                        "floor_y": float(occupancy.floor_y),
                        "frame": frame,
                        "in_map": in_map,
                        "target": np.stack([
                            revealed & (local_after == FREE),
                            revealed & (local_after == OCCUPIED),
                        ]).astype(np.float32),
                        "area_gt_m2": float(reference.sum() * cfg.mapping.resolution ** 2),
                    })

                chosen = policy.select(frontiers, rng)
                actions = explorer.planner.try_plan(
                    chosen.approach_point(float(cfg.frontiers.approach_offset_m))
                )
                if actions is None:
                    break
                explorer._execute(actions, occupancy, result, int(cfg.episode.max_steps))
                explorer._face(chosen.yaw, occupancy, result, int(cfg.episode.max_steps))
            print(f"    {label}: {len(branches)} branches", end="\r", flush=True)
    finally:
        env.close()
    print()
    return branches


# -- conversion ------------------------------------------------------------


def run_condition(branch, condition, envelope, params, estimator, cfg, predicted_depth):
    """Convert one branch under one cumulative condition."""
    max_depth = float(cfg.simulator.max_depth)
    resolution = float(cfg.mapping.resolution)

    if condition == "gt_depth_original":
        depths, use_robust = branch["sensor_depth"], False
    elif condition == "gt_depth_robust":
        depths, use_robust = branch["sensor_depth"], True
    elif condition == "original":
        depths, use_robust = apply_scale(predicted_depth, 1.0, max_depth), False
    else:
        depths = apply_scale(predicted_depth, branch["scale_first_frame"], max_depth)
        use_robust = condition != "+first_frame"

    if not use_robust:
        rollout = accumulate_rollout(
            depths=depths, rotations=branch["rotations"],
            translations=branch["translations"], frame=branch["frame"],
            grid_before=branch["grid_before"], map_origin=branch["map_origin"],
            map_resolution=resolution, hfov_deg=float(cfg.simulator.hfov),
            max_depth=max_depth, floor_y=branch["floor_y"],
        )
        return rollout.as_target_tensor(), rollout.valid, rollout.revealed_area_m2

    index = CONDITIONS.index(condition) if condition in CONDITIONS else len(CONDITIONS)
    config = RobustIntegratorConfig(
        k_free=params["k_free"], k_occupied=params["k_occupied"],
        conservative_free=index >= CONDITIONS.index("+conservative_carving"),
        soft_occupied=index >= CONDITIONS.index("+soft_occupied"),
        confidence_weighting=index >= CONDITIONS.index("+confidence_weighting"),
        edge_filter=index >= CONDITIONS.index("+rejection_filtering"),
        evidence_weight=params["w_free"],
    )
    if condition == "gt_depth_robust":
        config = RobustIntegratorConfig(
            k_free=params["k_free"], k_occupied=params["k_occupied"],
            evidence_weight=params["w_free"],
        )

    mapper = RobustOccupancyMap(
        branch["geometry"], envelope, config, floor_y=branch["floor_y"]
    )
    for i in range(depths.shape[0]):
        mapper.integrate(
            depths[i], branch["rotations"][i], branch["translations"][i],
            branch["translations"][i], float(cfg.simulator.hfov), max_depth,
        )
    grid_after = mapper.to_grid(params["tau_free"], params["tau_occ"])
    newly = (branch["grid_before"] == UNKNOWN) & (grid_after != UNKNOWN)

    local_after, in_map = resample_grid(
        grid_after, branch["map_origin"], resolution, branch["frame"], fill=UNKNOWN
    )
    local_newly, _ = resample_grid(
        newly.astype(np.uint8), branch["map_origin"], resolution, branch["frame"], fill=0
    )
    revealed = local_newly.astype(bool) & in_map
    predicted = np.stack([
        revealed & (local_after == FREE),
        revealed & (local_after == OCCUPIED),
        np.zeros_like(revealed),
    ]).astype(np.float32)
    return predicted, in_map, float(newly.sum() * resolution ** 2)


def branch_hashes(branches) -> dict:
    """Hash the captured inputs, so the ablation is pinned to what it compared.

    Every condition consumes the SAME captured branches. Without recording
    their hashes, a rerun that happened to capture different branches would be
    indistinguishable from a converter difference.
    """
    import hashlib

    digest = hashlib.sha256()
    for branch in branches:
        for key in ("rgb", "sensor_depth", "rotations", "translations", "target"):
            digest.update(np.ascontiguousarray(branch[key]).tobytes())
        digest.update(branch["group_id"].encode())
    return {"n_branches": len(branches), "sha256": digest.hexdigest()[:32]}


def score(branches, predictions):
    """Aggregate metrics. Area ratio is the ratio of TOTALS, per the register."""
    from scipy import ndimage

    free_ious, occ_ious, occ_tol, precisions, recalls, per_branch_ratio = [], [], [], [], [], []
    total_predicted = total_gt = 0.0

    for branch, (predicted, valid, area) in zip(branches, predictions):
        window = branch["in_map"] & valid
        target = branch["target"]
        free_ious.append(masked_iou(predicted[0], target[0], window))
        occ_ious.append(masked_iou(predicted[1], target[1], window))

        p = (predicted[1] >= 0.5) & window
        a = (target[1] >= 0.5) & window
        if p.any() or a.any():
            occ_tol.append(
                ((p & ndimage.binary_dilation(a, iterations=2)).sum()
                 + (a & ndimage.binary_dilation(p, iterations=2)).sum())
                / max(p.sum() + a.sum(), 1)
            )
        precisions.append(float((p & a).sum() / p.sum()) if p.sum() else np.nan)
        recalls.append(float((p & a).sum() / a.sum()) if a.sum() else np.nan)

        total_predicted += area
        total_gt += branch["area_gt_m2"]
        if branch["area_gt_m2"] > 1e-6:
            per_branch_ratio.append(area / branch["area_gt_m2"])

    free = float(np.nanmean(free_ious))
    occupied = float(np.nanmean(occ_ious))
    return {
        "free_iou": free,
        "occupied_iou": occupied,
        "macro_iou": float(np.nanmean([free, occupied])),
        "occupied_iou_tolerant": float(np.nanmean(occ_tol)) if occ_tol else float("nan"),
        "occupied_precision": float(np.nanmean(precisions)),
        "occupied_recall": float(np.nanmean(recalls)),
        "area_ratio": total_predicted / total_gt if total_gt > 1e-6 else float("nan"),
        "area_ratio_median_per_branch": (
            float(np.median(per_branch_ratio)) if per_branch_ratio else float("nan")
        ),
        "n": len(branches),
        "per_branch_free_iou": free_ious,
        "per_branch_occ_iou": occ_ious,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    problems = check_data_paths(cfg)
    if problems:
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    seed_everything(int(cfg.seed.value), torch_deterministic=False)
    run_dir = Path(cfg.experiment.output_dir) / "phase9c" / make_run_id("ablation", "9c")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    from frontierworld.data.manifests import SplitManifest

    pool = SplitManifest.load(args.pool)
    all_scenes = sorted(pool.scene_ids("pool"))
    calibration_scenes = all_scenes[: args.calibration_scenes]
    validation_scenes = all_scenes[
        args.calibration_scenes : args.calibration_scenes + args.validation_scenes
    ]
    assert not set(calibration_scenes) & set(validation_scenes)

    estimator = MetricDepthEstimator(max_depth=float(cfg.simulator.max_depth))
    print(f"run_dir: {run_dir}")
    print(f"estimator: {estimator.model_name}")
    print(f"calibration scenes {len(calibration_scenes)} | "
          f"converter-validation scenes {len(validation_scenes)} | disjoint: True")
    print(f"registered area interval [{AREA_LOW}, {AREA_HIGH}], ratio of totals\n")

    print("collecting calibration branches...")
    calibration = collect_branches(
        cfg, calibration_scenes, args.episodes_per_scene, args.decisions,
        args.generator_frames, estimator, "calib",
    )
    print("collecting converter-validation branches...")
    validation = collect_branches(
        cfg, validation_scenes, args.episodes_per_scene, args.decisions,
        args.generator_frames, estimator, "val",
    )
    if not calibration or not validation:
        print("insufficient branches", file=sys.stderr)
        return 1

    # Predict depth once per branch and reuse.
    for group in (calibration, validation):
        for branch in group:
            branch["predicted_depth"] = estimator.predict_batch(branch["rgb"])

    # sigma(d_hat) from CALIBRATION residuals only, after first-frame scaling.
    predicted_all = np.concatenate([
        (b["predicted_depth"] * b["scale_first_frame"]).ravel() for b in calibration
    ])
    truth_all = np.concatenate([b["sensor_depth"].ravel() for b in calibration])
    envelope = fit_error_envelope(
        predicted_all, truth_all, max_depth=float(cfg.simulator.max_depth)
    )
    print(f"\nsigma(d) fitted on calibration: {np.round(envelope.sigma, 3).tolist()}")
    (run_dir / "error_envelope.json").write_text(json.dumps(envelope.to_dict(), indent=2))

    # ---- joint parameter search on CALIBRATION only ---------------------
    print("\njoint calibration of (k_f, k_o, w_free, tau_free, tau_occ)...")
    grid = [
        {"k_free": kf, "k_occupied": ko, "w_free": w, "tau_free": tf, "tau_occ": to}
        for kf in (0.5, 1.0)
        for ko in (0.5, 1.0)
        for w in (0.5, 1.0)
        for tf in (0.5,)
        for to in (0.15, 0.3, 0.5)
    ]
    grid_records = []
    for params in grid:
        predictions = [
            run_condition(b, "+rejection_filtering", envelope, params, estimator, cfg,
                          b["predicted_depth"])
            for b in calibration
        ]
        metrics = score(calibration, predictions)
        grid_records.append({
            "params": params,
            "inside_interval": bool(AREA_LOW <= metrics["area_ratio"] <= AREA_HIGH),
            "macro_iou": metrics["macro_iou"],
            "free_iou": metrics["free_iou"],
            "occupied_iou": metrics["occupied_iou"],
            "occupied_iou_tolerant": metrics["occupied_iou_tolerant"],
            "occupied_recall": metrics["occupied_recall"],
            "area_ratio": metrics["area_ratio"],
            "area_ratio_median_per_branch": metrics["area_ratio_median_per_branch"],
        })

    def sort_key(record):
        params = record["params"]
        return (
            not record["inside_interval"],
            -(record["macro_iou"] if np.isfinite(record["macro_iou"]) else -np.inf),
            -(record["occupied_iou_tolerant"]
              if np.isfinite(record["occupied_iou_tolerant"]) else -np.inf),
            (params["k_free"], params["k_occupied"], params["w_free"],
             params["tau_free"], params["tau_occ"]),
        )

    inside_any = [r for r in grid_records if r["inside_interval"]]
    if inside_any:
        ranked = sorted(inside_any, key=sort_key)
        fallback = False
    else:
        print("  no configuration inside the area interval; applying the "
              "registered fallback")
        ranked = sorted(
            grid_records,
            key=lambda r: (
                abs(r["area_ratio"] - 1.0) if np.isfinite(r["area_ratio"]) else np.inf,
                -(r["macro_iou"] if np.isfinite(r["macro_iou"]) else -np.inf),
            ),
        )
        fallback = True
    best = ranked[0]["params"]
    print(f"  {len(grid_records)} configurations scored, "
          f"{len(inside_any)} inside the area interval")
    print(f"  selected {best}")
    print(f"    calibration macro-IoU {ranked[0]['macro_iou']:.3f}  "
          f"area ratio {ranked[0]['area_ratio']:.3f}")
    (run_dir / "grid_calibration.json").write_text(json.dumps({
        "tie_break_order": TIE_BREAK_ORDER,
        "fallback_used": fallback,
        "configurations": grid_records,
        "ranking": [r["params"] for r in ranked],
        "selected": best,
    }, indent=2, default=str))

    # ---- cumulative ablation on CONVERTER-VALIDATION --------------------
    print("\ncumulative ablation on converter-validation:\n")
    header = (f"{'condition':<24}{'freeIoU':>9}{'occIoU':>8}{'macro':>8}"
              f"{'occTol':>8}{'occP':>7}{'occR':>7}{'areaR':>8}")
    print(header)
    print("-" * len(header))

    rows, per_branch = {}, {}
    for condition in CONDITIONS + ["gt_depth_original", "gt_depth_robust"]:
        predictions = [
            run_condition(b, condition, envelope, best, estimator, cfg, b["predicted_depth"])
            for b in validation
        ]
        metrics = score(validation, predictions)
        rows[condition] = {k: v for k, v in metrics.items() if not k.startswith("per_branch")}
        per_branch[condition] = metrics
        print(f"{condition:<24}{metrics['free_iou']:>9.3f}{metrics['occupied_iou']:>8.3f}"
              f"{metrics['macro_iou']:>8.3f}{metrics['occupied_iou_tolerant']:>8.3f}"
              f"{metrics['occupied_precision']:>7.3f}{metrics['occupied_recall']:>7.3f}"
              f"{metrics['area_ratio']:>8.3f}")

    # ---- bootstrap by decision group ------------------------------------
    group_ids = np.asarray([b["group_id"] for b in validation])
    reference = "+first_frame"
    full = "+rejection_filtering"
    comparison = {
        "occupied_iou": paired_bootstrap(
            np.asarray(per_branch[full]["per_branch_occ_iou"]),
            np.asarray(per_branch[reference]["per_branch_occ_iou"]), group_ids,
            seed=BOOTSTRAP_SEED,
        ),
        "free_iou": paired_bootstrap(
            np.asarray(per_branch[full]["per_branch_free_iou"]),
            np.asarray(per_branch[reference]["per_branch_free_iou"]), group_ids,
            seed=BOOTSTRAP_SEED,
        ),
    }
    print(f"\npaired bootstrap, full robust vs {reference} "
          f"({comparison['occupied_iou']['n_groups']} groups, "
          f"{comparison['occupied_iou']['n_branches']} branches):")
    for name, result in comparison.items():
        print(f"  {name:<14} {result['mean_difference']:+.3f} "
              f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}] "
              f"{'SIGNIFICANT' if result['significant'] else 'n.s.'}")

    # ---- registered gate -------------------------------------------------
    robust, baseline = rows[full], rows[reference]
    ceiling, conservatism = rows["gt_depth_original"], rows["gt_depth_robust"]
    area_ok = AREA_LOW <= robust["area_ratio"] <= AREA_HIGH
    macro_ok = robust["macro_iou"] > baseline["macro_iou"]
    recall_ok = robust["occupied_recall"] > baseline["occupied_recall"]
    clean_macro_ok = (
        conservatism["macro_iou"] >= ceiling["macro_iou"] - CLEAN_MACRO_TOLERANCE
    )
    clean_area_ok = AREA_LOW <= conservatism["area_ratio"] <= AREA_HIGH
    clean_ok = clean_macro_ok and clean_area_ok
    passed = area_ok and macro_ok and recall_ok and clean_ok

    print(f"\nregistered gate:")
    print(f"  area ratio {robust['area_ratio']:.3f} in [{AREA_LOW}, {AREA_HIGH}]: {area_ok}")
    print(f"  macro IoU {robust['macro_iou']:.3f} > {baseline['macro_iou']:.3f}: {macro_ok}")
    print(f"  occupied recall {robust['occupied_recall']:.3f} > "
          f"{baseline['occupied_recall']:.3f}: {recall_ok}")
    print(f"  perfect-depth macro {conservatism['macro_iou']:.3f} >= "
          f"{ceiling['macro_iou'] - CLEAN_MACRO_TOLERANCE:.3f} "
          f"(ceiling {ceiling['macro_iou']:.3f} - {CLEAN_MACRO_TOLERANCE}): {clean_macro_ok}")
    print(f"  perfect-depth area ratio {conservatism['area_ratio']:.3f} in "
          f"[{AREA_LOW}, {AREA_HIGH}]: {clean_area_ok}")
    print(f"GATE: {'PASS -- freeze this converter' if passed else 'FAIL -- freeze anyway and move to Cosmos'}")

    (run_dir / "ablation.json").write_text(json.dumps({
        "registered": {"area_interval": [AREA_LOW, AREA_HIGH],
                       "aggregation": "ratio_of_totals",
                       "clean_macro_tolerance": CLEAN_MACRO_TOLERANCE,
                       "bootstrap_seed": BOOTSTRAP_SEED,
                       "tie_break_order": TIE_BREAK_ORDER,
                       "selection": "max exact macro-IoU inside interval; "
                                    "tolerant occupied IoU tie-breaker only"},
        "branch_hashes": {
            "calibration": branch_hashes(calibration),
            "validation": branch_hashes(validation),
        },
        "group_ids": sorted(set(group_ids.tolist())),
        "grid_calibration": grid_records,
        "calibration_scenes": calibration_scenes,
        "validation_scenes": validation_scenes,
        "selected_params": best,
        "error_envelope": envelope.to_dict(),
        "rows": rows,
        "bootstrap": comparison,
        "gate": {"area_ok": area_ok, "macro_ok": macro_ok, "recall_ok": recall_ok,
                 "clean_ok": clean_ok, "passed": passed},
    }, indent=2, default=str))
    print(f"\nresults: {run_dir / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
