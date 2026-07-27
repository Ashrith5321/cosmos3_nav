"""Phase 10 (10.2-10.5): frozen deep ensemble, uncertainty, calibration, evaluation.

The three archived Phase 8 seeds are loaded and never updated. The only fitted
quantities are calibration parameters, and they see calibration scenes only.
Validation scenes are touched once, at the end, to produce the table.

Order matters and is enforced by the structure of `main`:

    predict(calibration) -> fit calibrators -> predict(validation) -> evaluate

Per-seed logits, probabilities and outputs are all stored, not just their mean,
so the ensemble can be re-analysed without re-running the models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frontierworld.config import load_config
from frontierworld.data.dataset import FrontierRevealDataset
from frontierworld.data.tensors import (
    FrameSpec,
    TGT_REVEALED_FREE,
    TGT_REVEALED_OCCUPIED,
    TGT_REVEALED_SEMANTIC,
    build_group_tensors,
)
from frontierworld.models.ensemble import (
    ConformalAreaInterval,
    PlattScaler,
    TemperatureScaler,
    brier,
    ensemble_mean,
    expected_calibration_error,
    mutual_information,
    negative_log_likelihood,
    pairwise_cosine_disagreement,
    predictive_entropy,
    ranking_stability,
    risk_coverage_curve,
    seed_variance,
    spearman,
)
from frontierworld.models.metrics import auprc, auroc, paired_bootstrap
from frontierworld.models.predictor import RevelationPredictor, goal_index

CHANNEL_NAMES = {
    TGT_REVEALED_FREE: "free",
    TGT_REVEALED_OCCUPIED: "occupied",
    TGT_REVEALED_SEMANTIC: "semantic",
}


def load_seeds(paths: list[Path], device: torch.device) -> list[RevelationPredictor]:
    models = []
    for path in paths:
        payload = torch.load(path, map_location=device)
        state = payload.get("model", payload.get("state_dict", payload))
        model = RevelationPredictor().to(device)
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    return models


@torch.no_grad()
def predict_split(
    models: list[RevelationPredictor], dataset: FrontierRevealDataset, device: torch.device
) -> dict:
    """Run every seed over every branch. Returns per-seed arrays, no averaging."""
    spec = FrameSpec()
    n_seeds = len(models)

    spatial_logits: list[np.ndarray] = []
    target_logits: list[np.ndarray] = []
    crossing_logits: list[np.ndarray] = []
    areas: list[np.ndarray] = []

    truth = {"free": [], "occupied": [], "semantic": [], "valid": [],
             "target": [], "crossing": [], "area": []}
    group_ids: list[str] = []
    scene_ids: list[str] = []

    for index in range(len(dataset)):
        group = build_group_tensors(dataset[index], spec)
        n = int(group["candidate_mask"].sum())
        if n == 0:
            continue

        inputs = torch.from_numpy(group["inputs"][:n]).to(device)
        options = torch.from_numpy(group["options"][:n]).to(device)
        goal = torch.full((n,), goal_index(group["goal"]), dtype=torch.long, device=device)

        seed_spatial, seed_target, seed_crossing, seed_area = [], [], [], []
        for model in models:
            output = model(inputs, options, goal)
            seed_spatial.append(output["revealed_logits"].float().cpu().numpy())
            seed_target.append(output["target_logit"].float().cpu().numpy())
            seed_crossing.append(output["crossing_logit"].float().cpu().numpy())
            seed_area.append(output["area"].float().cpu().numpy())

        spatial_logits.append(np.stack(seed_spatial))      # (S, n, T, H, W)
        target_logits.append(np.stack(seed_target))        # (S, n)
        crossing_logits.append(np.stack(seed_crossing))
        areas.append(np.stack(seed_area))

        targets = group["targets"][:n]
        truth["free"].append(targets[:, TGT_REVEALED_FREE])
        truth["occupied"].append(targets[:, TGT_REVEALED_OCCUPIED])
        truth["semantic"].append(targets[:, TGT_REVEALED_SEMANTIC])
        truth["valid"].append(group["target_valid"][:n])
        truth["target"].append(group["scalars"]["target_present"][:n])
        truth["crossing"].append(group["scalars"]["crossing_success"][:n])
        truth["area"].append(group["scalars"]["revealed_area_m2"][:n])

        group_ids.extend([group["group_id"]] * n)
        scene_ids.extend([group["scene_id"]] * n)

    if not group_ids:
        raise SystemExit("no usable branches in split")

    concat = lambda parts: np.concatenate(parts, axis=1)  # noqa: E731 - seed axis is 0
    return {
        "n_seeds": n_seeds,
        "spatial_logits": concat(spatial_logits),
        "target_logits": concat(target_logits),
        "crossing_logits": concat(crossing_logits),
        "areas": concat(areas),
        "truth": {k: np.concatenate(v, axis=0) for k, v in truth.items()},
        "group_ids": np.asarray(group_ids),
        "scene_ids": np.asarray(scene_ids),
        "n_branches": len(group_ids),
        "n_groups": len(set(group_ids)),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def iou(probability: np.ndarray, labels: np.ndarray, valid: np.ndarray, threshold: float = 0.5) -> float:
    predicted = (probability >= threshold) & valid
    actual = (labels > 0.5) & valid
    union = (predicted | actual).sum()
    return float((predicted & actual).sum() / union) if union else float("nan")


def occupancy_block(seed_logits: np.ndarray, truth: dict, channel: int, calibrator=None) -> dict:
    """Metrics and uncertainty for one spatial channel, over valid cells only."""
    name = CHANNEL_NAMES[channel]
    valid = truth["valid"].astype(bool)
    labels = truth[name]

    seed_probabilities = sigmoid(seed_logits[:, :, channel])
    if calibrator is not None:
        seed_probabilities = np.stack(
            [calibrator.transform(seed_logits[s, :, channel]) for s in range(seed_logits.shape[0])]
        )
    mean = ensemble_mean(seed_probabilities)

    flat_valid = valid.ravel()
    p = mean.ravel()[flat_valid]
    y = labels.ravel()[flat_valid]

    return {
        "iou": iou(mean, labels, valid),
        "brier": brier(p, y),
        "nll": negative_log_likelihood(p, y),
        "ece": expected_calibration_error(p, y),
        "positive_rate": float(y.mean()),
        "n_valid_cells": int(flat_valid.sum()),
        # Per-branch uncertainty: mean over the branch's valid cells.
        "_per_branch_entropy": _per_branch(predictive_entropy(seed_probabilities), valid),
        "_per_branch_mi": _per_branch(mutual_information(seed_probabilities), valid),
        "_per_branch_variance": _per_branch(seed_variance(seed_probabilities), valid),
        "_per_branch_error": _per_branch_error(mean, labels, valid),
    }


def _per_branch(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    counts = valid.reshape(valid.shape[0], -1).sum(axis=1)
    totals = (values.reshape(values.shape[0], -1) * valid.reshape(valid.shape[0], -1)).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def _per_branch_error(mean: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    error = (mean - labels) ** 2
    return _per_branch(error, valid)


def scalar_block(
    seed_logits: np.ndarray, labels: np.ndarray, calibrator=None, name: str = ""
) -> dict:
    seed_probabilities = (
        np.stack([calibrator.transform(seed_logits[s]) for s in range(seed_logits.shape[0])])
        if calibrator is not None
        else sigmoid(seed_logits)
    )
    mean = ensemble_mean(seed_probabilities)
    labels = np.asarray(labels, dtype=np.float64)

    block = {
        "brier": brier(mean, labels),
        "nll": negative_log_likelihood(mean, labels),
        "ece": expected_calibration_error(mean, labels),
        "positive_rate": float(labels.mean()),
        "accuracy": float(((mean >= 0.5) == (labels > 0.5)).mean()),
        "_probability": mean,
        "_entropy": predictive_entropy(seed_probabilities),
        "_mi": mutual_information(seed_probabilities),
        "_variance": seed_variance(seed_probabilities),
        "_error": np.abs(mean - labels),
    }
    if len(np.unique(labels)) > 1:
        block["auroc"] = float(auroc(mean, labels))
        block["auprc"] = float(auprc(mean, labels))
    else:
        block["auroc"] = float("nan")
        block["auprc"] = float("nan")
    return block


def constant_prior_block(rate: float, labels: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=np.float64)
    prediction = np.full(labels.shape, float(rate))
    return {
        "brier": brier(prediction, labels),
        "nll": negative_log_likelihood(prediction, labels),
        "ece": expected_calibration_error(prediction, labels),
        "auroc": float("nan"),  # a constant cannot rank
        "auprc": float(labels.mean()),
        "accuracy": float(((prediction >= 0.5) == (labels > 0.5)).mean()),
    }


def strip_private(block: dict) -> dict:
    return {k: v for k, v in block.items() if not k.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("outputs/phase5/full_train/config.yaml"))
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    load_config(args.config)
    models = load_seeds(list(args.checkpoints), device)

    calibration_dataset = FrontierRevealDataset(args.calibration)
    validation_dataset = FrontierRevealDataset(args.validation)

    overlap = sorted(set(calibration_dataset.scenes()) & set(validation_dataset.scenes()))
    if overlap:
        raise SystemExit(f"calibration and validation share scenes: {overlap[:5]}")

    print(f"seeds       : {len(models)}")
    print(f"calibration : {len(calibration_dataset)} groups, {len(calibration_dataset.scenes())} scenes")
    print(f"validation  : {len(validation_dataset)} groups, {len(validation_dataset.scenes())} scenes")

    calibration = predict_split(models, calibration_dataset, device)
    validation = predict_split(models, validation_dataset, device)

    # -- 10.4 fit calibrators on CALIBRATION only --------------------------
    calibrators = {
        "target_temperature": TemperatureScaler().fit(
            ensemble_mean(calibration["target_logits"]), calibration["truth"]["target"]
        ),
        "target_platt": PlattScaler().fit(
            ensemble_mean(calibration["target_logits"]), calibration["truth"]["target"]
        ),
        "crossing_temperature": TemperatureScaler().fit(
            ensemble_mean(calibration["crossing_logits"]), calibration["truth"]["crossing"]
        ),
    }
    for channel, name in CHANNEL_NAMES.items():
        valid = calibration["truth"]["valid"].astype(bool).ravel()
        logits = ensemble_mean(calibration["spatial_logits"][:, :, channel]).ravel()[valid]
        labels = calibration["truth"][name].ravel()[valid]
        calibrators[f"{name}_temperature"] = TemperatureScaler().fit(logits, labels)

    calibration_area_mean = ensemble_mean(calibration["areas"])
    calibration_area_spread = np.std(calibration["areas"], axis=0)
    conformal = ConformalAreaInterval(alpha=args.alpha).fit(
        calibration["truth"]["area"], calibration_area_mean, calibration_area_spread
    )

    priors = {
        "target": float(calibration["truth"]["target"].mean()),
        "crossing": float(calibration["truth"]["crossing"].mean()),
        "area": float(calibration["truth"]["area"].mean()),
    }

    # -- 10.5 evaluate on VALIDATION --------------------------------------
    results: dict = {
        "n_seeds": len(models),
        "calibration": {"n_groups": calibration["n_groups"], "n_branches": calibration["n_branches"],
                        "n_scenes": len(calibration_dataset.scenes())},
        "validation": {"n_groups": validation["n_groups"], "n_branches": validation["n_branches"],
                       "n_scenes": len(validation_dataset.scenes())},
        "fitted_calibrators": {
            "target_temperature": calibrators["target_temperature"].temperature,
            "target_platt": {"slope": calibrators["target_platt"].slope,
                             "intercept": calibrators["target_platt"].intercept},
            "crossing_temperature": calibrators["crossing_temperature"].temperature,
            **{f"{n}_temperature": calibrators[f"{n}_temperature"].temperature
               for n in CHANNEL_NAMES.values()},
            "conformal_quantile": conformal.quantile,
            "conformal_n_calibration": conformal.n_calibration,
        },
        "calibration_priors": priors,
    }

    truth = validation["truth"]
    groups = validation["group_ids"]

    # occupancy / semantic channels
    occupancy: dict = {}
    for channel, name in CHANNEL_NAMES.items():
        raw = occupancy_block(validation["spatial_logits"], truth, channel)
        calibrated = occupancy_block(
            validation["spatial_logits"], truth, channel, calibrators[f"{name}_temperature"]
        )
        occupancy[name] = {
            "ensemble_raw": strip_private(raw),
            "ensemble_calibrated": strip_private(calibrated),
            "uncertainty_error_spearman": {
                "entropy": spearman(raw["_per_branch_entropy"], raw["_per_branch_error"]),
                "mutual_information": spearman(raw["_per_branch_mi"], raw["_per_branch_error"]),
                "variance": spearman(raw["_per_branch_variance"], raw["_per_branch_error"]),
            },
            "risk_coverage_mutual_information": risk_coverage_curve(
                raw["_per_branch_mi"], raw["_per_branch_error"]
            ),
        }
        # per-seed, for the individual-model rows
        occupancy[name]["individual_seeds"] = []
        for s in range(validation["n_seeds"]):
            single = sigmoid(validation["spatial_logits"][s : s + 1, :, channel])
            valid = truth["valid"].astype(bool)
            p = single[0].ravel()[valid.ravel()]
            y = truth[name].ravel()[valid.ravel()]
            occupancy[name]["individual_seeds"].append(
                {"iou": iou(single[0], truth[name], valid), "brier": brier(p, y),
                 "nll": negative_log_likelihood(p, y)}
            )
    results["occupancy"] = occupancy

    # semantic disagreement across seeds
    semantic_probabilities = sigmoid(validation["spatial_logits"][:, :, TGT_REVEALED_SEMANTIC])
    results["semantic_cosine_disagreement"] = {
        "mean": float(np.nanmean(pairwise_cosine_disagreement(semantic_probabilities))),
        "median": float(np.nanmedian(pairwise_cosine_disagreement(semantic_probabilities))),
    }

    # target and crossing heads
    for head, logits_key, calibrator_key in (
        ("target", "target_logits", "target_temperature"),
        ("crossing", "crossing_logits", "crossing_temperature"),
    ):
        labels = truth[head]
        raw = scalar_block(validation[logits_key], labels)
        calibrated = scalar_block(validation[logits_key], labels, calibrators[calibrator_key])
        entry = {
            "constant_prior": constant_prior_block(priors[head], labels),
            "ensemble_raw": strip_private(raw),
            "ensemble_calibrated": strip_private(calibrated),
            "uncertainty_error_spearman": {
                "entropy": spearman(raw["_entropy"], raw["_error"]),
                "mutual_information": spearman(raw["_mi"], raw["_error"]),
                "variance": spearman(raw["_variance"], raw["_error"]),
            },
            "risk_coverage_mutual_information": risk_coverage_curve(raw["_mi"], raw["_error"]),
            "individual_seeds": [
                strip_private(scalar_block(validation[logits_key][s : s + 1], labels))
                for s in range(validation["n_seeds"])
            ],
        }
        if head == "target":
            entry["ensemble_platt"] = strip_private(
                scalar_block(validation[logits_key], labels, calibrators["target_platt"])
            )
        # Bootstrap the calibrated-vs-raw Brier difference by decision group.
        entry["bootstrap_brier_calibrated_minus_raw"] = paired_bootstrap(
            (calibrated["_probability"] - labels) ** 2,
            (raw["_probability"] - labels) ** 2,
            groups,
            iterations=args.bootstrap,
            seed=args.seed,
        )
        results[head] = entry

    # area head
    area_mean = ensemble_mean(validation["areas"])
    area_spread = np.std(validation["areas"], axis=0)
    area_truth = truth["area"]
    area_error = np.abs(area_truth - area_mean)
    results["area"] = {
        "constant_prior_mae": float(np.mean(np.abs(area_truth - priors["area"]))),
        "ensemble_mae": float(np.mean(area_error)),
        "ensemble_rmse": float(np.sqrt(np.mean((area_truth - area_mean) ** 2))),
        "pearson": float(np.corrcoef(area_mean, area_truth)[0, 1]) if np.std(area_mean) > 1e-9 else float("nan"),
        "spearman": spearman(area_mean, area_truth),
        "individual_seeds_mae": [float(np.mean(np.abs(area_truth - validation["areas"][s])))
                                 for s in range(validation["n_seeds"])],
        "oracle_seed_mae": float(np.mean(np.min(np.abs(area_truth[None, :] - validation["areas"]), axis=0))),
        "oracle_note": "diagnostic only; picks the best seed per branch using the truth",
        "conformal_interval": conformal.coverage(area_truth, area_mean, area_spread),
        "uncertainty_error_spearman_seed_std": spearman(area_spread, area_error),
        "risk_coverage_seed_std": risk_coverage_curve(area_spread, area_error),
    }

    # ranking stability across seeds, using predicted area as the ranking score
    results["ranking_stability"] = ranking_stability(validation["areas"], groups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("calibration", "validation", "fitted_calibrators",
                               "ranking_stability", "semantic_cosine_disagreement")},
                     indent=2, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
