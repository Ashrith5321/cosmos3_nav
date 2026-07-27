#!/usr/bin/env python
"""Phase 8 v1 diagnosis: why the occupied and target heads failed.

TRAIN AND VALIDATION ONLY. The v0 test split is consumed -- its numbers have
been read, so anything designed against it now would be biased. Nothing here
opens outputs/phase5/full_test.

Occupied prediction:
  * occupied-cell prevalence per scene and split
  * precision / recall / probability histograms
  * whether thin obstacle boundaries make exact IoU hypersensitive to one-cell
    misalignment, via a distance-tolerant boundary metric
  * whether validation checkpoint selection sacrificed occupied IoU
  * a factorised alternative, p(revealed) * p(occupied | revealed)

Target presence:
  * against a constant-prior Brier and AUPRC
  * calibration curve and confidence histogram
  * per-category performance
  * whether positive branches carry visual evidence in the input at all

    python scripts/diagnose_heads.py --train outputs/phase5/full_train \\
        --val outputs/phase5/full_val --checkpoint archive/phase8_v0/checkpoint_seed0.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontierworld.config import load_config  # noqa: E402
from frontierworld.data.dataset import FrontierRevealDataset  # noqa: E402
from frontierworld.data.tensors import (  # noqa: E402
    TGT_REVEALED_FREE,
    TGT_REVEALED_OCCUPIED,
    FrameSpec,
)
from frontierworld.evaluation import make_run_id  # noqa: E402
from frontierworld.models.metrics import auprc, auroc  # noqa: E402
from frontierworld.models.predictor import PredictorConfig, RevelationPredictor  # noqa: E402
from diagnose_predictor import collect  # noqa: E402
from train_predictor import batches, tensorise  # noqa: E402

FORBIDDEN = "full_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# -- occupied head ---------------------------------------------------------


def prevalence(groups: list[dict], label: str) -> dict:
    """How much of the revealed region is occupied, per group and overall."""
    free_fraction, occupied_fraction, per_scene = [], [], defaultdict(list)
    for group in groups:
        for index in range(group["targets"].shape[0]):
            valid = group["target_valid"][index]
            free = (group["targets"][index, TGT_REVEALED_FREE] > 0.5) & valid
            occupied = (group["targets"][index, TGT_REVEALED_OCCUPIED] > 0.5) & valid
            revealed = free | occupied
            if not revealed.any():
                continue
            free_fraction.append(free.sum() / revealed.sum())
            occupied_fraction.append(occupied.sum() / revealed.sum())
            per_scene[group.get("scene_id", "?")].append(
                occupied.sum() / revealed.sum()
            )
    return {
        "split": label,
        "mean_free_fraction_of_revealed": float(np.mean(free_fraction)),
        "mean_occupied_fraction_of_revealed": float(np.mean(occupied_fraction)),
        "occupied_fraction_sd": float(np.std(occupied_fraction)),
        "n_branches": len(occupied_fraction),
        "per_scene_occupied_fraction": {
            k: float(np.mean(v)) for k, v in sorted(per_scene.items())
        },
    }


def precision_recall(probabilities, targets, valid, channel, threshold=0.5) -> dict:
    predicted = (probabilities[:, channel] >= threshold) & valid
    actual = (targets[:, channel] >= 0.5) & valid
    tp = float((predicted & actual).sum())
    fp = float((predicted & ~actual).sum())
    fn = float((~predicted & actual).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "predicted_positive_rate": float(predicted.sum() / max(valid.sum(), 1)),
        "actual_positive_rate": float(actual.sum() / max(valid.sum(), 1)),
        "max_probability": float(probabilities[:, channel].max()),
        "mean_probability_on_positives": (
            float(probabilities[:, channel][actual].mean()) if actual.any() else float("nan")
        ),
        "mean_probability_on_negatives": (
            float(probabilities[:, channel][valid & ~actual].mean())
            if (valid & ~actual).any() else float("nan")
        ),
    }


def tolerant_boundary_iou(
    probabilities, targets, valid, channel, tolerance_cells=2, threshold=0.5
) -> float:
    """IoU allowing a few cells of misalignment.

    Revealed-occupied cells are thin wall strips. Exact IoU on a one-cell-wide
    structure is dominated by alignment: a prediction offset by a single cell
    scores near zero while being visually right. Comparing tolerant against
    exact separates "wrong place" from "wrong shape".
    """
    from scipy import ndimage

    scores = []
    for index in range(probabilities.shape[0]):
        predicted = (probabilities[index, channel] >= threshold) & valid[index]
        actual = (targets[index, channel] >= 0.5) & valid[index]
        if not (predicted.any() or actual.any()):
            continue
        predicted_dilated = ndimage.binary_dilation(predicted, iterations=tolerance_cells)
        actual_dilated = ndimage.binary_dilation(actual, iterations=tolerance_cells)
        intersection = (predicted & actual_dilated).sum() + (actual & predicted_dilated).sum()
        union = predicted.sum() + actual.sum()
        scores.append(intersection / union if union else np.nan)
    return float(np.nanmean(scores)) if scores else float("nan")


def factorised_occupied(probabilities, channel_free, channel_occupied) -> np.ndarray:
    """p(revealed) * p(occupied | revealed), from the existing heads.

    The current heads predict revealed-free and revealed-occupied
    independently, so nothing ties them to a coherent p(revealed). This
    recombination tests whether the failure is in the representation rather
    than in the features: if factorising helps without retraining, v1 should
    predict revelation and material separately.
    """
    revealed = np.clip(probabilities[:, channel_free] + probabilities[:, channel_occupied], 0, 1)
    conditional = np.where(
        revealed > 1e-6, probabilities[:, channel_occupied] / np.maximum(revealed, 1e-6), 0.0
    )
    return revealed * conditional


# -- target head -----------------------------------------------------------


def calibration(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (scores >= low) & (scores < high if high < 1 else scores <= 1)
        if not mask.any():
            continue
        rows.append({
            "bin": f"[{low:.1f},{high:.1f})",
            "n": int(mask.sum()),
            "mean_confidence": float(scores[mask].mean()),
            "empirical_rate": float(labels[mask].mean()),
        })
    return rows


def visual_evidence(groups: list[dict]) -> dict:
    """Do target-positive branches look different in the INPUT?

    If positives and negatives are indistinguishable at the decision state,
    no amount of loss reweighting will fix the target head -- the evidence is
    simply not in the current observation, which is the argument for the
    lineage-aware visual memory of Phase 11.
    """
    from frontierworld.data.tensors import CH_FREE, CH_OCCUPIED, CH_UNKNOWN

    positive, negative = defaultdict(list), defaultdict(list)
    for group in groups:
        for index in range(group["inputs"].shape[0]):
            bucket = positive if group["scalars"]["target_present"][index] > 0.5 else negative
            inputs = group["inputs"][index]
            bucket["unknown_area"].append(float(inputs[CH_UNKNOWN].sum()))
            bucket["free_area"].append(float(inputs[CH_FREE].sum()))
            bucket["occupied_area"].append(float(inputs[CH_OCCUPIED].sum()))
            bucket["info_gain"].append(float(group["options"][index][5]))
            bucket["travel_cost"].append(float(group["options"][index][4]))

    summary = {}
    for key in positive:
        p, n = np.asarray(positive[key]), np.asarray(negative[key])
        pooled = np.sqrt((p.var() + n.var()) / 2) if p.size and n.size else 0.0
        summary[key] = {
            "positive_mean": float(p.mean()) if p.size else float("nan"),
            "negative_mean": float(n.mean()) if n.size else float("nan"),
            # Cohen's d: how separable positives are from negatives on this
            # feature alone. Near zero means the input carries no signal.
            "cohens_d": float((p.mean() - n.mean()) / pooled) if pooled > 1e-9 else 0.0,
        }
    summary["n_positive"] = len(positive["unknown_area"])
    summary["n_negative"] = len(negative["unknown_area"])
    return summary


def per_category(groups: list[dict], scores: np.ndarray, labels: np.ndarray) -> dict:
    goals = []
    for group in groups:
        goals.extend([group["goal"]] * group["inputs"].shape[0])
    goals = np.asarray(goals[: len(scores)])

    out = {}
    for goal in sorted({g for g in goals if g}):
        mask = goals == goal
        if mask.sum() < 5:
            continue
        out[goal] = {
            "n": int(mask.sum()),
            "positive_rate": float(labels[mask].mean()),
            "auroc": auroc(scores[mask], labels[mask]),
            "auprc": auprc(scores[mask], labels[mask]),
            "brier": float(np.mean((scores[mask] - labels[mask]) ** 2)),
        }
    return out


def main() -> int:
    args = parse_args()
    for path in (args.train, args.val):
        if FORBIDDEN in str(path):
            print(f"refusing to read the consumed v0 test split: {path}", file=sys.stderr)
            return 1

    cfg = load_config(None)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    spec = FrameSpec()
    run_dir = Path(cfg.experiment.output_dir) / "phase8_v1_diagnosis" / make_run_id("diag", "v1")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_groups = tensorise(FrontierRevealDataset(args.train), spec, 0)
    val_groups = tensorise(FrontierRevealDataset(args.val), spec, 0)
    print(f"run_dir: {run_dir}")
    print(f"train {len(train_groups)} groups | val {len(val_groups)} groups "
          f"(test split NOT opened)\n")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = PredictorConfig(**checkpoint["config"])
    model = RevelationPredictor(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"checkpoint: {args.checkpoint} (selected epoch "
          f"{checkpoint.get('selected_epoch')}, val score {checkpoint.get('val_score'):.4f})\n")

    results = {}

    # ---- occupied prediction --------------------------------------------
    print("=" * 70)
    print("OCCUPIED HEAD")
    print("=" * 70)

    for label, groups in (("train", train_groups), ("val", val_groups)):
        stats = prevalence(groups, label)
        print(f"  {label}: revealed cells are {stats['mean_occupied_fraction_of_revealed']:.1%} "
              f"occupied (sd {stats['occupied_fraction_sd']:.1%}), "
              f"{stats['mean_free_fraction_of_revealed']:.1%} free  "
              f"[{stats['n_branches']} branches]")
        results[f"prevalence_{label}"] = stats

    val = collect(model, val_groups, device)
    for channel, name in ((TGT_REVEALED_FREE, "free"), (TGT_REVEALED_OCCUPIED, "occupied")):
        pr = precision_recall(val["probabilities"], val["targets"], val["valid"], channel)
        print(f"\n  {name} head on VAL:")
        print(f"    precision {pr['precision']:.3f}  recall {pr['recall']:.3f}")
        print(f"    predicts positive on {pr['predicted_positive_rate']:.1%} of cells; "
              f"actual {pr['actual_positive_rate']:.1%}")
        print(f"    max probability {pr['max_probability']:.3f}; "
              f"mean on positives {pr['mean_probability_on_positives']:.3f}, "
              f"on negatives {pr['mean_probability_on_negatives']:.3f}")
        results[f"precision_recall_{name}"] = pr

    from frontierworld.models.metrics import masked_iou

    exact = float(np.nanmean([
        masked_iou(val["probabilities"][i, TGT_REVEALED_OCCUPIED],
                   val["targets"][i, TGT_REVEALED_OCCUPIED], val["valid"][i])
        for i in range(val["probabilities"].shape[0])
    ]))
    tolerant = tolerant_boundary_iou(
        val["probabilities"], val["targets"], val["valid"], TGT_REVEALED_OCCUPIED
    )
    print(f"\n  occupied IoU exact {exact:.3f} vs 2-cell-tolerant {tolerant:.3f}")
    print("    -> a large gap means thin-structure misalignment, not wrong shape")
    results["occupied_iou_exact"] = exact
    results["occupied_iou_tolerant"] = tolerant

    factorised = factorised_occupied(val["probabilities"], TGT_REVEALED_FREE, TGT_REVEALED_OCCUPIED)
    factorised_iou = float(np.nanmean([
        masked_iou(factorised[i], val["targets"][i, TGT_REVEALED_OCCUPIED], val["valid"][i])
        for i in range(factorised.shape[0])
    ]))
    print(f"  factorised p(revealed)p(occ|revealed) IoU {factorised_iou:.3f} "
          f"(vs direct {exact:.3f})")
    results["occupied_iou_factorised"] = factorised_iou

    # ---- target presence -------------------------------------------------
    print("\n" + "=" * 70)
    print("TARGET PRESENCE HEAD")
    print("=" * 70)

    scores, labels = val["target_scores"], val["target_labels"]
    prior = float(labels.mean())
    constant_brier = float(np.mean((prior - labels) ** 2))
    print(f"  val positive rate {prior:.3f}")
    print(f"  model  AUROC {auroc(scores, labels):.3f}  AUPRC {auprc(scores, labels):.3f}  "
          f"Brier {np.mean((scores - labels) ** 2):.3f}")
    print(f"  constant-prior baseline: AUPRC {prior:.3f} (chance)  Brier {constant_brier:.3f}")
    beats_prior_brier = float(np.mean((scores - labels) ** 2)) < constant_brier
    print(f"  beats the constant prior on Brier: {beats_prior_brier}")
    results["target"] = {
        "positive_rate": prior,
        "auroc": auroc(scores, labels),
        "auprc": auprc(scores, labels),
        "brier": float(np.mean((scores - labels) ** 2)),
        "constant_prior_brier": constant_brier,
        "constant_prior_auprc": prior,
        "beats_constant_prior_brier": bool(beats_prior_brier),
    }

    print("\n  calibration:")
    rows = calibration(scores, labels)
    for row in rows:
        print(f"    {row['bin']:<12} n={row['n']:5d}  confidence {row['mean_confidence']:.3f}  "
              f"actual {row['empirical_rate']:.3f}")
    results["calibration"] = rows
    print(f"  confidence range [{scores.min():.3f}, {scores.max():.3f}], "
          f"sd {scores.std():.3f}")

    print("\n  per goal category:")
    categories = per_category(val_groups, scores, labels)
    for goal, m in sorted(categories.items(), key=lambda kv: -kv[1]["n"]):
        print(f"    {goal:<12} n={m['n']:4d}  pos {m['positive_rate']:.2f}  "
              f"AUROC {m['auroc']:.3f}  AUPRC {m['auprc']:.3f}  Brier {m['brier']:.3f}")
    results["per_category"] = categories

    print("\n  is the evidence even in the input? (Cohen's d, positives vs negatives)")
    evidence = visual_evidence(val_groups)
    for key, value in evidence.items():
        if isinstance(value, dict):
            print(f"    {key:<16} d={value['cohens_d']:+.3f}  "
                  f"(pos {value['positive_mean']:.1f} vs neg {value['negative_mean']:.1f})")
    print(f"    {evidence['n_positive']} positive / {evidence['n_negative']} negative branches")
    results["visual_evidence"] = evidence

    (run_dir / "diagnosis.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults: {run_dir / 'diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
