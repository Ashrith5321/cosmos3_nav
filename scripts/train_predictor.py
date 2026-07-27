#!/usr/bin/env python
"""Phase 8: train the deterministic structured revelation predictor.

Runs the checklist's debug progression in order, because each step isolates a
different failure:

    overfit-1    can the model represent a single answer at all?
    overfit-32   does it fit a small set, or is capacity/optimisation broken?
    train        does it generalise within the training scenes?
    heldout      does it generalise to scenes it has never seen?
    action-test  does the prediction change when the option changes?

    python scripts/train_predictor.py --train outputs/phase5/full_train \\
        --val outputs/phase5/full_val --epochs 30
    python scripts/train_predictor.py --train outputs/phase5/pilot_500 --stage overfit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.config import config_hash, load_config, save_config  # noqa: E402
from frontierworld.data.dataset import FrontierRevealDataset  # noqa: E402
from frontierworld.data.tensors import (  # noqa: E402
    FrameSpec,
    build_group_tensors,
    collate_tensor_groups,
)
from frontierworld.evaluation import environment_provenance, make_run_id  # noqa: E402
from frontierworld.models.baselines import (  # noqa: E402
    BASELINES,
    PredictionMetrics,
    binary_auc,
    masked_iou,
)
from frontierworld.models.predictor import (  # noqa: E402
    PredictorConfig,
    RevelationPredictor,
    goal_index,
    revelation_loss,
)
from frontierworld.seeding import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--train", required=True, help="training dataset root")
    parser.add_argument("--val", default=None, help="held-out dataset root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--extent", type=float, default=8.0)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--max-groups", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--stage", default="all", choices=["all", "overfit", "train"],
        help="'overfit' runs only the two overfitting checks",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


# -- data ------------------------------------------------------------------


def tensorise(dataset, spec: FrameSpec, limit: int = 0) -> list[dict]:
    groups = []
    total = len(dataset) if limit <= 0 else min(limit, len(dataset))
    for index in range(total):
        try:
            groups.append(build_group_tensors(dataset[index], spec))
        except Exception as exc:  # noqa: BLE001 - one bad group must not stop training
            print(f"  skipped group {index}: {exc}", file=sys.stderr)
    return groups


def batches(groups: list[dict], size: int, shuffle: bool = True, seed: int = 0):
    order = np.arange(len(groups))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), size):
        chunk = [groups[i] for i in order[start : start + size]]
        if chunk:
            yield collate_tensor_groups(chunk)


def to_torch(batch: dict, device: torch.device) -> dict:
    """Flatten (group, candidate) into one batch axis, dropping padded slots."""
    mask = batch["candidate_mask"]
    goals = []
    for group_index, goal in enumerate(batch["goals"]):
        goals.extend([goal_index(goal)] * int(mask[group_index].sum()))
    return {
        "inputs": torch.from_numpy(batch["inputs"][mask]).to(device),
        "targets": torch.from_numpy(batch["targets"][mask]).to(device),
        "valid": torch.from_numpy(batch["target_valid"][mask]).to(device),
        "options": torch.from_numpy(batch["options"][mask]).to(device),
        "goals": torch.tensor(goals, dtype=torch.long, device=device),
        "scalars": {
            key: torch.from_numpy(values[mask]).to(device)
            for key, values in batch["scalars"].items()
        },
    }


# -- evaluation ------------------------------------------------------------


@torch.no_grad()
def evaluate_model(model, groups, device, batch_size: int = 4) -> PredictionMetrics:
    model.eval()
    occ_ious, sem_ious, areas = [], [], []
    target_scores, target_labels = [], []
    crossing_correct = crossing_total = 0

    for batch in batches(groups, batch_size, shuffle=False):
        data = to_torch(batch, device)
        if data["inputs"].shape[0] == 0:
            continue
        prediction = model(data["inputs"], data["options"], data["goals"])
        probabilities = torch.sigmoid(prediction["revealed_logits"]).cpu().numpy()
        targets = data["targets"].cpu().numpy()
        valid = data["valid"].cpu().numpy()

        for index in range(probabilities.shape[0]):
            occ_ious.append(masked_iou(probabilities[index, 0], targets[index, 0], valid[index]))
            sem_ious.append(masked_iou(probabilities[index, 2], targets[index, 2], valid[index]))
        areas.append(
            np.abs(
                prediction["area"].cpu().numpy()
                - data["scalars"]["revealed_area_m2"].cpu().numpy()
            )
        )
        target_scores.append(torch.sigmoid(prediction["target_logit"]).cpu().numpy())
        target_labels.append(data["scalars"]["target_present"].cpu().numpy())
        predicted = (prediction["crossing_logit"] > 0).float().cpu().numpy()
        actual = data["scalars"]["crossing_success"].cpu().numpy()
        crossing_correct += int((predicted == actual).sum())
        crossing_total += int(actual.size)

    scores = np.concatenate(target_scores) if target_scores else np.zeros(0)
    labels = np.concatenate(target_labels) if target_labels else np.zeros(0)
    return PredictionMetrics(
        occupancy_iou=float(np.nanmean(occ_ious)) if occ_ious else float("nan"),
        semantic_iou=float(np.nanmean(sem_ious)) if sem_ious else float("nan"),
        target_auc=binary_auc(scores, labels),
        target_brier=float(np.mean((scores - labels) ** 2)) if scores.size else float("nan"),
        crossing_accuracy=crossing_correct / max(crossing_total, 1),
        area_mae=float(np.concatenate(areas).mean()) if areas else float("nan"),
        n=crossing_total,
    )


def evaluate_baseline(baseline, groups, batch_size: int = 4) -> PredictionMetrics:
    occ_ious, sem_ious, areas = [], [], []
    target_scores, target_labels = [], []
    crossing_correct = crossing_total = 0

    for batch in batches(groups, batch_size, shuffle=False):
        mask = batch["candidate_mask"]
        inputs, targets = batch["inputs"][mask], batch["targets"][mask]
        valid, options = batch["target_valid"][mask], batch["options"][mask]
        scalars = {k: v[mask] for k, v in batch["scalars"].items()}

        for index in range(inputs.shape[0]):
            prediction = baseline.predict(inputs[index], options[index], None)
            if prediction.revealed is not None:
                occ_ious.append(masked_iou(prediction.revealed[0], targets[index, 0], valid[index]))
                sem_ious.append(masked_iou(prediction.revealed[2], targets[index, 2], valid[index]))
            areas.append(abs(prediction.revealed_area_m2 - scalars["revealed_area_m2"][index]))
            target_scores.append(prediction.target_present)
            target_labels.append(scalars["target_present"][index])
            crossing_correct += int(
                (prediction.crossing_success > 0.5) == bool(scalars["crossing_success"][index])
            )
            crossing_total += 1

    scores, labels = np.asarray(target_scores), np.asarray(target_labels)
    return PredictionMetrics(
        occupancy_iou=float(np.nanmean(occ_ious)) if occ_ious else float("nan"),
        semantic_iou=float(np.nanmean(sem_ious)) if sem_ious else float("nan"),
        target_auc=binary_auc(scores, labels),
        target_brier=float(np.mean((scores - labels) ** 2)) if scores.size else float("nan"),
        crossing_accuracy=crossing_correct / max(crossing_total, 1),
        area_mae=float(np.mean(areas)) if areas else float("nan"),
        n=crossing_total,
    )


@torch.no_grad()
def action_sensitivity(model, groups, device, batch_size: int = 4) -> dict:
    """Does the prediction move when the option changes?

    Permuting options within a batch keeps every map identical and swaps only
    the action. If the predictions do not move, the model has learned the state
    and ignored the candidate -- every candidate would then rank the same, and
    the ranking problem this paper is about would be vacuous.
    """
    model.eval()
    deltas, area_deltas = [], []
    for batch in batches(groups, batch_size, shuffle=False):
        data = to_torch(batch, device)
        if data["inputs"].shape[0] < 2:
            continue
        original = model(data["inputs"], data["options"], data["goals"])
        permutation = torch.roll(torch.arange(data["options"].shape[0]), 1)
        permuted = model(data["inputs"], data["options"][permutation], data["goals"])
        deltas.append(
            float(
                (
                    torch.sigmoid(original["revealed_logits"])
                    - torch.sigmoid(permuted["revealed_logits"])
                ).abs().mean()
            )
        )
        area_deltas.append(float((original["area"] - permuted["area"]).abs().mean()))

    return {
        "mean_spatial_change": float(np.mean(deltas)) if deltas else 0.0,
        "mean_area_change_m2": float(np.mean(area_deltas)) if area_deltas else 0.0,
        "uses_action_conditioning": bool(np.mean(deltas) > 1e-3) if deltas else False,
    }


# -- training --------------------------------------------------------------


def train(model, groups, device, epochs: int, lr: float, batch_size: int,
          label: str, val_groups=None, log_every: int = 5) -> list[dict]:
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in batches(groups, batch_size, shuffle=True, seed=epoch):
            data = to_torch(batch, device)
            if data["inputs"].shape[0] == 0:
                continue
            prediction = model(data["inputs"], data["options"], data["goals"])
            loss, parts = revelation_loss(
                prediction, data["targets"], data["valid"], data["scalars"]
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(parts)

        if not losses:
            break
        mean_loss = float(np.mean([p["loss"] for p in losses]))
        history.append(
            {"epoch": epoch, "loss": mean_loss,
             **{k: float(np.mean([p[k] for p in losses])) for k in losses[0] if k != "loss"}}
        )
        if epoch % log_every == 0 or epoch == epochs - 1:
            message = f"  [{label}] epoch {epoch:3d}  loss {mean_loss:.4f}"
            if val_groups and (epoch % (log_every * 2) == 0 or epoch == epochs - 1):
                metrics = evaluate_model(model, val_groups, device)
                message += (
                    f"  | val occIoU {metrics.occupancy_iou:.3f} "
                    f"tgtAUC {metrics.target_auc:.3f} areaMAE {metrics.area_mae:.2f}"
                )
            print(message, flush=True)
    return history


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)
    seed_everything(int(cfg.seed.value), torch_deterministic=False)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    spec = FrameSpec(extent_m=args.extent, resolution=args.resolution)

    run_id = make_run_id("phase8_predictor", config_hash(cfg))
    run_dir = Path(cfg.experiment.output_dir) / "phase8" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(json.dumps(environment_provenance(), indent=2))

    print(f"run_dir: {run_dir}")
    print(f"device : {device}")

    train_dataset = FrontierRevealDataset(args.train)
    print(f"train  : {args.train}  {len(train_dataset)} groups, "
          f"{len(train_dataset.scenes())} scenes")
    train_groups = tensorise(train_dataset, spec, args.max_groups)
    if not train_groups:
        print("no usable training groups", file=sys.stderr)
        return 1

    val_groups = None
    if args.val:
        val_dataset = FrontierRevealDataset(args.val)
        overlap = train_dataset.scenes() & val_dataset.scenes()
        print(f"val    : {args.val}  {len(val_dataset)} groups, "
              f"{len(val_dataset.scenes())} scenes")
        print(f"scene overlap train/val: {len(overlap)} "
              f"{'OK' if not overlap else 'LEAK'}")
        if overlap:
            print("held-out evaluation is invalid with overlapping scenes", file=sys.stderr)
            return 1
        val_groups = tensorise(val_dataset, spec, 0)

    option_dim = train_groups[0]["options"].shape[1]
    results: dict = {
        "device": str(device),
        "frame": {"extent_m": spec.extent_m, "resolution": spec.resolution},
        "train_groups": len(train_groups),
        "val_groups": len(val_groups) if val_groups else 0,
    }

    # -- overfit one group -------------------------------------------------
    print("\n== overfit 1 decision group ==")
    model = RevelationPredictor(PredictorConfig(option_dim=option_dim)).to(device)
    history = train(model, train_groups[:1], device, 120, 1e-3, 1, "overfit-1", log_every=40)
    first, last = history[0]["loss"], history[-1]["loss"]
    overfit1_ok = last < first * 0.35
    print(f"  loss {first:.4f} -> {last:.4f}  {'PASS' if overfit1_ok else 'FAIL'}")
    results["overfit_1"] = {"first": first, "last": last, "passed": overfit1_ok}

    # -- overfit 32 groups --------------------------------------------------
    print("\n== overfit 32 decision groups ==")
    model = RevelationPredictor(PredictorConfig(option_dim=option_dim)).to(device)
    subset = train_groups[:32]
    history = train(model, subset, device, 120, 1e-3, 4, "overfit-32", log_every=40)
    first, last = history[0]["loss"], history[-1]["loss"]
    metrics = evaluate_model(model, subset, device)
    overfit32_ok = last < first * 0.5 and metrics.occupancy_iou > 0.3
    print(f"  loss {first:.4f} -> {last:.4f}  train occIoU {metrics.occupancy_iou:.3f}  "
          f"{'PASS' if overfit32_ok else 'FAIL'}")
    results["overfit_32"] = {
        "first": first, "last": last,
        "occupancy_iou": metrics.occupancy_iou, "passed": overfit32_ok,
    }

    if args.stage == "overfit":
        (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nresults: {run_dir / 'results.json'}")
        return 0 if overfit1_ok and overfit32_ok else 1

    # -- full training ------------------------------------------------------
    print(f"\n== train on {len(train_groups)} groups ==")
    model = RevelationPredictor(PredictorConfig(option_dim=option_dim)).to(device)
    started = time.perf_counter()
    results["train_history"] = train(
        model, train_groups, device, args.epochs, args.lr,
        args.batch_groups, "train", val_groups,
    )
    results["train_seconds"] = time.perf_counter() - started
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.__dict__},
        run_dir / "checkpoint.pt",
    )

    # -- held-out comparison ------------------------------------------------
    evaluation_groups = val_groups if val_groups else train_groups
    label = "held-out scenes" if val_groups else "TRAIN scenes (no held-out set)"
    print(f"\n== evaluation on {label}: {len(evaluation_groups)} groups ==")

    model_metrics = evaluate_model(model, evaluation_groups, device)
    rows = {"model": model_metrics.to_dict()}
    train_batches = list(batches(train_groups, 4, shuffle=False))
    for name, factory in BASELINES.items():
        rows[name] = evaluate_baseline(factory().fit(train_batches), evaluation_groups).to_dict()

    header = (f"{'method':<28}{'occIoU':>9}{'semIoU':>9}{'tgtAUC':>9}"
              f"{'tgtBrier':>10}{'crossAcc':>10}{'areaMAE':>9}")
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in rows.items():
        print(
            f"{name:<28}{metrics['occupancy_iou']:>9.3f}{metrics['semantic_iou']:>9.3f}"
            f"{metrics['target_auc']:>9.3f}{metrics['target_brier']:>10.3f}"
            f"{metrics['crossing_accuracy']:>10.3f}{metrics['area_mae']:>9.2f}"
        )
    results["evaluation"] = rows
    results["evaluated_on_heldout"] = val_groups is not None

    sensitivity = action_sensitivity(model, evaluation_groups, device)
    print(f"\naction conditioning: spatial change {sensitivity['mean_spatial_change']:.4f}, "
          f"area change {sensitivity['mean_area_change_m2']:.3f} m2 "
          f"-> uses option: {sensitivity['uses_action_conditioning']}")
    results["action_sensitivity"] = sensitivity

    prior, observation = rows["dataset_prior"], rows["current_observation_only"]
    beats_prior = (
        model_metrics.occupancy_iou > prior["occupancy_iou"]
        or model_metrics.area_mae < prior["area_mae"]
    )
    beats_observation = (
        model_metrics.occupancy_iou > observation["occupancy_iou"]
        or model_metrics.area_mae < observation["area_mae"]
    )
    passed = overfit32_ok and beats_prior and beats_observation
    results["gate"] = {
        "overfit_32": overfit32_ok,
        "beats_dataset_prior": beats_prior,
        "beats_current_observation": beats_observation,
        "passed": passed,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\ngate: overfit-32 {overfit32_ok}, beats prior {beats_prior}, "
          f"beats current-observation {beats_observation}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    print(f"\nresults: {run_dir / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
