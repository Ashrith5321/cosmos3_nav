#!/usr/bin/env python
"""Phase 8 diagnostics: is the overfit result actually airtight?

The aggregate loss can fall while a dense head stays weak -- the scalar heads
converge fast and dominate the total. These checks separate the heads and
verify the model is using the frontier option rather than the state alone.

    1  visualise predicted vs ground-truth occupancy (one-group run)
    2  report every head's metric separately
    3  free-space and occupied-space IoU on the one-group run
    4  invalid / unrevealed pixels excluded from both loss and metrics
    5  swapping omega between two real candidates changes the output
    6  shuffling options across candidates degrades performance
    7  candidate ordering does not affect predictions

    python scripts/diagnose_predictor.py --train outputs/phase5/pilot_500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.config import config_hash, load_config  # noqa: E402
from frontierworld.data.dataset import FrontierRevealDataset  # noqa: E402
from frontierworld.data.tensors import (  # noqa: E402
    CH_FREE,
    CH_OCCUPIED,
    CH_UNKNOWN,
    TGT_REVEALED_FREE,
    TGT_REVEALED_OCCUPIED,
    TGT_REVEALED_SEMANTIC,
    FrameSpec,
    build_group_tensors,
)
from frontierworld.evaluation import make_run_id  # noqa: E402
from frontierworld.models.metrics import compute_head_metrics  # noqa: E402
from frontierworld.models.predictor import (  # noqa: E402
    PredictorConfig,
    RevelationPredictor,
    goal_index,
    revelation_loss,
)
from frontierworld.seeding import seed_everything  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_predictor import batches, tensorise, to_torch, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--train", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def collect(model, groups, device, batch_size: int = 4, option_permute: bool = False):
    """Run the model over groups, returning stacked predictions and labels."""
    model.eval()
    probabilities, targets, valid = [], [], []
    target_scores, target_labels = [], []
    crossing_scores, crossing_labels = [], []
    areas, area_labels = [], []

    for batch in batches(groups, batch_size, shuffle=False):
        data = to_torch(batch, device)
        if data["inputs"].shape[0] == 0:
            continue
        options = data["options"]
        if option_permute and options.shape[0] > 1:
            # Roll within the batch: every candidate keeps its map but is given
            # another candidate's action.
            options = options[torch.roll(torch.arange(options.shape[0]), 1)]

        prediction = model(data["inputs"], options, data["goals"])
        probabilities.append(torch.sigmoid(prediction["revealed_logits"]).cpu().numpy())
        targets.append(data["targets"].cpu().numpy())
        valid.append(data["valid"].cpu().numpy())
        target_scores.append(torch.sigmoid(prediction["target_logit"]).cpu().numpy())
        target_labels.append(data["scalars"]["target_present"].cpu().numpy())
        crossing_scores.append(torch.sigmoid(prediction["crossing_logit"]).cpu().numpy())
        crossing_labels.append(data["scalars"]["crossing_success"].cpu().numpy())
        areas.append(prediction["area"].cpu().numpy())
        area_labels.append(data["scalars"]["revealed_area_m2"].cpu().numpy())

    return dict(
        probabilities=np.concatenate(probabilities),
        targets=np.concatenate(targets),
        valid=np.concatenate(valid),
        target_scores=np.concatenate(target_scores),
        target_labels=np.concatenate(target_labels),
        crossing_scores=np.concatenate(crossing_scores),
        crossing_labels=np.concatenate(crossing_labels),
        areas=np.concatenate(areas),
        area_labels=np.concatenate(area_labels),
    )


def metrics_from(collected: dict):
    return compute_head_metrics(
        collected["probabilities"], collected["targets"], collected["valid"],
        collected["target_scores"], collected["target_labels"],
        collected["crossing_scores"], collected["crossing_labels"],
        collected["areas"], collected["area_labels"],
    )


def visualise(collected: dict, path: Path, n: int = 4) -> Path:
    """Predicted vs ground-truth occupancy, side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(n, collected["probabilities"].shape[0])
    figure, axes = plt.subplots(3, n, figsize=(3.6 * n, 11), constrained_layout=True)
    axes = np.atleast_2d(axes)
    if n == 1:
        axes = axes.reshape(3, 1)

    for index in range(n):
        prediction = collected["probabilities"][index]
        target = collected["targets"][index]
        window = collected["valid"][index]

        axes[0, index].imshow(
            np.clip(np.stack([target[TGT_REVEALED_OCCUPIED],
                              target[TGT_REVEALED_FREE],
                              target[TGT_REVEALED_SEMANTIC]], -1), 0, 1),
            origin="lower",
        )
        axes[0, index].set_title(f"GT #{index}\nR=occ G=free B=sem", fontsize=9)

        axes[1, index].imshow(
            np.clip(np.stack([prediction[TGT_REVEALED_OCCUPIED],
                              prediction[TGT_REVEALED_FREE],
                              prediction[TGT_REVEALED_SEMANTIC]], -1), 0, 1),
            origin="lower",
        )
        axes[1, index].set_title("prediction", fontsize=9)

        # Where the free-space head agrees and disagrees.
        predicted_free = (prediction[TGT_REVEALED_FREE] >= 0.5) & window
        actual_free = (target[TGT_REVEALED_FREE] >= 0.5) & window
        overlay = np.zeros((*window.shape, 3))
        overlay[actual_free & predicted_free] = (0.2, 0.8, 0.2)   # hit
        overlay[actual_free & ~predicted_free] = (0.9, 0.2, 0.2)  # miss
        overlay[~actual_free & predicted_free] = (0.2, 0.4, 0.95)  # false positive
        overlay[~window] = (0.35, 0.35, 0.35)                      # not scored
        axes[2, index].imshow(overlay, origin="lower")
        union = (actual_free | predicted_free).sum()
        axes[2, index].set_title(
            f"free IoU {((actual_free & predicted_free).sum() / max(union,1)):.3f}\n"
            "green hit / red miss / blue FP / grey unscored",
            fontsize=8,
        )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle("Phase 8: predicted vs ground-truth revelation", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.seed.value), torch_deterministic=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    spec = FrameSpec()

    run_dir = Path(cfg.experiment.output_dir) / "phase8_diagnostics" / make_run_id(
        "diag", config_hash(cfg)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = FrontierRevealDataset(args.train)
    groups = tensorise(dataset, spec, args.groups)
    option_dim = groups[0]["options"].shape[1]
    print(f"run_dir: {run_dir}\ndevice : {device}\ngroups : {len(groups)}\n")

    results: dict = {}

    # ---- one-group overfit ------------------------------------------------
    print("== overfitting 1 decision group ==")
    single = RevelationPredictor(PredictorConfig(option_dim=option_dim)).to(device)
    train(single, groups[:1], device, args.epochs, 1e-3, 1, "one-group", log_every=100)
    one = collect(single, groups[:1], device)
    one_metrics = metrics_from(one)

    print("\n-- check 1/3: per-head metrics on the overfit group --")
    for name, value in one_metrics.table_rows():
        print(f"  {name:<28} {value}")
    results["one_group"] = one_metrics.to_dict()

    dense_ok = one_metrics.free_iou > 0.7 and one_metrics.occupied_iou > 0.4
    print(f"\n  dense heads high (free>0.7, occ>0.4): "
          f"{'PASS' if dense_ok else 'FAIL'}")
    results["one_group_dense_ok"] = bool(dense_ok)

    figure = visualise(one, run_dir / "one_group_prediction.png", n=min(4, one["probabilities"].shape[0]))
    print(f"  check 1: visualisation -> {figure}")

    # ---- check 4: masking --------------------------------------------------
    print("\n-- check 4: invalid / unrevealed pixels excluded --")
    invalid = ~one["valid"]
    print(f"  invalid cells in window        : {int(invalid.sum())} "
          f"({100 * invalid.mean():.1f}%)")

    # A group whose window sits inside the map has no invalid cells, and then
    # corrupting "invalid" cells corrupts nothing and the check passes
    # vacuously. Mark a band invalid explicitly so the masking path is really
    # exercised, and record whether real invalid cells were present too.
    data = to_torch(next(batches(groups[:1], 1, shuffle=False)), device)
    synthetic_valid = data["valid"].clone()
    height = synthetic_valid.shape[1]
    synthetic_valid[:, height // 2 :, :] = False

    with torch.no_grad():
        base = single(data["inputs"], data["options"], data["goals"])
    corrupted = dict(base)
    corrupted["revealed_logits"] = torch.where(
        synthetic_valid.unsqueeze(1),
        base["revealed_logits"],
        base["revealed_logits"] + 50.0,  # wildly wrong, but only where invalid
    )
    loss_base, _ = revelation_loss(
        base, data["targets"], synthetic_valid, data["scalars"]
    )
    loss_corrupt, _ = revelation_loss(
        corrupted, data["targets"], synthetic_valid, data["scalars"]
    )
    loss_unchanged = abs(float(loss_base) - float(loss_corrupt)) < 1e-6
    print(f"  loss unchanged by corrupting invalid cells : "
          f"{'PASS' if loss_unchanged else 'FAIL'} "
          f"({float(loss_base):.6f} vs {float(loss_corrupt):.6f})")

    synthetic_np = one["valid"].copy()
    synthetic_np[:, synthetic_np.shape[1] // 2 :, :] = False
    reference = compute_head_metrics(
        one["probabilities"], one["targets"], synthetic_np,
        one["target_scores"], one["target_labels"],
        one["crossing_scores"], one["crossing_labels"],
        one["areas"], one["area_labels"],
    )
    corrupted_np = one["probabilities"].copy()
    corrupted_np[~np.repeat(synthetic_np[:, None], corrupted_np.shape[1], axis=1)] = 1.0
    metrics_corrupt = compute_head_metrics(
        corrupted_np, one["targets"], synthetic_np,
        one["target_scores"], one["target_labels"],
        one["crossing_scores"], one["crossing_labels"],
        one["areas"], one["area_labels"],
    )
    metrics_unchanged = abs(metrics_corrupt.free_iou - reference.free_iou) < 1e-9
    print(f"  metrics unchanged by the same corruption   : "
          f"{'PASS' if metrics_unchanged else 'FAIL'} "
          f"({reference.free_iou:.6f} vs {metrics_corrupt.free_iou:.6f})")

    # Sanity: the corruption must be large enough that an UNMASKED loss moves,
    # otherwise "unchanged" proves nothing about the mask.
    all_valid = torch.ones_like(synthetic_valid)
    unmasked_base, _ = revelation_loss(base, data["targets"], all_valid, data["scalars"])
    unmasked_corrupt, _ = revelation_loss(
        corrupted, data["targets"], all_valid, data["scalars"]
    )
    corruption_detectable = abs(float(unmasked_base) - float(unmasked_corrupt)) > 1e-3
    print(f"  corruption is detectable when unmasked     : "
          f"{'PASS' if corruption_detectable else 'FAIL (test is vacuous)'}")

    results["masking"] = {
        "loss_unchanged": bool(loss_unchanged),
        "metrics_unchanged": bool(metrics_unchanged),
        "corruption_detectable_when_unmasked": bool(corruption_detectable),
        "real_invalid_fraction": float(invalid.mean()),
        "used_synthetic_invalid_band": True,
    }
    masking_ok = loss_unchanged and metrics_unchanged and corruption_detectable

    # ---- 32-group model for the action checks ------------------------------
    print(f"\n== training on {len(groups)} groups for the action checks ==")
    model = RevelationPredictor(PredictorConfig(option_dim=option_dim)).to(device)
    train(model, groups, device, args.epochs, 1e-3, 4, "multi-group", log_every=100)

    normal = collect(model, groups, device)
    normal_metrics = metrics_from(normal)
    print("\n-- check 2: per-head metrics (32 groups, training fit) --")
    for name, value in normal_metrics.table_rows():
        print(f"  {name:<28} {value}")
    results["multi_group"] = normal_metrics.to_dict()

    # ---- check 5: swap omega between two real candidates -------------------
    print("\n-- check 5: swapping omega between two real candidates --")
    group = next(g for g in groups if g["inputs"].shape[0] >= 2)
    inputs = torch.from_numpy(group["inputs"][:2]).to(device)
    options = torch.from_numpy(group["options"][:2]).to(device)
    goals = torch.tensor([goal_index(group["goal"])] * 2, dtype=torch.long, device=device)
    with torch.no_grad():
        original = model(inputs, options, goals)
        swapped = model(inputs, options.flip(0), goals)
    spatial_change = float(
        (torch.sigmoid(original["revealed_logits"]) - torch.sigmoid(swapped["revealed_logits"]))
        .abs().mean()
    )
    area_change = float((original["area"] - swapped["area"]).abs().mean())
    swap_ok = spatial_change > 1e-3
    print(f"  spatial change {spatial_change:.5f}, area change {area_change:.4f} m2  "
          f"{'PASS' if swap_ok else 'FAIL'}")
    results["option_swap"] = {
        "spatial_change": spatial_change, "area_change_m2": area_change, "passed": bool(swap_ok)
    }

    # ---- check 6: shuffling options should degrade -------------------------
    print("\n-- check 6: shuffling options across candidates --")
    shuffled = collect(model, groups, device, option_permute=True)
    shuffled_metrics = metrics_from(shuffled)
    degradation = {
        "free_iou": normal_metrics.free_iou - shuffled_metrics.free_iou,
        "occupied_iou": normal_metrics.occupied_iou - shuffled_metrics.occupied_iou,
        "area_mae": shuffled_metrics.area_mae - normal_metrics.area_mae,
    }
    print(f"  free IoU     {normal_metrics.free_iou:.3f} -> {shuffled_metrics.free_iou:.3f} "
          f"(drop {degradation['free_iou']:+.3f})")
    print(f"  occupied IoU {normal_metrics.occupied_iou:.3f} -> {shuffled_metrics.occupied_iou:.3f} "
          f"(drop {degradation['occupied_iou']:+.3f})")
    print(f"  area MAE     {normal_metrics.area_mae:.2f} -> {shuffled_metrics.area_mae:.2f} "
          f"(rise {degradation['area_mae']:+.2f})")
    shuffle_ok = degradation["free_iou"] > 0 or degradation["area_mae"] > 0
    print(f"  performance degrades on shuffle: {'PASS' if shuffle_ok else 'FAIL'}")
    results["option_shuffle"] = {**degradation, "passed": bool(shuffle_ok)}

    # ---- check 7: candidate ordering must not matter -----------------------
    print("\n-- check 7: candidate ordering invariance --")
    group = next(g for g in groups if g["inputs"].shape[0] >= 3)
    n = group["inputs"].shape[0]
    inputs = torch.from_numpy(group["inputs"]).to(device)
    options = torch.from_numpy(group["options"]).to(device)
    goals = torch.tensor([goal_index(group["goal"])] * n, dtype=torch.long, device=device)
    order = torch.randperm(n)
    with torch.no_grad():
        straight = model(inputs, options, goals)
        permuted = model(inputs[order], options[order], goals[order])
    difference = float(
        (straight["revealed_logits"][order] - permuted["revealed_logits"]).abs().max()
    )
    order_ok = difference < 1e-4
    print(f"  max |difference| under permutation: {difference:.2e}  "
          f"{'PASS' if order_ok else 'FAIL'}")
    results["order_invariance"] = {"max_difference": difference, "passed": bool(order_ok)}

    # ---- summary -----------------------------------------------------------
    checks = {
        "1 visualisation written": True,
        "2 per-head metrics reported": True,
        "3 one-group dense heads high": dense_ok,
        "4 invalid cells excluded from loss and metrics": masking_ok,
        "5 swapping omega changes the output": swap_ok,
        "6 shuffling options degrades performance": shuffle_ok,
        "7 candidate order does not matter": order_ok,
    }
    print("\n" + "=" * 62)
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    results["checks"] = {k: bool(v) for k, v in checks.items()}
    (run_dir / "diagnostics.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nresults: {run_dir / 'diagnostics.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
