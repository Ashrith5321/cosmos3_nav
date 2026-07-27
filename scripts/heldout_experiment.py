#!/usr/bin/env python
"""Phase 8 held-out experiment.

Protocol, followed exactly:

    1  freeze architecture, losses, thresholds and metric definitions
    2  save the exact train / val / test manifests with the results
    3  train three seeds independently on TRAIN scenes only
    4  select one checkpoint per seed using VAL only
    5  evaluate each selected checkpoint exactly ONCE on TEST
    6  run every baseline on the identical test examples
    7  paired bootstrap CIs resampling DECISION GROUPS, not branches
    8  change nothing based on test performance

The test split is touched once, at the end. Nothing in this script reads a test
metric before that point, and no decision -- checkpoint, threshold, or
architecture -- depends on it.

    python scripts/heldout_experiment.py \\
        --train outputs/phase5/full_train \\
        --val   outputs/phase5/full_val \\
        --test  outputs/phase5/full_test --seeds 0 1 2
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frontierworld.config import config_hash, load_config, save_config  # noqa: E402
from frontierworld.data.dataset import FrontierRevealDataset  # noqa: E402
from frontierworld.data.tensors import FrameSpec  # noqa: E402
from frontierworld.evaluation import environment_provenance, make_run_id  # noqa: E402
from frontierworld.models.baselines import BASELINES  # noqa: E402
from frontierworld.models.metrics import compute_head_metrics, paired_bootstrap  # noqa: E402
from frontierworld.models.predictor import (  # noqa: E402
    PredictorConfig,
    RevelationPredictor,
    revelation_loss,
)
from frontierworld.seeding import seed_everything  # noqa: E402
from diagnose_predictor import collect, metrics_from  # noqa: E402
from train_predictor import batches, tensorise, to_torch  # noqa: E402

# FROZEN for this experiment. Recorded in the results so the table can never be
# read without them.
DECISION_THRESHOLD = 0.5
LOSS_WEIGHTS = {"occ": 1.0, "sem": 0.5, "goal": 1.0, "cross": 0.5, "area": 0.1}
SELECTION_METRIC = "free_iou"  # on VAL only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--manifest", default="manifests/full.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-groups", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def group_ids_for(groups: list[dict]) -> np.ndarray:
    """One group id per branch, aligned with the collected prediction order."""
    ids = []
    for batch in batches(groups, 4, shuffle=False):
        mask = batch["candidate_mask"]
        for index, group_id in enumerate(batch["group_ids"]):
            ids.extend([group_id] * int(mask[index].sum()))
    return np.asarray(ids)


def per_branch_scores(collected: dict, metric: str = "free_iou") -> np.ndarray:
    """Per-branch score, for the bootstrap. Group-level aggregation happens in
    the resampler, so this stays at branch granularity."""
    from frontierworld.models.metrics import masked_iou

    channel = {"free_iou": 0, "occupied_iou": 1, "semantic_iou": 2}[metric]
    return np.asarray(
        [
            masked_iou(
                collected["probabilities"][i, channel],
                collected["targets"][i, channel],
                collected["valid"][i],
                DECISION_THRESHOLD,
            )
            for i in range(collected["probabilities"].shape[0])
        ]
    )


def baseline_scores(baseline, groups: list[dict], metric: str = "free_iou") -> tuple:
    """Per-branch score and area error for a baseline, in collect() order."""
    from frontierworld.models.metrics import masked_iou

    channel = {"free_iou": 0, "occupied_iou": 1, "semantic_iou": 2}[metric]
    scores, area_errors = [], []
    probabilities, targets, valid = [], [], []
    target_scores, target_labels = [], []
    crossing_scores, crossing_labels = [], []
    areas, area_labels = [], []

    for batch in batches(groups, 4, shuffle=False):
        mask = batch["candidate_mask"]
        inputs, target = batch["inputs"][mask], batch["targets"][mask]
        window, options = batch["target_valid"][mask], batch["options"][mask]
        scalars = {k: v[mask] for k, v in batch["scalars"].items()}

        for index in range(inputs.shape[0]):
            prediction = baseline.predict(inputs[index], options[index], None)
            revealed = (
                prediction.revealed
                if prediction.revealed is not None
                else np.zeros_like(target[index])
            )
            scores.append(
                masked_iou(revealed[channel], target[index, channel], window[index],
                           DECISION_THRESHOLD)
            )
            area_errors.append(
                abs(prediction.revealed_area_m2 - scalars["revealed_area_m2"][index])
            )
            probabilities.append(revealed)
            targets.append(target[index])
            valid.append(window[index])
            target_scores.append(prediction.target_present)
            target_labels.append(scalars["target_present"][index])
            crossing_scores.append(prediction.crossing_success)
            crossing_labels.append(scalars["crossing_success"][index])
            areas.append(prediction.revealed_area_m2)
            area_labels.append(scalars["revealed_area_m2"][index])

    metrics = compute_head_metrics(
        np.stack(probabilities), np.stack(targets), np.stack(valid),
        np.asarray(target_scores), np.asarray(target_labels),
        np.asarray(crossing_scores), np.asarray(crossing_labels),
        np.asarray(areas), np.asarray(area_labels),
    )
    return np.asarray(scores), np.asarray(area_errors), metrics


def train_one_seed(seed, train_groups, val_groups, args, device, option_dim, run_dir):
    """Train one seed; select the checkpoint on VAL only."""
    seed_everything(seed, torch_deterministic=False)
    model = RevelationPredictor(PredictorConfig(option_dim=option_dim)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = {"score": -np.inf, "epoch": -1, "state": None}
    history = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in batches(train_groups, args.batch_groups, shuffle=True, seed=seed * 1000 + epoch):
            data = to_torch(batch, device)
            if data["inputs"].shape[0] == 0:
                continue
            prediction = model(data["inputs"], data["options"], data["goals"])
            loss, parts = revelation_loss(
                prediction, data["targets"], data["valid"], data["scalars"], LOSS_WEIGHTS
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(parts["loss"])

        record = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            validation = metrics_from(collect(model, val_groups, device))
            score = getattr(validation, SELECTION_METRIC)
            record["val_" + SELECTION_METRIC] = float(score)
            record["val_area_mae"] = float(validation.area_mae)
            if score > best["score"]:
                best = {
                    "score": float(score),
                    "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                }
            print(f"  seed {seed} epoch {epoch:3d}  loss {record['train_loss']:.4f}  "
                  f"val {SELECTION_METRIC} {score:.4f}"
                  f"{'  <- best' if best['epoch'] == epoch else ''}", flush=True)
        history.append(record)

    model.load_state_dict(best["state"])
    torch.save(
        {"state_dict": best["state"], "config": model.config.__dict__,
         "seed": seed, "selected_epoch": best["epoch"], "val_score": best["score"]},
        run_dir / f"checkpoint_seed{seed}.pt",
    )
    print(f"  seed {seed}: selected epoch {best['epoch']} "
          f"(val {SELECTION_METRIC} {best['score']:.4f})")
    return model, {"seed": seed, "selected_epoch": best["epoch"],
                   "val_score": best["score"], "history": history}


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    spec = FrameSpec()

    run_dir = Path(cfg.experiment.output_dir) / "phase8_heldout" / make_run_id(
        "heldout", config_hash(cfg)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "provenance.json").write_text(json.dumps(environment_provenance(), indent=2))

    # -- step 2: freeze and save the manifests ------------------------------
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        (run_dir / "manifest.json").write_text(manifest_path.read_text())

    train_dataset = FrontierRevealDataset(args.train)
    val_dataset = FrontierRevealDataset(args.val)
    test_dataset = FrontierRevealDataset(args.test)

    splits = {
        "train": sorted(train_dataset.scenes()),
        "val": sorted(val_dataset.scenes()),
        "test": sorted(test_dataset.scenes()),
    }
    overlaps = {
        "train_val": sorted(set(splits["train"]) & set(splits["val"])),
        "train_test": sorted(set(splits["train"]) & set(splits["test"])),
        "val_test": sorted(set(splits["val"]) & set(splits["test"])),
    }
    (run_dir / "splits.json").write_text(json.dumps({"splits": splits, "overlaps": overlaps}, indent=2))

    print(f"run_dir: {run_dir}\ndevice : {device}")
    for name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        print(f"  {name:5s} {len(dataset):4d} groups  {len(dataset.scenes()):3d} scenes")
    for name, shared in overlaps.items():
        print(f"  overlap {name}: {len(shared)} {'OK' if not shared else 'LEAK ' + str(shared[:3])}")
    if any(overlaps.values()):
        print("scene leakage between splits; aborting", file=sys.stderr)
        return 1

    train_groups = tensorise(train_dataset, spec, 0)
    val_groups = tensorise(val_dataset, spec, 0)
    test_groups = tensorise(test_dataset, spec, 0)
    option_dim = train_groups[0]["options"].shape[1]

    frozen = {
        "architecture": PredictorConfig(option_dim=option_dim).__dict__,
        "loss_weights": LOSS_WEIGHTS,
        "decision_threshold": DECISION_THRESHOLD,
        "selection_metric": SELECTION_METRIC,
        "frame": {"extent_m": spec.extent_m, "resolution": spec.resolution},
    }
    (run_dir / "frozen.json").write_text(json.dumps(frozen, indent=2, default=str))
    print(f"\nfrozen: threshold {DECISION_THRESHOLD}, selection on val "
          f"{SELECTION_METRIC}, weights {LOSS_WEIGHTS}")

    # -- steps 3-4: train seeds, select on val ------------------------------
    print(f"\n== training {len(args.seeds)} seeds on TRAIN, selecting on VAL ==")
    started = time.perf_counter()
    models, seed_records = [], []
    for seed in args.seeds:
        model, record = train_one_seed(
            seed, train_groups, val_groups, args, device, option_dim, run_dir
        )
        models.append(model)
        seed_records.append(record)

    # -- step 5: ONE evaluation on test -------------------------------------
    print(f"\n== evaluating on TEST ({len(test_groups)} groups) -- first and only look ==")
    test_group_ids = group_ids_for(test_groups)
    per_seed, model_scores, model_area_errors = [], [], []
    for seed, model in zip(args.seeds, models):
        collected = collect(model, test_groups, device)
        metrics = metrics_from(collected)
        per_seed.append({"seed": seed, **metrics.to_dict()})
        model_scores.append(per_branch_scores(collected, "free_iou"))
        model_area_errors.append(np.abs(collected["areas"] - collected["area_labels"]))
        print(f"  seed {seed}: free IoU {metrics.free_iou:.3f}  occ IoU {metrics.occupied_iou:.3f}  "
              f"tgt AUROC {metrics.target_auroc:.3f}  area MAE {metrics.area_mae:.2f}")

    def mean_std(key: str) -> tuple[float, float]:
        values = [record[key] for record in per_seed if np.isfinite(record[key])]
        return (float(np.mean(values)), float(np.std(values))) if values else (float("nan"),) * 2

    # -- step 6: baselines on identical test examples ------------------------
    print("\n== baselines on the identical test examples ==")
    train_batches = list(batches(train_groups, 4, shuffle=False))
    baseline_rows, baseline_score_sets = {}, {}
    for name, factory in BASELINES.items():
        fitted = factory().fit(train_batches)  # fitted on TRAIN only
        scores, area_errors, metrics = baseline_scores(fitted, test_groups, "free_iou")
        baseline_rows[name] = metrics.to_dict()
        baseline_score_sets[name] = (scores, area_errors)
        print(f"  {name:<28} free IoU {metrics.free_iou:.3f}  occ IoU {metrics.occupied_iou:.3f}  "
              f"area MAE {metrics.area_mae:.2f}")

    # -- step 7: paired bootstrap over decision GROUPS -----------------------
    print("\n== paired bootstrap (resampling decision groups) ==")
    mean_model_scores = np.mean(model_scores, axis=0)
    mean_model_area = np.mean(model_area_errors, axis=0)
    comparisons = {}
    for name, (scores, area_errors) in baseline_score_sets.items():
        iou = paired_bootstrap(mean_model_scores, scores, test_group_ids)
        area = paired_bootstrap(area_errors, mean_model_area, test_group_ids)  # baseline - model
        comparisons[name] = {"free_iou": iou, "area_mae_reduction": area}
        print(f"  vs {name:<28} free IoU  {iou['mean_difference']:+.3f} "
              f"[{iou['ci_low']:+.3f}, {iou['ci_high']:+.3f}] "
              f"{'SIGNIFICANT' if iou['significant'] else 'n.s.'}  "
              f"({iou['n_groups']} groups, {iou['n_branches']} branches)")
        print(f"  {'':31} area MAE  {area['mean_difference']:+.2f} m2 "
              f"[{area['ci_low']:+.2f}, {area['ci_high']:+.2f}] "
              f"{'SIGNIFICANT' if area['significant'] else 'n.s.'}")

    # -- report --------------------------------------------------------------
    print("\n" + "=" * 74)
    print("PHASE 8 HELD-OUT RESULTS (mean +/- sd over "
          f"{len(args.seeds)} seeds, test scenes)")
    print("=" * 74)
    rows = [
        ("Revelation occupancy", "free IoU", "free_iou"),
        ("", "occupied IoU", "occupied_iou"),
        ("", "macro IoU", "macro_iou"),
        ("Semantics (revealed)", "IoU", "semantic_iou"),
        ("", "cosine", "semantic_cosine"),
        ("Target presence", "AUROC", "target_auroc"),
        ("", "AUPRC", "target_auprc"),
        ("", "Brier", "target_brier"),
        ("Crossing", "accuracy", "crossing_accuracy"),
        ("", "AUROC", "crossing_auroc"),
        ("Revealed area", "MAE (m2)", "area_mae"),
        ("", "correlation", "area_correlation"),
    ]
    print(f"{'Head':<22}{'Metric':<16}{'mean':>9}{'sd':>8}   per-seed")
    print("-" * 74)
    for head, metric, key in rows:
        mean, sd = mean_std(key)
        values = "  ".join(f"{record[key]:.3f}" for record in per_seed)
        print(f"{head:<22}{metric:<16}{mean:>9.3f}{sd:>8.3f}   {values}")

    strongest = max(
        baseline_rows,
        key=lambda n: baseline_rows[n]["free_iou"]
        if np.isfinite(baseline_rows[n]["free_iou"]) else -np.inf,
    )
    beats_occupancy = comparisons[strongest]["free_iou"]["significant"] and (
        comparisons[strongest]["free_iou"]["mean_difference"] > 0
    )
    scalar_wins = {
        "area_mae": comparisons[strongest]["area_mae_reduction"]["significant"]
        and comparisons[strongest]["area_mae_reduction"]["mean_difference"] > 0,
        "target_auroc": mean_std("target_auroc")[0]
        > (baseline_rows[strongest]["target_auroc"]
           if np.isfinite(baseline_rows[strongest]["target_auroc"]) else 0.5),
    }
    passed = beats_occupancy and any(scalar_wins.values())

    print(f"\nstrongest non-learned baseline: {strongest}")
    print(f"  beats it on occupancy (paired, group-level CI): {beats_occupancy}")
    print(f"  decision-relevant scalar wins: {scalar_wins}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")

    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "frozen": frozen,
                "splits": splits,
                "overlaps": overlaps,
                "seeds": seed_records,
                "test_per_seed": per_seed,
                "test_mean": {key: mean_std(key)[0] for _, _, key in rows},
                "test_sd": {key: mean_std(key)[1] for _, _, key in rows},
                "baselines": baseline_rows,
                "comparisons": comparisons,
                "strongest_baseline": strongest,
                "gate": {
                    "beats_occupancy": bool(beats_occupancy),
                    "scalar_wins": {k: bool(v) for k, v in scalar_wins.items()},
                    "passed": bool(passed),
                },
                "train_seconds": time.perf_counter() - started,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nresults: {run_dir / 'results.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
