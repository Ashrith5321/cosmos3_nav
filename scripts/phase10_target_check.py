"""Is the calibrated target head actually better than a constant prior?

Gate criterion 4 compared two Brier scores and found a margin of 0.00026. A
margin that small is only meaningful if it survives resampling, and the whole
point of a gate is that it should not be passable by a difference in the fourth
decimal place. This re-runs the scalar heads only and bootstraps the comparison
by decision group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from frontierworld.data.dataset import FrontierRevealDataset
from frontierworld.models.ensemble import (
    PlattScaler,
    TemperatureScaler,
    brier,
    ensemble_mean,
    expected_calibration_error,
    negative_log_likelihood,
)
from frontierworld.models.metrics import auroc, paired_bootstrap

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase10_ensemble import load_seeds, predict_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_seeds(list(args.checkpoints), device)

    calibration = predict_split(models, FrontierRevealDataset(args.calibration), device)
    validation = predict_split(models, FrontierRevealDataset(args.validation), device)

    report: dict = {}
    for head, key in (("target", "target_logits"), ("crossing", "crossing_logits")):
        calibration_logits = ensemble_mean(calibration[key])
        calibration_labels = calibration["truth"][head]
        validation_logits = ensemble_mean(validation[key])
        labels = validation["truth"][head]
        groups = validation["group_ids"]

        temperature = TemperatureScaler().fit(calibration_logits, calibration_labels)
        platt = PlattScaler().fit(calibration_logits, calibration_labels)

        # The prior is a property of the CALIBRATION split, not the validation
        # labels; using the validation base rate would give the baseline
        # information the model never had.
        prior_rate = float(calibration_labels.mean())
        prior = np.full(labels.shape, prior_rate)

        candidates = {
            "temperature": temperature.transform(validation_logits),
            "platt": platt.transform(validation_logits),
        }

        entry = {
            "calibration_base_rate": prior_rate,
            "validation_base_rate": float(labels.mean()),
            "constant_prior": {
                "brier": brier(prior, labels),
                "nll": negative_log_likelihood(prior, labels),
                "ece": expected_calibration_error(prior, labels),
            },
            "auroc_ensemble": float(auroc(ensemble_mean(1 / (1 + np.exp(-validation[key]))), labels)),
        }

        for name, probability in candidates.items():
            comparison = paired_bootstrap(
                (probability - labels) ** 2,   # calibrated squared error
                (prior - labels) ** 2,         # prior squared error
                groups,
                iterations=args.bootstrap,
                seed=args.seed,
            )
            entry[name] = {
                "brier": brier(probability, labels),
                "nll": negative_log_likelihood(probability, labels),
                "ece": expected_calibration_error(probability, labels),
                "brier_margin_over_prior": entry["constant_prior"]["brier"] - brier(probability, labels),
                "bootstrap_calibrated_minus_prior": comparison,
                "beats_prior_significantly": bool(
                    comparison["significant"] and comparison["mean_difference"] < 0
                ),
            }

        entry["verdict"] = (
            "beats prior"
            if any(entry[n]["beats_prior_significantly"] for n in candidates)
            else "does NOT significantly beat the constant prior"
        )
        report[head] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float))
    for head, entry in report.items():
        print(f"\n=== {head} (AUROC {entry['auroc_ensemble']:.4f}) ===")
        print(f"  constant prior Brier {entry['constant_prior']['brier']:.5f}")
        for name in ("temperature", "platt"):
            block = entry[name]
            ci = block["bootstrap_calibrated_minus_prior"]
            print(
                f"  {name:12s} Brier {block['brier']:.5f}  margin {block['brier_margin_over_prior']:+.5f}  "
                f"CI [{ci['ci_low']:+.5f}, {ci['ci_high']:+.5f}]  significant={ci['significant']}"
            )
        print(f"  verdict: {entry['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
