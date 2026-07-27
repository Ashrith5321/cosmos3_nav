"""Evaluate the six registered Phase 10 gate criteria against the results file.

Thresholds live here, in one place, so a gate cannot be quietly reinterpreted
after the table is seen. Each criterion reports the measured number alongside
its verdict; a PASS with no number attached is not auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Registered tolerances.
COVERAGE_TOLERANCE = 0.05      # |empirical - nominal| for the conformal interval
DEGRADATION_TOLERANCE = 0.02   # relative IoU loss the ensemble may cost vs the best seed
MIN_POSITIVE_SPEARMAN = 0.0    # uncertainty must correlate positively, not merely differ


def criterion_uncertainty_tracks_error(results: dict) -> dict:
    """1. Ensemble uncertainty positively correlates with realised error."""
    measured = {}
    for name, block in results["occupancy"].items():
        measured[f"occupancy_{name}_mi"] = block["uncertainty_error_spearman"]["mutual_information"]
    for head in ("target", "crossing"):
        measured[f"{head}_mi"] = results[head]["uncertainty_error_spearman"]["mutual_information"]
    measured["area_seed_std"] = results["area"]["uncertainty_error_spearman_seed_std"]

    positive = {k: v for k, v in measured.items() if v is not None and v > MIN_POSITIVE_SPEARMAN}
    return {
        "criterion": "uncertainty positively correlates with realised error",
        "measured_spearman": measured,
        "n_positive": len(positive),
        "n_total": len(measured),
        "pass": len(positive) > len(measured) / 2,
        "note": "majority of heads must show positive rank correlation",
    }


def criterion_risk_coverage(results: dict) -> dict:
    """2. Low-uncertainty predictions have lower error (AURC beats random)."""
    comparisons = {}
    for name, block in results["occupancy"].items():
        curve = block["risk_coverage_mutual_information"]
        comparisons[f"occupancy_{name}"] = {"aurc": curve["aurc"], "random": curve["random_aurc"]}
    for head in ("target", "crossing"):
        curve = results[head]["risk_coverage_mutual_information"]
        comparisons[head] = {"aurc": curve["aurc"], "random": curve["random_aurc"]}
    curve = results["area"]["risk_coverage_seed_std"]
    comparisons["area"] = {"aurc": curve["aurc"], "random": curve["random_aurc"]}

    better = {k: v for k, v in comparisons.items() if v["aurc"] < v["random"]}
    return {
        "criterion": "risk-coverage: selective prediction beats random ordering",
        "comparisons": comparisons,
        "n_better": len(better),
        "n_total": len(comparisons),
        "pass": len(better) > len(comparisons) / 2,
    }


def criterion_calibration_improves(results: dict) -> dict:
    """3. Calibration improves Brier or NLL over the raw ensemble."""
    improvements = {}
    for head in ("target", "crossing"):
        raw, calibrated = results[head]["ensemble_raw"], results[head]["ensemble_calibrated"]
        improvements[head] = {
            "brier_raw": raw["brier"], "brier_calibrated": calibrated["brier"],
            "nll_raw": raw["nll"], "nll_calibrated": calibrated["nll"],
            "improved": bool(calibrated["brier"] < raw["brier"] or calibrated["nll"] < raw["nll"]),
        }
    for name, block in results["occupancy"].items():
        raw, calibrated = block["ensemble_raw"], block["ensemble_calibrated"]
        improvements[f"occupancy_{name}"] = {
            "brier_raw": raw["brier"], "brier_calibrated": calibrated["brier"],
            "nll_raw": raw["nll"], "nll_calibrated": calibrated["nll"],
            "improved": bool(calibrated["brier"] < raw["brier"] or calibrated["nll"] < raw["nll"]),
        }
    improved = [k for k, v in improvements.items() if v["improved"]]
    return {
        "criterion": "calibration improves Brier or NLL over the raw ensemble",
        "per_head": improvements,
        "n_improved": len(improved),
        "n_total": len(improvements),
        "pass": len(improved) > len(improvements) / 2,
    }


def criterion_target_beats_prior(results: dict) -> dict:
    """4. Calibrated target Brier beats the constant-prior baseline."""
    prior = results["target"]["constant_prior"]["brier"]
    calibrated = results["target"]["ensemble_calibrated"]["brier"]
    platt = results["target"].get("ensemble_platt", {}).get("brier")
    best = min(x for x in (calibrated, platt) if x is not None)
    bootstrap = results["target"].get("bootstrap_brier_calibrated_minus_raw", {})
    return {
        "criterion": "calibrated target Brier beats the constant prior",
        "constant_prior_brier": prior,
        "temperature_brier": calibrated,
        "platt_brier": platt,
        "best_calibrated_brier": best,
        "margin": prior - best,
        "bootstrap_calibrated_minus_raw": bootstrap,
        "pass": bool(best < prior),
    }


def criterion_area_coverage(results: dict) -> dict:
    """5. Area intervals achieve approximately their registered coverage."""
    interval = results["area"]["conformal_interval"]
    deviation = abs(interval["coverage"] - interval["target_coverage"])
    return {
        "criterion": "conformal area intervals achieve nominal coverage",
        "empirical_coverage": interval["coverage"],
        "target_coverage": interval["target_coverage"],
        "deviation": deviation,
        "tolerance": COVERAGE_TOLERANCE,
        "mean_width_m2": interval["mean_width"],
        "pass": bool(deviation <= COVERAGE_TOLERANCE),
    }


def criterion_no_degradation(results: dict) -> dict:
    """6. Ensemble averaging does not materially degrade deterministic metrics."""
    comparisons = {}
    for name, block in results["occupancy"].items():
        seeds = [s["iou"] for s in block["individual_seeds"]]
        best_seed = max(seeds)
        ensemble = block["ensemble_raw"]["iou"]
        relative = (best_seed - ensemble) / best_seed if best_seed else 0.0
        comparisons[f"occupancy_{name}_iou"] = {
            "best_seed": best_seed, "mean_seed": sum(seeds) / len(seeds),
            "ensemble": ensemble, "relative_loss": relative,
            "ok": bool(relative <= DEGRADATION_TOLERANCE),
        }
    for head in ("target", "crossing"):
        seeds = [s["auroc"] for s in results[head]["individual_seeds"]]
        best_seed = max(seeds)
        ensemble = results[head]["ensemble_raw"]["auroc"]
        relative = (best_seed - ensemble) / best_seed if best_seed else 0.0
        comparisons[f"{head}_auroc"] = {
            "best_seed": best_seed, "mean_seed": sum(seeds) / len(seeds),
            "ensemble": ensemble, "relative_loss": relative,
            "ok": bool(relative <= DEGRADATION_TOLERANCE),
        }
    seeds_mae = results["area"]["individual_seeds_mae"]
    ensemble_mae = results["area"]["ensemble_mae"]
    best_mae = min(seeds_mae)
    relative = (ensemble_mae - best_mae) / best_mae if best_mae else 0.0
    comparisons["area_mae"] = {
        "best_seed": best_mae, "mean_seed": sum(seeds_mae) / len(seeds_mae),
        "ensemble": ensemble_mae, "relative_loss": relative,
        "ok": bool(relative <= DEGRADATION_TOLERANCE),
    }

    failing = [k for k, v in comparisons.items() if not v["ok"]]
    return {
        "criterion": "ensemble averaging does not materially degrade deterministic metrics",
        "comparisons": comparisons,
        "tolerance": DEGRADATION_TOLERANCE,
        "failing": failing,
        "pass": not failing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    gate = {
        "1_uncertainty_tracks_error": criterion_uncertainty_tracks_error(results),
        "2_risk_coverage": criterion_risk_coverage(results),
        "3_calibration_improves": criterion_calibration_improves(results),
        "4_target_beats_prior": criterion_target_beats_prior(results),
        "5_area_coverage": criterion_area_coverage(results),
        "6_no_degradation": criterion_no_degradation(results),
    }
    passed = [k for k, v in gate.items() if v["pass"]]
    gate["summary"] = {
        "n_passed": len(passed),
        "n_total": len(gate),
        "passed": passed,
        "failed": [k for k in gate if k != "summary" and not gate[k]["pass"]],
        "overall_pass": len(passed) == 6,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gate, indent=2, default=float))

    for name, block in gate.items():
        if name == "summary":
            continue
        print(f"{'PASS' if block['pass'] else 'FAIL'}  {name}: {block['criterion']}")
    print(f"\noverall: {gate['summary']['n_passed']}/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
