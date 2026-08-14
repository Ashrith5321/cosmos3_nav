#!/usr/bin/env python
"""
Offline frontier-ranking evaluation against the oracle (design section 38).

Reads wm_state.jsonl files written by episodes run with --wm-oracle and a
world-model config, and reports:
  - Spearman rank correlation between predicted utility Q and oracle value
  - top-1 agreement (argmax Q == argmax oracle)
  - the same metrics for the current-frame VLM prior alone (p_obs) and for
    the world-model score alone (wm_mean), to attribute the gain

Usage:
    python scripts/analyze_wm_ranking.py 'output/<run>/**/wm_state.jsonl'
"""

import glob
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def decisions(paths):
    for path in paths:
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                oracle = entry.get("oracle")
                if not oracle:
                    continue
                rows = [
                    r
                    for r in oracle["frontiers"]
                    if r.get("oracle_value") is not None
                    and r.get("utility") is not None
                ]
                if len(rows) >= 2:
                    yield rows


def evaluate(paths):
    stats = {
        key: {"rho": [], "top1": []}
        for key in ("utility", "p_obs", "wm_mean")
    }
    n_decisions = 0
    for rows in decisions(paths):
        n_decisions += 1
        oracle = np.array([r["oracle_value"] for r in rows], dtype=float)
        for key in stats:
            vals = [r.get(key) for r in rows]
            if any(v is None for v in vals):
                continue
            pred = np.array(vals, dtype=float)
            if np.ptp(pred) < 1e-12 or np.ptp(oracle) < 1e-12:
                continue
            rho = spearmanr(pred, oracle).correlation
            if np.isfinite(rho):
                stats[key]["rho"].append(rho)
            stats[key]["top1"].append(
                1.0 if int(np.argmax(pred)) == int(np.argmax(oracle)) else 0.0
            )

    print(f"decisions with oracle: {n_decisions}")
    for key, s in stats.items():
        if not s["top1"]:
            print(f"{key:>10}: no data")
            continue
        rho = np.mean(s["rho"]) if s["rho"] else float("nan")
        print(
            f"{key:>10}: spearman rho = {rho:+.3f} "
            f"(n={len(s['rho'])}), top-1 = {np.mean(s['top1']):.3f} "
            f"(n={len(s['top1'])})"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    paths = sorted(
        p for pattern in sys.argv[1:] for p in glob.glob(pattern, recursive=True)
    )
    if not paths:
        print("No wm_state.jsonl files matched")
        sys.exit(1)
    print(f"analyzing {len(paths)} files")
    evaluate(paths)
