#!/usr/bin/env python
"""Phase 5.5: dataset integrity checks and per-channel summary plots.

    python scripts/check_dataset.py outputs/phase5/<run>
    python scripts/check_dataset.py outputs/phase5/<run> --manifest manifests/pilot_debug.json

Exits nonzero if any integrity check fails, so this can gate downstream use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.data import integrity  # noqa: E402
from frontierworld.data.dataset import FrontierRevealDataset, collate_groups  # noqa: E402

PLOT_CHANNELS = [
    ("newly_observed_area_m2", "newly revealed area (m2)"),
    ("newly_free_area_m2", "newly revealed free area (m2)"),
    ("newly_semantic_in_revealed_area_m2", "semantics in revealed area (m2)"),
    ("n_new_frontiers", "new frontiers exposed"),
    ("collisions", "collisions per branch"),
    ("distance_travelled_m", "distance travelled (m)"),
    ("actions_executed", "actions executed"),
    ("geodesic_distance_to_target_m", "geodesic distance to target (m)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset root containing index.jsonl")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    return parser.parse_args()


def summary_plots(dataset: FrontierRevealDataset, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values: dict[str, list[float]] = {key: [] for key, _ in PLOT_CHANNELS}
    branches_per_group: list[int] = []
    target_positive = 0
    total = 0
    spread: list[float] = []

    for group in dataset:
        branches_per_group.append(group.n_branches)
        areas = []
        for revelation in group.revelations():
            total += 1
            target_positive += int(bool(revelation.get("target_became_visible")))
            areas.append(float(revelation.get("newly_observed_area_m2", 0.0)))
            for key, _ in PLOT_CHANNELS:
                value = revelation.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values[key].append(float(value))
        if len(areas) >= 2 and min(areas) > 0.01:
            spread.append(max(areas) / min(areas))

    rows, cols = 3, 4
    figure, axes = plt.subplots(rows, cols, figsize=(19, 11), constrained_layout=True)
    axes = axes.ravel()

    for index, (key, label) in enumerate(PLOT_CHANNELS):
        axis = axes[index]
        data = [v for v in values[key] if np.isfinite(v)]
        if data:
            axis.hist(data, bins=30, color="#2b7bba", edgecolor="white", linewidth=0.4)
            axis.axvline(np.mean(data), color="#d1495b", linewidth=1.5, label=f"mean {np.mean(data):.2f}")
            axis.legend(fontsize=7)
        axis.set_title(label, fontsize=9)
        axis.tick_params(labelsize=7)

    axis = axes[len(PLOT_CHANNELS)]
    axis.hist(branches_per_group, bins=range(1, max(branches_per_group or [2]) + 2),
              color="#4c956c", edgecolor="white", linewidth=0.4, align="left")
    axis.set_title("branches per decision group", fontsize=9)
    axis.tick_params(labelsize=7)

    # The spread is the premise of the paper: if candidates from one state all
    # revealed the same amount, there would be nothing to predict.
    axis = axes[len(PLOT_CHANNELS) + 1]
    if spread:
        axis.hist(np.log10(spread), bins=30, color="#e07a5f", edgecolor="white", linewidth=0.4)
        axis.set_xlabel("log10(max/min revealed area within a group)", fontsize=7)
        axis.axvline(np.median(np.log10(spread)), color="#3d405b", linewidth=1.5,
                     label=f"median {np.median(spread):.1f}x")
        axis.legend(fontsize=7)
    axis.set_title("within-group counterfactual spread", fontsize=9)
    axis.tick_params(labelsize=7)

    axis = axes[len(PLOT_CHANNELS) + 2]
    rate = target_positive / total if total else 0.0
    axis.bar(["target seen", "not seen"], [rate, 1 - rate], color=["#2a9d8f", "#adb5bd"])
    axis.set_ylim(0, 1)
    axis.set_title(f"target revelation rate ({rate:.1%})", fontsize=9)
    axis.tick_params(labelsize=7)

    for axis in axes[len(PLOT_CHANNELS) + 3 :]:
        axis.axis("off")

    figure.suptitle(
        f"FrontierReveal channel summary -- {len(dataset)} groups, {total} branches",
        fontsize=12,
    )
    path = out / "channel_summary.png"
    figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    root = Path(args.dataset)

    manifest = None
    if args.manifest:
        from frontierworld.data.manifests import SplitManifest

        manifest = SplitManifest.load(args.manifest)

    dataset = FrontierRevealDataset(root, split=args.split, manifest=manifest)
    print(f"dataset: {root}")
    print(f"groups : {len(dataset)}")
    if len(dataset) == 0:
        print("empty dataset", file=sys.stderr)
        return 1

    # -- the Phase 5 gate ---------------------------------------------------
    group = dataset[0]
    print(f"\ngate: dataloader returns all branches of one common state")
    print(f"  group {group.group_id}: {group.n_branches} branches, "
          f"frontier ids {group.frontier_ids()}")
    batch = collate_groups([dataset[i] for i in range(min(4, len(dataset)))])
    print(f"  collated batch: mask {batch['branch_mask'].shape}, "
          f"branches per group {batch['n_branches'].tolist()}")

    # -- integrity ----------------------------------------------------------
    print("\nintegrity checks:")
    results = integrity.run_all(dataset, manifest=manifest)
    for result in results:
        print(f"  {result}")
    failed = [r for r in results if not r.passed]

    # -- statistics ---------------------------------------------------------
    stats = dataset.statistics()
    print(f"\nstatistics:")
    print(f"  scenes                : {stats['scenes']}")
    print(f"  groups                : {stats['groups']}")
    print(f"  branches              : {stats['branches']}")
    print(f"  branches per group    : {stats['branches_per_group_mean']:.2f} "
          f"(min {stats['branches_per_group_min']}, max {stats['branches_per_group_max']})")
    print(f"  target-positive rate  : {stats['target_positive_rate']:.1%}")
    print(f"  crossing success rate : {stats['crossing_success_rate']:.1%}")
    for key in ("newly_observed_area_m2", "newly_semantic_in_revealed_area_m2",
                "n_new_frontiers", "collisions"):
        channel = stats["channels"].get(key)
        if channel:
            print(f"  {key:36s} mean {channel['mean']:7.2f}  sd {channel['std']:6.2f}  "
                  f"max {channel['max']:7.2f}")

    (root / "statistics.json").write_text(json.dumps(stats, indent=2))
    (root / "integrity.json").write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    "n_checked": r.n_checked,
                    "failures": r.failures[:50],
                }
                for r in results
            ],
            indent=2,
        )
    )

    if args.plots:
        path = summary_plots(dataset, root)
        print(f"\nchannel summary plot: {path}")

    print(f"\nstatistics: {root / 'statistics.json'}")
    if failed:
        print(f"\n{len(failed)} INTEGRITY CHECK(S) FAILED", file=sys.stderr)
        return 1
    print("all integrity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
