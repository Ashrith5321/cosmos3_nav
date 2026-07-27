#!/usr/bin/env python
"""Phase 5.5: render decision groups for manual inspection.

One figure per decision group showing the shared decision state and what each
candidate branch revealed, side by side. Rendered from the stored arrays, so no
simulator is needed.

    python scripts/inspect_groups.py outputs/phase5/<run> --groups 50
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.data.dataset import FrontierRevealDataset  # noqa: E402
from frontierworld.data.records import unpack_mask  # noqa: E402
from frontierworld.evaluation.visualize import visualise_decision_group  # noqa: E402
from frontierworld.mapping.occupancy import MapGeometry  # noqa: E402


@dataclass
class StoredFrontier:
    """Enough of a Frontier for the visualiser, rebuilt from the record."""

    frontier_id: int
    cells: np.ndarray
    centroid_world: np.ndarray
    orientation: np.ndarray


@dataclass
class StoredRevelation:
    frontier_id: int
    newly_observed_area_m2: float
    crossed: bool
    collisions: int
    n_new_frontiers: int
    target_became_visible: bool
    newly_semantic_in_revealed_area_m2: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stride", type=int, default=1, help="sample every Nth group")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset)
    out = Path(args.out) if args.out else root / "inspection"
    out.mkdir(parents=True, exist_ok=True)

    dataset = FrontierRevealDataset(root)
    indices = list(range(0, len(dataset), max(1, args.stride)))[: args.groups]
    print(f"dataset: {root}  groups: {len(dataset)}  rendering: {len(indices)}")

    spreads = []
    written = []
    for count, index in enumerate(indices, start=1):
        group = dataset[index]
        arrays = group.arrays
        grid_before = arrays["grid_before"]
        geometry = MapGeometry(
            resolution=float(arrays["resolution"]),
            size_cells=grid_before.shape[0],
            origin_x=float(arrays["map_origin"][0]),
            origin_z=float(arrays["map_origin"][1]),
        )

        frontiers, masks, revelations = [], [], []
        for example in group.examples:
            geom = example["frontier_geometry"]
            targets = example.get("target_arrays", {})
            revelation = example["future_revelation"]

            frontiers.append(
                StoredFrontier(
                    frontier_id=geom["frontier_id"],
                    cells=np.asarray(geom.get("boundary_cells", []), dtype=np.int32).reshape(-1, 2),
                    centroid_world=np.asarray(geom["centroid_world"], dtype=float),
                    orientation=np.asarray(geom["orientation"], dtype=float),
                )
            )
            key = targets.get("revealed_mask")
            masks.append(
                unpack_mask(arrays[key], targets["packed_shape"])
                if key in arrays
                else np.zeros_like(grid_before, dtype=bool)
            )
            revelations.append(
                StoredRevelation(
                    frontier_id=revelation["frontier_id"],
                    newly_observed_area_m2=revelation["newly_observed_area_m2"],
                    crossed=revelation["crossed"],
                    collisions=revelation["collisions"],
                    n_new_frontiers=revelation["n_new_frontiers"],
                    target_became_visible=revelation["target_became_visible"],
                    newly_semantic_in_revealed_area_m2=revelation.get(
                        "newly_semantic_in_revealed_area_m2", 0.0
                    ),
                )
            )

        areas = [r.newly_observed_area_m2 for r in revelations]
        if len(areas) >= 2 and min(areas) > 0.01:
            spreads.append(max(areas) / min(areas))

        path = visualise_decision_group(
            path=out / f"{group.group_id}.png",
            grid_before=grid_before,
            geometry=geometry,
            frontiers=frontiers,
            branch_masks=masks,
            revelations=revelations,
            rgb_before=None,
            meta={
                "group_id": group.group_id,
                "navigation_goal": group.navigation_goal,
                "decision_timestep": group.decision_timestep,
                "collection_policy": group.examples[0].get("collection_policy"),
            },
        )
        written.append(path)
        if count % 10 == 0:
            print(f"  {count}/{len(indices)}", flush=True)

    manifest = {
        "dataset": str(root),
        "rendered": len(written),
        "figures": [str(p.relative_to(out)) for p in written],
        "spread_median": float(np.median(spreads)) if spreads else None,
        "spread_max": float(np.max(spreads)) if spreads else None,
    }
    (out / "inspection.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nrendered {len(written)} group figures to {out}")
    if spreads:
        print(
            f"within-group revealed-area spread: median {np.median(spreads):.1f}x  "
            f"max {np.max(spreads):.0f}x"
        )
    print(f"inspect with: xdg-open {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
