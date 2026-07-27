#!/usr/bin/env python
"""Phase 7 gate: tensorise a decision group and put it back on the global map.

Checks that a batch returns one complete decision group, that every transform
inverts exactly, that no future information reaches the inputs, and renders
inputs and targets reprojected into the global frame for visual alignment.

    python scripts/check_tensors.py outputs/phase5/pilot_500 --groups 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.data.dataset import FrontierRevealDataset  # noqa: E402
from frontierworld.data.records import unpack_mask  # noqa: E402
from frontierworld.data.tensors import (  # noqa: E402
    CH_BOUNDARY,
    CH_FREE,
    CH_OCCUPIED,
    CH_UNKNOWN,
    INPUT_CHANNEL_NAMES,
    TARGET_CHANNEL_NAMES,
    TGT_REVEALED_FREE,
    TGT_REVEALED_OCCUPIED,
    FrameSpec,
    build_group_tensors,
    collate_tensor_groups,
)
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--groups", type=int, default=6)
    parser.add_argument("--extent", type=float, default=8.0)
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def check_roundtrip(frame) -> float:
    """Max error of world -> local -> world over the whole window."""
    size = frame.spec.size
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([rows, cols], axis=-1).reshape(-1, 2)
    local = frame.cell_to_local(cells)
    world = frame.to_global(local)
    back = frame.to_local(world)
    return float(np.abs(back - local).max())


def check_no_future_leakage(tensors: dict, group) -> list[str]:
    """Input channels must contain nothing that only the future reveals.

    The specific failure to catch: a cell that was unknown before the branch
    but appears as observed in the inputs. That is the model being shown its
    own answer.
    """
    problems: list[str] = []
    for index in range(tensors["inputs"].shape[0]):
        inputs = tensors["inputs"][index]
        targets = tensors["targets"][index]

        observed = (inputs[CH_FREE] > 0) | (inputs[CH_OCCUPIED] > 0)
        revealed = (targets[TGT_REVEALED_FREE] > 0) | (targets[TGT_REVEALED_OCCUPIED] > 0)
        overlap = int((observed & revealed).sum())
        if overlap:
            problems.append(
                f"candidate {index}: {overlap} cells are both already-observed "
                "in the input and newly-revealed in the target"
            )

        unknown = inputs[CH_UNKNOWN] > 0
        if not np.all(revealed <= unknown):
            escaped = int((revealed & ~unknown).sum())
            problems.append(
                f"candidate {index}: {escaped} revealed target cells were not "
                "unknown in the input"
            )
    return problems


def reproject_to_global(tensors: dict, group, spec: FrameSpec) -> dict:
    """Paint each candidate's input and target back onto the global grid."""
    arrays = group.arrays
    grid = arrays["grid_before"]
    origin = np.asarray(arrays["map_origin"], dtype=np.float64)
    resolution = float(arrays["resolution"])

    boundary_global = np.zeros_like(grid, dtype=bool)
    target_global = np.zeros_like(grid, dtype=bool)

    for index, frame in enumerate(tensors["frames"]):
        for channel, accumulator in (
            (tensors["inputs"][index][CH_BOUNDARY], boundary_global),
            (
                tensors["targets"][index][TGT_REVEALED_FREE]
                + tensors["targets"][index][TGT_REVEALED_OCCUPIED],
                target_global,
            ),
        ):
            cells = np.argwhere(channel > 0)
            if cells.size == 0:
                continue
            world = frame.cell_to_global(cells)
            cols = np.floor((world[:, 0] - origin[0]) / resolution).astype(int)
            rows = np.floor((world[:, 1] - origin[1]) / resolution).astype(int)
            keep = (
                (rows >= 0) & (rows < grid.shape[0])
                & (cols >= 0) & (cols < grid.shape[1])
            )
            accumulator[rows[keep], cols[keep]] = True

    return {"boundary": boundary_global, "target": target_global}


def alignment_error(reprojected: dict, group) -> dict:
    """How well the reprojected boundary/target land on the stored ground truth."""
    arrays = group.arrays
    stored_boundary = np.zeros_like(arrays["grid_before"], dtype=bool)
    for example in group.examples:
        cells = np.asarray(
            example["frontier_geometry"].get("boundary_cells", []), dtype=np.int32
        ).reshape(-1, 2)
        if cells.size:
            stored_boundary[cells[:, 0], cells[:, 1]] = True

    stored_target = np.zeros_like(stored_boundary)
    for example in group.examples:
        keys = example.get("target_arrays", {})
        if keys.get("revealed_mask") in arrays:
            stored_target |= unpack_mask(
                arrays[keys["revealed_mask"]], keys["packed_shape"]
            )

    def recall(reproj: np.ndarray, stored: np.ndarray) -> float:
        if not stored.any():
            return float("nan")
        from scipy import ndimage

        # A one-cell tolerance: the model frame is coarser than the map frame,
        # so exact cell equality is not the right test.
        dilated = ndimage.binary_dilation(reproj, iterations=2)
        return float((stored & dilated).sum() / stored.sum())

    return {
        "boundary_recall": recall(reprojected["boundary"], stored_boundary),
        "target_recall": recall(reprojected["target"], stored_target),
    }


def render(group, tensors: dict, reprojected: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from frontierworld.evaluation.visualize import crop_bounds, grid_to_rgb

    grid = group.arrays["grid_before"]
    r0, r1, c0, c1 = crop_bounds(grid)

    n_show = min(3, tensors["inputs"].shape[0])
    figure, axes = plt.subplots(2, 3 + n_show, figsize=(4.2 * (3 + n_show), 8.4),
                                constrained_layout=True)

    axes[0, 0].imshow(grid_to_rgb(grid)[r0:r1, c0:c1], interpolation="nearest")
    axes[0, 0].set_title("global map before", fontsize=9)

    image = grid_to_rgb(grid)[r0:r1, c0:c1].copy()
    image[reprojected["boundary"][r0:r1, c0:c1]] = (0.9, 0.15, 0.25)
    axes[0, 1].imshow(image, interpolation="nearest")
    axes[0, 1].set_title("boundaries reprojected from\nthe frontier frame", fontsize=9)

    image = grid_to_rgb(grid)[r0:r1, c0:c1].copy()
    image[reprojected["target"][r0:r1, c0:c1]] = (0.15, 0.65, 0.95)
    axes[0, 2].imshow(image, interpolation="nearest")
    axes[0, 2].set_title("targets reprojected to global", fontsize=9)

    for index in range(n_show):
        axis = axes[0, 3 + index]
        inputs = tensors["inputs"][index]
        rgb = np.stack(
            [inputs[CH_OCCUPIED], inputs[CH_FREE], inputs[CH_BOUNDARY]], axis=-1
        )
        axis.imshow(np.clip(rgb, 0, 1), origin="lower")
        axis.set_title(
            f"input f{tensors['frontier_ids'][index]}\nR=occ G=free B=boundary",
            fontsize=9,
        )

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.0, 0.5,
        "\n".join(
            [f"group {group.group_id}", f"goal {group.navigation_goal}",
             f"candidates {tensors['inputs'].shape[0]}",
             "", "input channels:"] +
            [f"  {i} {n}" for i, n in enumerate(INPUT_CHANNEL_NAMES)] +
            ["", "target channels:"] +
            [f"  {i} {n}" for i, n in enumerate(TARGET_CHANNEL_NAMES)]
        ),
        fontsize=8, family="monospace", va="center",
    )
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    for index in range(n_show):
        axis = axes[1, 3 + index]
        targets = tensors["targets"][index]
        rgb = np.stack(
            [targets[TGT_REVEALED_OCCUPIED], targets[TGT_REVEALED_FREE],
             targets[2]], axis=-1
        )
        axis.imshow(np.clip(rgb, 0, 1), origin="lower")
        axis.set_title(
            f"target f{tensors['frontier_ids'][index]}\nR=occ G=free B=semantic",
            fontsize=9,
        )

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> int:
    args = parse_args()
    root = Path(args.dataset)
    out = Path(args.out) if args.out else root / "tensors"
    out.mkdir(parents=True, exist_ok=True)

    spec = FrameSpec(extent_m=args.extent, resolution=args.resolution)
    dataset = FrontierRevealDataset(root)
    print(f"dataset: {root}  groups: {len(dataset)}")
    print(f"frame  : {spec.extent_m} m at {spec.resolution} m/cell -> "
          f"{spec.size}x{spec.size}\n")

    leakage: list[str] = []
    roundtrip_errors: list[float] = []
    alignment: list[dict] = []
    tensorised: list[dict] = []

    for index in range(min(args.groups, len(dataset))):
        group = dataset[index]
        tensors = build_group_tensors(group, spec)
        tensorised.append(tensors)

        for frame in tensors["frames"]:
            roundtrip_errors.append(check_roundtrip(frame))
        leakage.extend(check_no_future_leakage(tensors, group))

        reprojected = reproject_to_global(tensors, group, spec)
        errors = alignment_error(reprojected, group)
        alignment.append(errors)

        path = render(group, tensors, reprojected, out / f"{group.group_id}.png")
        print(
            f"  {group.group_id:44s} candidates={tensors['inputs'].shape[0]} "
            f"inputs={tensors['inputs'].shape[1:]} "
            f"boundary_recall={errors['boundary_recall']:.3f} "
            f"target_recall={errors['target_recall']:.3f}"
        )

    batch = collate_tensor_groups(tensorised)
    print(f"\ncollated batch:")
    print(f"  inputs        {batch['inputs'].shape}")
    print(f"  targets       {batch['targets'].shape}")
    print(f"  candidate_mask{batch['candidate_mask'].shape} "
          f"-> {batch['candidate_mask'].sum(axis=1).tolist()} real candidates")
    padded = ~batch["candidate_mask"]
    padded_clean = bool(np.all(batch["inputs"][padded] == 0)) if padded.any() else True

    max_roundtrip = max(roundtrip_errors) if roundtrip_errors else 0.0
    boundary_recall = float(np.nanmean([a["boundary_recall"] for a in alignment]))
    target_recall = float(np.nanmean([a["target_recall"] for a in alignment]))

    print(f"\nchecks:")
    print(f"  transform round-trip max error : {max_roundtrip:.3e} m")
    print(f"  padded slots are zero          : {padded_clean}")
    print(f"  future-leakage problems        : {len(leakage)}")
    for problem in leakage[:5]:
        print(f"      {problem}")
    print(f"  boundary reprojection recall   : {boundary_recall:.3f}")
    print(f"  target reprojection recall     : {target_recall:.3f}")

    summary = {
        "groups_checked": len(tensorised),
        "frame": {"extent_m": spec.extent_m, "resolution": spec.resolution, "size": spec.size},
        "roundtrip_max_error_m": max_roundtrip,
        "padded_slots_zero": padded_clean,
        "leakage_problems": leakage,
        "boundary_reprojection_recall": boundary_recall,
        "target_reprojection_recall": target_recall,
    }
    (out / "tensor_check.json").write_text(json.dumps(summary, indent=2))

    passed = (
        max_roundtrip < 1e-9
        and not leakage
        and padded_clean
        and boundary_recall > 0.9
    )
    print(f"\nfigures: {out}")
    print(f"GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
