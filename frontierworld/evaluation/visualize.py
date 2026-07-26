"""Side-by-side visualisation of one branch and what it revealed.

The Phase 4 gate is manual verification, so the figure has to make every claim
in the revelation record checkable by eye: where the agent was, which frontier
it chose, the path it took, and exactly which cells went from unknown to known
as a result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN  # noqa: E402

COLOUR_UNKNOWN = (0.60, 0.60, 0.62)
COLOUR_FREE = (0.97, 0.97, 0.97)
COLOUR_OCCUPIED = (0.13, 0.13, 0.15)
COLOUR_REVEALED = (0.15, 0.65, 0.95)
COLOUR_SEMANTIC = (0.95, 0.55, 0.10)
COLOUR_CHOSEN = (0.90, 0.15, 0.25)
COLOUR_OTHER = (0.35, 0.75, 0.40)
COLOUR_PATH = (0.95, 0.35, 0.75)


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    image = np.zeros((*grid.shape, 3), dtype=np.float32)
    image[grid == UNKNOWN] = COLOUR_UNKNOWN
    image[grid == FREE] = COLOUR_FREE
    image[grid == OCCUPIED] = COLOUR_OCCUPIED
    return image


def crop_bounds(grid: np.ndarray, margin: int = 25) -> tuple[int, int, int, int]:
    """Tight crop around observed cells, so the map is legible not mostly grey."""
    known = np.argwhere(grid != UNKNOWN)
    if known.size == 0:
        return 0, grid.shape[0], 0, grid.shape[1]
    r0, c0 = known.min(axis=0)
    r1, c1 = known.max(axis=0)
    return (
        max(0, int(r0) - margin),
        min(grid.shape[0], int(r1) + margin),
        max(0, int(c0) - margin),
        min(grid.shape[1], int(c1) + margin),
    )


def world_to_cell(geometry, x: float, z: float) -> tuple[float, float]:
    col = (x - geometry.origin_x) / geometry.resolution
    row = (z - geometry.origin_z) / geometry.resolution
    return row, col


def visualise_branch(
    path: str | Path,
    grid_before: np.ndarray,
    grid_after: np.ndarray,
    revealed_mask: np.ndarray,
    semantic_mask: np.ndarray,
    geometry,
    chosen_frontier,
    other_frontiers,
    trajectory: list[dict],
    rgb_before: np.ndarray | None,
    rgb_after: np.ndarray | None,
    revelation,
    meta: dict,
) -> Path:
    """Write one six-panel figure for a single branch."""
    bounds = crop_bounds(grid_after)
    r0, r1, c0, c1 = bounds

    def crop(array: np.ndarray) -> np.ndarray:
        return array[r0:r1, c0:c1]

    figure = plt.figure(figsize=(19, 9.5), constrained_layout=True)
    spec = figure.add_gridspec(2, 3)

    start = trajectory[0]["position"] if trajectory else None
    end = trajectory[-1]["position"] if trajectory else None

    # 1. map before crossing, with the candidate frontiers
    axis = figure.add_subplot(spec[0, 0])
    axis.imshow(crop(grid_to_rgb(grid_before)), interpolation="nearest")
    for frontier in other_frontiers:
        _plot_frontier(axis, frontier, geometry, bounds, COLOUR_OTHER, 6)
    if chosen_frontier is not None:
        _plot_frontier(axis, chosen_frontier, geometry, bounds, COLOUR_CHOSEN, 10)
        _plot_arrow(axis, chosen_frontier, geometry, bounds)
    if start is not None:
        _plot_agent(axis, start, geometry, bounds, "start")
    axis.set_title("1. Map BEFORE crossing\nred = chosen frontier, green = other candidates")
    _clean(axis)

    # 2. map after, with what was revealed
    axis = figure.add_subplot(spec[0, 1])
    image = crop(grid_to_rgb(grid_after))
    overlay = crop(revealed_mask)
    image[overlay] = COLOUR_REVEALED
    axis.imshow(image, interpolation="nearest")
    if start is not None:
        _plot_agent(axis, start, geometry, bounds, "start")
    if end is not None:
        _plot_agent(axis, end, geometry, bounds, "end", marker="*")
    axis.set_title(
        f"2. Map AFTER crossing\nblue = newly revealed "
        f"({revelation.newly_observed_area_m2:.1f} m2)"
    )
    _clean(axis)

    # 3. trajectory over the before-map, so the path is read against what the
    #    agent actually knew when it committed
    axis = figure.add_subplot(spec[0, 2])
    axis.imshow(crop(grid_to_rgb(grid_before)), interpolation="nearest")
    _plot_trajectory(axis, trajectory, geometry, bounds)
    if chosen_frontier is not None:
        _plot_frontier(axis, chosen_frontier, geometry, bounds, COLOUR_CHOSEN, 8)
    axis.set_title(
        f"3. Executed trajectory ({len(trajectory)} actions)\n"
        f"solid = approach, dashed = crossing probe"
    )
    _clean(axis)

    # 4. newly observed semantics
    axis = figure.add_subplot(spec[1, 0])
    image = crop(grid_to_rgb(grid_after))
    semantic_overlay = crop(semantic_mask)
    image[semantic_overlay] = COLOUR_SEMANTIC
    axis.imshow(image, interpolation="nearest")
    axis.set_title(
        f"4. Newly observed semantics\norange = {revelation.newly_semantic_in_revealed_area_m2:.1f} m2 labelled in revealed area"
    )
    _clean(axis)

    # 5/6. what the agent saw, before and after
    axis = figure.add_subplot(spec[1, 1])
    if rgb_before is not None:
        axis.imshow(np.asarray(rgb_before)[..., :3].astype(np.uint8))
    axis.set_title("5. RGB at decision time")
    _clean(axis)

    axis = figure.add_subplot(spec[1, 2])
    if rgb_after is not None:
        axis.imshow(np.asarray(rgb_after)[..., :3].astype(np.uint8))
    axis.set_title("6. RGB at end of crossing")
    _clean(axis)

    figure.suptitle(_headline(revelation, meta), fontsize=11, fontfamily="monospace")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=95, bbox_inches="tight")
    plt.close(figure)
    return path


def _headline(revelation, meta: dict) -> str:
    categories = ", ".join(
        f"{name}({count})"
        for name, count in sorted(
            revelation.revealed_categories.items(),
            key=lambda item: -item[1],
        )[:6]
    )
    return (
        f"{meta.get('scene')}  ep={meta.get('episode_id')}  t={meta.get('decision_timestep')}  "
        f"goal={meta.get('navigation_goal')}  frontier={revelation.frontier_id}\n"
        f"revealed={revelation.newly_observed_area_m2:.1f} m2 "
        f"(free {revelation.newly_free_area_m2:.1f})  "
        f"semantic(in revealed)={revelation.newly_semantic_in_revealed_area_m2:.1f} m2  "
        f"crossed={revelation.crossed}  collisions={revelation.collisions}  "
        f"new frontiers={revelation.n_new_frontiers}\n"
        f"room={revelation.room_category} (derived)  "
        f"target visible={revelation.target_became_visible}  "
        f"geodesic to target={revelation.geodesic_distance_to_target_m:.2f} m\n"
        f"new categories: {categories or 'none'}"
    )


def _plot_frontier(axis, frontier, geometry, bounds, colour, size) -> None:
    r0, _, c0, _ = bounds
    cells = frontier.cells
    axis.scatter(
        cells[:, 1] - c0, cells[:, 0] - r0, s=1.2, c=[colour], alpha=0.65, linewidths=0
    )
    row, col = world_to_cell(
        geometry, frontier.centroid_world[0], frontier.centroid_world[2]
    )
    axis.scatter([col - c0], [row - r0], s=size * 6, c=[colour], marker="o",
                 edgecolors="black", linewidths=0.6, zorder=5)


def _plot_arrow(axis, frontier, geometry, bounds) -> None:
    r0, _, c0, _ = bounds
    row, col = world_to_cell(
        geometry, frontier.centroid_world[0], frontier.centroid_world[2]
    )
    length = 22.0
    axis.arrow(
        col - c0,
        row - r0,
        frontier.orientation[0] * length,
        frontier.orientation[1] * length,
        width=1.6,
        color=COLOUR_CHOSEN,
        length_includes_head=True,
        zorder=6,
    )


def _plot_agent(axis, position, geometry, bounds, label, marker="o") -> None:
    r0, _, c0, _ = bounds
    row, col = world_to_cell(geometry, position[0], position[2])
    axis.scatter(
        [col - c0], [row - r0], s=150, marker=marker, c="gold",
        edgecolors="black", linewidths=1.0, zorder=8, label=label,
    )
    axis.legend(loc="upper right", fontsize=7, framealpha=0.85)


def _plot_trajectory(axis, trajectory, geometry, bounds) -> None:
    r0, _, c0, _ = bounds
    for phase, style in (("approach", "-"), ("cross", "--")):
        points = [p for p in trajectory if p.get("phase") == phase]
        if len(points) < 2:
            continue
        rows, cols = zip(
            *[world_to_cell(geometry, p["position"][0], p["position"][2]) for p in points]
        )
        axis.plot(
            np.asarray(cols) - c0,
            np.asarray(rows) - r0,
            style,
            color=COLOUR_PATH,
            linewidth=2.2,
            zorder=7,
            label=phase,
        )
    handles = [
        mpatches.Patch(color=COLOUR_PATH, label="approach (solid) / probe (dashed)")
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)


def _clean(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)


def contact_sheet(figure_paths: list[Path], output: str | Path, columns: int = 2) -> Path:
    """Stack per-branch figures into one sheet for quick scanning."""
    import imageio.v2 as imageio

    images = [imageio.imread(p) for p in figure_paths]
    if not images:
        raise ValueError("no figures to combine")

    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)

    for index, image in enumerate(images):
        row, col = divmod(index, columns)
        patch = image[..., :3]
        sheet[
            row * height : row * height + patch.shape[0],
            col * width : col * width + patch.shape[1],
        ] = patch

    output = Path(output)
    imageio.imwrite(output, sheet)
    return output


def visualise_decision_group(
    path: str | Path,
    grid_before: np.ndarray,
    geometry,
    frontiers,
    branch_masks: list[np.ndarray],
    revelations: list,
    rgb_before: np.ndarray | None,
    meta: dict,
    max_panels: int = 8,
) -> Path:
    """One figure per decision group: every candidate branch side by side.

    Phase 5 asks for 50 groups to be inspected by hand. Per-branch figures make
    that a slog and, more importantly, hide the thing worth checking -- whether
    the candidates from a shared state really do differ in what they reveal.
    """
    bounds = crop_bounds(grid_before)
    r0, r1, c0, c1 = bounds

    def crop(array: np.ndarray) -> np.ndarray:
        return array[r0:r1, c0:c1]

    n = min(len(revelations), max_panels)
    columns = min(4, n + 1)
    rows = int(np.ceil((n + 1) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 4.9 * rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    # Panel 0: the shared decision state with every candidate marked.
    axis = axes[0]
    axis.imshow(crop(grid_to_rgb(grid_before)), interpolation="nearest")
    for frontier in frontiers:
        _plot_frontier(axis, frontier, geometry, bounds, COLOUR_OTHER, 6)
        row, col = world_to_cell(
            geometry, frontier.centroid_world[0], frontier.centroid_world[2]
        )
        axis.annotate(
            str(frontier.frontier_id),
            (col - c0, row - r0),
            fontsize=8,
            color="black",
            weight="bold",
            ha="center",
            va="center",
        )
    axis.set_title(f"decision state: {len(frontiers)} candidates", fontsize=9)
    _clean(axis)

    areas = [float(r.newly_observed_area_m2) for r in revelations[:n]]
    best = int(np.argmax(areas)) if areas else -1

    for index in range(n):
        axis = axes[index + 1]
        image = crop(grid_to_rgb(grid_before))
        image[crop(branch_masks[index])] = COLOUR_REVEALED
        axis.imshow(image, interpolation="nearest")
        revelation = revelations[index]
        marker = "  <= most revealed" if index == best else ""
        axis.set_title(
            f"f{revelation.frontier_id}: {revelation.newly_observed_area_m2:.1f} m2{marker}\n"
            f"crossed={revelation.crossed} coll={revelation.collisions} "
            f"newF={revelation.n_new_frontiers} tgt={int(revelation.target_became_visible)}",
            fontsize=8,
        )
        _clean(axis)

    for axis in axes[n + 1 :]:
        axis.axis("off")

    ratio = (max(areas) / max(min(areas), 0.01)) if len(areas) >= 2 else float("nan")
    figure.suptitle(
        f"{meta.get('group_id')}   goal={meta.get('navigation_goal')}   "
        f"t={meta.get('decision_timestep')}   policy={meta.get('collection_policy')}   "
        f"spread={ratio:.1f}x",
        fontsize=11,
        fontfamily="monospace",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(figure)
    return path
