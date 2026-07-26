"""Frontier extraction from the occupancy map.

A frontier is a connected run of free cells adjacent to unknown space. This is
the classical geometric definition (Yamauchi 1997); Phase 7 replaces the
identity-by-position assumption here with the lineage graph, and the paper's
FrontierNet variant replaces the detector itself. Everything downstream --
options, memory, revelation targets -- keys off the fields on Frontier, so
they are fixed now.

All world coordinates are habitat-native: x right, y up, z forward-negative.
The map plane is (x, z).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import ndimage

from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, MapGeometry


@dataclass
class Frontier:
    """One connected free-unknown boundary segment."""

    frontier_id: int
    cells: np.ndarray  # (N, 2) int32 grid cells, rows then columns
    centroid_cell: np.ndarray  # (2,) float, row/col
    centroid_world: np.ndarray  # (3,) float, habitat world x/y/z
    orientation: np.ndarray  # (2,) unit vector in world (x, z), into unknown
    size_cells: int
    boundary_length_m: float
    information_gain_m2: float
    # Filled in by the planner; None means "not yet queried".
    geodesic_distance_m: float | None = None
    reachable: bool | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def yaw(self) -> float:
        """Heading into unknown space, radians, atan2(x, -z) as habitat uses."""
        return float(np.arctan2(self.orientation[0], -self.orientation[1]))

    def approach_point(self, offset_m: float = 0.25) -> np.ndarray:
        """A point just inside known free space, backed off from the boundary.

        Targeting the centroid directly aims at the boundary itself, which
        often snaps into unknown territory; backing off keeps the approach in
        space the agent has actually observed.
        """
        world = self.centroid_world.copy()
        world[0] -= float(offset_m) * self.orientation[0]
        world[2] -= float(offset_m) * self.orientation[1]
        return world

    def to_dict(self) -> dict:
        return {
            "frontier_id": self.frontier_id,
            "centroid_world": self.centroid_world.tolist(),
            "orientation": self.orientation.tolist(),
            "yaw": self.yaw,
            "size_cells": self.size_cells,
            "boundary_length_m": self.boundary_length_m,
            "information_gain_m2": self.information_gain_m2,
            "geodesic_distance_m": self.geodesic_distance_m,
            "reachable": self.reachable,
        }


def boundary_mask(grid: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Free cells that touch unknown space."""
    unknown = grid == UNKNOWN
    structure = (
        ndimage.generate_binary_structure(2, 2)
        if connectivity == 8
        else ndimage.generate_binary_structure(2, 1)
    )
    unknown_dilated = ndimage.binary_dilation(unknown, structure=structure)
    return (grid == FREE) & unknown_dilated


def obstacle_clearance_mask(grid: np.ndarray, resolution: float, radius_m: float) -> np.ndarray:
    """Cells at least radius_m away from any obstacle.

    A frontier centroid inside the agent's own radius of a wall is not
    approachable, so those cells are dropped before clustering rather than
    failing later in the planner.
    """
    if radius_m <= 0.0:
        return np.ones_like(grid, dtype=bool)
    radius_cells = int(np.ceil(radius_m / resolution))
    occupied = grid == OCCUPIED
    inflated = ndimage.binary_dilation(
        occupied, structure=ndimage.generate_binary_structure(2, 2), iterations=radius_cells
    )
    return ~inflated


def information_gain(
    grid: np.ndarray, centroid_cell: np.ndarray, resolution: float, radius_m: float
) -> float:
    """Unknown area within radius_m of a frontier, in square metres.

    The standard geometric proxy for "how much would I learn by going here".
    Phase 9 replaces this with a predicted revelation; keeping the same units
    makes the two directly comparable.
    """
    radius_cells = int(np.ceil(radius_m / resolution))
    row, col = int(round(centroid_cell[0])), int(round(centroid_cell[1]))
    row0, row1 = max(0, row - radius_cells), min(grid.shape[0], row + radius_cells + 1)
    col0, col1 = max(0, col - radius_cells), min(grid.shape[1], col + radius_cells + 1)
    patch = grid[row0:row1, col0:col1]
    if patch.size == 0:
        return 0.0

    rows = np.arange(row0, row1)[:, None]
    cols = np.arange(col0, col1)[None, :]
    within = ((rows - row) ** 2 + (cols - col) ** 2) <= radius_cells**2
    unknown_cells = int(((patch == UNKNOWN) & within).sum())
    return float(unknown_cells * resolution**2)


def extract_frontiers(
    grid: np.ndarray,
    geometry: MapGeometry,
    floor_y: float = 0.0,
    min_size_cells: int = 8,
    agent_radius_m: float = 0.18,
    info_gain_radius_m: float = 2.0,
    connectivity: int = 8,
) -> list[Frontier]:
    """Cluster free-unknown boundary cells into frontiers.

    Returns frontiers sorted by descending size. Reachability is not decided
    here -- that needs the planner; see planning.habitat_planner.
    """
    candidates = boundary_mask(grid, connectivity=connectivity)
    candidates &= obstacle_clearance_mask(grid, geometry.resolution, agent_radius_m)
    if not candidates.any():
        return []

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    labels, count = ndimage.label(candidates, structure=structure)
    if count == 0:
        return []

    unknown = grid == UNKNOWN
    frontiers: list[Frontier] = []
    for label_id in range(1, count + 1):
        cells = np.argwhere(labels == label_id).astype(np.int32)
        if cells.shape[0] < min_size_cells:
            continue

        centroid_cell = cells.mean(axis=0)
        world_x = geometry.origin_x + (centroid_cell[1] + 0.5) * geometry.resolution
        world_z = geometry.origin_z + (centroid_cell[0] + 0.5) * geometry.resolution
        orientation = _unknown_direction(cells, unknown)

        frontiers.append(
            Frontier(
                frontier_id=len(frontiers),
                cells=cells,
                centroid_cell=centroid_cell,
                centroid_world=np.array([world_x, floor_y, world_z], dtype=np.float64),
                orientation=orientation,
                size_cells=int(cells.shape[0]),
                boundary_length_m=float(cells.shape[0]) * geometry.resolution,
                information_gain_m2=information_gain(
                    grid, centroid_cell, geometry.resolution, info_gain_radius_m
                ),
            )
        )

    frontiers.sort(key=lambda f: f.size_cells, reverse=True)
    for index, frontier in enumerate(frontiers):
        frontier.frontier_id = index
    return frontiers


def _unknown_direction(cells: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """Mean direction from frontier cells toward the unknown cells they touch.

    Returned in world (x, z), i.e. (column, row) in grid terms.
    """
    height, width = unknown.shape
    accumulator = np.zeros(2, dtype=np.float64)
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            if delta_row == 0 and delta_col == 0:
                continue
            rows = cells[:, 0] + delta_row
            cols = cells[:, 1] + delta_col
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            if not valid.any():
                continue
            hits = int(unknown[rows[valid], cols[valid]].sum())
            if hits:
                accumulator += hits * np.array([delta_col, delta_row], dtype=np.float64)

    norm = np.linalg.norm(accumulator)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return accumulator / norm


def frontiers_to_records(frontiers: Iterable[Frontier]) -> list[dict]:
    return [frontier.to_dict() for frontier in frontiers]
