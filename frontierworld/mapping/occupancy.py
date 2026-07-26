"""2D occupancy mapping from depth and ground-truth pose.

Phase 1 needs a map to save; Phase 2 builds frontiers on top of it, so the
representation here is already the one frontier extraction will consume:
a top-down grid of free / occupied / unknown cells in habitat world
coordinates, plus per-cell observation counts.

Habitat world axes are x right, y up, z forward-negative. The map plane is
(x, z); heights are measured relative to the agent's floor level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


@dataclass
class MapGeometry:
    """Grid extent and the transform between world metres and cell indices."""

    resolution: float
    size_cells: int
    origin_x: float  # world x of cell (0, 0)
    origin_z: float  # world z of cell (0, 0)

    def world_to_cell(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((x - self.origin_x) / self.resolution).astype(np.int32)
        row = np.floor((z - self.origin_z) / self.resolution).astype(np.int32)
        return row, col

    def in_bounds(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        return (
            (row >= 0) & (row < self.size_cells) & (col >= 0) & (col < self.size_cells)
        )


class OccupancyMap:
    """Counting occupancy grid built by projecting depth images.

    Free space is carved by ray casting: for each subsampled image column the
    nearest obstacle hit bounds a ray from the agent, and cells along that ray
    are counted free. This is what makes free/unknown boundaries -- and hence
    frontiers -- well defined, rather than only marking observed surfaces.
    """

    def __init__(
        self,
        resolution: float = 0.05,
        size_m: float = 40.0,
        obstacle_height_min: float = 0.20,
        obstacle_height_max: float = 1.50,
        column_stride: int = 2,
        min_observations: int = 1,
        center: tuple[float, float] = (0.0, 0.0),
        floor_y: float = 0.0,
    ) -> None:
        size_cells = int(round(size_m / resolution))
        self.geometry = MapGeometry(
            resolution=resolution,
            size_cells=size_cells,
            origin_x=center[0] - size_m / 2.0,
            origin_z=center[1] - size_m / 2.0,
        )
        self.obstacle_height_min = obstacle_height_min
        self.obstacle_height_max = obstacle_height_max
        self.column_stride = max(1, int(column_stride))
        self.min_observations = max(1, int(min_observations))
        self.floor_y = floor_y

        self.free_counts = np.zeros((size_cells, size_cells), dtype=np.int32)
        self.occupied_counts = np.zeros((size_cells, size_cells), dtype=np.int32)
        self._cached_intrinsics: tuple[int, int, float, float] | None = None

    # -- projection ------------------------------------------------------

    def _intrinsics(self, width: int, height: int, hfov_deg: float) -> tuple[float, float]:
        """Focal lengths in pixels. Habitat applies hfov horizontally with
        square pixels, so fy == fx."""
        key = (width, height, hfov_deg, 0.0)
        if self._cached_intrinsics is not None and self._cached_intrinsics[:3] == key[:3]:
            fx = self._cached_intrinsics[3]
            return fx, fx
        fx = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        self._cached_intrinsics = (width, height, hfov_deg, fx)
        return fx, fx

    def unproject(
        self,
        depth: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        hfov_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Depth image to world points.

        Returns (points, valid_mask) where points has shape (H, W, 3) in world
        coordinates and valid_mask marks pixels with usable depth.
        """
        depth = np.squeeze(np.asarray(depth, dtype=np.float32))
        height, width = depth.shape
        fx, fy = self._intrinsics(width, height, hfov_deg)
        cx, cy = width / 2.0, height / 2.0

        us, vs = np.meshgrid(np.arange(width), np.arange(height))
        # Habitat camera frame: x right, y up, z backward.
        x_cam = (us - cx) / fx * depth
        y_cam = -(vs - cy) / fy * depth
        z_cam = -depth
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        points_world = points_cam @ rotation.T + translation
        valid = depth > 0.0
        return points_world.astype(np.float32), valid

    # -- integration -----------------------------------------------------

    def integrate(
        self,
        depth: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        agent_position: np.ndarray,
        hfov_deg: float,
        max_depth: float,
    ) -> None:
        """Fold one depth observation into the map."""
        points, valid = self.unproject(depth, rotation, translation, hfov_deg)
        depth2d = np.squeeze(np.asarray(depth, dtype=np.float32))

        # Depth exactly at max range is a miss, not a surface: treating it as
        # an obstacle would wall off the unexplored region we care about.
        valid &= depth2d < (max_depth - 1e-3)

        stride = self.column_stride
        points = points[::stride, ::stride]
        valid = valid[::stride, ::stride]

        heights = points[..., 1] - self.floor_y
        is_obstacle = (
            valid
            & (heights >= self.obstacle_height_min)
            & (heights <= self.obstacle_height_max)
        )
        # Anything observed that is not an obstacle and sits below the band is
        # traversable ground; ceilings above the band bound nothing useful.
        is_ground = valid & (heights < self.obstacle_height_min)

        geom = self.geometry
        agent_row, agent_col = geom.world_to_cell(
            np.asarray([agent_position[0]]), np.asarray([agent_position[2]])
        )
        agent_rc = (int(agent_row[0]), int(agent_col[0]))
        if not geom.in_bounds(np.asarray([agent_rc[0]]), np.asarray([agent_rc[1]]))[0]:
            return

        # Per image column, the nearest obstacle bounds free space; with no
        # obstacle the farthest ground return does.
        ranges = np.linalg.norm(
            points[..., [0, 2]] - np.asarray([agent_position[0], agent_position[2]]),
            axis=-1,
        )

        endpoints: list[tuple[int, int]] = []
        for col_idx in range(points.shape[1]):
            obstacle_rows = np.flatnonzero(is_obstacle[:, col_idx])
            if obstacle_rows.size:
                row_idx = obstacle_rows[np.argmin(ranges[obstacle_rows, col_idx])]
            else:
                ground_rows = np.flatnonzero(is_ground[:, col_idx])
                if not ground_rows.size:
                    continue
                row_idx = ground_rows[np.argmax(ranges[ground_rows, col_idx])]
            endpoints.append((int(row_idx), int(col_idx)))

        if not endpoints:
            return

        rows = np.asarray([e[0] for e in endpoints])
        cols = np.asarray([e[1] for e in endpoints])
        end_world = points[rows, cols]
        end_row, end_col = geom.world_to_cell(end_world[:, 0], end_world[:, 2])

        self._draw_free_rays(agent_rc, end_row, end_col)

        # Obstacle cells: every point in the height band, not just the nearest
        # per column, so walls seen at an angle are filled in.
        obs_points = points[is_obstacle]
        if obs_points.size:
            obs_row, obs_col = geom.world_to_cell(obs_points[:, 0], obs_points[:, 2])
            keep = geom.in_bounds(obs_row, obs_col)
            np.add.at(self.occupied_counts, (obs_row[keep], obs_col[keep]), 1)

    def _draw_free_rays(
        self, start: tuple[int, int], end_rows: np.ndarray, end_cols: np.ndarray
    ) -> None:
        """Count cells along each ray as free, excluding the endpoint."""
        geom = self.geometry
        for end_row, end_col in zip(end_rows.tolist(), end_cols.tolist()):
            cells = _bresenham(start[0], start[1], int(end_row), int(end_col))
            if len(cells) <= 1:
                continue
            cells = cells[:-1]  # endpoint is a surface, not free space
            rr = np.asarray([c[0] for c in cells])
            cc = np.asarray([c[1] for c in cells])
            keep = geom.in_bounds(rr, cc)
            np.add.at(self.free_counts, (rr[keep], cc[keep]), 1)

    # -- readout ---------------------------------------------------------

    def to_grid(self) -> np.ndarray:
        """Ternary map of UNKNOWN / FREE / OCCUPIED cells."""
        grid = np.full(self.free_counts.shape, UNKNOWN, dtype=np.uint8)
        seen_free = self.free_counts >= self.min_observations
        seen_occupied = self.occupied_counts >= self.min_observations
        grid[seen_free] = FREE
        # A single obstacle return outweighs free counts: under-reporting
        # obstacles is the dangerous direction for a planner.
        grid[seen_occupied] = OCCUPIED
        return grid

    def stats(self) -> dict[str, float]:
        grid = self.to_grid()
        cell_area = self.geometry.resolution**2
        return {
            "free_cells": int((grid == FREE).sum()),
            "occupied_cells": int((grid == OCCUPIED).sum()),
            "unknown_cells": int((grid == UNKNOWN).sum()),
            "explored_area_m2": float(((grid != UNKNOWN).sum()) * cell_area),
            "free_area_m2": float(((grid == FREE).sum()) * cell_area),
        }

    def to_preview(self) -> np.ndarray:
        """RGB image of the map for visual inspection."""
        grid = self.to_grid()
        preview = np.zeros((*grid.shape, 3), dtype=np.uint8)
        preview[grid == UNKNOWN] = (128, 128, 128)
        preview[grid == FREE] = (255, 255, 255)
        preview[grid == OCCUPIED] = (0, 0, 0)
        return preview


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer line from (r0, c0) to (r1, c1), inclusive of both ends."""
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            return cells
        err2 = 2 * err
        if err2 > -dc:
            err -= dc
            r += sr
        if err2 < dr:
            err += dr
            c += sc
