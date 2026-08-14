"""
Map-based geometric frontier extraction - the guaranteed-recall supplement
to FrontierNet.

FrontierNet is view-dependent: it marks frontiers only in the current camera
frame and can miss openings (behind the robot, low-texture, partial views).
"Missed useful frontiers" is one of OpenFrontier's own failure categories.
This module extracts frontiers directly from the wavemap occupancy state:
a frontier is a cluster of free cells adjacent to unknown space, sized by
the width of the opening - if the map knows about an opening, it WILL be
proposed.

Geometric frontiers carry:
  - pos3d / view_direction (into the unknown), pose6d via the manager
  - gain proportional to the opening size (m^3-equivalent, tuned to the
    same scale as FrontierNet's predicted volumes)
  - no pixel_pos (they may be outside the current view), so they skip the
    Set-of-Marks VLM prompt and keep the neutral prior; the world model
    scores them from scene context instead.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from frontier.frontier import Frontier

NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS_8 = NEIGHBORS_4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _to_cells(points: np.ndarray, voxel_size: float) -> Set[Tuple[int, int]]:
    if points is None or len(points) == 0:
        return set()
    q = np.floor(points[:, :2] / voxel_size).astype(np.int64)
    return set(map(tuple, q))


def extract_geometric_frontiers(
    free_pts: np.ndarray,
    occ_pts: np.ndarray,
    voxel_size: float,
    nav_level: float,
    params: Optional[dict] = None,
) -> List[Frontier]:
    """Extract frontier clusters from the free/occupied point sets.

    Args:
        free_pts: (N,3) free-space voxel centers (wavemap).
        occ_pts: (M,3) occupied voxel centers.
        voxel_size: grid resolution in meters.
        nav_level: floor height of the current navigation level (world z).
        params keys (all optional):
            slab_below/slab_above: z-slab around nav_level for the 2D projection
            min_cells: minimum cluster size (cells) to count as an opening
            max_cells: clusters larger than this are split by centroid chunks
            gain_per_cell: m^3-equivalent gain per frontier cell
            frontier_z: z offset of the emitted frontier above nav_level
    """
    p = params or {}
    slab_below = float(p.get("slab_below", 0.2))
    slab_above = float(p.get("slab_above", 0.8))
    min_cells = int(p.get("min_cells", 5))
    gain_per_cell = float(p.get("gain_per_cell", 0.15))
    frontier_z = float(p.get("frontier_z", 0.4))

    if free_pts is None or len(free_pts) == 0:
        return []

    free_pts = np.asarray(free_pts)
    occ_pts = np.asarray(occ_pts) if occ_pts is not None else np.zeros((0, 3))

    # z-slab at the navigation level: free space the robot can stand in,
    # obstacles anywhere in the robot's height band
    free_slab = free_pts[
        (free_pts[:, 2] > nav_level - slab_below)
        & (free_pts[:, 2] < nav_level + slab_above)
    ]
    occ_slab = occ_pts[
        (occ_pts[:, 2] > nav_level - slab_below)
        & (occ_pts[:, 2] < nav_level + 1.5)
    ]

    free_cells = _to_cells(free_slab, voxel_size)
    occ_cells = _to_cells(occ_slab, voxel_size)
    if not free_cells:
        return []

    known = free_cells | occ_cells

    # frontier cells: free, and bordering unknown (not free, not occupied)
    frontier_cells: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for cell in free_cells:
        if cell in occ_cells:
            continue
        unknown_dirs = []
        for d in NEIGHBORS_4:
            nb = (cell[0] + d[0], cell[1] + d[1])
            if nb not in known:
                unknown_dirs.append(d)
        if unknown_dirs:
            frontier_cells[cell] = unknown_dirs

    if not frontier_cells:
        return []

    # connected components (8-connectivity) over the sparse frontier set
    unvisited = set(frontier_cells)
    clusters: List[List[Tuple[int, int]]] = []
    while unvisited:
        seed = unvisited.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            cur = stack.pop()
            for d in NEIGHBORS_8:
                nb = (cur[0] + d[0], cur[1] + d[1])
                if nb in unvisited:
                    unvisited.remove(nb)
                    comp.append(nb)
                    stack.append(nb)
        clusters.append(comp)

    frontiers: List[Frontier] = []
    for comp in clusters:
        if len(comp) < min_cells:
            continue
        cells = np.array(comp, dtype=np.float64)
        centroid_xy = (cells.mean(axis=0) + 0.5) * voxel_size

        # direction into the unknown: mean of the unknown-neighbor offsets
        dirs = np.array(
            [d for cell in comp for d in frontier_cells[cell]], dtype=np.float64
        )
        mean_dir = dirs.mean(axis=0)
        norm = np.linalg.norm(mean_dir)
        if norm < 1e-6:
            continue
        mean_dir = mean_dir / norm

        ft = Frontier()
        ft.pos3d = np.array(
            [centroid_xy[0], centroid_xy[1], nav_level + frontier_z], dtype=float
        )
        ft.view_direction = np.array([mean_dir[0], mean_dir[1], 0.0], dtype=float)
        ft.direct_angle = float(np.arctan2(mean_dir[1], mean_dir[0]))
        ft.pixel_pos = None  # not tied to any camera frame
        ft.gain = ft.u_gain = float(len(comp) * gain_per_cell)
        ft.justification = "Geometric Frontier"
        ft.set_valid()
        frontiers.append(ft)

    return frontiers


def filter_near_existing(
    candidates: List[Frontier],
    existing_positions: List[np.ndarray],
    min_separation: float = 1.0,
) -> List[Frontier]:
    """Drop candidates within min_separation (xy) of an existing frontier."""
    if not existing_positions:
        return candidates
    existing = np.asarray(existing_positions, dtype=float)[:, :2]
    kept = []
    for ft in candidates:
        d = np.linalg.norm(existing - np.asarray(ft.pos3d)[:2][None, :], axis=1)
        if d.min() >= min_separation:
            kept.append(ft)
    return kept
