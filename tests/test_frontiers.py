"""Frontier extraction tests.

Built on hand-made grids whose frontiers are known by construction, so a
regression shows up as a wrong count or a flipped orientation rather than as
mysteriously worse navigation.
"""

from __future__ import annotations

import numpy as np

from frontierworld.frontiers import extract_frontiers
from frontierworld.frontiers.extraction import (
    boundary_mask,
    information_gain,
    obstacle_clearance_mask,
)
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, MapGeometry

RESOLUTION = 0.1


def geometry(size: int = 100) -> MapGeometry:
    return MapGeometry(
        resolution=RESOLUTION, size_cells=size, origin_x=-5.0, origin_z=-5.0
    )


def room_grid(size: int = 100) -> np.ndarray:
    """Free square in the middle of unknown space, walled except one gap."""
    grid = np.full((size, size), UNKNOWN, dtype=np.uint8)
    grid[30:70, 30:70] = FREE
    grid[30, 30:70] = OCCUPIED  # top wall
    grid[69, 30:70] = OCCUPIED  # bottom wall
    grid[30:70, 30] = OCCUPIED  # left wall
    grid[30:70, 69] = OCCUPIED  # right wall
    return grid


def test_no_frontiers_in_a_sealed_room():
    grid = room_grid()
    assert extract_frontiers(grid, geometry(), agent_radius_m=0.0) == []


def test_single_gap_gives_one_frontier():
    grid = room_grid()
    grid[30, 45:55] = FREE  # cut a doorway in the top wall
    grid[25:30, 45:55] = UNKNOWN

    frontiers = extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=3)

    assert len(frontiers) == 1
    assert frontiers[0].size_cells >= 3


def test_two_gaps_give_two_frontiers():
    grid = room_grid()
    grid[30, 40:45] = FREE
    grid[30, 55:60] = FREE

    frontiers = extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=3)
    assert len(frontiers) == 2


def test_min_size_filters_small_boundaries():
    grid = room_grid()
    grid[30, 49:51] = FREE  # a 2-cell gap

    assert extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=8) == []
    assert extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=2)


def test_orientation_points_into_unknown():
    """A gap in the top wall must face toward -z (decreasing row)."""
    grid = room_grid()
    grid[30, 45:55] = FREE

    frontier = extract_frontiers(
        grid, geometry(), agent_radius_m=0.0, min_size_cells=3
    )[0]

    # orientation is (world x, world z) == (column, row) direction.
    assert frontier.orientation[1] < 0, "should point toward decreasing row (-z)"
    assert abs(frontier.orientation[0]) < 0.6, "should be mostly axis-aligned"


def test_centroid_maps_to_world_coordinates():
    grid = room_grid()
    grid[30, 45:55] = FREE
    geom = geometry()

    frontier = extract_frontiers(grid, geom, agent_radius_m=0.0, min_size_cells=3)[0]

    expected_x = geom.origin_x + (49.5 + 0.5) * RESOLUTION
    assert abs(frontier.centroid_world[0] - expected_x) < 0.3
    assert frontier.centroid_world.shape == (3,)


def test_boundary_mask_only_marks_free_cells():
    grid = room_grid()
    grid[30, 45:55] = FREE
    mask = boundary_mask(grid)
    assert (grid[mask] == FREE).all()


def test_obstacle_clearance_excludes_cells_near_walls():
    grid = room_grid()
    clearance = obstacle_clearance_mask(grid, RESOLUTION, radius_m=0.3)
    # A cell adjacent to the left wall must be excluded.
    assert not clearance[50, 31]
    # The middle of the room must survive.
    assert clearance[50, 50]


def test_information_gain_counts_unknown_area():
    grid = np.full((100, 100), UNKNOWN, dtype=np.uint8)
    gain_all_unknown = information_gain(grid, np.array([50.0, 50.0]), RESOLUTION, 1.0)

    grid[:] = FREE
    gain_none_unknown = information_gain(grid, np.array([50.0, 50.0]), RESOLUTION, 1.0)

    assert gain_all_unknown > gain_none_unknown == 0.0
    # Disc of radius 1 m is pi m2; allow for discretisation.
    assert 2.5 < gain_all_unknown < 3.6


def test_frontiers_sorted_by_size_with_sequential_ids():
    grid = room_grid()
    grid[30, 40:50] = FREE  # wide gap
    grid[30, 55:58] = FREE  # narrow gap

    frontiers = extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=2)

    assert [f.frontier_id for f in frontiers] == list(range(len(frontiers)))
    sizes = [f.size_cells for f in frontiers]
    assert sizes == sorted(sizes, reverse=True)


def test_approach_point_sits_behind_the_boundary():
    grid = room_grid()
    grid[30, 45:55] = FREE
    frontier = extract_frontiers(
        grid, geometry(), agent_radius_m=0.0, min_size_cells=3
    )[0]

    approach = frontier.approach_point(0.5)
    # Backing off along -orientation moves away from unknown space.
    assert approach[2] > frontier.centroid_world[2]


def test_extraction_is_deterministic():
    grid = room_grid()
    grid[30, 40:50] = FREE
    grid[69, 55:60] = FREE

    first = extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=2)
    second = extract_frontiers(grid, geometry(), agent_radius_m=0.0, min_size_cells=2)

    assert [f.to_dict() for f in first] == [f.to_dict() for f in second]
