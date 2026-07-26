"""Occupancy mapping tests.

The mapper is checked against a synthetic scene whose answer is known: a flat
floor with a wall at a fixed distance. Getting free/occupied/unknown right here
is what makes free-unknown boundary extraction meaningful in Phase 2.
"""

from __future__ import annotations

import numpy as np

from frontierworld.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyMap
from frontierworld.mapping.occupancy import _bresenham


def make_map(**kwargs) -> OccupancyMap:
    defaults = dict(resolution=0.1, size_m=20.0, column_stride=1, floor_y=0.0)
    defaults.update(kwargs)
    return OccupancyMap(**defaults)


def test_grid_starts_unknown():
    grid = make_map().to_grid()
    assert grid.shape == (200, 200)
    assert (grid == UNKNOWN).all()


def test_world_to_cell_roundtrip():
    occupancy = make_map(center=(0.0, 0.0))
    row, col = occupancy.geometry.world_to_cell(np.array([0.0]), np.array([0.0]))
    # Centre of a 20 m map at 0.1 m/cell is cell (100, 100).
    assert (int(row[0]), int(col[0])) == (100, 100)


def test_unproject_places_wall_at_correct_range():
    """A constant-depth image must land at that distance in front of the camera."""
    occupancy = make_map()
    depth = np.full((8, 8), 3.0, dtype=np.float32)
    rotation = np.eye(3)
    translation = np.array([0.0, 0.88, 0.0])

    points, valid = occupancy.unproject(depth, rotation, translation, hfov_deg=90.0)

    assert valid.all()
    # Camera looks down -z, so the centre pixel sits 3 m along -z.
    centre = points[4, 4]
    assert np.isclose(centre[2], -3.0, atol=0.5)
    assert np.isclose(centre[1], 0.88, atol=0.5)


def test_wall_is_occupied_and_space_before_it_is_free():
    occupancy = make_map(obstacle_height_min=0.2, obstacle_height_max=1.5)
    depth = np.full((16, 16), 3.0, dtype=np.float32)

    occupancy.integrate(
        depth=depth,
        rotation=np.eye(3),
        translation=np.array([0.0, 0.88, 0.0]),
        agent_position=np.array([0.0, 0.0, 0.0]),
        hfov_deg=79.0,
        max_depth=5.0,
    )
    grid = occupancy.to_grid()

    assert (grid == OCCUPIED).any(), "constant-depth wall should mark occupied cells"
    assert (grid == FREE).any(), "space between agent and wall should be free"
    assert (grid == UNKNOWN).any(), "space beyond the wall should stay unknown"

    # Cells just past the wall are unobserved, so they must remain unknown.
    geometry = occupancy.geometry
    row, col = geometry.world_to_cell(np.array([0.0]), np.array([-4.5]))
    assert grid[int(row[0]), int(col[0])] == UNKNOWN


def test_max_range_returns_do_not_create_walls():
    """Depth at the sensor limit is a miss; treating it as a surface would
    wall off exactly the unexplored region frontiers are defined by."""
    occupancy = make_map()
    depth = np.full((16, 16), 5.0, dtype=np.float32)

    occupancy.integrate(
        depth=depth,
        rotation=np.eye(3),
        translation=np.array([0.0, 0.88, 0.0]),
        agent_position=np.array([0.0, 0.0, 0.0]),
        hfov_deg=79.0,
        max_depth=5.0,
    )

    assert not (occupancy.to_grid() == OCCUPIED).any()


def test_stats_report_explored_area():
    occupancy = make_map()
    depth = np.full((16, 16), 2.0, dtype=np.float32)
    occupancy.integrate(
        depth=depth,
        rotation=np.eye(3),
        translation=np.array([0.0, 0.88, 0.0]),
        agent_position=np.array([0.0, 0.0, 0.0]),
        hfov_deg=79.0,
        max_depth=5.0,
    )
    stats = occupancy.stats()

    assert stats["explored_area_m2"] > 0.0
    assert stats["free_cells"] + stats["occupied_cells"] + stats["unknown_cells"] == 200 * 200


def test_integration_is_deterministic():
    depth = np.random.default_rng(0).uniform(1.0, 4.0, size=(16, 16)).astype(np.float32)
    grids = []
    for _ in range(2):
        occupancy = make_map()
        occupancy.integrate(
            depth=depth,
            rotation=np.eye(3),
            translation=np.array([0.0, 0.88, 0.0]),
            agent_position=np.array([0.0, 0.0, 0.0]),
            hfov_deg=79.0,
            max_depth=5.0,
        )
        grids.append(occupancy.to_grid())
    np.testing.assert_array_equal(grids[0], grids[1])


def test_bresenham_endpoints_and_continuity():
    cells = _bresenham(0, 0, 5, 3)
    assert cells[0] == (0, 0)
    assert cells[-1] == (5, 3)
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1
