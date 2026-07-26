"""Ground-truth revelation tests.

The revelation is the prediction target for the whole project, so its
definition is pinned here on grids whose answer is known by construction. Two
of these tests exist because manual verification of the Phase 4 figures caught
the corresponding defects.
"""

from __future__ import annotations

import numpy as np

from frontierworld.data.revelation import (
    frontier_delta,
    occupancy_delta,
    semantic_delta,
)
from frontierworld.frontiers.extraction import Frontier
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN
from frontierworld.mapping.semantic import UNLABELLED, SemanticMap
from frontierworld.mapping.occupancy import MapGeometry

RESOLUTION = 0.1


def make_frontier(fid: int, x: float, z: float, gain: float = 1.0) -> Frontier:
    return Frontier(
        frontier_id=fid,
        cells=np.zeros((10, 2), dtype=np.int32),
        centroid_cell=np.array([0.0, 0.0]),
        centroid_world=np.array([x, 0.0, z]),
        orientation=np.array([1.0, 0.0]),
        size_cells=10,
        boundary_length_m=0.5,
        information_gain_m2=gain,
    )


# -- occupancy delta -------------------------------------------------------


def test_occupancy_delta_counts_only_newly_known_cells():
    before = np.full((10, 10), UNKNOWN, dtype=np.uint8)
    after = before.copy()
    after[0:2, 0:5] = FREE      # 10 newly free
    after[2, 0:5] = OCCUPIED    # 5 newly occupied

    stats = occupancy_delta(before, after, RESOLUTION)

    assert stats["newly_observed_cells"] == 15
    assert stats["newly_free_cells"] == 10
    assert stats["newly_occupied_cells"] == 5
    assert abs(stats["newly_observed_area_m2"] - 15 * RESOLUTION**2) < 1e-9


def test_cells_already_known_are_not_revelation():
    """Space the agent had already mapped cannot be revealed again."""
    before = np.full((10, 10), FREE, dtype=np.uint8)
    after = np.full((10, 10), OCCUPIED, dtype=np.uint8)

    stats = occupancy_delta(before, after, RESOLUTION)
    assert stats["newly_observed_cells"] == 0


def test_occupancy_delta_is_a_one_way_difference():
    """Cells going known -> unknown must not count as negative revelation."""
    before = np.full((10, 10), FREE, dtype=np.uint8)
    after = np.full((10, 10), UNKNOWN, dtype=np.uint8)
    assert occupancy_delta(before, after, RESOLUTION)["newly_observed_cells"] == 0


# -- semantic delta --------------------------------------------------------


def semantic_map_with(labels: np.ndarray) -> SemanticMap:
    geometry = MapGeometry(RESOLUTION, labels.shape[0], 0.0, 0.0)
    semantic = SemanticMap(geometry)
    semantic.category_id("chair")
    semantic.category_grid[...] = labels
    return semantic


def test_semantic_delta_restricted_to_revealed_area():
    """Regression: labelling catching up on already-mapped space is not
    revelation. Unrestricted, this counted 3.3x the true figure."""
    before = np.full((10, 10), UNLABELLED, dtype=np.int16)
    after = np.zeros((10, 10), dtype=np.int16)  # every cell labelled 'chair'

    revealed = np.zeros((10, 10), dtype=bool)
    revealed[0:2, 0:5] = True  # only 10 cells are genuinely new territory

    stats = semantic_delta(
        before, after, semantic_map_with(after), RESOLUTION, revealed_mask=revealed
    )

    assert stats["newly_semantic_cells"] == 100
    assert stats["newly_semantic_in_revealed_cells"] == 10
    assert stats["newly_semantic_in_revealed_cells"] < stats["newly_semantic_cells"]


def test_semantic_in_revealed_never_exceeds_revealed_area():
    before = np.full((8, 8), UNLABELLED, dtype=np.int16)
    after = np.zeros((8, 8), dtype=np.int16)
    revealed = np.zeros((8, 8), dtype=bool)
    revealed[3:5, 3:5] = True

    stats = semantic_delta(
        before, after, semantic_map_with(after), RESOLUTION, revealed_mask=revealed
    )
    assert stats["newly_semantic_in_revealed_cells"] <= int(revealed.sum())


def test_already_labelled_cells_are_not_new():
    before = np.zeros((6, 6), dtype=np.int16)
    after = np.zeros((6, 6), dtype=np.int16)
    stats = semantic_delta(before, after, semantic_map_with(after), RESOLUTION)
    assert stats["newly_semantic_cells"] == 0


# -- frontier delta --------------------------------------------------------


def test_new_frontiers_detected():
    before = [make_frontier(0, 0.0, 0.0)]
    after = [make_frontier(0, 0.0, 0.0), make_frontier(1, 5.0, 5.0, gain=3.0)]

    stats = frontier_delta(before, after)

    assert stats["n_new_frontiers"] == 1
    assert abs(stats["new_frontier_gain_m2"] - 3.0) < 1e-9


def test_shifted_frontier_is_not_counted_as_new():
    """A frontier drifting a few cells is the same opening, not a new one."""
    before = [make_frontier(0, 0.0, 0.0)]
    after = [make_frontier(0, 0.2, 0.1)]
    assert frontier_delta(before, after, match_radius_m=0.75)["n_new_frontiers"] == 0


def test_frontier_beyond_match_radius_is_new():
    before = [make_frontier(0, 0.0, 0.0)]
    after = [make_frontier(0, 2.0, 0.0)]
    assert frontier_delta(before, after, match_radius_m=0.75)["n_new_frontiers"] == 1


def test_disappearing_frontiers_do_not_count_as_new():
    before = [make_frontier(0, 0.0, 0.0), make_frontier(1, 9.0, 9.0)]
    after = [make_frontier(0, 0.0, 0.0)]

    stats = frontier_delta(before, after)
    assert stats["n_new_frontiers"] == 0
    assert stats["n_frontiers_before"] == 2
    assert stats["n_frontiers_after"] == 1
