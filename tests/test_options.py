"""Frontier-crossing option tests.

The pure-geometry parts of option construction are tested here without a
simulator. Whether a real branch actually crosses is a simulator question and
is covered by scripts/demo_branches.py, which is the Phase 3 gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontierworld.frontiers.extraction import Frontier
from frontierworld.mapping.occupancy import FREE, UNKNOWN, OccupancyMap
from frontierworld.planning.options import (
    MOVE_FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    FrontierOption,
    is_in_observed_free_space,
    turn_actions_to,
    wrap_angle,
)


def make_frontier(orientation=(1.0, 0.0)) -> Frontier:
    return Frontier(
        frontier_id=0,
        cells=np.zeros((10, 2), dtype=np.int32),
        centroid_cell=np.array([50.0, 50.0]),
        centroid_world=np.array([0.0, 0.0, 0.0]),
        orientation=np.array(orientation, dtype=np.float64),
        size_cells=10,
        boundary_length_m=0.5,
        information_gain_m2=1.0,
    )


# -- angle helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    "angle", [0.0, 0.5, -0.5, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 7.0, -7.0]
)
def test_wrap_angle_lands_in_range_and_preserves_the_heading(angle):
    wrapped = wrap_angle(angle)
    assert -np.pi <= wrapped < np.pi + 1e-12
    # Same physical heading: the difference must be a whole number of turns.
    turns = (angle - wrapped) / (2 * np.pi)
    assert abs(turns - round(turns)) < 1e-9


def test_no_turns_when_already_facing_the_target():
    assert turn_actions_to(0.0, 0.0, 30.0, turn_sign=1) == []


def test_turn_count_matches_the_angle():
    # 90 degrees at 30 degrees per turn is three actions.
    actions = turn_actions_to(0.0, np.pi / 2, 30.0, turn_sign=1)
    assert len(actions) == 3
    assert len(set(actions)) == 1, "should turn one way, not oscillate"


def test_turn_sign_selects_the_direction():
    left = turn_actions_to(0.0, np.pi / 2, 30.0, turn_sign=1)
    right = turn_actions_to(0.0, np.pi / 2, 30.0, turn_sign=-1)
    assert left[0] == TURN_LEFT
    assert right[0] == TURN_RIGHT


def test_turns_take_the_short_way_round():
    """Turning 350 degrees the wrong way would waste most of the horizon."""
    actions = turn_actions_to(0.0, np.deg2rad(-30), 30.0, turn_sign=1)
    assert len(actions) == 1


# -- observed free space ---------------------------------------------------


def occupancy_with_free_patch() -> OccupancyMap:
    occupancy = OccupancyMap(resolution=0.1, size_m=20.0, center=(0.0, 0.0))
    # Mark a patch around the origin as observed free space.
    occupancy.free_counts[95:105, 95:105] = 5
    return occupancy


def test_point_inside_mapped_free_space_is_accepted():
    assert is_in_observed_free_space(np.array([0.0, 0.0, 0.0]), occupancy_with_free_patch())


def test_point_in_unknown_space_is_rejected():
    """The navmesh covers the whole scene, so an approach pose has to be
    checked against what the agent has actually observed."""
    occupancy = occupancy_with_free_patch()
    assert not is_in_observed_free_space(np.array([5.0, 0.0, 5.0]), occupancy)


def test_point_outside_the_map_is_rejected():
    occupancy = occupancy_with_free_patch()
    assert not is_in_observed_free_space(np.array([500.0, 0.0, 500.0]), occupancy)


# -- option shape ----------------------------------------------------------


def test_actions_concatenate_approach_then_cross():
    option = FrontierOption(
        frontier_id=0,
        frontier=make_frontier(),
        approach_world=np.zeros(3),
        crossing_direction=np.array([1.0, 0.0]),
        crossing_yaw=0.0,
        probe_distance_m=2.0,
        horizon=12,
        approach_actions=[MOVE_FORWARD, TURN_LEFT],
        cross_actions=[MOVE_FORWARD] * 8,
    )
    assert option.actions == [MOVE_FORWARD, TURN_LEFT] + [MOVE_FORWARD] * 8
    assert len(option.actions) == 10


def test_rejected_option_serialises_its_reason():
    option = FrontierOption(
        frontier_id=3,
        frontier=make_frontier(),
        approach_world=np.zeros(3),
        crossing_direction=np.array([1.0, 0.0]),
        crossing_yaw=0.0,
        probe_distance_m=2.0,
        horizon=12,
        valid=False,
        reject_reason="unreachable before crossing",
    )
    payload = option.to_dict()
    assert payload["valid"] is False
    assert payload["reject_reason"] == "unreachable before crossing"
    assert payload["frontier_id"] == 3


def test_frontier_yaw_matches_orientation():
    """The crossing heading must point the way the frontier faces."""
    east = make_frontier(orientation=(1.0, 0.0))
    assert abs(east.yaw - np.pi / 2) < 1e-6

    north = make_frontier(orientation=(0.0, -1.0))
    assert abs(wrap_angle(north.yaw - 0.0)) < 1e-6
