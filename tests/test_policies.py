"""Frontier-policy tests.

The policies are the baselines the revelation model has to beat, so their
ranking rules need to be exactly right: a nearest policy that quietly ignores
distance would make the Phase 12 regret comparison meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontierworld.frontiers.extraction import Frontier
from frontierworld.planning.policies import POLICIES, make_policy


def frontier(fid: int, gain: float, distance: float | None) -> Frontier:
    return Frontier(
        frontier_id=fid,
        cells=np.zeros((10, 2), dtype=np.int32),
        centroid_cell=np.array([0.0, 0.0]),
        centroid_world=np.array([float(fid), 0.0, 0.0]),
        orientation=np.array([1.0, 0.0]),
        size_cells=10,
        boundary_length_m=0.5,
        information_gain_m2=gain,
        geodesic_distance_m=distance,
        reachable=True,
    )


@pytest.fixture
def candidates() -> list[Frontier]:
    return [
        frontier(0, gain=1.0, distance=1.0),   # near, low gain
        frontier(1, gain=10.0, distance=9.0),  # far, high gain
        frontier(2, gain=4.0, distance=2.0),   # best gain-minus-cost
    ]


def test_nearest_picks_smallest_distance(candidates):
    chosen = make_policy("nearest").select(candidates, np.random.default_rng(0))
    assert chosen.frontier_id == 0


def test_max_info_gain_ignores_distance(candidates):
    chosen = make_policy("max_info_gain").select(candidates, np.random.default_rng(0))
    assert chosen.frontier_id == 1


def test_info_gain_minus_cost_trades_off(candidates):
    policy = make_policy("info_gain_minus_cost", cost_weight=1.0)
    chosen = policy.select(candidates, np.random.default_rng(0))
    # scores: 1-1=0, 10-9=1, 4-2=2
    assert chosen.frontier_id == 2


def test_cost_weight_changes_the_ranking(candidates):
    cheap_travel = make_policy("info_gain_minus_cost", cost_weight=0.0)
    assert cheap_travel.select(candidates, np.random.default_rng(0)).frontier_id == 1

    dear_travel = make_policy("info_gain_minus_cost", cost_weight=5.0)
    assert dear_travel.select(candidates, np.random.default_rng(0)).frontier_id == 0


def test_random_is_reproducible_and_in_range(candidates):
    first = make_policy("random").select(candidates, np.random.default_rng(7))
    second = make_policy("random").select(candidates, np.random.default_rng(7))
    assert first.frontier_id == second.frontier_id
    assert first.frontier_id in {0, 1, 2}


def test_random_uses_the_whole_candidate_set(candidates):
    rng = np.random.default_rng(0)
    picks = {make_policy("random").select(candidates, rng).frontier_id for _ in range(50)}
    assert picks == {0, 1, 2}


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_empty_candidates_return_none(name):
    assert make_policy(name).select([], np.random.default_rng(0)) is None


def test_unqueried_distance_is_treated_as_unreachable():
    """A frontier with no distance must not beat one with a real distance."""
    unqueried = frontier(0, gain=100.0, distance=None)
    known = frontier(1, gain=1.0, distance=2.0)

    assert make_policy("nearest").select(
        [unqueried, known], np.random.default_rng(0)
    ).frontier_id == 1
    assert make_policy("info_gain_minus_cost").select(
        [unqueried, known], np.random.default_rng(0)
    ).frontier_id == 1


def test_unknown_policy_name_raises():
    with pytest.raises(KeyError):
        make_policy("does_not_exist")
