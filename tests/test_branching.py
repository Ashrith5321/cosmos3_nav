"""Snapshot and restore tests.

The whole branch-complete protocol rests on one property: after a branch runs,
the world is exactly as it was. These tests pin that down with a fake agent,
so a regression fails here rather than silently biasing every counterfactual
label in Phase 5.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from frontierworld.mapping.occupancy import OccupancyMap
from frontierworld.planning.branching import SimulatorSnapshot


@dataclass
class FakeQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass
class FakeAgentState:
    position: np.ndarray
    rotation: FakeQuaternion


class FakeAgent:
    def __init__(self, state: FakeAgentState) -> None:
        self._state = state

    def get_state(self) -> FakeAgentState:
        return self._state

    def set_state(self, state: FakeAgentState) -> None:
        self._state = state


class FakeSim:
    def __init__(self, state: FakeAgentState) -> None:
        self._agent = FakeAgent(state)

    def get_agent(self, index: int = 0) -> FakeAgent:
        return self._agent


def make_world():
    state = FakeAgentState(
        position=np.array([1.0, 2.0, 3.0]), rotation=FakeQuaternion(1.0, 0.0, 0.0, 0.0)
    )
    sim = FakeSim(state)
    occupancy = OccupancyMap(resolution=0.1, size_m=10.0)
    occupancy.free_counts[10:20, 10:20] = 3
    occupancy.occupied_counts[30:35, 30:35] = 7
    return sim, occupancy


def test_snapshot_restores_agent_position():
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)

    sim.get_agent(0).set_state(
        FakeAgentState(np.array([9.0, 9.0, 9.0]), FakeQuaternion(0.0, 1.0, 0.0, 0.0))
    )
    snapshot.restore(sim, occupancy)

    np.testing.assert_array_equal(
        sim.get_agent(0).get_state().position, np.array([1.0, 2.0, 3.0])
    )


def test_snapshot_restores_the_map():
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)

    occupancy.free_counts[:] = 99
    occupancy.occupied_counts[:] = 99
    snapshot.restore(sim, occupancy)

    assert occupancy.free_counts[15, 15] == 3
    assert occupancy.occupied_counts[32, 32] == 7
    assert occupancy.free_counts[0, 0] == 0


def test_snapshot_is_not_aliased_to_the_live_map():
    """Capturing must copy: a snapshot sharing memory with the live map would
    silently track the very changes it is supposed to undo."""
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)

    occupancy.free_counts[15, 15] = 42
    assert snapshot.free_counts[15, 15] == 3


def test_snapshot_is_not_aliased_to_the_live_agent_state():
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)

    sim.get_agent(0).get_state().position[0] = 77.0
    assert snapshot.agent_state.position[0] == 1.0


def test_matches_agent_detects_movement():
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)
    assert snapshot.matches_agent(sim)

    sim.get_agent(0).set_state(
        FakeAgentState(np.array([1.0, 2.0, 3.001]), FakeQuaternion(1.0, 0.0, 0.0, 0.0))
    )
    assert not snapshot.matches_agent(sim)


def test_matches_agent_detects_rotation():
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)

    sim.get_agent(0).set_state(
        FakeAgentState(np.array([1.0, 2.0, 3.0]), FakeQuaternion(0.7, 0.0, 0.7, 0.0))
    )
    assert not snapshot.matches_agent(sim)


def test_repeated_restore_is_idempotent():
    """Phase 5 restores once per branch; drift across many branches would
    accumulate into a different starting state for later candidates."""
    sim, occupancy = make_world()
    snapshot = SimulatorSnapshot.capture(sim, occupancy)

    for step in range(10):
        sim.get_agent(0).set_state(
            FakeAgentState(
                np.array([float(step), 0.0, 0.0]), FakeQuaternion(0.0, 0.0, 1.0, 0.0)
            )
        )
        occupancy.free_counts[:] = step
        snapshot.restore(sim, occupancy)
        assert snapshot.matches_agent(sim)
        assert occupancy.free_counts[15, 15] == 3
