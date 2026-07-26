"""Forking the simulator to execute counterfactual frontier options.

Phase 5 needs every candidate frontier executed from an *identical* navigation
state. That requires two things this module provides: an exact snapshot of the
state a branch starts from, and a rollout that does not disturb the live
episode.

Rollouts drive `agent.act` and read `sim.get_sensor_observations` directly
rather than going through `env.step`. Habitat's Env counts elapsed steps and
can end the episode; a counterfactual branch must cost the episode nothing,
because it never happened as far as the agent's real trajectory is concerned.

The scene is static, so a snapshot is the agent state plus whatever the agent
itself accumulated -- here, the occupancy map. Both are restored on exit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from frontierworld.mapping.occupancy import UNKNOWN, OccupancyMap
from frontierworld.planning.options import FrontierOption, MOVE_FORWARD


@dataclass
class SimulatorSnapshot:
    """Everything needed to put the world back exactly as it was."""

    agent_state: object
    free_counts: np.ndarray
    occupied_counts: np.ndarray

    @classmethod
    def capture(cls, sim, occupancy: OccupancyMap) -> "SimulatorSnapshot":
        return cls(
            agent_state=copy.deepcopy(sim.get_agent(0).get_state()),
            free_counts=occupancy.free_counts.copy(),
            occupied_counts=occupancy.occupied_counts.copy(),
        )

    def restore(self, sim, occupancy: OccupancyMap) -> None:
        sim.get_agent(0).set_state(copy.deepcopy(self.agent_state))
        occupancy.free_counts[...] = self.free_counts
        occupancy.occupied_counts[...] = self.occupied_counts

    def matches_agent(self, sim, tolerance: float = 0.0) -> bool:
        """Is the live agent state identical to this snapshot?"""
        state = sim.get_agent(0).get_state()
        position_equal = np.allclose(
            np.asarray(state.position, dtype=np.float64),
            np.asarray(self.agent_state.position, dtype=np.float64),
            atol=tolerance,
            rtol=0.0,
        )
        rotation_equal = np.allclose(
            _quat_array(state.rotation), _quat_array(self.agent_state.rotation),
            atol=tolerance, rtol=0.0,
        )
        return bool(position_equal and rotation_equal)


@dataclass
class BranchResult:
    """What executing one option actually did."""

    frontier_id: int
    valid: bool
    reject_reason: str | None = None
    executed_actions: list[int] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    n_approach_executed: int = 0
    n_cross_executed: int = 0
    collisions: int = 0
    crossed: bool = False
    distance_travelled_m: float = 0.0
    displacement_m: float = 0.0
    newly_observed_area_m2: float = 0.0
    newly_free_area_m2: float = 0.0
    final_position: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {k: v for k, v in self.__dict__.items() if k != "trajectory"}
        payload["n_trajectory_poses"] = len(self.trajectory)
        return payload


def sim_observations(sim, cfg) -> dict:
    """Raw sensor observations, clipped to match habitat-lab's depth sensor.

    habitat-lab clips depth to [min_depth, max_depth]; the raw simulator does
    not. Skipping this would feed the mapper unclipped ranges and turn
    beyond-range misses into solid walls, sealing off the unknown region the
    branch exists to reveal.
    """
    observations = sim.get_sensor_observations()
    depth = np.asarray(observations["depth"], dtype=np.float32)
    observations = dict(observations)
    observations["depth"] = np.clip(
        depth, float(cfg.simulator.min_depth), float(cfg.simulator.max_depth)
    )
    return observations


def execute_option(
    sim,
    option: FrontierOption,
    occupancy: OccupancyMap,
    cfg,
    record_trajectory: bool = True,
) -> BranchResult:
    """Run one option from the current state, mapping as it goes.

    The caller is responsible for snapshotting and restoring; see
    `run_branches`, which does both.
    """
    from frontierworld.habitat_env import sensor_extrinsics

    result = BranchResult(
        frontier_id=option.frontier_id,
        valid=option.valid,
        reject_reason=option.reject_reason,
    )
    if not option.valid:
        return result

    agent = sim.get_agent(0)
    grid_before = occupancy.to_grid()
    start_position = np.asarray(agent.get_state().position, dtype=np.float64)
    previous = start_position

    def integrate() -> None:
        rotation, translation = sensor_extrinsics(sim, "depth")
        occupancy.integrate(
            depth=sim_observations(sim, cfg)["depth"],
            rotation=rotation,
            translation=translation,
            agent_position=np.asarray(agent.get_state().position),
            hfov_deg=float(cfg.simulator.hfov),
            max_depth=float(cfg.simulator.max_depth),
        )

    n_approach = len(option.approach_actions)
    for index, action in enumerate(option.actions):
        agent.act(action)
        position = np.asarray(agent.get_state().position, dtype=np.float64)

        moved = float(np.linalg.norm(position - previous))
        if action == MOVE_FORWARD and moved < 1e-3:
            result.collisions += 1
        result.distance_travelled_m += moved
        previous = position

        integrate()
        result.executed_actions.append(int(action))
        if index < n_approach:
            result.n_approach_executed += 1
        else:
            result.n_cross_executed += 1

        if record_trajectory:
            state = agent.get_state()
            result.trajectory.append(
                {
                    "action": int(action),
                    "position": [float(v) for v in state.position],
                    "rotation": _quat_array(state.rotation).tolist(),
                    "phase": "approach" if index < n_approach else "cross",
                }
            )

    final_position = np.asarray(agent.get_state().position, dtype=np.float64)
    result.final_position = [float(v) for v in final_position]
    result.displacement_m = float(np.linalg.norm(final_position - start_position))

    # Crossing succeeded if the agent ended up somewhere that was unknown
    # before the branch started.
    geometry = occupancy.geometry
    row, col = geometry.world_to_cell(
        np.array([final_position[0]]), np.array([final_position[2]])
    )
    row, col = int(row[0]), int(col[0])
    if geometry.in_bounds(np.array([row]), np.array([col]))[0]:
        result.crossed = bool(grid_before[row, col] == UNKNOWN)

    grid_after = occupancy.to_grid()
    cell_area = geometry.resolution**2
    result.newly_observed_area_m2 = float(
        ((grid_before == UNKNOWN) & (grid_after != UNKNOWN)).sum() * cell_area
    )
    result.newly_free_area_m2 = float(
        ((grid_before == UNKNOWN) & (grid_after == 1)).sum() * cell_area
    )
    return result


def run_branches(
    sim,
    options: list[FrontierOption],
    occupancy: OccupancyMap,
    cfg,
    verify_identical_start: bool = True,
) -> tuple[list[BranchResult], SimulatorSnapshot]:
    """Execute every option from the same state, restoring between branches.

    Returns the results and the snapshot they all started from. Raises if a
    branch does not begin from the recorded state -- a silent violation here
    would poison every counterfactual comparison built on top of it.
    """
    snapshot = SimulatorSnapshot.capture(sim, occupancy)
    results: list[BranchResult] = []

    for option in options:
        if verify_identical_start and not snapshot.matches_agent(sim):
            raise RuntimeError(
                f"branch for frontier {option.frontier_id} would not start from "
                "the recorded state"
            )
        try:
            results.append(execute_option(sim, option, occupancy, cfg))
        finally:
            snapshot.restore(sim, occupancy)

    return results, snapshot


def _quat_array(quaternion_value) -> np.ndarray:
    return np.array(
        [
            float(quaternion_value.w),
            float(quaternion_value.x),
            float(quaternion_value.y),
            float(quaternion_value.z),
        ],
        dtype=np.float64,
    )
