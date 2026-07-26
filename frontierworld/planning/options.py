"""Frontier-crossing options.

An option is the canonical macro-action for one frontier,

    omega_i = (tau_approach, tau_cross, H)

a path to an approach pose in *observed* free space, a probe that crosses the
boundary into unknown space, and an action horizon. It is canonical in the
sense that it is fixed by the frontier geometry and the configuration alone --
not by what happens when it runs. That is what makes the Phase 5 branches
comparable: every candidate is offered the same kind of chance.

The approach pose is required to lie in free space the agent has actually
observed. Snapping to the navmesh alone is not enough: the navmesh covers the
whole scene, so a snapped point can sit in territory the agent has never seen,
which would smuggle privileged geometry into the option definition itself.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from frontierworld.frontiers.extraction import Frontier
from frontierworld.mapping.occupancy import FREE
from frontierworld.planning.habitat_planner import PlannerFailure

# habitat discrete actions; see planning.exploration for the ordering check.
STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT = 0, 1, 2, 3


@dataclass
class FrontierOption:
    """The canonical macro-action for one frontier."""

    frontier_id: int
    frontier: Frontier
    approach_world: np.ndarray  # snapped approach point, habitat world x/y/z
    crossing_direction: np.ndarray  # (2,) unit vector in world (x, z)
    crossing_yaw: float
    probe_distance_m: float
    horizon: int
    approach_actions: list[int] = field(default_factory=list)
    cross_actions: list[int] = field(default_factory=list)
    valid: bool = True
    reject_reason: str | None = None

    @property
    def actions(self) -> list[int]:
        """The full option, approach then crossing."""
        return list(self.approach_actions) + list(self.cross_actions)

    def to_dict(self) -> dict:
        return {
            "frontier_id": self.frontier_id,
            "approach_world": self.approach_world.tolist(),
            "crossing_direction": self.crossing_direction.tolist(),
            "crossing_yaw": float(self.crossing_yaw),
            "probe_distance_m": float(self.probe_distance_m),
            "horizon": int(self.horizon),
            "n_approach_actions": len(self.approach_actions),
            "n_cross_actions": len(self.cross_actions),
            "cross_actions": list(self.cross_actions),
            "valid": bool(self.valid),
            "reject_reason": self.reject_reason,
            "frontier": self.frontier.to_dict(),
        }


def agent_yaw_from_state(state) -> float:
    """Heading of an agent state, matching Frontier.yaw's convention."""
    import quaternion  # noqa: F401

    forward = quaternion.as_rotation_matrix(state.rotation) @ np.array([0.0, 0.0, -1.0])
    return float(np.arctan2(forward[0], -forward[2]))


def wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi). Both bounds denote the same heading."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def turn_actions_to(
    current_yaw: float, target_yaw: float, turn_angle_deg: float, turn_sign: int
) -> list[int]:
    """Discrete turns that bring current_yaw closest to target_yaw."""
    step = np.deg2rad(turn_angle_deg)
    error = wrap_angle(target_yaw - current_yaw)
    count = int(round(abs(error) / step))
    if count == 0:
        return []
    action = TURN_LEFT if (error > 0) == (turn_sign > 0) else TURN_RIGHT
    return [action] * count


def is_in_observed_free_space(
    point: np.ndarray, occupancy, tolerance_cells: int = 1
) -> bool:
    """Is a world point inside free space the agent has actually mapped?"""
    grid = occupancy.to_grid()
    geometry = occupancy.geometry
    row, col = geometry.world_to_cell(np.array([point[0]]), np.array([point[2]]))
    row, col = int(row[0]), int(col[0])
    if not geometry.in_bounds(np.array([row]), np.array([col]))[0]:
        return False
    lo_r, hi_r = max(0, row - tolerance_cells), min(grid.shape[0], row + tolerance_cells + 1)
    lo_c, hi_c = max(0, col - tolerance_cells), min(grid.shape[1], col + tolerance_cells + 1)
    return bool((grid[lo_r:hi_r, lo_c:hi_c] == FREE).any())


def build_option(
    frontier: Frontier,
    planner,
    occupancy,
    sim,
    cfg,
    turn_sign: int,
) -> FrontierOption:
    """Construct the canonical option for one frontier.

    Rejects, rather than raises, when the frontier cannot be approached: the
    caller needs the reason recorded so Phase 5 can report why a candidate was
    dropped instead of silently thinning the branch set.
    """
    options_cfg = cfg.options
    approach_raw = frontier.approach_point(float(cfg.frontiers.approach_offset_m))
    approach = planner.snap_point(approach_raw)

    option = FrontierOption(
        frontier_id=frontier.frontier_id,
        frontier=frontier,
        approach_world=np.asarray(approach, dtype=np.float64),
        crossing_direction=np.asarray(frontier.orientation, dtype=np.float64),
        crossing_yaw=frontier.yaw,
        probe_distance_m=float(options_cfg.probe_distance_m),
        horizon=int(options_cfg.horizon_actions),
    )

    if np.isnan(approach).any():
        return _reject(option, "approach point did not snap onto the navmesh")

    drift = float(np.linalg.norm(approach[[0, 2]] - approach_raw[[0, 2]]))
    if drift > float(options_cfg.max_snap_drift_m):
        return _reject(option, f"snap moved the approach point {drift:.2f} m")

    if not is_in_observed_free_space(approach, occupancy):
        return _reject(option, "approach pose is not in observed free space")

    # tau_approach
    try:
        option.approach_actions = list(planner.plan(approach))
    except PlannerFailure as exc:
        return _reject(option, f"unreachable before crossing: {exc}")

    # tau_cross has to be computed from the pose the approach ends at, so
    # roll the approach forward on a copy of the agent state and restore.
    agent = sim.get_agent(0)
    saved = copy.deepcopy(agent.get_state())
    try:
        for action in option.approach_actions:
            agent.act(action)
        arrival_yaw = agent_yaw_from_state(agent.get_state())
    finally:
        agent.set_state(saved)

    turns = turn_actions_to(
        arrival_yaw, option.crossing_yaw, float(cfg.simulator.turn_angle), turn_sign
    )
    steps = int(round(option.probe_distance_m / float(cfg.simulator.forward_step_size)))
    cross = turns + [MOVE_FORWARD] * steps

    if len(cross) > option.horizon:
        # Truncating turns first would leave the agent facing the wrong way,
        # so keep the turns and shorten the probe.
        if len(turns) >= option.horizon:
            return _reject(
                option,
                f"turning to the crossing heading needs {len(turns)} actions, "
                f"over the horizon of {option.horizon}",
            )
        cross = turns + [MOVE_FORWARD] * (option.horizon - len(turns))

    option.cross_actions = cross
    return option


def _reject(option: FrontierOption, reason: str) -> FrontierOption:
    option.valid = False
    option.reject_reason = reason
    option.approach_actions = []
    option.cross_actions = []
    return option


def build_options(
    frontiers, planner, occupancy, sim, cfg, turn_sign: int
) -> list[FrontierOption]:
    return [
        build_option(frontier, planner, occupancy, sim, cfg, turn_sign)
        for frontier in frontiers
    ]
