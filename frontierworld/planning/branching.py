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
    semantic_grid: np.ndarray | None = None
    # Phase 6: the lineage graph is agent state too. Observations made inside
    # one branch must never reach another, or the memory Phase 11 attaches to a
    # lineage id would be built from futures that never happened.
    lineage: object | None = None
    lineage_fingerprint: str | None = None

    @classmethod
    def capture(
        cls, sim, occupancy: OccupancyMap, semantic_map=None, lineage=None
    ) -> "SimulatorSnapshot":
        return cls(
            agent_state=copy.deepcopy(sim.get_agent(0).get_state()),
            free_counts=occupancy.free_counts.copy(),
            occupied_counts=occupancy.occupied_counts.copy(),
            semantic_grid=(
                semantic_map.copy_counts() if semantic_map is not None else None
            ),
            lineage=lineage.copy() if lineage is not None else None,
            lineage_fingerprint=(
                lineage.fingerprint() if lineage is not None else None
            ),
        )

    def restore(self, sim, occupancy: OccupancyMap, semantic_map=None, lineage=None):
        sim.get_agent(0).set_state(copy.deepcopy(self.agent_state))
        occupancy.free_counts[...] = self.free_counts
        occupancy.occupied_counts[...] = self.occupied_counts
        if semantic_map is not None and self.semantic_grid is not None:
            semantic_map.restore_counts(self.semantic_grid)
        if lineage is not None and self.lineage is not None:
            restored = self.lineage.copy()
            lineage.nodes = restored.nodes
            lineage.edges = restored.edges
            lineage.active = restored.active
            lineage.next_lineage_id = restored.next_lineage_id
            lineage.timestep = restored.timestep
            lineage.event_counts = restored.event_counts
        return lineage

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
    # Semantic instances seen anywhere along the branch; the room label and the
    # target-visibility flag are both derived from these.
    observed_instances: list[int] = field(default_factory=list)
    target_seen_at_step: int | None = None
    target_pixel_count: int = 0
    # Retained RGB-D for later experiments (checklist Phase 4).
    rgb_frames: list[np.ndarray] = field(default_factory=list, repr=False)
    depth_frames: list[np.ndarray] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        skip = {"trajectory", "rgb_frames", "depth_frames", "observed_instances"}
        payload = {k: v for k, v in self.__dict__.items() if k not in skip}
        payload["n_trajectory_poses"] = len(self.trajectory)
        payload["n_observed_instances"] = len(self.observed_instances)
        payload["n_rgb_frames"] = len(self.rgb_frames)
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
    semantic_map=None,
    instance_to_category: dict | None = None,
    target_instance_ids: set | None = None,
    keep_frames: bool = False,
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

    seen_instances: set[int] = set()

    def integrate(step_index: int) -> None:
        observations = sim_observations(sim, cfg)
        rotation, translation = sensor_extrinsics(sim, "depth")
        depth = observations["depth"]
        occupancy.integrate(
            depth=depth,
            rotation=rotation,
            translation=translation,
            agent_position=np.asarray(agent.get_state().position),
            hfov_deg=float(cfg.simulator.hfov),
            max_depth=float(cfg.simulator.max_depth),
        )

        if keep_frames:
            result.rgb_frames.append(np.asarray(observations["rgb"])[..., :3].copy())
            result.depth_frames.append(np.asarray(depth, dtype=np.float32).copy())

        semantic = observations.get("semantic")
        if semantic is None:
            return
        semantic = np.squeeze(np.asarray(semantic)).astype(np.int64)

        if semantic_map is not None and instance_to_category is not None:
            points, valid = occupancy.unproject(
                depth, rotation, translation, float(cfg.simulator.hfov)
            )
            valid &= depth < (float(cfg.simulator.max_depth) - 1e-3)
            semantic_map.integrate(semantic, points, valid, instance_to_category)

        seen_instances.update(int(v) for v in np.unique(semantic))

        if target_instance_ids:
            hits = int(np.isin(semantic, list(target_instance_ids)).sum())
            if hits > result.target_pixel_count:
                result.target_pixel_count = hits
            if hits >= int(cfg.goal_detector.min_pixels) and result.target_seen_at_step is None:
                result.target_seen_at_step = step_index

    n_approach = len(option.approach_actions)
    for index, action in enumerate(option.actions):
        agent.act(action)
        position = np.asarray(agent.get_state().position, dtype=np.float64)

        moved = float(np.linalg.norm(position - previous))
        if action == MOVE_FORWARD and moved < 1e-3:
            result.collisions += 1
        result.distance_travelled_m += moved
        previous = position

        integrate(index)
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

    result.observed_instances = sorted(seen_instances)
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
    semantic_map=None,
    **execute_kwargs,
) -> tuple[list[BranchResult], SimulatorSnapshot]:
    """Execute every option from the same state, restoring between branches.

    Returns the results and the snapshot they all started from. Raises if a
    branch does not begin from the recorded state -- a silent violation here
    would poison every counterfactual comparison built on top of it.
    """
    snapshot = SimulatorSnapshot.capture(sim, occupancy, semantic_map)
    results: list[BranchResult] = []

    for option in options:
        if verify_identical_start and not snapshot.matches_agent(sim):
            raise RuntimeError(
                f"branch for frontier {option.frontier_id} would not start from "
                "the recorded state"
            )
        try:
            results.append(
                execute_option(
                    sim, option, occupancy, cfg,
                    semantic_map=semantic_map, **execute_kwargs,
                )
            )
        finally:
            snapshot.restore(sim, occupancy, semantic_map)

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
