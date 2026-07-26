"""Navmesh planner, ported from OpenFrontier's ``planner/habitat_planner.py``.

The mechanism is theirs, unchanged:

    pathfinder.snap_point -> ShortestPath -> GreedyGeodesicFollower
    -> discrete action list, with the agent state saved and restored around
       the follower rollout so planning never perturbs the live episode.

Two deliberate differences from the original:

1. Coordinates stay habitat-native (x right, y up, z forward-negative).
   OpenFrontier converts to a z-up frame via ``utils.transform.from_habitat_position``;
   carrying a second convention alongside the habitat-native occupancy map
   would be a standing source of sign errors, and nothing here needs their frame.
2. Only the parts Phase 2 uses are ported -- the OMPL/RRT branches, the
   pointnav planner and the 3D plotting are left behind.

PRIVILEGED INFORMATION: the navmesh is ground truth for the whole scene,
including regions the agent has never observed. This is the same assumption
OpenFrontier's baseline makes, and it is what isolates frontier-selection
quality from local-controller quality. It must be labelled as such in any
reported result, and it is never available at deployment.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np


class PlannerFailure(Exception):
    """Raised when no path exists to a requested goal."""


class HabitatNavmeshPlanner:
    """Plans discrete ObjectNav actions to a world-space goal."""

    def __init__(
        self,
        sim,
        goal_radius: float | None = None,
        max_actions: int = 500,
    ) -> None:
        from habitat_sim.nav import GreedyGeodesicFollower, ShortestPath

        self.sim = sim
        self.pathfinder = sim.pathfinder
        self.sim_agent = sim.get_agent(0)
        self.shortest_path = ShortestPath()
        self.follower = GreedyGeodesicFollower(
            self.pathfinder,
            self.sim_agent,
            **({"goal_radius": goal_radius} if goal_radius is not None else {}),
        )
        self.max_actions = int(max_actions)
        self.actions: list[Any] = []
        self.last_failure: str | None = None

    # -- geometry --------------------------------------------------------

    def snap_point(self, point: Sequence[float]) -> np.ndarray:
        """Snap a world point onto the navmesh."""
        snapped = self.pathfinder.snap_point(np.asarray(point, dtype=np.float32))
        return np.asarray(snapped, dtype=np.float64)

    def is_navigable(self, point: Sequence[float]) -> bool:
        return bool(self.pathfinder.is_navigable(np.asarray(point, dtype=np.float32)))

    def geodesic_distance(self, goal: Sequence[float], start: Sequence[float] | None = None) -> float:
        """Geodesic distance along the navmesh, or inf if unreachable.

        Cheaper than planning actions, so policies use this for travel cost and
        for the reachability filter.
        """
        start_position = (
            np.asarray(self.sim_agent.get_state().position, dtype=np.float32)
            if start is None
            else np.asarray(start, dtype=np.float32)
        )
        goal_snapped = self.snap_point(goal)
        if np.isnan(goal_snapped).any():
            return float("inf")

        self.shortest_path.requested_start = start_position
        self.shortest_path.requested_end = np.asarray(goal_snapped, dtype=np.float32)
        if not self.pathfinder.find_path(self.shortest_path):
            return float("inf")
        distance = float(self.shortest_path.geodesic_distance)
        return distance if np.isfinite(distance) else float("inf")

    # -- planning --------------------------------------------------------

    def plan(self, goal: Sequence[float]) -> list[Any]:
        """Discrete actions from the agent's current pose to ``goal``.

        Returns an empty list when already at the goal. Raises PlannerFailure
        when no path exists.
        """
        self.actions = []
        self.last_failure = None

        start_position = np.asarray(self.sim_agent.get_state().position, dtype=np.float32)
        goal_snapped = self.snap_point(goal)
        if np.isnan(goal_snapped).any():
            self.last_failure = "goal did not snap onto the navmesh"
            raise PlannerFailure(self.last_failure)

        if np.allclose(start_position, goal_snapped, atol=0.25):
            return []

        if not self.pathfinder.is_navigable(start_position):
            self.last_failure = f"start not navigable: {start_position}"
            raise PlannerFailure(self.last_failure)
        if not self.pathfinder.is_navigable(np.asarray(goal_snapped, dtype=np.float32)):
            self.last_failure = f"goal not navigable: {goal_snapped}"
            raise PlannerFailure(self.last_failure)

        self.shortest_path.requested_start = start_position
        self.shortest_path.requested_end = np.asarray(goal_snapped, dtype=np.float32)
        if not self.pathfinder.find_path(self.shortest_path):
            self.last_failure = "pathfinder found no path"
            raise PlannerFailure(self.last_failure)

        waypoints = self.shortest_path.points
        actions: list[Any] = []

        # The follower drives the real sim agent, so snapshot and restore.
        saved_state = copy.deepcopy(self.sim_agent.get_state())
        try:
            for index in range(1, len(waypoints)):
                section = self.follower.find_path(waypoints[index])
                for action in section:
                    if action is None:
                        break
                    self.sim_agent.act(action)
                    actions.append(action)
                    if len(actions) >= self.max_actions:
                        break
                if len(actions) >= self.max_actions:
                    break
        except Exception as exc:  # noqa: BLE001 - follower raises on thrashing
            self.last_failure = f"follower failed: {exc}"
            actions = []
        finally:
            self.sim_agent.set_state(saved_state)

        if not actions:
            self.last_failure = self.last_failure or "follower produced no actions"
            raise PlannerFailure(self.last_failure)

        self.actions = actions
        return actions

    def try_plan(self, goal: Sequence[float]) -> list[Any] | None:
        """plan() but returning None instead of raising."""
        try:
            return self.plan(goal)
        except PlannerFailure:
            return None

    # -- frontier helpers ------------------------------------------------

    def annotate_reachability(
        self,
        frontiers,
        approach_offset_m: float = 0.25,
        max_distance_m: float = float("inf"),
    ) -> list:
        """Fill in geodesic_distance_m and reachable on each frontier.

        Uses the geodesic query rather than a full action rollout: the same
        reachability answer at a fraction of the cost, which matters when a
        decision state has a dozen candidates.
        """
        for frontier in frontiers:
            target = frontier.approach_point(approach_offset_m)
            snapped = self.snap_point(target)
            if np.isnan(snapped).any():
                frontier.geodesic_distance_m = float("inf")
                frontier.reachable = False
                continue
            # Reject snaps that jumped somewhere else entirely.
            snap_error = float(np.linalg.norm(snapped[[0, 2]] - target[[0, 2]]))
            distance = self.geodesic_distance(snapped)
            frontier.geodesic_distance_m = distance
            frontier.reachable = bool(
                np.isfinite(distance) and distance <= max_distance_m and snap_error < 1.0
            )
            frontier.metadata["snap_error_m"] = snap_error
            frontier.metadata["snapped_world"] = snapped.tolist()
        return frontiers


def reachable_frontiers(frontiers) -> list:
    return [f for f in frontiers if f.reachable]
