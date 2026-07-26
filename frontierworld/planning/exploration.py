"""The frontier exploration loop.

One decision cycle:

    integrate observation -> extract frontiers -> filter reachable
    -> policy picks one -> planner returns actions -> execute -> repeat

This is the closed loop the whole project is about. Phase 11 swaps the policy
for a prediction-based score and Phase 12 reports the same metrics, so the
loop, the counters and the per-decision record are written to survive both.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from frontierworld.frontiers.extraction import Frontier, extract_frontiers
from frontierworld.mapping.occupancy import OccupancyMap
from frontierworld.planning.goal_detector import OracleSemanticGoalDetector
from frontierworld.planning.habitat_planner import (
    HabitatNavmeshPlanner,
    PlannerFailure,
)

# habitat's discrete navigation actions. The simulator agent's action space is
# integer-keyed (verified in _check_action_space), and GreedyGeodesicFollower
# returns these same integers -- which is why OpenFrontier's planner treats
# 2 as turn_left and 3 as turn_right.
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

EXPECTED_ACTION_ORDER = ["stop", "move_forward", "turn_left", "turn_right"]


@dataclass
class EpisodeResult:
    scene_id: str
    episode_id: str
    object_category: str | None
    steps: int = 0
    decisions: int = 0
    planner_failures: int = 0
    frontier_exhaustions: int = 0
    collisions: int = 0
    revisits: int = 0
    detected: bool = False
    detection_step: int | None = None
    called_stop: bool = False
    success: float = 0.0
    spl: float = 0.0
    soft_spl: float = 0.0
    distance_to_goal: float = float("nan")
    explored_area_m2: float = 0.0
    free_area_m2: float = 0.0
    wall_time_s: float = 0.0
    decision_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {k: v for k, v in self.__dict__.items() if k != "decision_log"}
        payload["decisions_per_episode"] = self.decisions
        return payload


class FrontierExplorer:
    """Runs one ObjectNav episode under a frontier-selection policy."""

    def __init__(
        self,
        env,
        cfg,
        policy,
        writer=None,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.policy = policy
        self.writer = writer
        self.planner: HabitatNavmeshPlanner | None = None
        self.detector = OracleSemanticGoalDetector(
            min_pixels=cfg.goal_detector.min_pixels,
            max_detection_distance_m=cfg.goal_detector.max_detection_distance_m,
        )
        self._turn_sign: int | None = None

    def _bind_to_scene(self) -> None:
        """(Re)build simulator-derived state after a reset.

        The pathfinder and the agent handle belong to the loaded scene graph.
        Grabbing them before env.reset() leaves them pointing at the previous
        scene, and the next call fails with "Attached Object is invalid" --
        which only shows up once a run spans more than one scene.
        """
        cfg = self.cfg
        self.planner = HabitatNavmeshPlanner(
            self.env.sim,
            goal_radius=cfg.planning.goal_radius,
            max_actions=cfg.planning.max_actions_per_decision,
        )
        self._turn_sign = None
        self._check_action_space()

    def _check_action_space(self) -> None:
        """Fail loudly if habitat's action indices are not what we assume.

        A silent mismatch here would swap turning for stopping, which shows up
        as inexplicably bad navigation rather than as an error.
        """
        names = list(self.env.task.actions.keys())[: len(EXPECTED_ACTION_ORDER)]
        if names != EXPECTED_ACTION_ORDER:
            raise RuntimeError(
                f"unexpected habitat action order {names}; "
                f"expected {EXPECTED_ACTION_ORDER}"
            )

    # -- orientation -----------------------------------------------------

    def _agent_yaw(self) -> float:
        """Agent heading, in the same convention as Frontier.yaw."""
        import quaternion  # noqa: F401

        state = self.env.sim.get_agent_state()
        forward = quaternion.as_rotation_matrix(state.rotation) @ np.array([0.0, 0.0, -1.0])
        return float(np.arctan2(forward[0], -forward[2]))

    def _calibrate_turn_sign(self) -> int:
        """Which turn action increases yaw.

        Measured rather than derived: the sign depends on habitat's rotation
        convention interacting with ours, and getting it wrong silently makes
        the agent turn the long way round on every decision.
        """
        if self._turn_sign is not None:
            return self._turn_sign
        import copy

        agent = self.env.sim.get_agent(0)
        saved = copy.deepcopy(agent.get_state())
        before = self._agent_yaw()
        try:
            agent.act(TURN_LEFT)
            delta = _wrap_angle(self._agent_yaw() - before)
        finally:
            agent.set_state(saved)
        self._turn_sign = 1 if delta > 0 else -1
        return self._turn_sign

    def _face(
        self, target_yaw: float, occupancy: OccupancyMap, result: EpisodeResult, max_steps: int
    ) -> int:
        """Turn in place toward a heading. Returns steps taken.

        Needed because a frontier the agent is already standing on produces an
        empty plan: without turning to look at it, the map never grows and the
        episode starves.
        """
        turn_angle = np.deg2rad(float(self.cfg.simulator.turn_angle))
        sign = self._calibrate_turn_sign()
        error = _wrap_angle(target_yaw - self._agent_yaw())
        n_turns = int(round(abs(error) / turn_angle))
        if n_turns == 0:
            return 0
        action = TURN_LEFT if (error > 0) == (sign > 0) else TURN_RIGHT
        return self._execute([action] * n_turns, occupancy, result, max_steps)

    def _scan(
        self, occupancy: OccupancyMap, result: EpisodeResult, max_steps: int
    ) -> int:
        """Turn a full circle to build a local map before the first decision.

        One camera frustum is too little map to extract meaningful frontiers
        from, so the first decision would otherwise be made almost blind.
        """
        turns = int(self.cfg.episode.initial_scan_turns)
        if turns <= 0:
            return 0
        return self._execute([TURN_LEFT] * turns, occupancy, result, max_steps)

    # -- helpers ---------------------------------------------------------

    def _integrate(self, observations, occupancy: OccupancyMap) -> None:
        from frontierworld.habitat_env import sensor_extrinsics

        rotation, translation = sensor_extrinsics(self.env.sim, "depth")
        occupancy.integrate(
            depth=observations["depth"],
            rotation=rotation,
            translation=translation,
            agent_position=np.asarray(self.env.sim.get_agent_state().position),
            hfov_deg=float(self.cfg.simulator.hfov),
            max_depth=float(self.cfg.simulator.max_depth),
        )

    def _frontiers(self, occupancy: OccupancyMap) -> list[Frontier]:
        frontiers = extract_frontiers(
            occupancy.to_grid(),
            occupancy.geometry,
            floor_y=occupancy.floor_y,
            min_size_cells=int(self.cfg.frontiers.min_size_cells),
            agent_radius_m=float(self.cfg.simulator.agent_radius),
            info_gain_radius_m=float(self.cfg.frontiers.info_gain_radius_m),
            connectivity=int(self.cfg.frontiers.connectivity),
        )
        self.planner.annotate_reachability(
            frontiers,
            approach_offset_m=float(self.cfg.frontiers.approach_offset_m),
            max_distance_m=float(self.cfg.frontiers.max_travel_distance_m),
        )
        return frontiers

    def _blacklist_key(self, frontier: Frontier) -> tuple[int, int]:
        """Frontiers move slightly between updates; quantise before blacklisting.

        Phase 7 replaces this with real lineage identity. Until then, a
        half-metre grid is enough to stop the loop re-selecting a frontier the
        planner has already refused.
        """
        cell = 0.5
        return (
            int(round(frontier.centroid_world[0] / cell)),
            int(round(frontier.centroid_world[2] / cell)),
        )

    # -- main loop -------------------------------------------------------

    def run(self, rng: np.random.Generator) -> EpisodeResult:
        cfg = self.cfg
        env = self.env
        started = time.perf_counter()

        observations = env.reset()
        episode = env.current_episode
        self._bind_to_scene()
        self.detector.reset(episode)

        start_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        occupancy = OccupancyMap(
            resolution=cfg.mapping.resolution,
            size_m=cfg.mapping.size_m,
            obstacle_height_min=cfg.mapping.obstacle_height_min,
            obstacle_height_max=cfg.mapping.obstacle_height_max,
            column_stride=cfg.mapping.column_stride,
            min_observations=cfg.mapping.min_observations,
            center=(float(start_position[0]), float(start_position[2])),
            floor_y=float(start_position[1]),
        )

        result = EpisodeResult(
            scene_id=str(episode.scene_id),
            episode_id=str(episode.episode_id),
            object_category=getattr(episode, "object_category", None),
        )

        self._last_observations = observations
        self._target_position = None
        self._integrate(observations, occupancy)
        if self.writer is not None:
            from frontierworld.habitat_env import agent_pose

            self.writer.write_step(0, observations, agent_pose(env.sim), action=None)

        blacklist: set[tuple[int, int]] = set()
        attempts: dict[tuple[int, int], int] = {}
        visited_keys: list[tuple[int, int]] = []
        max_steps = int(cfg.episode.max_steps)

        self._scan(occupancy, result, max_steps)

        while not env.episode_over and result.steps < max_steps:
            self._check_detection(self._last_observations, occupancy, result)

            # Once the target is located, drive to it and stop.
            if self._target_position is not None and cfg.goal_detector.stop_on_detection:
                stopped = self._approach_target(
                    self._target_position, occupancy, result, max_steps
                )
                if stopped:
                    break
                # Approach failed; drop the fix and keep exploring rather than
                # retrying the same unreachable point every cycle.
                self._target_position = None
                continue

            frontiers = self._frontiers(occupancy)
            candidates = [
                f
                for f in frontiers
                if f.reachable and self._blacklist_key(f) not in blacklist
            ]
            if not candidates:
                result.frontier_exhaustions += 1
                break

            chosen = self.policy.select(candidates, rng)
            if chosen is None:
                result.frontier_exhaustions += 1
                break

            key = self._blacklist_key(chosen)
            if key in visited_keys:
                result.revisits += 1
            visited_keys.append(key)

            goal = chosen.approach_point(float(cfg.frontiers.approach_offset_m))
            try:
                actions = self.planner.plan(goal)
            except PlannerFailure as exc:
                result.planner_failures += 1
                blacklist.add(key)
                result.decision_log.append(
                    {
                        "step": result.steps,
                        "event": "planner_failure",
                        "reason": str(exc),
                        "frontier": chosen.to_dict(),
                    }
                )
                continue

            result.decisions += 1
            result.decision_log.append(
                {
                    "step": result.steps,
                    "event": "decision",
                    "policy": self.policy.name,
                    "n_candidates": len(candidates),
                    "chosen": chosen.to_dict(),
                    "plan_length": len(actions),
                    **occupancy.stats(),
                }
            )

            executed = self._execute(actions, occupancy, result, max_steps)
            # Always look into the unknown region after arriving: a frontier
            # the agent is already standing on plans to an empty action list,
            # and without turning to face it the map would never grow.
            executed += self._face(chosen.yaw, occupancy, result, max_steps)

            attempts[key] = attempts.get(key, 0) + 1
            if executed == 0 and attempts[key] >= 2:
                # Two visits that moved nothing: this frontier is a dead end
                # for the controller, not just an unlucky plan.
                blacklist.add(key)

        metrics = env.get_metrics()
        result.success = float(metrics.get("success", 0.0))
        result.spl = float(metrics.get("spl", 0.0))
        result.soft_spl = float(metrics.get("soft_spl", 0.0))
        result.distance_to_goal = float(metrics.get("distance_to_goal", float("nan")))
        stats = occupancy.stats()
        result.explored_area_m2 = float(stats["explored_area_m2"])
        result.free_area_m2 = float(stats["free_area_m2"])
        result.wall_time_s = time.perf_counter() - started

        if self.writer is not None:
            self.writer.write_map(occupancy)

        self.occupancy = occupancy
        return result

    # -- execution -------------------------------------------------------

    def _execute(
        self,
        actions: list[Any],
        occupancy: OccupancyMap,
        result: EpisodeResult,
        max_steps: int,
    ) -> int:
        """Run a plan, mapping as we go. Returns the number of steps taken."""
        from frontierworld.habitat_env import agent_pose

        executed = 0
        for action in actions:
            if self.env.episode_over or result.steps >= max_steps:
                break
            if action is None or action == STOP:
                break

            before = np.asarray(self.env.sim.get_agent_state().position)
            observations = self.env.step(action)
            result.steps += 1
            executed += 1
            after = np.asarray(self.env.sim.get_agent_state().position)

            if _is_forward(action) and float(np.linalg.norm(after - before)) < 1e-3:
                result.collisions += 1

            self._integrate(observations, occupancy)
            self._last_observations = observations

            if self.writer is not None:
                self.writer.write_step(
                    result.steps,
                    observations,
                    agent_pose(self.env.sim),
                    action=str(action),
                    info=self.env.get_metrics(),
                )

            self._check_detection(observations, occupancy, result)

        return executed

    def _approach_target(
        self,
        target_position: np.ndarray,
        occupancy: OccupancyMap,
        result: EpisodeResult,
        max_steps: int,
    ) -> bool:
        """Navigate to a detected target and call STOP. True if STOP was called."""
        try:
            actions = self.planner.plan(target_position)
        except PlannerFailure as exc:
            result.planner_failures += 1
            result.decision_log.append(
                {
                    "step": result.steps,
                    "event": "target_plan_failure",
                    "reason": str(exc),
                }
            )
            return False

        self._execute(actions, occupancy, result, max_steps)

        position = np.asarray(self.env.sim.get_agent_state().position)
        distance = float(np.linalg.norm(position[[0, 2]] - target_position[[0, 2]]))
        if distance <= float(self.cfg.goal_detector.stop_distance_m):
            self.env.step(STOP)
            result.steps += 1
            result.called_stop = True
            result.decision_log.append(
                {"step": result.steps, "event": "stop", "distance_to_target": distance}
            )
            return True
        return False

    def _check_detection(
        self, observations, occupancy: OccupancyMap, result: EpisodeResult
    ) -> None:
        if observations is None or result.detected:
            return
        from frontierworld.habitat_env import sensor_extrinsics

        rotation, translation = sensor_extrinsics(self.env.sim, "depth")
        detection = self.detector.detect(
            semantic=observations["semantic"],
            depth=observations["depth"],
            occupancy_map=occupancy,
            rotation=rotation,
            translation=translation,
            hfov_deg=float(self.cfg.simulator.hfov),
        )
        if detection.seen:
            result.detected = True
            result.detection_step = result.steps
            self._target_position = detection.position
            result.decision_log.append(
                {
                    "step": result.steps,
                    "event": "detection",
                    "pixel_count": detection.pixel_count,
                    "distance_m": detection.distance_m,
                    "target_position": (
                        detection.position.tolist()
                        if detection.position is not None
                        else None
                    ),
                }
            )


def _is_forward(action: Any) -> bool:
    return action == MOVE_FORWARD


def _wrap_angle(angle: float) -> float:
    """Wrap to [-pi, pi). Both bounds denote the same heading."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
