"""
Oracle frontier ranking evaluation (design section 38).

Uses simulator ground truth (navmesh geodesic distance from each frontier
to the nearest episode goal) to define an oracle frontier value, logged
alongside the predicted Q_i on every decision so that ranking quality
(rank correlation, top-1 agreement) can be computed offline with
scripts/analyze_wm_ranking.py - independent of low-level navigation
failures.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OracleFrontierEvaluator:
    """Wraps a habitat pathfinder + episode goals; world-frame in, values out."""

    def __init__(self, pathfinder, goal_positions_habitat: List[np.ndarray]):
        self.pathfinder = pathfinder
        self.goals = [np.asarray(g, dtype=np.float32) for g in goal_positions_habitat]

    @classmethod
    def from_episode(cls, sim, episode) -> "OracleFrontierEvaluator":
        goals = []
        for goal in episode.goals:
            # ObjectNav goals may expose view points (agent-reachable); prefer them
            view_points = getattr(goal, "view_points", None) or []
            if view_points:
                goals.extend(
                    np.asarray(vp.agent_state.position) for vp in view_points
                )
            else:
                goals.append(np.asarray(goal.position))
        return cls(sim.pathfinder, goals)

    def _geodesic(self, start_h: np.ndarray, end_h: np.ndarray) -> float:
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = self.pathfinder.snap_point(start_h)
        path.requested_end = self.pathfinder.snap_point(end_h)
        if self.pathfinder.find_path(path):
            return float(path.geodesic_distance)
        return float("inf")

    def oracle_value(self, ft_pos_world: np.ndarray) -> Optional[float]:
        """-min geodesic distance from the frontier to any goal viewpoint."""
        from utils.transform import to_habitat_position

        pos_h = to_habitat_position(np.asarray(ft_pos_world, dtype=float))
        dists = [self._geodesic(pos_h, g) for g in self.goals]
        finite = [d for d in dists if np.isfinite(d)]
        if not finite:
            return None
        return -float(min(finite))

    def evaluate(self, frontiers: List, current_pose: np.ndarray) -> dict:
        """Per-decision snapshot pairing predicted utility with oracle value."""
        rows = []
        for ft in frontiers:
            if ft.is_object or ft.pos3d is None:
                continue
            val = self.oracle_value(ft.pos3d)
            rows.append(
                {
                    "id": ft.id,
                    "uid": ft.features.get("uid"),
                    "utility": None if ft.utility is None else float(ft.utility),
                    "wm_mean": ft.features.get("wm_mean"),
                    "p_obs": None if ft.probability is None else float(ft.probability),
                    "oracle_value": val,
                }
            )
        return {"frontiers": rows}
