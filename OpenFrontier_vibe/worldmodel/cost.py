"""
Path-cost estimation for frontier ranking (design section 15).

Replaces OpenFrontier's straight-line distance with planner path cost
where affordable: geodesic costs are computed only for the top-M world
model candidates (planner solves are expensive) and cached per
(frontier uid, quantized robot cell).
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PathCostEstimator:
    def __init__(self, manager, memory, params: Optional[dict] = None):
        p = params or {}
        self.manager = manager
        self.memory = memory
        self.mode = p.get("mode", "geodesic")  # euclidean | geodesic
        self.unreachable_penalty = float(p.get("unreachable_penalty", 3.0))
        self.cell = float(p.get("cache_cell", 0.75))  # robot-position quantization

    def euclidean(self, ft_pos: np.ndarray, current_pos: np.ndarray) -> float:
        return float(
            np.linalg.norm(np.asarray(ft_pos, dtype=float) - current_pos.reshape(3))
        )

    def _cache_key(self, current_pos: np.ndarray) -> str:
        q = np.round(current_pos / self.cell).astype(int)
        return f"{q[0]}_{q[1]}_{q[2]}"

    def _graph_distance(self, ft) -> float:
        """Shortest path robot->frontier through the R/F graph (meters)."""
        robot_id = self.manager.current_robot_id
        if robot_id is None:
            return float("inf")
        path, distance = self.manager.graph.get_shortest_R_to_F(
            robot_id=robot_id, frontier_id=ft.id
        )
        if not path or distance is None:
            return float("inf")
        return float(distance)

    def cost(
        self,
        ft,
        current_pose: np.ndarray,
        allow_geodesic: bool = True,
    ) -> float:
        """Path cost in meters from the robot to frontier ``ft``."""
        current_pos = np.asarray(current_pose[:3, 3], dtype=float)
        d_euclid = self.euclidean(ft.pos3d, current_pos)

        if self.mode != "geodesic" or not allow_geodesic:
            return d_euclid

        uid = ft.features.get("uid")
        rec = self.memory.get(uid) if uid else None
        key = self._cache_key(current_pos)
        if rec is not None and rec.geodesic_cost_key == key and rec.geodesic_cost is not None:
            return rec.geodesic_cost

        cost = d_euclid
        try:
            if getattr(self.manager.planner, "is_pointnav_planner", False):
                # PointNav planner has no path solver; use the frontier graph's
                # topological distance through visited poses instead (edge
                # weights are metric, so this approximates traversable length)
                length = self._graph_distance(ft)
            else:
                goal_pose = ft.pose6d if ft.pose6d is not None else None
                length = (
                    self.manager.get_optimal_path_length(current_pose, goal_pose)
                    if goal_pose is not None
                    else float("inf")
                )
            if np.isfinite(length) and length > 0:
                # graph/planner paths can undershoot the straight line when
                # endpoints get snapped; keep the max as a lower bound
                cost = max(float(length), d_euclid)
            else:
                cost = d_euclid * self.unreachable_penalty
        except Exception as e:  # noqa: BLE001 - planner failures degrade gracefully
            logger.debug("Geodesic cost failed for %s: %s", uid, e)
            cost = d_euclid

        if rec is not None:
            rec.geodesic_cost = cost
            rec.geodesic_cost_key = key
        return cost
