"""Target detection and the stopping rule.

ObjectNav needs the agent to decide *when* it has found the goal. Phase 2 has
no learned detector, so this reads ground-truth semantics: an episode's goal
objects are known by instance id, and the target counts as seen when enough of
its pixels are in view.

PRIVILEGED INFORMATION: ground-truth semantic instance ids. Phase 8 replaces
this with the predicted target-presence head, and Phase 15 counts stopping
failures separately from exploration failures. Any reported number that uses
this detector must say so.

Detection is deliberately separated from stopping: `detect` answers "is the
target visible", `target_position` answers "where is it", and the explorer
decides what to do. Keeping them apart is what lets Phase 15 attribute a
failure to perception rather than to control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    seen: bool
    pixel_count: int = 0
    position: np.ndarray | None = None  # world (x, y, z) of the target centroid
    distance_m: float = float("inf")


class OracleSemanticGoalDetector:
    """Detects the episode's goal objects in the semantic observation."""

    def __init__(
        self,
        min_pixels: int = 100,
        max_detection_distance_m: float = 5.0,
    ) -> None:
        self.min_pixels = int(min_pixels)
        self.max_detection_distance_m = float(max_detection_distance_m)
        self.target_instance_ids: set[int] = set()

    def reset(self, episode) -> None:
        """Record which semantic instances count as the goal for this episode."""
        ids: set[int] = set()
        for goal in getattr(episode, "goals", []) or []:
            object_id = getattr(goal, "object_id", None)
            if object_id is None:
                continue
            try:
                ids.add(int(object_id))
            except (TypeError, ValueError):
                continue
        self.target_instance_ids = ids

    def detect(
        self,
        semantic: np.ndarray,
        depth: np.ndarray,
        occupancy_map,
        rotation: np.ndarray,
        translation: np.ndarray,
        hfov_deg: float,
    ) -> Detection:
        """Look for goal instances and locate them in world coordinates."""
        if not self.target_instance_ids:
            return Detection(seen=False)

        semantic = np.squeeze(np.asarray(semantic))
        mask = np.isin(semantic, list(self.target_instance_ids))
        pixel_count = int(mask.sum())
        if pixel_count < self.min_pixels:
            return Detection(seen=False, pixel_count=pixel_count)

        points, valid = occupancy_map.unproject(depth, rotation, translation, hfov_deg)
        selected = mask & valid
        if not selected.any():
            return Detection(seen=False, pixel_count=pixel_count)

        target_points = points[selected]
        # Median rather than mean: robust to a few stray pixels bleeding onto
        # a distant surface behind the object.
        position = np.median(target_points, axis=0).astype(np.float64)
        distance = float(np.linalg.norm(position[[0, 2]] - translation[[0, 2]]))

        if distance > self.max_detection_distance_m:
            return Detection(seen=False, pixel_count=pixel_count)

        return Detection(
            seen=True,
            pixel_count=pixel_count,
            position=position,
            distance_m=distance,
        )
