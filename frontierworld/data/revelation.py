"""Ground-truth revelation: what crossing a frontier actually revealed.

This is the prediction target for the whole project, so it is defined once,
here, before any model exists. For a branch that ran option omega_i from
decision time t over horizon H:

    dM_occ  = M_{t+H}^observed \\ M_t^observed     newly visible occupancy
    dM_sem  = newly labelled semantic cells
    area    = newly revealed square metres
    r       = room category encountered (DERIVED, see mapping.semantic)
    y_goal  = whether the target became visible
    d_goal  = distance from the final position to the target
    chi     = collision / traversability outcome
    F_new   = frontiers created that did not exist before

Everything is a set difference against the pre-branch state, which is why the
snapshot in planning.branching has to be exact: a leak between branches would
show up here as revelation that some other branch produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN


@dataclass
class Revelation:
    """Ground-truth outcome of executing one frontier-crossing option."""

    frontier_id: int

    # dM_occ: newly visible occupancy
    newly_observed_cells: int = 0
    newly_free_cells: int = 0
    newly_occupied_cells: int = 0
    newly_observed_area_m2: float = 0.0
    newly_free_area_m2: float = 0.0

    # dM_sem: newly observed semantics.
    #
    # Two quantities, because they mean different things. The *revealed* pair
    # counts labels gained inside dM_occ -- genuinely new territory, and the
    # correct prediction target. The plain pair also counts cells the agent had
    # already mapped geometrically but never labelled, which measures labelling
    # catching up on known space. Measured at 3.3x the revealed figure, so
    # conflating them would train the model to predict re-labelling of space it
    # has already seen.
    newly_semantic_cells: int = 0
    newly_semantic_area_m2: float = 0.0
    newly_semantic_in_revealed_cells: int = 0
    newly_semantic_in_revealed_area_m2: float = 0.0
    newly_observed_categories: dict[str, int] = field(default_factory=dict)
    revealed_categories: dict[str, int] = field(default_factory=dict)

    # r: room category encountered (derived from object composition)
    room_category: str | None = None
    room_categories_seen: dict[str, int] = field(default_factory=dict)
    room_category_is_derived: bool = True

    # y_goal: target evidence
    target_became_visible: bool = False
    target_pixel_count: int = 0
    target_visible_at_step: int | None = None

    # d_goal: distance from the final position to the target
    distance_to_target_m: float = float("nan")
    geodesic_distance_to_target_m: float = float("nan")
    distance_to_target_reduced_m: float = float("nan")

    # chi: traversability
    crossed: bool = False
    collisions: int = 0
    collision_rate: float = 0.0
    distance_travelled_m: float = 0.0
    displacement_m: float = 0.0
    actions_executed: int = 0

    # F_new: newly created frontiers
    n_frontiers_before: int = 0
    n_frontiers_after: int = 0
    n_new_frontiers: int = 0
    new_frontier_gain_m2: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def occupancy_delta(grid_before: np.ndarray, grid_after: np.ndarray, resolution: float) -> dict:
    """Set difference between two occupancy grids."""
    was_unknown = grid_before == UNKNOWN
    now_known = grid_after != UNKNOWN
    newly_observed = was_unknown & now_known
    cell_area = resolution**2
    return {
        "newly_observed_mask": newly_observed,
        "newly_observed_cells": int(newly_observed.sum()),
        "newly_free_cells": int((newly_observed & (grid_after == FREE)).sum()),
        "newly_occupied_cells": int((newly_observed & (grid_after == OCCUPIED)).sum()),
        "newly_observed_area_m2": float(newly_observed.sum() * cell_area),
        "newly_free_area_m2": float(
            (newly_observed & (grid_after == FREE)).sum() * cell_area
        ),
    }


def semantic_delta(
    semantic_before: np.ndarray,
    semantic_after: np.ndarray,
    semantic_map,
    resolution: float,
    revealed_mask: np.ndarray | None = None,
) -> dict:
    """Cells that gained a semantic label during the branch.

    `revealed_mask` restricts the primary figures to dM_occ. Without it the
    count is dominated by already-mapped cells finally receiving a label,
    which is not revelation.
    """
    from frontierworld.mapping.semantic import UNLABELLED

    newly_labelled = (semantic_before == UNLABELLED) & (semantic_after != UNLABELLED)
    in_revealed = (
        newly_labelled & revealed_mask if revealed_mask is not None else newly_labelled
    )
    return {
        "newly_semantic_mask": newly_labelled,
        "newly_semantic_in_revealed_mask": in_revealed,
        "newly_semantic_cells": int(newly_labelled.sum()),
        "newly_semantic_area_m2": float(newly_labelled.sum() * resolution**2),
        "newly_semantic_in_revealed_cells": int(in_revealed.sum()),
        "newly_semantic_in_revealed_area_m2": float(in_revealed.sum() * resolution**2),
        "newly_observed_categories": semantic_map.categories_in(newly_labelled),
        "revealed_categories": semantic_map.categories_in(in_revealed),
    }


def frontier_delta(frontiers_before, frontiers_after, match_radius_m: float = 0.75) -> dict:
    """Frontiers that appeared during the branch.

    Matched by centroid proximity. Phase 6 replaces this with lineage identity;
    until then a radius test is enough to tell a genuinely new opening from the
    same frontier having shifted a few cells.
    """
    before_centroids = [f.centroid_world[[0, 2]] for f in frontiers_before]
    new_frontiers = []
    for frontier in frontiers_after:
        centre = frontier.centroid_world[[0, 2]]
        if all(
            float(np.linalg.norm(centre - other)) > match_radius_m
            for other in before_centroids
        ):
            new_frontiers.append(frontier)

    return {
        "n_frontiers_before": len(frontiers_before),
        "n_frontiers_after": len(frontiers_after),
        "n_new_frontiers": len(new_frontiers),
        "new_frontier_gain_m2": float(
            sum(f.information_gain_m2 for f in new_frontiers)
        ),
        "new_frontiers": new_frontiers,
    }


def compute_revelation(
    frontier_id: int,
    grid_before: np.ndarray,
    grid_after: np.ndarray,
    semantic_before: np.ndarray,
    semantic_after: np.ndarray,
    semantic_map,
    frontiers_before,
    frontiers_after,
    branch,
    resolution: float,
    room_labeller=None,
    target_detection=None,
    distance_to_target_before: float | None = None,
    geodesic_to_target: float | None = None,
    instances_before=None,
) -> tuple[Revelation, dict]:
    """Assemble the full revelation for one executed branch.

    Returns the revelation and the intermediate masks, which the visualiser
    needs to draw the newly revealed region.
    """
    revelation = Revelation(frontier_id=frontier_id)

    occupancy_stats = occupancy_delta(grid_before, grid_after, resolution)
    semantic_stats = semantic_delta(
        semantic_before, semantic_after, semantic_map, resolution,
        revealed_mask=occupancy_stats["newly_observed_mask"],
    )
    frontier_stats = frontier_delta(frontiers_before, frontiers_after)

    for key, value in occupancy_stats.items():
        if key.endswith("_mask"):
            continue
        setattr(revelation, key, value)
    for key, value in semantic_stats.items():
        if key.endswith("_mask"):
            continue
        setattr(revelation, key, value)
    for key in (
        "n_frontiers_before",
        "n_frontiers_after",
        "n_new_frontiers",
        "new_frontier_gain_m2",
    ):
        setattr(revelation, key, frontier_stats[key])

    # Traversability
    revelation.crossed = bool(branch.crossed)
    revelation.collisions = int(branch.collisions)
    revelation.actions_executed = len(branch.executed_actions)
    revelation.collision_rate = float(
        branch.collisions / max(1, len(branch.executed_actions))
    )
    revelation.distance_travelled_m = float(branch.distance_travelled_m)
    revelation.displacement_m = float(branch.displacement_m)

    # Room category, derived only from instances this branch newly saw.
    # Using every instance visible during the branch would fold in the room the
    # agent started in, which biased the label heavily toward one room type.
    if room_labeller is not None:
        instances = set(getattr(branch, "observed_instances", None) or [])
        if instances_before is not None:
            instances = instances - set(instances_before)
        rooms = room_labeller.rooms_from_instances(instances)
        revelation.room_categories_seen = rooms
        revelation.room_category = (
            max(rooms.items(), key=lambda item: item[1])[0] if rooms else None
        )

    # Target evidence
    if target_detection is not None:
        revelation.target_became_visible = bool(target_detection.get("seen", False))
        revelation.target_pixel_count = int(target_detection.get("pixel_count", 0))
        revelation.target_visible_at_step = target_detection.get("step")
        revelation.distance_to_target_m = float(
            target_detection.get("distance_m", float("nan"))
        )
    if geodesic_to_target is not None:
        revelation.geodesic_distance_to_target_m = float(geodesic_to_target)
        if distance_to_target_before is not None:
            revelation.distance_to_target_reduced_m = float(
                distance_to_target_before - geodesic_to_target
            )

    masks = {
        "newly_observed": occupancy_stats["newly_observed_mask"],
        "newly_semantic": semantic_stats["newly_semantic_in_revealed_mask"],
        "newly_semantic_all": semantic_stats["newly_semantic_mask"],
        "new_frontiers": frontier_stats["new_frontiers"],
    }
    return revelation, masks
