"""Associating frontier observations through time.

Builds a soft association matrix A_ij between the previous timestep's active
observations and the current ones, then resolves it in three passes:

    1. one-to-one   optimal assignment on the confident pairs
    2. splits       one parent still strongly matching several children
    3. merges       several parents still strongly matching one child

Order matters. Resolving splits before the confident one-to-one matches lets a
single high-scoring pair get absorbed into a spurious split, which shows up
later as an identity switch rather than as an error here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frontierworld.lineage.graph import (
    ACTIVE,
    BIRTH,
    MERGE,
    RETIRE,
    RETIRED,
    SPLIT,
    UPDATE,
    FrontierObservation,
    LineageGraph,
)


@dataclass
class MatchingConfig:
    """Weights and thresholds for association.

    Boundary overlap dominates because it is the only cue that directly says
    "this is the same opening"; the rest disambiguate when a boundary has moved
    far enough that overlap alone is silent.
    """

    weight_overlap: float = 0.45
    weight_distance: float = 0.20
    weight_orientation: float = 0.15
    weight_region: float = 0.20
    weight_appearance: float = 0.0  # enabled once RGB descriptors exist

    dilation_cells: int = 3
    max_centroid_distance_m: float = 2.0
    match_threshold: float = 0.35
    split_threshold: float = 0.30
    merge_threshold: float = 0.30


def boundary_overlap(
    a: FrontierObservation, b: FrontierObservation, dilation: int
) -> float:
    """IoU of two boundary sets after dilation, in grid coordinates.

    Dilating first is what makes this robust to a boundary shifting by a cell
    or two between updates, which happens constantly as the map fills in.
    """
    if a.boundary_cells.size == 0 or b.boundary_cells.size == 0:
        return 0.0

    def dilate(cells: np.ndarray) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for row, col in cells:
            for dr in range(-dilation, dilation + 1):
                for dc in range(-dilation, dilation + 1):
                    out.add((int(row) + dr, int(col) + dc))
        return out

    set_a, set_b = dilate(a.boundary_cells), dilate(b.boundary_cells)
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union else 0.0


def centroid_score(a: FrontierObservation, b: FrontierObservation, max_distance: float) -> float:
    distance = float(np.linalg.norm(a.centroid - b.centroid))
    return max(0.0, 1.0 - distance / max_distance) if max_distance > 0 else 0.0


def orientation_score(a: FrontierObservation, b: FrontierObservation) -> float:
    """Cosine agreement between crossing normals, mapped to [0, 1]."""
    na, nb = a.normal, b.normal
    denominator = float(np.linalg.norm(na) * np.linalg.norm(nb))
    if denominator < 1e-9:
        return 0.0
    return float(np.clip((np.dot(na, nb) / denominator + 1.0) / 2.0, 0.0, 1.0))


def region_score(a: FrontierObservation, b: FrontierObservation) -> float:
    """Do both border the same unknown region?

    Unknown components are relabelled every timestep, so the ids are not
    comparable across time. What is comparable is relative size: two views of
    the same room agree on roughly how much is left unexplored, and the region
    behind a doorway shrinks smoothly rather than jumping.
    """
    if a.unknown_component < 0 or b.unknown_component < 0:
        return 0.0
    larger = max(a.unknown_area_m2, b.unknown_area_m2)
    if larger <= 0.0:
        return 0.0
    smaller = min(a.unknown_area_m2, b.unknown_area_m2)
    return float(smaller / larger)


def appearance_score(a: FrontierObservation, b: FrontierObservation) -> float:
    if a.appearance is None or b.appearance is None:
        return 0.0
    denominator = float(np.linalg.norm(a.appearance) * np.linalg.norm(b.appearance))
    if denominator < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a.appearance, b.appearance) / denominator, 0.0, 1.0))


def association_matrix(
    previous: list[FrontierObservation],
    current: list[FrontierObservation],
    config: MatchingConfig,
) -> np.ndarray:
    """Soft association weights A_ij in [0, 1]."""
    matrix = np.zeros((len(previous), len(current)), dtype=np.float64)
    for i, a in enumerate(previous):
        for j, b in enumerate(current):
            distance = float(np.linalg.norm(a.centroid - b.centroid))
            if distance > config.max_centroid_distance_m:
                continue  # far enough apart that no cue should rescue it
            matrix[i, j] = (
                config.weight_overlap * boundary_overlap(a, b, config.dilation_cells)
                + config.weight_distance
                * centroid_score(a, b, config.max_centroid_distance_m)
                + config.weight_orientation * orientation_score(a, b)
                + config.weight_region * region_score(a, b)
                + config.weight_appearance * appearance_score(a, b)
            )
    return matrix


def update_lineage(
    graph: LineageGraph,
    observations: list[FrontierObservation],
    timestep: int,
    config: MatchingConfig | None = None,
) -> dict:
    """Fold one timestep of observations into the graph.

    Returns a per-step event summary.
    """
    from scipy.optimize import linear_sum_assignment

    config = config or MatchingConfig()
    previous = graph.active_observations()
    matrix = association_matrix(previous, observations, config)

    matched_parents: set[int] = set()
    matched_children: set[int] = set()
    events = {BIRTH: 0, UPDATE: 0, SPLIT: 0, MERGE: 0, RETIRE: 0}

    # 1. confident one-to-one assignment
    if matrix.size:
        rows, cols = linear_sum_assignment(-matrix)
        for i, j in zip(rows, cols):
            if matrix[i, j] < config.match_threshold:
                continue
            parent, child = previous[i], observations[j]
            child.lineage_id = parent.lineage_id
            graph.add_node(child)
            graph.add_edge(parent.node_id, child.node_id, UPDATE, matrix[i, j])
            matched_parents.add(i)
            matched_children.add(j)
            events[UPDATE] += 1

    # 2. splits: a matched parent that also explains an unmatched child
    for i in sorted(matched_parents):
        for j in range(len(observations)):
            if j in matched_children or matrix[i, j] < config.split_threshold:
                continue
            parent, child = previous[i], observations[j]
            child.lineage_id = graph.new_lineage_id()  # a split child is a new identity
            graph.add_node(child)
            graph.add_edge(parent.node_id, child.node_id, SPLIT, matrix[i, j])
            matched_children.add(j)
            events[SPLIT] += 1

    # 3. merges: an unmatched parent explained by an already-matched child
    for i in range(len(previous)):
        if i in matched_parents:
            continue
        best_j, best_score = -1, 0.0
        for j in sorted(matched_children):
            if matrix[i, j] > best_score:
                best_j, best_score = j, matrix[i, j]
        if best_j >= 0 and best_score >= config.merge_threshold:
            graph.add_edge(
                previous[i].node_id, observations[best_j].node_id, MERGE, best_score
            )
            previous[i].status = RETIRED
            matched_parents.add(i)
            events[MERGE] += 1

    # 4. births and retirements
    for j, child in enumerate(observations):
        if j in matched_children:
            continue
        child.lineage_id = graph.new_lineage_id()
        graph.add_node(child)
        graph.add_edge("", child.node_id, BIRTH, 1.0)
        events[BIRTH] += 1

    for i, parent in enumerate(previous):
        if i in matched_parents:
            continue
        parent.status = RETIRED
        graph.add_edge(parent.node_id, "", RETIRE, 1.0)
        events[RETIRE] += 1

    graph.active = [o.node_id for o in observations]
    for observation in observations:
        observation.status = ACTIVE
    graph.timestep = timestep
    return events


def nearest_centroid_baseline(
    previous: list[FrontierObservation],
    current: list[FrontierObservation],
    max_distance_m: float = 2.0,
) -> dict[int, int]:
    """Phase 14's weakest association baseline: nearest centroid, greedy.

    Kept here so the ablation compares against a real implementation rather
    than a described one.
    """
    assignments: dict[int, int] = {}
    taken: set[int] = set()
    for j, child in enumerate(current):
        best_i, best_distance = -1, max_distance_m
        for i, parent in enumerate(previous):
            if i in taken:
                continue
            distance = float(np.linalg.norm(parent.centroid - child.centroid))
            if distance < best_distance:
                best_i, best_distance = i, distance
        if best_i >= 0:
            assignments[j] = best_i
            taken.add(best_i)
    return assignments
