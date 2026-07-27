"""Frontier lineage graph: birth, update, split, merge, retire. Phase 6."""

from frontierworld.lineage.graph import (
    ACTIVE,
    BIRTH,
    CROSSED,
    MERGE,
    RETIRE,
    RETIRED,
    SPLIT,
    UPDATE,
    FrontierObservation,
    LineageEdge,
    LineageGraph,
    observations_from_frontiers,
    unknown_components,
)
from frontierworld.lineage.matching import (
    MatchingConfig,
    association_matrix,
    nearest_centroid_baseline,
    update_lineage,
)
from frontierworld.lineage.metrics import LineageMetrics, evaluate_lineage
from frontierworld.lineage.oracle import oracle_lineage, long_absence_events

__all__ = [
    "LineageGraph",
    "FrontierObservation",
    "LineageEdge",
    "observations_from_frontiers",
    "unknown_components",
    "update_lineage",
    "association_matrix",
    "MatchingConfig",
    "nearest_centroid_baseline",
    "evaluate_lineage",
    "LineageMetrics",
    "oracle_lineage",
    "long_absence_events",
    "BIRTH", "UPDATE", "SPLIT", "MERGE", "RETIRE", "CROSSED", "ACTIVE", "RETIRED",
]
