"""The frontier lineage graph.

A frontier is not a stable object. As the map grows, a boundary moves, splits
into two doorways, merges with another, or disappears when the region behind it
is resolved. Nearest-centroid matching assumes one frontier at time t maps to
one frontier at t+1, which is exactly the assumption that fails.

Nodes here are time-indexed frontier *observations*; edges record what happened
between consecutive timesteps:

    birth   a boundary with no predecessor
    update  one-to-one continuation
    split   one parent, several children
    merge   several parents, one child
    retire  a parent with no successor

A lineage id is the persistent identity that survives all of these. Phase 11
attaches memory to that id, so an identity mistake here silently corrupts the
memory built on top of it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

BIRTH = "birth"
UPDATE = "update"
SPLIT = "split"
MERGE = "merge"
RETIRE = "retire"
CROSSED = "crossed"

ACTIVE = "active"
RETIRED = "retired"


@dataclass
class FrontierObservation:
    """One frontier as seen at one timestep: v_i^t."""

    node_id: str
    timestep: int
    lineage_id: int
    boundary_cells: np.ndarray  # b: (N, 2) grid cells
    centroid: np.ndarray  # c: world (x, z)
    normal: np.ndarray  # n: unit crossing direction in world (x, z)
    unknown_component: int = -1  # u: id of the unknown region it borders
    unknown_area_m2: float = 0.0  # u: size of that region
    appearance: np.ndarray | None = None  # z: optional visual descriptor
    status: str = ACTIVE  # sigma
    size_cells: int = 0
    information_gain_m2: float = 0.0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "timestep": self.timestep,
            "lineage_id": self.lineage_id,
            "centroid": self.centroid.tolist(),
            "normal": self.normal.tolist(),
            "unknown_component": int(self.unknown_component),
            "unknown_area_m2": float(self.unknown_area_m2),
            "status": self.status,
            "size_cells": int(self.size_cells),
            "information_gain_m2": float(self.information_gain_m2),
            "n_boundary_cells": int(len(self.boundary_cells)),
        }


@dataclass
class LineageEdge:
    parent: str
    child: str
    event: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "parent": self.parent,
            "child": self.child,
            "event": self.event,
            "weight": float(self.weight),
        }


@dataclass
class LineageGraph:
    """Time-indexed frontier observations and the events linking them."""

    nodes: dict[str, FrontierObservation] = field(default_factory=dict)
    edges: list[LineageEdge] = field(default_factory=list)
    active: list[str] = field(default_factory=list)  # node ids at the last step
    next_lineage_id: int = 0
    timestep: int = -1
    event_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # -- construction ----------------------------------------------------

    def new_lineage_id(self) -> int:
        value = self.next_lineage_id
        self.next_lineage_id += 1
        return value

    def add_node(self, observation: FrontierObservation) -> FrontierObservation:
        self.nodes[observation.node_id] = observation
        return observation

    def add_edge(self, parent: str, child: str, event: str, weight: float = 1.0) -> None:
        self.edges.append(LineageEdge(parent, child, event, weight))
        self.event_counts[event] += 1

    # -- queries ---------------------------------------------------------

    def parents_of(self, node_id: str) -> list[str]:
        return [e.parent for e in self.edges if e.child == node_id]

    def children_of(self, node_id: str) -> list[str]:
        return [e.child for e in self.edges if e.parent == node_id]

    def lineage_of(self, node_id: str) -> int:
        return self.nodes[node_id].lineage_id

    def nodes_in_lineage(self, lineage_id: int) -> list[FrontierObservation]:
        return sorted(
            (n for n in self.nodes.values() if n.lineage_id == lineage_id),
            key=lambda n: n.timestep,
        )

    def active_observations(self) -> list[FrontierObservation]:
        return [self.nodes[i] for i in self.active if i in self.nodes]

    def lineage_ids(self) -> set[int]:
        return {n.lineage_id for n in self.nodes.values()}

    def history(self, lineage_id: int) -> list[str]:
        """Event sequence for a lineage, oldest first."""
        events: list[str] = []
        members = {n.node_id for n in self.nodes_in_lineage(lineage_id)}
        for edge in self.edges:
            if edge.child in members or edge.parent in members:
                events.append(edge.event)
        return events

    # -- snapshotting ----------------------------------------------------

    def copy(self) -> "LineageGraph":
        """Deep enough copy to isolate a counterfactual branch.

        Branch isolation is the point: observations made inside branch i must
        never reach branch j. Sharing node objects between copies would let a
        status flip leak across, so nodes are rebuilt.
        """
        clone = LineageGraph(
            nodes={
                key: FrontierObservation(
                    node_id=value.node_id,
                    timestep=value.timestep,
                    lineage_id=value.lineage_id,
                    boundary_cells=value.boundary_cells.copy(),
                    centroid=value.centroid.copy(),
                    normal=value.normal.copy(),
                    unknown_component=value.unknown_component,
                    unknown_area_m2=value.unknown_area_m2,
                    appearance=(
                        None if value.appearance is None else value.appearance.copy()
                    ),
                    status=value.status,
                    size_cells=value.size_cells,
                    information_gain_m2=value.information_gain_m2,
                )
                for key, value in self.nodes.items()
            },
            edges=[LineageEdge(e.parent, e.child, e.event, e.weight) for e in self.edges],
            active=list(self.active),
            next_lineage_id=self.next_lineage_id,
            timestep=self.timestep,
        )
        clone.event_counts = defaultdict(int, self.event_counts)
        return clone

    def fingerprint(self) -> str:
        """Stable hash, for verifying a branch left the graph untouched."""
        import hashlib

        digest = hashlib.sha256()
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            digest.update(
                f"{node_id}|{node.lineage_id}|{node.status}|{node.timestep}".encode()
            )
        for edge in sorted(self.edges, key=lambda e: (e.parent, e.child, e.event)):
            digest.update(f"{edge.parent}|{edge.child}|{edge.event}".encode())
        return digest.hexdigest()[:16]

    def summary(self) -> dict:
        return {
            "timestep": self.timestep,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_lineages": len(self.lineage_ids()),
            "n_active": len(self.active),
            "events": dict(self.event_counts),
        }

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "active": list(self.active),
            "summary": self.summary(),
        }


def unknown_components(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components of unknown space.

    This is what separates a physical unresolved region from the boundary
    segments through which it happens to be visible: two frontiers touching the
    same component are looking into the same room, even when their boundaries
    never overlap.
    """
    from scipy import ndimage

    from frontierworld.mapping.occupancy import UNKNOWN

    labels, count = ndimage.label(
        grid == UNKNOWN, structure=ndimage.generate_binary_structure(2, 2)
    )
    return labels, int(count)


def describe_unknown_region(
    frontier, component_labels: np.ndarray, resolution: float
) -> tuple[int, float]:
    """Which unknown component a frontier borders, and how large it is."""
    height, width = component_labels.shape
    touching: dict[int, int] = defaultdict(int)
    for delta_row in (-1, 0, 1):
        for delta_col in (-1, 0, 1):
            rows = frontier.cells[:, 0] + delta_row
            cols = frontier.cells[:, 1] + delta_col
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            if not valid.any():
                continue
            for label in component_labels[rows[valid], cols[valid]]:
                if label > 0:
                    touching[int(label)] += 1
    if not touching:
        return -1, 0.0
    component = max(touching.items(), key=lambda item: item[1])[0]
    area = float((component_labels == component).sum() * resolution**2)
    return component, area


def observations_from_frontiers(
    frontiers: Iterable,
    timestep: int,
    grid: np.ndarray,
    resolution: float,
) -> list[FrontierObservation]:
    """Build unassociated observations for one timestep."""
    labels, _ = unknown_components(grid)
    observations = []
    for index, frontier in enumerate(frontiers):
        component, area = describe_unknown_region(frontier, labels, resolution)
        observations.append(
            FrontierObservation(
                node_id=f"t{timestep}_f{index}",
                timestep=timestep,
                lineage_id=-1,  # assigned by the matcher
                boundary_cells=np.asarray(frontier.cells, dtype=np.int32),
                centroid=np.asarray(frontier.centroid_world, dtype=float)[[0, 2]],
                normal=np.asarray(frontier.orientation, dtype=float),
                unknown_component=component,
                unknown_area_m2=area,
                size_cells=int(frontier.size_cells),
                information_gain_m2=float(frontier.information_gain_m2),
            )
        )
    return observations
