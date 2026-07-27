"""Offline oracle lineage.

The online matcher sees only the map as it stands. The oracle is allowed the
whole episode: each observation is labelled by the space actually reachable
through that specific opening in the FINAL map, and observations whose
footprints coincide are the same frontier -- whatever their boundaries happened
to look like at the time.

Kept strictly separate from the predicted lineage. The oracle is an upper bound
and an evaluation target, never an input -- it uses the future, so wiring it
into the online path would leak.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from frontierworld.mapping.occupancy import FREE, UNKNOWN


def eventual_regions(
    grid_initial: np.ndarray, grid_final: np.ndarray, min_cells: int = 20
) -> tuple[np.ndarray, int]:
    """Label the regions that were unknown at the start and known by the end.

    These are the "latent rooms" the agent had yet to resolve: exactly the
    things frontiers are boundaries *of*.
    """
    from scipy import ndimage

    resolved = (grid_initial == UNKNOWN) & (grid_final != UNKNOWN)
    labels, count = ndimage.label(
        resolved, structure=ndimage.generate_binary_structure(2, 2)
    )
    # Drop specks so a few stray cells do not become their own "room".
    for label_id in range(1, count + 1):
        if int((labels == label_id).sum()) < min_cells:
            labels[labels == label_id] = 0
    return labels, count


def revelation_footprint(
    observation,
    grid_final: np.ndarray,
    radius_cells: int = 50,
    max_cells: int = 20000,
) -> set:
    """Cells reachable through this specific opening, in the final map.

    Labelling an observation by the connected unresolved component it borders
    does not work: early in an episode the whole unexplored remainder of the
    house is ONE component, so every frontier gets the same identity and the
    ground truth chains unrelated frontiers together. Measured on a real
    episode, six of seven distinct frontiers collapsed onto a single id.

    The footprint is local instead: flood fill outward from just beyond the
    boundary, bounded by a radius, so two doorways into different parts of the
    same region get different footprints while one doorway keeps its footprint
    across time.
    """
    from collections import deque

    from frontierworld.mapping.occupancy import OCCUPIED

    height, width = grid_final.shape
    normal = np.asarray(observation.normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-9:
        normal = normal / norm

    seeds = []
    for row, col in observation.boundary_cells:
        probe_row = int(round(row + normal[1] * 2))
        probe_col = int(round(col + normal[0] * 2))
        if 0 <= probe_row < height and 0 <= probe_col < width:
            if grid_final[probe_row, probe_col] != OCCUPIED:
                seeds.append((probe_row, probe_col))
    if not seeds:
        return set()

    centre = observation.boundary_cells.mean(axis=0)
    visited: set = set(seeds)
    queue = deque(seeds)
    while queue and len(visited) < max_cells:
        row, col = queue.popleft()
        for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + delta_row, col + delta_col
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if (next_row, next_col) in visited:
                continue
            if grid_final[next_row, next_col] == OCCUPIED:
                continue
            if abs(next_row - centre[0]) > radius_cells or abs(next_col - centre[1]) > radius_cells:
                continue
            visited.add((next_row, next_col))
            queue.append((next_row, next_col))
    return visited


def footprint_iou(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def oracle_identity(
    observation, region_labels: np.ndarray, search_radius: int = 6
) -> int:
    """Which eventual region an observation was facing.

    Retained for diagnostics and for the region-level view; identity for
    evaluation comes from revelation footprints, see oracle_lineage.
    """
    height, width = region_labels.shape
    votes: dict[int, int] = defaultdict(int)

    normal = np.asarray(observation.normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm > 1e-9:
        normal = normal / norm

    for row, col in observation.boundary_cells:
        for step in range(1, search_radius + 1):
            probe_row = int(round(row + normal[1] * step))
            probe_col = int(round(col + normal[0] * step))
            if not (0 <= probe_row < height and 0 <= probe_col < width):
                break
            label = int(region_labels[probe_row, probe_col])
            if label > 0:
                votes[label] += 1
                break

    if not votes:
        return -1
    return max(votes.items(), key=lambda item: item[1])[0]


def _cluster_footprints(
    nodes: list, footprints: dict[str, set], iou_threshold: float
) -> dict[str, int]:
    """Union-find over footprint overlap; each cluster is one true identity."""
    parent: dict[str, str] = {n.node_id: n.node_id for n in nodes}

    def find(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    # Only link across timesteps: two frontiers visible simultaneously are
    # different frontiers by construction, however similar they look.
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            if a.timestep == b.timestep:
                continue
            if footprint_iou(footprints[a.node_id], footprints[b.node_id]) >= iou_threshold:
                union(a.node_id, b.node_id)

    roots: dict[str, int] = {}
    identities: dict[str, int] = {}
    for node in nodes:
        if not footprints[node.node_id]:
            identities[node.node_id] = -1
            continue
        root = find(node.node_id)
        if root not in roots:
            roots[root] = len(roots) + 1
        identities[node.node_id] = roots[root]
    return identities


def oracle_lineage(
    observations_by_step: dict[int, list],
    grid_initial: np.ndarray,
    grid_final: np.ndarray,
    iou_threshold: float = 0.4,
) -> tuple[dict[str, int], set[tuple[str, str]]]:
    """Ground-truth identities and parent->child links for a whole episode.

    Returns (node_id -> identity, links). Observations with no footprint get
    identity -1 and are excluded from the links, since there is no ground truth
    to hold the matcher to.

    Identity comes from clustering revelation footprints across the whole
    episode: two observations are the same frontier when the space reachable
    through them substantially coincides.
    """
    footprints: dict[str, set] = {}
    ordered_nodes: list = []
    for step in sorted(observations_by_step):
        for observation in observations_by_step[step]:
            footprints[observation.node_id] = revelation_footprint(
                observation, grid_final
            )
            ordered_nodes.append(observation)

    identities = _cluster_footprints(ordered_nodes, footprints, iou_threshold)

    by_identity: dict[int, list] = defaultdict(list)
    for step in sorted(observations_by_step):
        for observation in observations_by_step[step]:
            identity = identities[observation.node_id]
            if identity > 0:
                by_identity[identity].append(observation)

    links: set[tuple[str, str]] = set()
    for nodes in by_identity.values():
        ordered = sorted(nodes, key=lambda n: n.timestep)
        for previous, following in zip(ordered, ordered[1:]):
            links.add((previous.node_id, following.node_id))

    return identities, links


def long_absence_events(
    observations_by_step: dict[int, list], identities: dict[str, int], gap: int = 2
) -> list[dict]:
    """Identities that disappear for several steps and come back.

    Reappearance is the case nearest-centroid matching cannot handle at all --
    the parent is long gone from the active set -- so it is worth counting
    separately rather than folding into the overall switch rate.
    """
    appearances: dict[int, list[int]] = defaultdict(list)
    for step in sorted(observations_by_step):
        for observation in observations_by_step[step]:
            identity = identities.get(observation.node_id, -1)
            if identity > 0:
                appearances[identity].append(step)

    events = []
    for identity, steps in appearances.items():
        ordered = sorted(set(steps))
        for previous, following in zip(ordered, ordered[1:]):
            if following - previous > gap:
                events.append(
                    {
                        "identity": identity,
                        "last_seen": previous,
                        "reappeared": following,
                        "gap": following - previous,
                    }
                )
    return events
