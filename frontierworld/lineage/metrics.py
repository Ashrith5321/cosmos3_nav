"""Lineage association metrics.

Borrowed from multi-object tracking, because the problem is the same: keep a
consistent identity for a thing that moves, splits and vanishes. IDF1 and ID
switches are the standard measures, and cache contamination is the one this
project adds -- it is what actually breaks Phase 11 memory.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LineageMetrics:
    association_precision: float = 0.0
    association_recall: float = 0.0
    association_f1: float = 0.0
    idf1: float = 0.0
    id_switches: int = 0
    n_predicted_links: int = 0
    n_true_links: int = 0
    split_precision: float = float("nan")
    split_recall: float = float("nan")
    merge_precision: float = float("nan")
    merge_recall: float = float("nan")
    cache_contamination_rate: float = 0.0
    per_event: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def __str__(self) -> str:
        return (
            f"assoc P/R/F1 {self.association_precision:.3f}/"
            f"{self.association_recall:.3f}/{self.association_f1:.3f}  "
            f"IDF1 {self.idf1:.3f}  switches {self.id_switches}  "
            f"contamination {self.cache_contamination_rate:.3f}"
        )


def association_scores(
    predicted_links: set[tuple[str, str]], true_links: set[tuple[str, str]]
) -> tuple[float, float, float]:
    """Precision, recall and F1 over parent->child links."""
    if not predicted_links and not true_links:
        return 1.0, 1.0, 1.0
    true_positive = len(predicted_links & true_links)
    precision = true_positive / len(predicted_links) if predicted_links else 0.0
    recall = true_positive / len(true_links) if true_links else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def count_id_switches(
    predicted: dict[str, int], truth: dict[str, int], order: list[str]
) -> int:
    """Times a true identity changes predicted id, walking nodes in time order.

    A switch is charged when a stable ground-truth object is handed a different
    predicted id than it had last time it was seen -- the failure that makes
    accumulated memory belong to the wrong frontier.
    """
    last_predicted: dict[int, int] = {}
    switches = 0
    for node_id in order:
        if node_id not in predicted or node_id not in truth:
            continue
        true_id, predicted_id = truth[node_id], predicted[node_id]
        if true_id in last_predicted and last_predicted[true_id] != predicted_id:
            switches += 1
        last_predicted[true_id] = predicted_id
    return switches


def compute_idf1(predicted: dict[str, int], truth: dict[str, int]) -> float:
    """IDF1 over identity assignments, via optimal id-to-id matching."""
    from scipy.optimize import linear_sum_assignment

    shared = [n for n in predicted if n in truth]
    if not shared:
        return 0.0

    predicted_ids = sorted({predicted[n] for n in shared})
    true_ids = sorted({truth[n] for n in shared})
    overlap = np.zeros((len(true_ids), len(predicted_ids)), dtype=np.int64)
    true_index = {v: i for i, v in enumerate(true_ids)}
    predicted_index = {v: i for i, v in enumerate(predicted_ids)}
    for node in shared:
        overlap[true_index[truth[node]], predicted_index[predicted[node]]] += 1

    rows, cols = linear_sum_assignment(-overlap)
    matched = int(overlap[rows, cols].sum())
    denominator = len(shared) + len(shared)
    return float(2 * matched / denominator) if denominator else 0.0


def cache_contamination(
    predicted: dict[str, int], truth: dict[str, int]
) -> tuple[float, dict[int, set[int]]]:
    """Fraction of predicted lineages that pool more than one true identity.

    This is the number that matters for Phase 11: a lineage mixing two real
    frontiers accumulates memory from both, and the memory is then wrong for
    whichever one it is asked about.
    """
    members: dict[int, set[int]] = defaultdict(set)
    for node_id, predicted_id in predicted.items():
        if node_id in truth:
            members[predicted_id].add(truth[node_id])
    if not members:
        return 0.0, {}
    contaminated = sum(1 for ids in members.values() if len(ids) > 1)
    return contaminated / len(members), {k: v for k, v in members.items() if len(v) > 1}


def evaluate_lineage(
    graph,
    truth: dict[str, int],
    true_links: set[tuple[str, str]] | None = None,
) -> LineageMetrics:
    """Score a predicted lineage graph against ground-truth identities.

    `truth` maps node_id -> ground-truth identity. `true_links` is the set of
    ground-truth parent->child pairs; when omitted it is derived from `truth`
    by linking consecutive observations of the same identity.
    """
    predicted = {n.node_id: n.lineage_id for n in graph.nodes.values()}

    if true_links is None:
        by_identity: dict[int, list] = defaultdict(list)
        for node_id, identity in truth.items():
            node = graph.nodes.get(node_id)
            if node is not None:
                by_identity[identity].append(node)
        true_links = set()
        for nodes in by_identity.values():
            ordered = sorted(nodes, key=lambda n: n.timestep)
            for previous, following in zip(ordered, ordered[1:]):
                true_links.add((previous.node_id, following.node_id))

    predicted_links = {
        (e.parent, e.child)
        for e in graph.edges
        if e.event in {"update", "split", "merge"} and e.parent and e.child
    }

    precision, recall, f1 = association_scores(predicted_links, true_links)
    order = [
        n.node_id for n in sorted(graph.nodes.values(), key=lambda n: n.timestep)
    ]
    contamination, _ = cache_contamination(predicted, truth)

    return LineageMetrics(
        association_precision=precision,
        association_recall=recall,
        association_f1=f1,
        idf1=compute_idf1(predicted, truth),
        id_switches=count_id_switches(predicted, truth, order),
        n_predicted_links=len(predicted_links),
        n_true_links=len(true_links),
        cache_contamination_rate=contamination,
        per_event=dict(graph.event_counts),
    )
