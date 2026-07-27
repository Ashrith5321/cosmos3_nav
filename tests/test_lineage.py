"""Lineage graph tests.

Built on synthetic frontier sequences whose correct identities are known by
construction. This is the direct test of the Phase 6 gate -- "a frontier that
persists, splits and later disappears maintains correct parent-child identities
without leaking branch information" -- and it does not depend on the offline
oracle, which is itself an approximation.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontierworld.lineage import (
    BIRTH,
    MERGE,
    RETIRE,
    RETIRED,
    SPLIT,
    UPDATE,
    FrontierObservation,
    LineageGraph,
    MatchingConfig,
    association_matrix,
    update_lineage,
)
from frontierworld.lineage.matching import (
    boundary_overlap,
    centroid_score,
    orientation_score,
)
from frontierworld.lineage.metrics import (
    cache_contamination,
    compute_idf1,
    count_id_switches,
)


def observation(
    node_id: str,
    timestep: int,
    row: int,
    col: int,
    length: int = 10,
    normal=(0.0, -1.0),
    horizontal: bool = True,
) -> FrontierObservation:
    """A straight boundary segment of `length` cells at (row, col)."""
    if horizontal:
        cells = np.array([[row, col + i] for i in range(length)], dtype=np.int32)
    else:
        cells = np.array([[row + i, col] for i in range(length)], dtype=np.int32)
    return FrontierObservation(
        node_id=node_id,
        timestep=timestep,
        lineage_id=-1,
        boundary_cells=cells,
        centroid=np.array([col * 0.05, row * 0.05]),
        normal=np.array(normal, dtype=float),
        unknown_component=1,
        unknown_area_m2=10.0,
        size_cells=length,
    )


# -- similarity cues -------------------------------------------------------


def test_identical_boundaries_overlap_fully():
    a = observation("a", 0, 100, 100)
    b = observation("b", 1, 100, 100)
    assert boundary_overlap(a, b, dilation=1) == pytest.approx(1.0)


def test_shifted_boundary_still_overlaps():
    """A boundary drifting a cell or two is the same opening."""
    a = observation("a", 0, 100, 100)
    b = observation("b", 1, 101, 101)
    assert boundary_overlap(a, b, dilation=3) > 0.4


def test_distant_boundaries_do_not_overlap():
    a = observation("a", 0, 100, 100)
    b = observation("b", 1, 300, 300)
    assert boundary_overlap(a, b, dilation=3) == 0.0


def test_centroid_score_decays_with_distance():
    a = observation("a", 0, 100, 100)
    near = observation("b", 1, 100, 105)
    far = observation("c", 1, 100, 140)
    assert centroid_score(a, near, 2.0) > centroid_score(a, far, 2.0)


def test_opposing_normals_score_low():
    a = observation("a", 0, 100, 100, normal=(0.0, -1.0))
    same = observation("b", 1, 100, 100, normal=(0.0, -1.0))
    opposed = observation("c", 1, 100, 100, normal=(0.0, 1.0))
    assert orientation_score(a, same) == pytest.approx(1.0)
    assert orientation_score(a, opposed) == pytest.approx(0.0)


def test_association_matrix_ignores_far_pairs():
    previous = [observation("a", 0, 100, 100)]
    current = [observation("b", 1, 700, 700)]
    matrix = association_matrix(previous, current, MatchingConfig())
    assert matrix[0, 0] == 0.0


# -- the gate: persist, split, disappear -----------------------------------


def test_frontier_persisting_keeps_one_identity():
    graph = LineageGraph()
    config = MatchingConfig()
    identities = []
    for step in range(5):
        obs = [observation(f"t{step}", step, 100 + step, 100, length=10)]
        update_lineage(graph, obs, step, config)
        identities.append(obs[0].lineage_id)

    assert len(set(identities)) == 1, f"identity changed: {identities}"
    assert graph.event_counts[BIRTH] == 1
    assert graph.event_counts[UPDATE] == 4


def test_split_records_parent_and_new_child():
    graph = LineageGraph()
    config = MatchingConfig()

    parent = [observation("t0", 0, 100, 100, length=20)]
    update_lineage(graph, parent, 0, config)
    parent_lineage = parent[0].lineage_id

    # The wide boundary becomes two narrower ones at the same place.
    children = [
        observation("t1_a", 1, 100, 100, length=8),
        observation("t1_b", 1, 100, 112, length=8),
    ]
    update_lineage(graph, children, 1, config)

    assert graph.event_counts[SPLIT] >= 1, dict(graph.event_counts)
    # One child continues the parent's identity, the other is a new lineage.
    lineages = {c.lineage_id for c in children}
    assert parent_lineage in lineages
    assert len(lineages) == 2, "a split must not collapse both children into one id"

    parents_of_new = [
        graph.parents_of(c.node_id) for c in children if c.lineage_id != parent_lineage
    ]
    assert parents_of_new and parents_of_new[0] == ["t0"]


def test_merge_links_both_parents_to_one_child():
    graph = LineageGraph()
    config = MatchingConfig()

    parents = [
        observation("t0_a", 0, 100, 100, length=8),
        observation("t0_b", 0, 100, 112, length=8),
    ]
    update_lineage(graph, parents, 0, config)

    child = [observation("t1", 1, 100, 100, length=20)]
    update_lineage(graph, child, 1, config)

    assert graph.event_counts[MERGE] >= 1, dict(graph.event_counts)
    assert len(graph.parents_of("t1")) == 2


def test_disappearing_frontier_is_retired_not_reassigned():
    """The failure that matters: a vanished frontier's identity must not be
    silently handed to an unrelated new boundary elsewhere."""
    graph = LineageGraph()
    config = MatchingConfig()

    first = [observation("t0", 0, 100, 100)]
    update_lineage(graph, first, 0, config)
    retired_lineage = first[0].lineage_id

    # A completely different boundary appears far away.
    second = [observation("t1", 1, 600, 600)]
    update_lineage(graph, second, 1, config)

    assert graph.event_counts[RETIRE] == 1
    assert graph.nodes["t0"].status == RETIRED
    assert second[0].lineage_id != retired_lineage
    assert graph.event_counts[BIRTH] == 2


def test_full_gate_persist_split_disappear():
    """The Phase 6 gate as one sequence."""
    graph = LineageGraph()
    config = MatchingConfig()

    for step in range(3):  # persist
        update_lineage(graph, [observation(f"p{step}", step, 100, 100, length=20)], step, config)
    persisted = graph.nodes["p2"].lineage_id

    children = [  # split
        observation("s_a", 3, 100, 100, length=8),
        observation("s_b", 3, 100, 112, length=8),
    ]
    update_lineage(graph, children, 3, config)

    update_lineage(graph, [], 4, config)  # both disappear

    assert graph.nodes["p0"].lineage_id == persisted
    assert persisted in {c.lineage_id for c in children}
    assert graph.event_counts[SPLIT] >= 1
    assert all(graph.nodes[c.node_id].status == RETIRED for c in children)
    assert graph.active == []


def test_simultaneous_frontiers_never_share_an_identity():
    graph = LineageGraph()
    config = MatchingConfig()
    obs = [
        observation("a", 0, 100, 100),
        observation("b", 0, 400, 400),
        observation("c", 0, 700, 700),
    ]
    update_lineage(graph, obs, 0, config)
    assert len({o.lineage_id for o in obs}) == 3


# -- branch isolation ------------------------------------------------------


def test_graph_copy_is_independent():
    graph = LineageGraph()
    update_lineage(graph, [observation("t0", 0, 100, 100)], 0, MatchingConfig())

    clone = graph.copy()
    clone.nodes["t0"].status = RETIRED
    clone.add_edge("t0", "x", UPDATE)

    assert graph.nodes["t0"].status != RETIRED
    assert len(graph.edges) != len(clone.edges)


def test_fingerprint_detects_any_change():
    graph = LineageGraph()
    update_lineage(graph, [observation("t0", 0, 100, 100)], 0, MatchingConfig())
    before = graph.fingerprint()

    assert graph.copy().fingerprint() == before
    update_lineage(graph, [observation("t1", 1, 100, 100)], 1, MatchingConfig())
    assert graph.fingerprint() != before


def test_branch_updates_do_not_leak_into_the_parent_graph():
    """A counterfactual branch must leave the lineage exactly as it was."""
    graph = LineageGraph()
    update_lineage(graph, [observation("t0", 0, 100, 100)], 0, MatchingConfig())
    before = graph.fingerprint()

    for branch in range(3):
        working = graph.copy()
        update_lineage(
            working, [observation(f"b{branch}", 1, 100 + branch * 50, 100)], 1,
            MatchingConfig(),
        )
        assert working.fingerprint() != before  # the branch really did change it

    assert graph.fingerprint() == before


# -- metrics ---------------------------------------------------------------


def test_id_switches_counted_when_identity_changes():
    truth = {"a": 1, "b": 1, "c": 1}
    stable = {"a": 10, "b": 10, "c": 10}
    switching = {"a": 10, "b": 11, "c": 12}

    assert count_id_switches(stable, truth, ["a", "b", "c"]) == 0
    assert count_id_switches(switching, truth, ["a", "b", "c"]) == 2


def test_perfect_assignment_has_idf1_of_one():
    truth = {"a": 1, "b": 1, "c": 2}
    predicted = {"a": 7, "b": 7, "c": 9}
    assert compute_idf1(predicted, truth) == pytest.approx(1.0)


def test_contamination_flags_lineages_pooling_two_identities():
    truth = {"a": 1, "b": 2}
    clean = {"a": 10, "b": 11}
    merged = {"a": 10, "b": 10}

    assert cache_contamination(clean, truth)[0] == 0.0
    assert cache_contamination(merged, truth)[0] == 1.0
