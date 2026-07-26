"""Dataset, manifest and integrity tests.

Built on a synthetic dataset written to a temp directory, so the invariants the
branch-complete protocol depends on are checked without a simulator.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from frontierworld.data import integrity
from frontierworld.data.dataset import FrontierRevealDataset, collate_groups
from frontierworld.data.manifests import (
    SCHEMA_VERSION,
    SplitManifest,
    stable_partition,
)
from frontierworld.data.records import RevelationExample, RevelationWriter, array_hash


def make_example(
    frontier_id: int,
    scene: str = "sceneA",
    area: float = 5.0,
    semantic: float = 2.0,
    target: bool = False,
    start=(1.0, 2.0, 3.0),
) -> RevelationExample:
    return RevelationExample(
        scene_id=scene,
        episode_id="7",
        decision_timestep=42,
        navigation_goal="chair",
        frontier_geometry={"frontier_id": frontier_id},
        candidate_option={"frontier_id": frontier_id},
        future_revelation={
            "frontier_id": frontier_id,
            "newly_observed_area_m2": area,
            "newly_semantic_in_revealed_area_m2": semantic,
            "target_became_visible": target,
            "n_new_frontiers": 2,
            "collisions": 1,
            "distance_travelled_m": 4.0,
            "crossed": True,
            "geodesic_distance_to_target_m": float("inf"),
        },
        current_observation={"rgb": "rgb_before.png"},
        current_map={"arrays": "arrays.npz", "key": "grid_before"},
        observation_history={
            "branch_start_position": list(start),
            "target_already_visible": False,
        },
        privileged={"planner": "habitat_navmesh"},
        git_commit="abc123",
        config_hash="deadbeef",
    )


@pytest.fixture
def dataset_root(tmp_path):
    writer = RevelationWriter(tmp_path / "ds")
    for group_index, (scene, n) in enumerate([("sceneA", 3), ("sceneB", 2), ("sceneA", 4)]):
        examples = [
            make_example(i, scene=scene, area=5.0 + i, target=(i == 0))
            for i in range(n)
        ]
        writer.write_group(
            f"g{group_index}",
            examples,
            arrays={"grid_before": np.zeros((4, 4), dtype=np.uint8)},
        )
    writer.close()
    return tmp_path / "ds"


# -- dataloader ------------------------------------------------------------


def test_dataset_returns_whole_decision_groups(dataset_root):
    """The Phase 5 gate: all counterfactual branches of one common state."""
    dataset = FrontierRevealDataset(dataset_root)
    assert len(dataset) == 3

    group = dataset[0]
    assert group.n_branches == 3
    assert group.frontier_ids() == [0, 1, 2]
    assert len(group.revelations()) == 3


def test_groups_below_min_branches_are_dropped(tmp_path):
    writer = RevelationWriter(tmp_path / "ds")
    writer.write_group("solo", [make_example(0)], arrays={"a": np.zeros(1)})
    writer.write_group("pair", [make_example(0), make_example(1)], arrays={"a": np.zeros(1)})
    writer.close()

    dataset = FrontierRevealDataset(tmp_path / "ds", min_branches=2)
    assert len(dataset) == 1
    assert dataset[0].group_id == "pair"


def test_collate_pads_and_masks_variable_branch_counts(dataset_root):
    dataset = FrontierRevealDataset(dataset_root)
    batch = collate_groups([dataset[0], dataset[1], dataset[2]])

    assert batch["branch_mask"].shape == (3, 4)
    assert batch["branch_mask"][0].tolist() == [True, True, True, False]
    assert batch["branch_mask"][1].tolist() == [True, True, False, False]
    assert batch["n_branches"].tolist() == [3, 2, 4]


def test_padded_slots_carry_no_signal(dataset_root):
    dataset = FrontierRevealDataset(dataset_root)
    batch = collate_groups([dataset[1]])  # 2 real branches
    padded = ~batch["branch_mask"]
    assert np.all(batch["newly_observed_area_m2"][padded] == 0.0)
    assert np.all(batch["utility"][padded] == 0.0)


def test_collate_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        collate_groups([])


def test_oracle_index_picks_the_best_branch(dataset_root):
    dataset = FrontierRevealDataset(dataset_root)
    group = dataset[0]
    # branch 0 alone reveals the target, which dominates the default weights.
    assert group.oracle_index() == 0


def test_manifest_filters_by_scene(dataset_root):
    manifest = SplitManifest(
        name="t", splits={"train": ["sceneA"], "val": ["sceneB"]}, scenes={}
    )
    train = FrontierRevealDataset(dataset_root, split="train", manifest=manifest)
    val = FrontierRevealDataset(dataset_root, split="val", manifest=manifest)

    assert len(train) == 2
    assert len(val) == 1
    assert train.scenes() == {"sceneA"}


def test_statistics_ignore_non_finite_channels(dataset_root):
    """geodesic distance is inf when the target is unreachable; it must not
    silently turn every aggregate into nan."""
    stats = FrontierRevealDataset(dataset_root).statistics()
    assert stats["groups"] == 3
    assert stats["branches"] == 9
    assert np.isfinite(stats["channels"]["newly_observed_area_m2"]["mean"])
    assert "geodesic_distance_to_target_m" not in stats["channels"]


# -- manifests -------------------------------------------------------------


def test_stable_partition_is_scene_disjoint():
    scenes = [f"scene{i}" for i in range(60)]
    splits = stable_partition(scenes, {"train": 0.6, "val": 0.2, "test": 0.2})

    everything = [s for group in splits.values() for s in group]
    assert sorted(everything) == sorted(scenes)
    assert len(everything) == len(set(everything)), "a scene landed in two splits"


def test_stable_partition_is_deterministic():
    scenes = [f"scene{i}" for i in range(30)]
    fractions = {"train": 0.5, "val": 0.5}
    assert stable_partition(scenes, fractions) == stable_partition(scenes, fractions)


def test_adding_a_scene_does_not_reshuffle_existing_ones():
    """Hash-based assignment, so a growing dataset stays comparable."""
    fractions = {"train": 0.5, "val": 0.5}
    before = stable_partition([f"scene{i}" for i in range(20)], fractions)
    after = stable_partition([f"scene{i}" for i in range(21)], fractions)

    for split, scenes in before.items():
        assert set(scenes).issubset(set(after[split]))


def test_partition_rejects_fractions_that_do_not_sum_to_one():
    with pytest.raises(ValueError):
        stable_partition(["a", "b"], {"train": 0.5, "val": 0.2})


def test_manifest_detects_scene_overlap():
    manifest = SplitManifest(
        name="bad", splits={"train": ["a", "b"], "test": ["b", "c"]}, scenes={}
    )
    overlaps = manifest.check_disjoint()
    assert len(overlaps) == 1
    assert "b" in overlaps[0]


def test_manifest_roundtrip(tmp_path):
    manifest = SplitManifest(
        name="m", pilot=True, pilot_reason="why", splits={"train": ["a"]}, scenes={}
    )
    path = manifest.save(tmp_path / "m.json")
    loaded = SplitManifest.load(path)
    assert loaded.name == "m"
    assert loaded.pilot is True
    assert loaded.schema_version == SCHEMA_VERSION


# -- integrity -------------------------------------------------------------


def test_integrity_all_pass_on_a_clean_dataset(dataset_root):
    dataset = FrontierRevealDataset(dataset_root)
    results = integrity.run_all(dataset)
    failed = [r.name for r in results if not r.passed]
    assert failed == [], failed


def test_divergent_start_states_are_caught(tmp_path):
    writer = RevelationWriter(tmp_path / "ds")
    writer.write_group(
        "bad",
        [make_example(0, start=(1.0, 2.0, 3.0)), make_example(1, start=(9.0, 9.0, 9.0))],
        arrays={"a": np.zeros(1)},
    )
    writer.close()

    result = integrity.check_common_start_state(FrontierRevealDataset(tmp_path / "ds"))
    assert not result.passed


def test_semantic_exceeding_revealed_area_is_caught(tmp_path):
    """The Phase 4 defect, as a permanent guard."""
    writer = RevelationWriter(tmp_path / "ds")
    writer.write_group(
        "bad",
        [make_example(0, area=2.0, semantic=9.0), make_example(1)],
        arrays={"a": np.zeros(1)},
    )
    writer.close()

    result = integrity.check_semantic_within_revealed(
        FrontierRevealDataset(tmp_path / "ds")
    )
    assert not result.passed
    assert "semantic 9.00 > revealed 2.00" in result.failures[0]


def test_duplicate_frontier_ids_are_caught(tmp_path):
    writer = RevelationWriter(tmp_path / "ds")
    writer.write_group(
        "bad", [make_example(0), make_example(0)], arrays={"a": np.zeros(1)}
    )
    writer.close()

    result = integrity.check_one_outcome_per_frontier(
        FrontierRevealDataset(tmp_path / "ds")
    )
    assert not result.passed


def test_missing_provenance_is_caught(tmp_path):
    example = make_example(0)
    example.git_commit = None
    writer = RevelationWriter(tmp_path / "ds")
    writer.write_group("bad", [example, make_example(1)], arrays={"a": np.zeros(1)})
    writer.close()

    result = integrity.check_provenance(FrontierRevealDataset(tmp_path / "ds"))
    assert not result.passed


def test_already_visible_target_is_caught(tmp_path):
    example = make_example(0)
    example.observation_history["target_already_visible"] = True
    writer = RevelationWriter(tmp_path / "ds")
    writer.write_group("bad", [example, make_example(1)], arrays={"a": np.zeros(1)})
    writer.close()

    result = integrity.check_target_not_already_visible(
        FrontierRevealDataset(tmp_path / "ds")
    )
    assert not result.passed


# -- hashing ---------------------------------------------------------------


def test_array_hash_is_stable_and_sensitive():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert array_hash(a) == array_hash(a.copy())

    b = a.copy()
    b[0, 0] += 1e-3
    assert array_hash(a) != array_hash(b)


def test_array_hash_distinguishes_shape_and_dtype():
    assert array_hash(np.zeros((2, 6))) != array_hash(np.zeros((3, 4)))
    assert array_hash(np.zeros(4, dtype=np.float32)) != array_hash(
        np.zeros(4, dtype=np.float64)
    )


def test_written_group_is_readable(dataset_root):
    payload = json.loads((dataset_root / "groups" / "g0" / "group.json").read_text())
    assert payload["n_examples"] == 3
    assert payload["examples"][0]["detector_type"] == "geometric"
    assert payload["examples"][0]["schema_version"] == SCHEMA_VERSION
