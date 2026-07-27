"""Frontier-centred tensorisation tests.

The transform has to be exactly invertible: a predicted revelation is only
useful once it is back on the global map, and an alignment error there is
invisible in the loss but wrong on the map. These tests pin the round-trip, the
channel semantics, the validity masking and the absence of future leakage.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontierworld.data.tensors import (
    CH_BOUNDARY,
    CH_FREE,
    CH_OCCUPIED,
    CH_UNKNOWN,
    N_INPUT_CHANNELS,
    N_TARGET_CHANNELS,
    TGT_REVEALED_FREE,
    TGT_REVEALED_OCCUPIED,
    FrameSpec,
    FrontierFrame,
    build_branch_tensors,
    collate_tensor_groups,
    encode_option,
    rasterise_points,
    resample_grid,
)
from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN


def make_frame(normal=(1.0, 0.0), centroid=(0.0, 0.0)) -> FrontierFrame:
    return FrontierFrame(
        centroid_world=np.asarray(centroid, dtype=float),
        normal=np.asarray(normal, dtype=float),
        spec=FrameSpec(extent_m=8.0, resolution=0.1),
    )


# -- the frame -------------------------------------------------------------


def test_frame_size_from_extent_and_resolution():
    spec = FrameSpec(extent_m=8.0, resolution=0.1)
    assert spec.size == 80
    assert spec.half == 4.0


def test_centroid_maps_to_the_frame_origin():
    frame = make_frame(centroid=(3.0, -2.0))
    local = frame.to_local(np.array([[3.0, -2.0]]))
    assert np.allclose(local, 0.0)


def test_crossing_normal_is_the_local_plus_v_axis():
    """The whole point of the frame: 'into the unknown' is always +v."""
    for raw in [(1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (1.0, 1.0), (-0.3, 0.9)]:
        # Normalise the probe direction: the frame normalises its own axis, so
        # a non-unit input would move the point by less than the distance asked.
        direction = np.asarray(raw, dtype=float)
        direction = direction / np.linalg.norm(direction)
        frame = make_frame(normal=raw)
        ahead = frame.centroid_world + direction * 2.0
        local = frame.to_local(ahead[None, :])[0]
        assert local[1] == pytest.approx(2.0, abs=1e-9), raw
        assert local[0] == pytest.approx(0.0, abs=1e-9), raw


@pytest.mark.parametrize("normal", [(1.0, 0.0), (0.0, 1.0), (-0.6, 0.8)])
def test_world_local_roundtrip_is_exact(normal):
    frame = make_frame(normal=normal, centroid=(5.0, -3.0))
    points = np.random.default_rng(0).uniform(-10, 10, size=(200, 2))
    back = frame.to_global(frame.to_local(points))
    assert np.abs(back - points).max() < 1e-12


def test_cell_and_local_roundtrip_within_half_a_cell():
    frame = make_frame()
    cells = np.array([[0, 0], [40, 40], [79, 79]])
    back = frame.local_to_cell(frame.cell_to_local(cells))
    np.testing.assert_array_equal(back, cells)


def test_rotation_is_orthonormal():
    rotation = make_frame(normal=(0.6, 0.8)).rotation
    assert np.allclose(rotation @ rotation.T, np.eye(2), atol=1e-12)
    assert abs(abs(np.linalg.det(rotation)) - 1.0) < 1e-12


def test_in_window_rejects_cells_outside():
    frame = make_frame()
    cells = np.array([[0, 0], [79, 79], [-1, 5], [80, 5]])
    assert frame.in_window(cells).tolist() == [True, True, False, False]


# -- resampling ------------------------------------------------------------


def test_resample_marks_cells_outside_the_map_invalid():
    """Outside the global map there is no evidence, and 'no evidence' must not
    be silently recorded as 'observed unknown'."""
    grid = np.full((20, 20), FREE, dtype=np.uint8)
    frame = make_frame(centroid=(0.0, 0.0))
    values, valid = resample_grid(grid, np.array([0.0, 0.0]), 0.05, frame)

    assert valid.shape == (frame.spec.size, frame.spec.size)
    assert not valid.all(), "window extends past a 1 m map, so some cells are invalid"
    assert (values[~valid] == UNKNOWN).all()


def test_resample_preserves_values_inside_the_map():
    grid = np.full((400, 400), OCCUPIED, dtype=np.uint8)
    frame = make_frame(centroid=(10.0, 10.0))
    values, valid = resample_grid(grid, np.array([0.0, 0.0]), 0.05, frame)
    assert (values[valid] == OCCUPIED).all()


def test_rasterise_points_lands_at_the_frame_centre():
    frame = make_frame(centroid=(2.0, 2.0))
    canvas = rasterise_points(np.array([[2.0, 2.0]]), frame)
    assert canvas.sum() == 1.0
    row, col = np.argwhere(canvas > 0)[0]
    assert abs(int(row) - frame.spec.size // 2) <= 1
    assert abs(int(col) - frame.spec.size // 2) <= 1


# -- branch tensors --------------------------------------------------------


def synthetic_example_and_arrays():
    """A 20x20 m map, free on one side of a boundary and unknown beyond."""
    size = 400
    resolution = 0.05
    grid_before = np.full((size, size), UNKNOWN, dtype=np.uint8)
    grid_before[:200, :] = FREE
    grid_before[198:200, 100:140] = FREE

    grid_after = grid_before.copy()
    grid_after[200:240, 100:140] = FREE  # the branch revealed this
    grid_after[240:242, 100:140] = OCCUPIED

    revealed = np.zeros_like(grid_before, dtype=bool)
    revealed[200:242, 100:140] = True
    semantic = np.zeros_like(revealed)
    semantic[210:230, 110:130] = True

    boundary = np.array([[199, c] for c in range(100, 140)], dtype=np.int32)
    centroid_world = [100 * resolution + 1.0, 0.0, 199 * resolution]

    arrays = {
        "grid_before": grid_before,
        "map_origin": np.array([0.0, 0.0], dtype=np.float32),
        "resolution": np.float32(resolution),
        "grid_after_f0": grid_after,
        "revealed_mask_f0": np.packbits(revealed),
        "semantic_revealed_mask_f0": np.packbits(semantic),
    }
    example = {
        "frontier_geometry": {
            "frontier_id": 0,
            "centroid_world": centroid_world,
            "orientation": [0.0, 1.0],  # into increasing z, i.e. the unknown side
            "boundary_cells": boundary.tolist(),
            "information_gain_m2": 4.0,
            "boundary_length_m": 2.0,
            "geodesic_distance_m": 3.0,
            "yaw": 0.0,
        },
        "candidate_option": {
            "n_approach_actions": 10,
            "n_cross_actions": 12,
            "probe_distance_m": 2.0,
            "horizon": 12,
        },
        "future_revelation": {
            "target_became_visible": True,
            "crossed": True,
            "newly_observed_area_m2": 4.2,
            "collisions": 1,
            "n_new_frontiers": 2,
        },
        "observation_history": {"branch_start_position": [5.0, 0.0, 9.0], "trajectory": []},
        "target_arrays": {
            "revealed_mask": "revealed_mask_f0",
            "semantic_revealed_mask": "semantic_revealed_mask_f0",
            "grid_after": "grid_after_f0",
            "semantic_after": "semantic_after_f0",
            "packed_shape": [size, size],
        },
        "navigation_goal": "chair",
    }
    return example, arrays


def test_branch_tensors_have_the_declared_shapes():
    example, arrays = synthetic_example_and_arrays()
    spec = FrameSpec(extent_m=8.0, resolution=0.1)
    branch = build_branch_tensors(example, arrays, spec)

    assert branch.inputs.shape == (N_INPUT_CHANNELS, spec.size, spec.size)
    assert branch.targets.shape == (N_TARGET_CHANNELS, spec.size, spec.size)
    assert branch.target_valid.shape == (spec.size, spec.size)


def test_occupancy_channels_are_mutually_exclusive():
    example, arrays = synthetic_example_and_arrays()
    branch = build_branch_tensors(example, arrays)
    stack = (
        branch.inputs[CH_FREE] + branch.inputs[CH_OCCUPIED] + branch.inputs[CH_UNKNOWN]
    )
    assert np.all(stack <= 1.0 + 1e-6)
    assert np.all(stack >= 1.0 - 1e-6), "every cell must be exactly one of the three"


def test_boundary_channel_is_populated():
    example, arrays = synthetic_example_and_arrays()
    branch = build_branch_tensors(example, arrays)
    assert branch.inputs[CH_BOUNDARY].sum() > 0


def test_targets_lie_only_where_the_input_was_unknown():
    """The leakage test: nothing the branch revealed may already be observed."""
    example, arrays = synthetic_example_and_arrays()
    branch = build_branch_tensors(example, arrays)

    observed = (branch.inputs[CH_FREE] > 0) | (branch.inputs[CH_OCCUPIED] > 0)
    revealed = (branch.targets[TGT_REVEALED_FREE] > 0) | (
        branch.targets[TGT_REVEALED_OCCUPIED] > 0
    )
    assert int((observed & revealed).sum()) == 0


def test_revealed_free_and_occupied_are_distinguished():
    example, arrays = synthetic_example_and_arrays()
    branch = build_branch_tensors(example, arrays)
    assert branch.targets[TGT_REVEALED_FREE].sum() > 0
    assert branch.targets[TGT_REVEALED_OCCUPIED].sum() > 0
    overlap = (branch.targets[TGT_REVEALED_FREE] > 0) & (
        branch.targets[TGT_REVEALED_OCCUPIED] > 0
    )
    assert int(overlap.sum()) == 0


def test_scalar_labels_come_through():
    example, arrays = synthetic_example_and_arrays()
    branch = build_branch_tensors(example, arrays)
    assert branch.scalars["target_present"] == 1.0
    assert branch.scalars["crossing_success"] == 1.0
    assert branch.scalars["n_new_frontiers"] == 2.0


def test_revealed_region_appears_ahead_of_the_frontier():
    """In the frontier frame the revealed area must be on the +v side."""
    example, arrays = synthetic_example_and_arrays()
    spec = FrameSpec(extent_m=8.0, resolution=0.1)
    branch = build_branch_tensors(example, arrays, spec)

    revealed = branch.targets[TGT_REVEALED_FREE] > 0
    rows = np.argwhere(revealed)[:, 0]
    assert rows.size > 0
    # +v is increasing row index, so the mass must sit past the middle.
    assert rows.mean() > spec.size / 2


def test_option_encoding_is_finite_and_sized():
    example, arrays = synthetic_example_and_arrays()
    frame = make_frame()
    encoded = encode_option(example, frame)
    assert encoded.shape == (8,)
    assert np.all(np.isfinite(encoded))


# -- collation -------------------------------------------------------------


def fake_group(n_candidates: int, size: int = 80) -> dict:
    return {
        "group_id": f"g{n_candidates}",
        "goal": "chair",
        "inputs": np.ones((n_candidates, N_INPUT_CHANNELS, size, size), dtype=np.float32),
        "targets": np.ones((n_candidates, N_TARGET_CHANNELS, size, size), dtype=np.float32),
        "target_valid": np.ones((n_candidates, size, size), dtype=bool),
        "options": np.ones((n_candidates, 8), dtype=np.float32),
        "scalars": {"target_present": np.ones(n_candidates, dtype=np.float32)},
        "candidate_mask": np.ones(n_candidates, dtype=bool),
    }


def test_collate_pads_to_the_largest_candidate_count():
    batch = collate_tensor_groups([fake_group(2), fake_group(5), fake_group(3)])
    assert batch["inputs"].shape[:2] == (3, 5)
    assert batch["candidate_mask"].sum(axis=1).tolist() == [2, 5, 3]


def test_padded_candidates_are_zero_and_masked_out():
    batch = collate_tensor_groups([fake_group(2), fake_group(5)])
    padded = ~batch["candidate_mask"]
    assert padded.any()
    assert np.all(batch["inputs"][padded] == 0)
    assert np.all(batch["targets"][padded] == 0)
    assert not batch["target_valid"][padded].any()


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        collate_tensor_groups([])
