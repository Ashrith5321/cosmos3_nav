"""Tests for nominal option pose derivation.

These are pure geometry and run without a GPU or the Cosmos checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontierworld.models.nominal_poses import (
    ACTION_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    Kinematics,
    agent_to_camera,
    heading_vector,
    integrate_nominal,
    nominal_camera_poses,
    quaternion_to_yaw,
    recover_start_pose,
    subsample_indices,
    yaw_to_matrix,
)

KIN = Kinematics(forward_step_size=0.25, turn_angle_deg=30.0)


def yaw_quaternion(yaw: float) -> list[float]:
    """Habitat (w, x, y, z) for a yaw-only rotation about +y."""
    return [float(np.cos(yaw / 2)), 0.0, float(np.sin(yaw / 2)), 0.0]


@pytest.mark.parametrize("yaw", [0.0, 0.4, -1.2, 2.9, -3.0])
def test_quaternion_yaw_roundtrip(yaw):
    assert quaternion_to_yaw(yaw_quaternion(yaw)) == pytest.approx(yaw, abs=1e-9)


def test_heading_is_negative_z_at_zero_yaw():
    """Habitat agents look down their local -z; a sign error here would drive
    every generated rollout backwards."""
    assert heading_vector(0.0) == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_positive_yaw_turns_left():
    """+90 degrees of yaw must point the agent along -x, not +x."""
    assert heading_vector(np.pi / 2) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-12)


def test_forward_advances_by_step_size():
    poses = integrate_nominal([ACTION_FORWARD] * 4, np.zeros(3), 0.0, KIN)
    assert poses.shape == (5, 4, 4)
    assert poses[-1][:3, 3] == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def test_turn_does_not_translate():
    poses = integrate_nominal([ACTION_TURN_LEFT] * 3, np.array([1.0, 2.0, 3.0]), 0.0, KIN)
    for pose in poses:
        assert pose[:3, 3] == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)


def test_turn_left_then_right_returns_to_start_heading():
    poses = integrate_nominal([ACTION_TURN_LEFT, ACTION_TURN_RIGHT], np.zeros(3), 0.7, KIN)
    assert poses[-1][:3, :3] == pytest.approx(yaw_to_matrix(0.7), abs=1e-12)


def test_height_is_held_constant():
    """The discrete action set cannot change height; drift would tilt the camera."""
    actions = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_FORWARD, ACTION_TURN_RIGHT]
    poses = integrate_nominal(actions, np.array([0.0, 1.31, 0.0]), 0.0, KIN)
    assert np.allclose(poses[:, 1, 3], 1.31)


def test_turning_then_forward_moves_along_new_heading():
    """Composition order matters: turning must affect the *subsequent* step."""
    poses = integrate_nominal([ACTION_TURN_LEFT, ACTION_FORWARD], np.zeros(3), 0.0, KIN)
    expected = heading_vector(np.deg2rad(30.0)) * 0.25
    assert poses[-1][:3, 3] == pytest.approx(expected, abs=1e-12)


def test_recover_start_pose_undoes_a_left_turn():
    trajectory = [{"action": ACTION_TURN_LEFT, "rotation": yaw_quaternion(np.deg2rad(30.0))}]
    _, yaw = recover_start_pose(trajectory, [0.0, 0.0, 0.0], KIN)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_recover_start_pose_undoes_a_right_turn():
    trajectory = [{"action": ACTION_TURN_RIGHT, "rotation": yaw_quaternion(np.deg2rad(-30.0))}]
    _, yaw = recover_start_pose(trajectory, [0.0, 0.0, 0.0], KIN)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_recover_start_pose_leaves_heading_for_forward():
    trajectory = [{"action": ACTION_FORWARD, "rotation": yaw_quaternion(0.9)}]
    _, yaw = recover_start_pose(trajectory, [1.0, 0.0, 2.0], KIN)
    assert yaw == pytest.approx(0.9, abs=1e-9)


def test_recovered_start_pose_reproduces_the_executed_first_step():
    """Round-trip: recovering the pre-action pose and replaying the action must
    land on the recorded post-action rotation."""
    executed_yaw = np.deg2rad(-30.0) + 0.4
    trajectory = [{"action": ACTION_TURN_RIGHT, "rotation": yaw_quaternion(executed_yaw)}]
    _, yaw = recover_start_pose(trajectory, [0.0, 0.0, 0.0], KIN)
    poses = integrate_nominal([ACTION_TURN_RIGHT], np.zeros(3), yaw, KIN)
    assert poses[-1][:3, :3] == pytest.approx(yaw_to_matrix(executed_yaw), abs=1e-9)


def test_camera_conversion_flips_y_and_z():
    """OpenCV cameras look down +z with y down; Habitat looks down -z with y up."""
    poses = integrate_nominal([], np.array([1.0, 2.0, 3.0]), 0.0, KIN)
    camera = agent_to_camera(poses, KIN)
    # Camera +z must equal the Habitat forward direction.
    assert camera[0][:3, 2] == pytest.approx(heading_vector(0.0), abs=1e-12)
    # Camera +y must point down, i.e. against world +y.
    assert camera[0][:3, 1] == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)
    # Translation is untouched by an axis flip.
    assert camera[0][:3, 3] == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)


def test_camera_conversion_preserves_rotation_validity():
    poses = integrate_nominal([ACTION_TURN_LEFT, ACTION_FORWARD], np.zeros(3), 1.1, KIN)
    camera = agent_to_camera(poses, KIN)
    for pose in camera:
        rotation = pose[:3, :3]
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_subsample_spans_endpoints():
    indices = subsample_indices(67, 17)
    assert indices[0] == 0
    assert indices[-1] == 66
    assert len(indices) == 17
    assert np.all(np.diff(indices) > 0)


def test_subsample_rejects_too_short_an_option():
    """Fewer nominal poses than requested frames must fail loudly rather than
    silently return a short action sequence."""
    with pytest.raises(ValueError):
        nominal_camera_poses(
            [{"action": ACTION_FORWARD, "rotation": yaw_quaternion(0.0)}],
            [0.0, 0.0, 0.0],
            n_frames=17,
        )


def test_nominal_camera_poses_shape_and_start():
    trajectory = [
        {"action": ACTION_FORWARD if i % 3 else ACTION_TURN_LEFT, "rotation": yaw_quaternion(0.0)}
        for i in range(66)
    ]
    poses, indices = nominal_camera_poses(trajectory, [0.5, 1.31, -2.0], n_frames=17)
    assert poses.shape == (17, 4, 4)
    assert indices[0] == 0 and indices[-1] == 66
    assert poses[0][:3, 3] == pytest.approx([0.5, 1.31, -2.0], abs=1e-12)


def test_nominal_ignores_executed_positions():
    """The executed positions in the trajectory must not influence the result --
    only the action ids do. Corrupting the recorded positions (as a collision
    would) must leave the nominal poses unchanged."""
    actions = [ACTION_FORWARD, ACTION_TURN_LEFT, ACTION_FORWARD] * 22
    clean = [{"action": a, "rotation": yaw_quaternion(0.0), "position": [0.0, 0.0, 0.0]} for a in actions]
    blocked = [
        {"action": a, "rotation": yaw_quaternion(0.0), "position": [99.0, 99.0, 99.0]}
        for a in actions
    ]
    a_poses, _ = nominal_camera_poses(clean, [0.0, 0.0, 0.0], n_frames=17)
    b_poses, _ = nominal_camera_poses(blocked, [0.0, 0.0, 0.0], n_frames=17)
    assert np.allclose(a_poses, b_poses)


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        integrate_nominal([7], np.zeros(3), 0.0, KIN)
