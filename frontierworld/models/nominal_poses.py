"""Nominal option poses: what the option *commands*, not what the simulator did.

Conditioning on the executed trajectory would leak the future. The executed
poses already encode which forward steps were blocked by geometry the agent had
not yet observed, so a model conditioned on them is being told part of the
answer. At deployment there is no executed trajectory -- there is only an option
the planner has decided to attempt.

So the action fed to Cosmos is the kinematic ideal:

    FORWARD      translate +forward_step_size along the current heading
    TURN_LEFT    yaw += turn_angle
    TURN_RIGHT   yaw -= turn_angle

integrated from the pose at the decision point. Collisions, sliding and
navmesh snapping are all deliberately absent. Where the option is blocked, the
nominal and executed trajectories diverge -- that divergence is a property of
the world the model is being asked to predict, not an input it may see.

Coordinates: Habitat is y-up with the agent looking down its local -z. Cosmos
`camera_pose` actions are camera-to-world transforms in the OpenCV convention
(x right, y down, z forward), in metres. The conversion is a fixed 180-degree
flip about x, applied on the right:

    T_cam2world = T_agent2world @ diag(1, -1, -1, 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Habitat discrete ObjectNav action ids.
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3

# OpenCV camera axes from Habitat agent axes: flip y and z.
_HABITAT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class Kinematics:
    """Motion primitives, taken from the generation config -- never guessed."""

    forward_step_size: float = 0.25
    turn_angle_deg: float = 30.0
    sensor_height: float = 0.88


def quaternion_to_yaw(rotation: list[float] | np.ndarray) -> float:
    """Yaw about the world +y axis, from a Habitat (w, x, y, z) quaternion.

    Habitat agent rotations are yaw-only, so the full quaternion carries one
    degree of freedom; extracting it directly avoids depending on a rotation
    library and keeps the inverse below exact.
    """
    w, x, y, z = (float(v) for v in rotation)
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def yaw_to_matrix(yaw: float) -> np.ndarray:
    """Rotation about +y by `yaw`."""
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float64
    )


def heading_vector(yaw: float) -> np.ndarray:
    """Unit forward direction. The agent looks down its local -z."""
    return yaw_to_matrix(yaw) @ np.array([0.0, 0.0, -1.0])


def recover_start_pose(
    trajectory: list[dict], branch_start_position: list[float], kinematics: Kinematics
) -> tuple[np.ndarray, float]:
    """Pose at the decision point, i.e. immediately *before* the first action.

    The dataset records each trajectory entry as the state *after* its action,
    so the decision-point pose is obtained by undoing the first action rather
    than by reading a stored field that does not exist.
    """
    if not trajectory:
        raise ValueError("empty trajectory; cannot recover the decision-point pose")

    first = trajectory[0]
    yaw_after = quaternion_to_yaw(first["rotation"])
    turn = np.deg2rad(kinematics.turn_angle_deg)

    action = int(first["action"])
    if action == ACTION_TURN_LEFT:
        yaw_before = yaw_after - turn
    elif action == ACTION_TURN_RIGHT:
        yaw_before = yaw_after + turn
    else:
        # FORWARD (or STOP) leaves the heading unchanged.
        yaw_before = yaw_after

    return np.asarray(branch_start_position, dtype=np.float64), float(yaw_before)


def integrate_nominal(
    actions: list[int],
    start_position: np.ndarray,
    start_yaw: float,
    kinematics: Kinematics,
) -> np.ndarray:
    """Kinematically ideal agent poses: `(len(actions) + 1, 4, 4)`, agent-to-world.

    Height is held at the start height: the discrete action set cannot change
    it, and letting it drift would silently tilt the generated camera.
    """
    position = np.asarray(start_position, dtype=np.float64).copy()
    yaw = float(start_yaw)
    turn = np.deg2rad(kinematics.turn_angle_deg)

    poses = []

    def emit() -> None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = yaw_to_matrix(yaw)
        pose[:3, 3] = position
        poses.append(pose)

    emit()
    for raw in actions:
        action = int(raw)
        if action == ACTION_FORWARD:
            position = position + heading_vector(yaw) * kinematics.forward_step_size
        elif action == ACTION_TURN_LEFT:
            yaw += turn
        elif action == ACTION_TURN_RIGHT:
            yaw -= turn
        elif action == ACTION_STOP:
            pass
        else:
            raise ValueError(f"unknown action id {action}")
        emit()

    return np.stack(poses)


def agent_to_camera(poses_agent: np.ndarray, kinematics: Kinematics) -> np.ndarray:
    """Agent-to-world poses -> camera-to-world poses in the OpenCV convention.

    The camera sits `sensor_height` above the agent origin. Habitat's agent
    origin is already at the sensor height in the recorded trajectories (the
    positions are sensor positions), so the offset is not applied twice here;
    it is accepted as a parameter only to make that assumption explicit.
    """
    return np.einsum("nij,jk->nik", np.asarray(poses_agent, dtype=np.float64), _HABITAT_TO_OPENCV)


def subsample_indices(n_poses: int, n_frames: int) -> np.ndarray:
    """Uniform frame indices spanning the whole option, endpoints included.

    A 17-frame rollout over a 66-action option cannot show every primitive
    action, so each generated frame advances by the net transform of several.
    Spanning the whole option (rather than truncating it) is what keeps the
    generated video comparable to a revelation computed over the whole option.
    """
    if n_frames < 2:
        raise ValueError("need at least two frames")
    if n_poses < 2:
        raise ValueError("need at least two poses")
    return np.unique(np.round(np.linspace(0, n_poses - 1, n_frames)).astype(int))


def nominal_camera_poses(
    trajectory: list[dict],
    branch_start_position: list[float],
    n_frames: int,
    kinematics: Kinematics | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end: dataset trajectory -> `(n_frames, 4, 4)` camera-to-world poses.

    Returns the poses and the indices into the nominal pose sequence they were
    sampled from, so the mapping from generated frame to option progress stays
    auditable.
    """
    kinematics = kinematics or Kinematics()
    actions = [int(step["action"]) for step in trajectory]
    start_position, start_yaw = recover_start_pose(
        trajectory, branch_start_position, kinematics
    )
    poses_agent = integrate_nominal(actions, start_position, start_yaw, kinematics)
    indices = subsample_indices(len(poses_agent), n_frames)
    if len(indices) != n_frames:
        raise ValueError(
            f"option has {len(poses_agent)} nominal poses, too few to sample "
            f"{n_frames} distinct frames"
        )
    return agent_to_camera(poses_agent[indices], kinematics), indices


def to_action_vectors(poses_camera: np.ndarray) -> np.ndarray:
    """`(T, 4, 4)` camera-to-world -> `(T-1, 9)` relative-pose actions.

    Delegates to the Cosmos framework's own converter. The layout is
    `[translation(3), rot6d(6)]` under the `backward_framewise` convention
    (`delta_T = T_i^-1 @ T_i+1`). Using the framework's utility rather than a
    reimplementation is deliberate: a sign or handedness mismatch at this
    boundary would produce video that looks plausible and is wrong.
    """
    from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel

    return pose_abs_to_rel(
        np.asarray(poses_camera, dtype=np.float32),
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    )
