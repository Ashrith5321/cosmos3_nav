"""Build the immutable conditioning manifest for a Cosmos rollout.

Runs in the **Habitat** environment. It never imports Cosmos. Its only output
is a directory of files; the Cosmos worker reads that directory and nothing
else. That file interface is the whole reason the two environments can hold
incompatible torch versions without either one degrading the other.

What it emits, per requested branch:

    manifest.json               every field needed to reproduce the request
    conditioning_rgb.png        the observed frame at the decision point
    conditioning_depth.npy      the observed sensor depth at the same instant,
                                used later for first-frame scale anchoring
    nominal_camera_poses.npy    (n_frames, 4, 4) camera-to-world, OpenCV, metres

The conditioning depth is re-rendered by restoring the agent to the recorded
decision-point pose. That is an *observed* quantity -- the agent is standing
there and its depth sensor is on -- so it is not future information. Nothing
downstream of the conditioning frame is read from the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from frontierworld.config import load_config
from frontierworld.models.nominal_poses import Kinematics, nominal_camera_poses


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def render_conditioning_depth(cfg, scene_id: str, position, rotation) -> np.ndarray:
    """Sensor depth at the decision-point pose, by restoring the agent there.

    Imported lazily so that the pure parts of this module stay importable
    without a simulator.
    """
    import habitat_sim

    from frontierworld.planning.branching import sim_observations

    simulator = cfg.simulator
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_id
    backend.scene_dataset_config_file = str(cfg.data.scene_dataset_config)
    backend.gpu_device_id = int(simulator.gpu_device_id)
    backend.enable_physics = False

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [int(simulator.height), int(simulator.width)]
    depth_spec.position = [0.0, float(simulator.sensor_height), 0.0]
    depth_spec.hfov = float(simulator.hfov)

    colour_spec = habitat_sim.CameraSensorSpec()
    colour_spec.uuid = "rgb"
    colour_spec.sensor_type = habitat_sim.SensorType.COLOR
    colour_spec.resolution = [int(simulator.height), int(simulator.width)]
    colour_spec.position = [0.0, float(simulator.sensor_height), 0.0]
    colour_spec.hfov = float(simulator.hfov)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [colour_spec, depth_spec]
    agent_cfg.height = float(simulator.agent_height)
    agent_cfg.radius = float(simulator.agent_radius)

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    try:
        state = sim.get_agent(0).get_state()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = np.quaternion(*[float(v) for v in rotation])
        sim.get_agent(0).set_state(state)
        observations = sim_observations(sim, cfg)
        return (
            np.asarray(observations["depth"], dtype=np.float32).copy(),
            np.asarray(observations["rgb"], dtype=np.uint8)[..., :3].copy(),
        )
    finally:
        sim.close()


def decision_point_pose(example: dict, kinematics: Kinematics):
    """Position and rotation the agent occupied when the decision was made."""
    from frontierworld.models.nominal_poses import recover_start_pose, yaw_to_matrix

    history = example["observation_history"]
    trajectory = history["trajectory"]
    position, yaw = recover_start_pose(
        trajectory, history["branch_start_position"], kinematics
    )
    # Habitat quaternion (w, x, y, z) for a yaw-only rotation about +y.
    rotation = [float(np.cos(yaw / 2)), 0.0, float(np.sin(yaw / 2)), 0.0]
    return position, rotation, yaw


def build(
    selection: dict,
    cfg,
    n_frames: int,
    out_dir: Path,
    render_depth: bool,
) -> dict:
    group_dir = Path(selection["group_dir"])
    group = json.loads((group_dir / "group.json").read_text())
    example = group["examples"][selection["branch_index"]]

    kinematics = Kinematics(
        forward_step_size=float(cfg.simulator.forward_step_size),
        turn_angle_deg=float(cfg.simulator.turn_angle),
        sensor_height=float(cfg.simulator.sensor_height),
    )

    history = example["observation_history"]
    poses, indices = nominal_camera_poses(
        history["trajectory"],
        history["branch_start_position"],
        n_frames=n_frames,
        kinematics=kinematics,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "nominal_camera_poses.npy", poses.astype(np.float32))

    source_rgb = group_dir / example["current_observation"]["rgb"]
    shutil.copyfile(source_rgb, out_dir / "conditioning_rgb.png")

    position, rotation, yaw = decision_point_pose(example, kinematics)

    depth_info: dict = {"rendered": False}
    if render_depth:
        depth, rgb = render_conditioning_depth(
            cfg, example["scene_id"], position, rotation
        )
        np.save(out_dir / "conditioning_depth.npy", depth)
        np.save(out_dir / "conditioning_rgb_rerendered.npy", rgb)
        valid = np.isfinite(depth) & (depth > float(cfg.simulator.min_depth))
        depth_info = {
            "rendered": True,
            "shape": list(depth.shape),
            "valid_fraction": float(valid.mean()),
            "min_m": float(depth[valid].min()) if valid.any() else None,
            "max_m": float(depth[valid].max()) if valid.any() else None,
            "median_m": float(np.median(depth[valid])) if valid.any() else None,
            "hash": array_hash(depth),
        }

    option = example["candidate_option"]
    actions = [int(step["action"]) for step in history["trajectory"]]

    manifest = {
        "schema_version": "frontierworld-cosmos-conditioning-1",
        "group_id": group["group_id"],
        "branch_index": selection["branch_index"],
        "frontier_id": option["frontier_id"],
        "scene_id": example["scene_id"],
        "episode_id": example["episode_id"],
        "decision_timestep": example["decision_timestep"],
        "source_group_dir": str(group_dir),
        "source_rgb_hash": file_hash(source_rgb),
        "decision_point_pose": {
            "position": [float(v) for v in position],
            "rotation_wxyz": rotation,
            "yaw_rad": float(yaw),
            "derivation": "first trajectory action inverted; NOT a stored field",
        },
        "option": {
            "n_approach_actions": option["n_approach_actions"],
            "n_cross_actions": option["n_cross_actions"],
            "horizon": option["horizon"],
            "probe_distance_m": option["probe_distance_m"],
            "crossing_yaw": option["crossing_yaw"],
            "approach_world": option["approach_world"],
            "action_sequence": actions,
            "n_actions_total": len(actions),
        },
        "poses": {
            "n_frames": int(poses.shape[0]),
            "sampled_action_indices": [int(i) for i in indices],
            "convention": "camera-to-world, OpenCV axes (x right, y down, z forward), metres",
            "source": "nominal kinematic integration of the option action sequence",
            "uses_executed_poses": False,
            "uses_future_groundtruth": False,
            "hash": array_hash(poses.astype(np.float32)),
        },
        "sensor": {
            "width": int(cfg.simulator.width),
            "height": int(cfg.simulator.height),
            "hfov_deg": float(cfg.simulator.hfov),
            "sensor_height_m": float(cfg.simulator.sensor_height),
            "min_depth_m": float(cfg.simulator.min_depth),
            "max_depth_m": float(cfg.simulator.max_depth),
        },
        "kinematics": {
            "forward_step_size_m": kinematics.forward_step_size,
            "turn_angle_deg": kinematics.turn_angle_deg,
        },
        "conditioning_depth": depth_info,
        "prompt_policy": {
            "objectnav_goal_in_prompt": False,
            "goal_for_audit_only": example["navigation_goal"],
            "note": "the navigation goal is recorded for auditing and never sent to the generator",
        },
        "ground_truth_revelation": {
            "newly_free_cells": example["future_revelation"]["newly_free_cells"],
            "newly_occupied_cells": example["future_revelation"]["newly_occupied_cells"],
            "newly_observed_area_m2": example["future_revelation"]["newly_observed_area_m2"],
            "crossed": example["future_revelation"]["crossed"],
            "note": "held for scoring only; never provided to the generator",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=Path("archive/phase9d/smoke_selection.json"))
    parser.add_argument("--config", type=Path, default=Path("outputs/phase5/full_train/config.yaml"))
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--out", type=Path, default=Path("outputs/phase9d/smoke_conditioning"))
    parser.add_argument("--no-depth", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())["selected"]
    cfg = load_config(args.config)
    manifest = build(selection, cfg, args.frames, args.out, render_depth=not args.no_depth)
    print(json.dumps(manifest, indent=2)[:2000])


if __name__ == "__main__":
    main()
