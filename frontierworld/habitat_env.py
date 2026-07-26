"""Building a Habitat ObjectNav environment from the FrontierWorld config.

Habitat's packaged objectnav_hm3d.yaml gives an RGB-D agent. We add a semantic
sensor and point the simulator at the annotated HM3D scene dataset config,
without which habitat loads the bare .glb stage and every semantic observation
comes back as zeros.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from frontierworld.config import episode_dataset_path

BASE_TASK_CONFIG = "benchmark/nav/objectnav/objectnav_hm3d.yaml"


def build_habitat_config(cfg: DictConfig) -> Any:
    """Compose the habitat config for an ObjectNav run."""
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import (
        HabitatSimSemanticSensorConfig,
    )
    from habitat.config.read_write import read_write

    sim = cfg.simulator
    overrides = [
        f"habitat.dataset.scenes_dir={cfg.data.scenes_dir}",
        f"habitat.dataset.data_path={episode_dataset_path(cfg)}",
        f"habitat.dataset.split={cfg.data.split}",
        f"habitat.environment.max_episode_steps={cfg.episode.max_steps}",
        f"habitat.simulator.turn_angle={sim.turn_angle}",
        f"habitat.simulator.forward_step_size={sim.forward_step_size}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={sim.gpu_device_id}",
        f"habitat.simulator.habitat_sim_v0.allow_sliding={sim.allow_sliding}",
        f"habitat.seed={cfg.seed.value}",
    ]
    habitat_cfg = get_config(BASE_TASK_CONFIG, overrides=overrides)

    with read_write(habitat_cfg):
        habitat_cfg.habitat.simulator.scene_dataset = str(cfg.data.scene_dataset_config)

        agent = habitat_cfg.habitat.simulator.agents.main_agent
        agent.height = sim.agent_height
        agent.radius = sim.agent_radius

        position = [0.0, float(sim.sensor_height), 0.0]
        for name in ("rgb_sensor", "depth_sensor"):
            sensor = agent.sim_sensors[name]
            sensor.width = sim.width
            sensor.height = sim.height
            sensor.hfov = sim.hfov
            sensor.position = position

        depth = agent.sim_sensors["depth_sensor"]
        depth.normalize_depth = bool(sim.normalize_depth)
        depth.min_depth = sim.min_depth
        depth.max_depth = sim.max_depth

        # The packaged RGB-D config has no semantic sensor; add one.
        agent.sim_sensors["semantic_sensor"] = HabitatSimSemanticSensorConfig(
            width=sim.width,
            height=sim.height,
            hfov=sim.hfov,
            position=position,
        )

    return habitat_cfg


def make_env(cfg: DictConfig):
    """Construct a habitat.Env. Caller is responsible for closing it."""
    import habitat

    return habitat.Env(config=build_habitat_config(cfg))


def agent_pose(sim) -> dict[str, Any]:
    """Current agent and sensor poses in habitat world coordinates (y up)."""
    state = sim.get_agent_state()
    pose: dict[str, Any] = {
        "position": [float(x) for x in state.position],
        "rotation": _quat_to_list(state.rotation),
        "sensors": {},
    }
    for name, sensor_state in state.sensor_states.items():
        pose["sensors"][name] = {
            "position": [float(x) for x in sensor_state.position],
            "rotation": _quat_to_list(sensor_state.rotation),
        }
    return pose


def sensor_extrinsics(sim, sensor: str = "depth") -> tuple[np.ndarray, np.ndarray]:
    """(rotation matrix, translation) taking sensor-frame points to world."""
    import quaternion  # noqa: F401  (registers the numpy quaternion dtype)

    state = sim.get_agent_state().sensor_states[sensor]
    rotation = quaternion.as_rotation_matrix(state.rotation)
    translation = np.asarray(state.position, dtype=np.float64)
    return rotation, translation


def semantic_id_to_category(sim) -> dict[int, str]:
    """Map semantic instance ids in the observation to category names.

    Returns an empty mapping when the scene has no loaded semantic annotations,
    which is the signal that data.scene_dataset_config is wrong.
    """
    scene = sim.semantic_scene
    if scene is None:
        return {}
    mapping: dict[int, str] = {}
    for obj in scene.objects:
        if obj is None or obj.category is None:
            continue
        try:
            instance_id = int(obj.semantic_id)
        except (TypeError, ValueError):
            continue
        mapping[instance_id] = obj.category.name()
    return mapping


def _quat_to_list(q) -> list[float]:
    return [float(q.w), float(q.x), float(q.y), float(q.z)]
