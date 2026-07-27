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


def scene_key(scene_id: str) -> str:
    """Scene name from any scene_id spelling.

    Episode datasets store scene_id RELATIVE to scenes_dir
    ("hm3d_v0.2/train/00009-vLpv2VX547B/vLpv2VX547B.basis.glb") while manifests
    built from the filesystem store absolute paths. Comparing the raw strings
    silently matches nothing, which shows up as a generation run producing zero
    groups rather than as an error, so all scene comparisons go through here.
    """
    return Path(str(scene_id)).name.replace(".basis.glb", "").replace(".glb", "")


def build_habitat_config(
    cfg: DictConfig,
    gpu_device_id: int | None = None,
    content_scenes: list[str] | None = None,
) -> Any:
    """Compose the habitat config for an ObjectNav run."""
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import (
        HabitatSimSemanticSensorConfig,
    )
    from habitat.config.read_write import read_write

    sim = cfg.simulator
    if gpu_device_id is None:
        gpu_device_id = int(sim.gpu_device_id)
    overrides = [
        f"habitat.dataset.scenes_dir={cfg.data.scenes_dir}",
        f"habitat.dataset.data_path={episode_dataset_path(cfg)}",
        f"habitat.dataset.split={cfg.data.split}",
        f"habitat.environment.max_episode_steps={cfg.episode.max_steps}",
        f"habitat.simulator.turn_angle={sim.turn_angle}",
        f"habitat.simulator.forward_step_size={sim.forward_step_size}",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={gpu_device_id}",
        f"habitat.simulator.habitat_sim_v0.allow_sliding={sim.allow_sliding}",
        f"habitat.seed={cfg.seed.value}",
        # Distance to the nearest goal viewpoint that counts as success.
        f"habitat.task.measurements.success.success_distance={cfg.task.success_distance}",
    ]
    habitat_cfg = get_config(BASE_TASK_CONFIG, overrides=overrides)

    with read_write(habitat_cfg):
        habitat_cfg.habitat.simulator.scene_dataset = str(cfg.data.scene_dataset_config)

        # Deterministic, repeatable episode order. Without this every policy
        # would be scored on a different episode list, which makes the whole
        # comparison meaningless.
        iterator = habitat_cfg.habitat.environment.iterator_options
        iterator.shuffle = False
        iterator.group_by_scene = True
        iterator.cycle = True
        iterator.max_scene_repeat_steps = -1
        iterator.max_scene_repeat_episodes = -1

        # Restrict which content files are parsed. HM3D train ships roughly
        # 50k episodes per scene across 145 scenes -- about 7 million episodes,
        # each carrying hundreds of viewpoints. Loading all of them costs
        # ~10 minutes and ~7 GB per process, so a sharded run must only read
        # the scenes it will actually use.
        if content_scenes:
            habitat_cfg.habitat.dataset.content_scenes = list(content_scenes)

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


def make_env(cfg: DictConfig, gpu_device_id: int | None = None):
    """Construct a habitat.Env. Caller is responsible for closing it."""
    import habitat

    return habitat.Env(config=build_habitat_config(cfg, gpu_device_id))


def select_episodes(
    cfg: DictConfig,
    count: int,
    shard: int = 0,
    num_shards: int = 1,
    scenes: list[str] | None = None,
    episodes_per_scene: int | None = None,
):
    """A deterministic episode subset, and a dataset restricted to it.

    Returns (dataset, episode_keys). Episodes are ordered by (scene, id) and
    sliced round-robin across shards, so every worker gets a disjoint set and
    every policy is scored on exactly the same episodes.

    `scenes` may be given in any scene_id spelling; matching is by scene name.
    When given, only those content files are parsed.
    """
    import habitat

    wanted = {scene_key(s) for s in scenes} if scenes else None
    habitat_cfg = build_habitat_config(
        cfg, content_scenes=sorted(wanted) if wanted else None
    )
    dataset = habitat.datasets.make_dataset(
        habitat_cfg.habitat.dataset.type, config=habitat_cfg.habitat.dataset
    )

    if wanted:
        dataset.episodes = [
            e for e in dataset.episodes if scene_key(e.scene_id) in wanted
        ]
    if episodes_per_scene:
        # HM3D train has ~50k episodes per scene; a handful each is plenty for
        # collecting decision states and keeps the pool from dominating memory.
        by_scene: dict = {}
        kept = []
        for episode in sorted(
            dataset.episodes, key=lambda e: (str(e.scene_id), int(e.episode_id))
        ):
            key = scene_key(episode.scene_id)
            if by_scene.get(key, 0) >= episodes_per_scene:
                continue
            by_scene[key] = by_scene.get(key, 0) + 1
            kept.append(episode)
        dataset.episodes = kept

    episodes = sorted(dataset.episodes, key=lambda e: (str(e.scene_id), int(e.episode_id)))
    episodes = episodes[: int(count)]
    if num_shards > 1:
        episodes = episodes[shard::num_shards]

    # Group by scene so the simulator reloads as rarely as possible.
    episodes.sort(key=lambda e: (str(e.scene_id), int(e.episode_id)))
    dataset.episodes = episodes
    keys = [(str(e.scene_id), str(e.episode_id)) for e in episodes]
    return dataset, keys


def make_env_with_dataset(cfg: DictConfig, dataset, gpu_device_id: int | None = None):
    import habitat

    return habitat.Env(config=build_habitat_config(cfg, gpu_device_id), dataset=dataset)


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
