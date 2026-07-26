"""Per-episode observation recording.

Layout produced for one episode::

    <run_dir>/
      config.yaml            resolved config for the whole run
      run.jsonl              one record per step, all episodes
      run.csv                same, flat columns
      summary.json           run-level result
      episodes/<scene>_<episode_id>/
        episode.json         metadata, goal, metrics, per-step index
        pose.jsonl           agent and sensor poses per step
        semantic_categories.json  instance id -> category name
        rgb/000000.png
        depth/000000.npz     float32 metres
        semantic/000000.npz  int32 instance ids
        map/occupancy.npz    final ternary grid + counts + geometry
        map/occupancy.png    preview
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class EpisodeWriter:
    """Writes the observation streams for a single episode."""

    def __init__(
        self,
        root: str | Path,
        scene_id: str,
        episode_id: str | int,
        recording_cfg: Any,
    ) -> None:
        scene_name = Path(str(scene_id)).stem.replace(".basis", "")
        self.dir = Path(root) / "episodes" / f"{scene_name}_{episode_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = recording_cfg
        self.scene_id = str(scene_id)
        self.episode_id = episode_id

        self._subdirs: dict[str, Path] = {}
        for name, enabled in (
            ("rgb", recording_cfg.save_rgb),
            ("depth", recording_cfg.save_depth),
            ("semantic", recording_cfg.save_semantic),
        ):
            if enabled:
                path = self.dir / name
                path.mkdir(exist_ok=True)
                self._subdirs[name] = path

        self._pose_file = (
            (self.dir / "pose.jsonl").open("w") if recording_cfg.save_pose else None
        )
        self.step_index: list[dict[str, Any]] = []

    # -- writing ---------------------------------------------------------

    def write_step(
        self,
        step: int,
        observations: dict[str, Any],
        pose: dict[str, Any],
        action: int | None = None,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record one timestep. Honours recording.stride."""
        stride = int(getattr(self.cfg, "stride", 1))
        saved = step % stride == 0
        record: dict[str, Any] = {"step": step, "action": action, "saved": saved}

        if saved:
            stem = f"{step:06d}"
            if "rgb" in self._subdirs and "rgb" in observations:
                _write_png(self._subdirs["rgb"] / f"{stem}.png", observations["rgb"])
                record["rgb"] = f"rgb/{stem}.png"
            if "depth" in self._subdirs and "depth" in observations:
                depth = np.squeeze(np.asarray(observations["depth"], dtype=np.float32))
                np.savez_compressed(self._subdirs["depth"] / f"{stem}.npz", depth=depth)
                record["depth"] = f"depth/{stem}.npz"
            if "semantic" in self._subdirs and "semantic" in observations:
                semantic = np.squeeze(
                    np.asarray(observations["semantic"], dtype=np.int32)
                )
                np.savez_compressed(
                    self._subdirs["semantic"] / f"{stem}.npz", semantic=semantic
                )
                record["semantic"] = f"semantic/{stem}.npz"

        if self._pose_file is not None:
            self._pose_file.write(json.dumps({"step": step, **pose}) + "\n")

        for key in ("gps", "compass", "objectgoal"):
            if key in observations:
                record[key] = np.asarray(observations[key]).tolist()
        if info:
            record["info"] = _jsonable(info)

        self.step_index.append(record)

    def write_semantic_categories(self, mapping: dict[int, str]) -> None:
        path = self.dir / "semantic_categories.json"
        path.write_text(json.dumps({str(k): v for k, v in mapping.items()}, indent=2))

    def write_map(self, occupancy_map) -> None:
        """Save the final occupancy grid, its counts and its geometry."""
        if not getattr(self.cfg, "save_map", True):
            return
        map_dir = self.dir / "map"
        map_dir.mkdir(exist_ok=True)
        geometry = occupancy_map.geometry
        np.savez_compressed(
            map_dir / "occupancy.npz",
            grid=occupancy_map.to_grid(),
            free_counts=occupancy_map.free_counts,
            occupied_counts=occupancy_map.occupied_counts,
            resolution=np.float32(geometry.resolution),
            origin=np.asarray([geometry.origin_x, geometry.origin_z], dtype=np.float32),
            floor_y=np.float32(occupancy_map.floor_y),
        )
        if getattr(self.cfg, "save_map_preview", True):
            _write_png(map_dir / "occupancy.png", occupancy_map.to_preview())

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        payload = {
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            **_jsonable(metadata),
            "steps": self.step_index,
        }
        (self.dir / "episode.json").write_text(json.dumps(payload, indent=2))

    def close(self) -> None:
        if self._pose_file is not None:
            self._pose_file.close()
            self._pose_file = None

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _write_png(path: Path, array: np.ndarray) -> None:
    import imageio.v2 as imageio

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    imageio.imwrite(path, array.astype(np.uint8))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
