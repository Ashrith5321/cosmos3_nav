"""The stored example schema.

One record per (decision state, candidate frontier), holding exactly the
fields the checklist names:

    scene_id, episode_id, decision_timestep, current_observation, current_map,
    navigation_goal, frontier_geometry, candidate_option, observation_history,
    future_revelation

Large arrays (maps, RGB-D) go to .npz beside the JSON rather than inside it, so
the index stays readable and loadable without pulling gigabytes into memory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RevelationExample:
    """One counterfactual training example."""

    scene_id: str
    episode_id: str
    decision_timestep: int
    navigation_goal: str | None

    frontier_geometry: dict = field(default_factory=dict)
    candidate_option: dict = field(default_factory=dict)
    future_revelation: dict = field(default_factory=dict)

    # Paths, relative to the record directory, for the heavy arrays.
    current_observation: dict = field(default_factory=dict)
    current_map: dict = field(default_factory=dict)
    observation_history: dict = field(default_factory=dict)

    # Provenance for the privileged components used to produce the labels.
    privileged: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RevelationWriter:
    """Writes examples grouped by decision state.

    Grouping matters: Phase 5's dataloader has to return all counterfactual
    branches belonging to one common decision state, so the group is the unit
    on disk, not the individual example.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._index = self.index_path.open("a")
        self.n_groups = 0
        self.n_examples = 0

    def write_group(
        self,
        group_id: str,
        examples: list[RevelationExample],
        arrays: dict[str, np.ndarray],
        rgb: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """Write one decision state: its examples and its shared arrays."""
        group_dir = self.root / "groups" / group_id
        group_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(group_dir / "arrays.npz", **arrays)
        if rgb:
            import imageio.v2 as imageio

            for name, image in rgb.items():
                imageio.imwrite(group_dir / f"{name}.png", np.asarray(image).astype(np.uint8))

        payload = {
            "group_id": group_id,
            "n_examples": len(examples),
            "examples": [example.to_dict() for example in examples],
        }
        (group_dir / "group.json").write_text(
            json.dumps(payload, indent=2, default=_default)
        )

        self._index.write(
            json.dumps(
                {
                    "group_id": group_id,
                    "scene_id": examples[0].scene_id if examples else None,
                    "episode_id": examples[0].episode_id if examples else None,
                    "decision_timestep": (
                        examples[0].decision_timestep if examples else None
                    ),
                    "navigation_goal": examples[0].navigation_goal if examples else None,
                    "n_examples": len(examples),
                    "path": str(group_dir.relative_to(self.root)),
                },
                default=_default,
            )
            + "\n"
        )
        self._index.flush()
        self.n_groups += 1
        self.n_examples += len(examples)
        return group_dir

    def close(self) -> None:
        self._index.close()

    def __enter__(self) -> "RevelationWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def load_group(group_dir: str | Path) -> dict:
    """Read one decision state back, arrays included."""
    group_dir = Path(group_dir)
    payload = json.loads((group_dir / "group.json").read_text())
    payload["arrays"] = dict(np.load(group_dir / "arrays.npz"))
    return payload


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)
