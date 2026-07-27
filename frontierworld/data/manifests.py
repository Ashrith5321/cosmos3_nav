"""Scene-disjoint split manifests.

Splits are defined over *scenes*, never episodes. Two episodes in the same
apartment share geometry, semantics and room layout, so an episode-level split
leaks the test set into training and every held-out number becomes optimistic.

Each manifest records the scenes it covers, their semantic instance counts, the
config hash and the code commit, so a dataset built from it can be traced back
to exactly this partition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = "frontierreveal-1"


@dataclass
class SceneRecord:
    scene_id: str
    scene_name: str
    split_source: str  # the HM3D split the scene actually comes from
    n_episodes: int
    n_semantic_instances: int = -1  # -1 means "not audited"

    @property
    def has_semantics(self) -> bool:
        return self.n_semantic_instances > 0


@dataclass
class SplitManifest:
    """One scene-disjoint partition of the available scenes."""

    name: str
    schema_version: str = SCHEMA_VERSION
    pilot: bool = False
    pilot_reason: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    splits: dict[str, list[str]] = field(default_factory=dict)  # split -> scene ids
    scenes: dict[str, dict] = field(default_factory=dict)  # scene id -> SceneRecord

    def scene_ids(self, split: str) -> list[str]:
        return list(self.splits.get(split, []))

    def check_disjoint(self) -> list[str]:
        """Return a list of scenes appearing in more than one split."""
        seen: dict[str, str] = {}
        overlaps: list[str] = []
        for split, scenes in self.splits.items():
            for scene in scenes:
                if scene in seen:
                    overlaps.append(f"{scene} in both {seen[scene]} and {split}")
                seen[scene] = split
        return overlaps

    def scenes_without_semantics(self) -> list[str]:
        return [
            scene_id
            for scene_id, record in self.scenes.items()
            if record.get("n_semantic_instances", -1) == 0
        ]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text())
        return cls(**payload)

    def summary(self) -> str:
        lines = [
            f"manifest: {self.name}  schema={self.schema_version}",
        ]
        if self.pilot:
            lines.append(f"  PILOT: {self.pilot_reason}")
        for split in sorted(self.splits):
            scenes = self.splits[split]
            episodes = sum(
                max(0, self.scenes.get(s, {}).get("n_episodes", 0)) for s in scenes
            )
            suffix = f"  {episodes:5d} episodes" if episodes else ""
            lines.append(f"  {split:6s} {len(scenes):3d} scenes{suffix}")
        overlaps = self.check_disjoint()
        lines.append(f"  scene-disjoint: {'yes' if not overlaps else 'NO -- ' + '; '.join(overlaps)}")
        missing = self.scenes_without_semantics()
        if missing:
            lines.append(f"  scenes WITHOUT semantics: {len(missing)}")
        return "\n".join(lines)


def stable_partition(
    scene_ids: list[str], fractions: dict[str, float], seed: int = 0
) -> dict[str, list[str]]:
    """Deterministically partition scenes by hashing their names.

    Hashing rather than shuffling means adding a scene to the pool does not
    reshuffle the ones already assigned, so a manifest stays comparable as more
    data arrives.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    ordered = sorted(fractions.items(), key=lambda item: item[0])
    boundaries: list[tuple[str, float]] = []
    cumulative = 0.0
    for name, fraction in ordered:
        cumulative += fraction
        boundaries.append((name, cumulative))

    splits: dict[str, list[str]] = {name: [] for name, _ in ordered}
    for scene_id in sorted(scene_ids):
        digest = hashlib.sha256(f"{seed}|{scene_id}".encode("utf-8")).hexdigest()
        position = int(digest[:16], 16) / float(1 << 64)
        for name, edge in boundaries:
            if position < edge:
                splits[name].append(scene_id)
                break
        else:
            splits[ordered[-1][0]].append(scene_id)
    return splits


def count_semantic_instances(scene_id: str, scene_dataset_config: str) -> int:
    """Load a scene headlessly and count annotated semantic objects.

    Cheaper than a full environment reset, and it is the check that catches the
    silent failure mode where habitat loads the bare .glb and every semantic
    observation is zeros.
    """
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_id
    backend.scene_dataset_config_file = str(scene_dataset_config)
    backend.load_semantic_mesh = True
    backend.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    try:
        with habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [agent_cfg])
        ) as sim:
            scene = sim.semantic_scene
            if scene is None:
                return 0
            return sum(
                1 for obj in scene.objects if obj is not None and obj.category is not None
            )
    except Exception:  # noqa: BLE001 - a scene that will not load counts as unusable
        return -1
