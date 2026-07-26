"""The FrontierReveal dataloader.

The unit of iteration is a **decision group**, not a branch. Every branch in a
group started from the same simulator state, and the whole point of the dataset
is to compare candidates against each other from that shared state; shuffling
branches independently would destroy exactly the structure the ranking loss
needs.

Variable frontier counts are padded with a mask rather than truncated, so a
state with nine candidates is not silently reduced to the same shape as one
with two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


@dataclass
class DecisionGroup:
    """All counterfactual branches from one common decision state."""

    group_id: str
    scene_id: str
    episode_id: str
    decision_timestep: int
    navigation_goal: str | None
    examples: list[dict]
    arrays: dict[str, np.ndarray]
    path: Path

    @property
    def n_branches(self) -> int:
        return len(self.examples)

    def revelations(self) -> list[dict]:
        return [e["future_revelation"] for e in self.examples]

    def frontier_ids(self) -> list[int]:
        return [e["frontier_geometry"]["frontier_id"] for e in self.examples]

    def utilities(self, weights: dict[str, float] | None = None) -> np.ndarray:
        """Realised utility per branch, from ground truth only.

        Phase 12 defines the coefficients; the default here is a neutral
        placeholder so ranking code can be exercised before they are fixed.
        """
        weights = weights or {
            "target": 1.0,
            "area": 0.05,
            "new_frontiers": 0.1,
            "cost": 0.02,
            "risk": 0.05,
        }
        scores = []
        for revelation in self.revelations():
            scores.append(
                weights["target"] * float(revelation.get("target_became_visible", 0.0))
                + weights["area"] * float(revelation.get("newly_observed_area_m2", 0.0))
                + weights["new_frontiers"] * float(revelation.get("n_new_frontiers", 0))
                - weights["cost"] * float(revelation.get("distance_travelled_m", 0.0))
                - weights["risk"] * float(revelation.get("collisions", 0))
            )
        return np.asarray(scores, dtype=np.float32)

    def oracle_index(self, weights: dict[str, float] | None = None) -> int:
        return int(np.argmax(self.utilities(weights)))


class FrontierRevealDataset:
    """Reads decision groups written by RevelationWriter.

    Deliberately not a torch Dataset: Phase 7 defines the tensorisation, and
    binding this to a framework before the representation is settled would make
    it harder to change. `collate_groups` gives the padded batch when needed.
    """

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        manifest: Any = None,
        min_branches: int = 2,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.manifest = manifest
        self.min_branches = int(min_branches)

        index_path = self.root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"no dataset index at {index_path}")

        allowed: set[str] | None = None
        if manifest is not None and split is not None:
            allowed = set(manifest.scene_ids(split))

        self.entries: list[dict] = []
        with index_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("n_examples", 0) < self.min_branches:
                    continue
                if allowed is not None and entry.get("scene_id") not in allowed:
                    continue
                self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> DecisionGroup:
        entry = self.entries[index]
        group_dir = self.root / entry["path"]
        payload = json.loads((group_dir / "group.json").read_text())
        arrays = dict(np.load(group_dir / "arrays.npz"))
        examples = payload["examples"]
        first = examples[0] if examples else {}
        return DecisionGroup(
            group_id=payload["group_id"],
            scene_id=first.get("scene_id", ""),
            episode_id=first.get("episode_id", ""),
            decision_timestep=first.get("decision_timestep", -1),
            navigation_goal=first.get("navigation_goal"),
            examples=examples,
            arrays=arrays,
            path=group_dir,
        )

    def __iter__(self) -> Iterator[DecisionGroup]:
        for index in range(len(self)):
            yield self[index]

    def scenes(self) -> set[str]:
        return {e["scene_id"] for e in self.entries if e.get("scene_id")}

    def statistics(self) -> dict:
        """Per-channel dataset statistics, for the Phase 5.5 summary plots."""
        branches = 0
        per_group = []
        channels: dict[str, list[float]] = {}
        targets = 0
        crossed = 0

        for group in self:
            per_group.append(group.n_branches)
            branches += group.n_branches
            for revelation in group.revelations():
                targets += int(bool(revelation.get("target_became_visible")))
                crossed += int(bool(revelation.get("crossed")))
                for key, value in revelation.items():
                    # inf/nan appear where a target is unreachable; they would
                    # poison the mean rather than being informative.
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and np.isfinite(value)
                    ):
                        channels.setdefault(key, []).append(float(value))

        summary = {
            "groups": len(self),
            "branches": branches,
            "scenes": len(self.scenes()),
            "branches_per_group_mean": float(np.mean(per_group)) if per_group else 0.0,
            "branches_per_group_min": int(min(per_group)) if per_group else 0,
            "branches_per_group_max": int(max(per_group)) if per_group else 0,
            "target_positive_rate": targets / branches if branches else 0.0,
            "crossing_success_rate": crossed / branches if branches else 0.0,
            "channels": {
                key: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for key, values in sorted(channels.items())
            },
        }
        return summary


def collate_groups(groups: Sequence[DecisionGroup]) -> dict[str, Any]:
    """Pad a batch of decision groups to a common branch count.

    Returns a `branch_mask` marking real branches; padded slots carry zeros and
    must be excluded from every loss and every ranking.
    """
    if not groups:
        raise ValueError("cannot collate an empty batch")

    max_branches = max(group.n_branches for group in groups)
    batch = len(groups)

    mask = np.zeros((batch, max_branches), dtype=bool)
    utilities = np.zeros((batch, max_branches), dtype=np.float32)
    revealed = np.zeros((batch, max_branches), dtype=np.float32)
    target = np.zeros((batch, max_branches), dtype=np.float32)

    for i, group in enumerate(groups):
        n = group.n_branches
        mask[i, :n] = True
        utilities[i, :n] = group.utilities()
        for j, revelation in enumerate(group.revelations()):
            revealed[i, j] = float(revelation.get("newly_observed_area_m2", 0.0))
            target[i, j] = float(revelation.get("target_became_visible", 0.0))

    return {
        "group_ids": [g.group_id for g in groups],
        "branch_mask": mask,
        "utility": utilities,
        "newly_observed_area_m2": revealed,
        "target_became_visible": target,
        "n_branches": np.asarray([g.n_branches for g in groups], dtype=np.int32),
    }
