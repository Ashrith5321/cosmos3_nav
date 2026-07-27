"""Phase 9 infrastructure: generated video -> structured revelation.

Cosmos 3 predicts pixels. The paper is scored on geometry and semantics, so a
generated rollout has to be turned into the same structured targets Phase 7
defines, in the same frontier-centred frame, or its output cannot be compared
against Phase 4 ground truth at all.

    V_hat -> depth -> world points -> occupancy / semantics -> frontier frame

Deliberately backend-agnostic. Cosmos 3 needs Python >= 3.10 (diffusers 0.37+)
while the simulator stack is pinned to 3.9, so generation will run out of
process; only the frames come back. Nothing here imports Cosmos, which also
means the conversion is testable today against *simulator* rollouts, and any
error found later is attributable to the generator rather than to this code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyMap


@dataclass
class RolloutSpec:
    """Everything that determines a generated rollout.

    Hashing this gives the cache key, so a rollout is reused only when the
    conditioning is identical. Including the seed means a "deterministic"
    regeneration is verifiable rather than assumed.
    """

    group_id: str
    frontier_id: int
    horizon: int
    n_frames: int
    seed: int
    model: str = "cosmos3-nano"
    guidance: float = 0.0
    conditioning_frames: int = 1
    extra: dict = field(default_factory=dict)

    def key(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]

    def to_dict(self) -> dict:
        return {**asdict(self), "key": self.key()}


class RolloutCache:
    """On-disk cache of generated rollouts, keyed by conditioning.

    Generation is the expensive step by orders of magnitude, and the same
    decision group is revisited constantly during evaluation and ablation. The
    cache also pins reproducibility: a stored rollout carries the exact spec
    that produced it.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path_for(self, spec: RolloutSpec) -> Path:
        return self.root / spec.key()

    def has(self, spec: RolloutSpec) -> bool:
        return (self.path_for(spec) / "frames.npz").exists()

    def load(self, spec: RolloutSpec) -> dict | None:
        directory = self.path_for(spec)
        if not (directory / "frames.npz").exists():
            self.misses += 1
            return None
        self.hits += 1
        payload = dict(np.load(directory / "frames.npz"))
        payload["spec"] = json.loads((directory / "spec.json").read_text())
        return payload

    def store(self, spec: RolloutSpec, frames: np.ndarray, **arrays) -> Path:
        directory = self.path_for(spec)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "frames.npz", frames=frames, **arrays)
        (directory / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
        return directory

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "entries": len(list(self.root.glob("*/frames.npz"))),
        }


def rollout_generator(seed: int, group_id: str, frontier_id: int) -> np.random.Generator:
    """Per-rollout RNG derived from (seed, group, frontier).

    Same rule as episode_rng: derived rather than drawn from a running stream,
    so a rollout reproduces regardless of how many were generated before it.
    Without this, regenerating one branch of a group would need the whole group
    replayed in order.
    """
    digest = int.from_bytes(
        hashlib.sha256(f"{group_id}|{frontier_id}".encode()).digest()[:8], "big"
    )
    return np.random.default_rng(np.random.SeedSequence([seed, digest % (2**63)]))


@dataclass
class StructuredRollout:
    """What a generated rollout implies about the revelation."""

    occupancy: np.ndarray  # frontier-frame grid, UNKNOWN/FREE/OCCUPIED
    revealed_free: np.ndarray  # bool
    revealed_occupied: np.ndarray  # bool
    semantic: np.ndarray  # bool, target-category evidence
    valid: np.ndarray  # bool, cells the rollout could speak to
    revealed_area_m2: float = 0.0
    target_present_score: float = 0.0
    n_frames: int = 0

    def as_target_tensor(self) -> np.ndarray:
        """Stack into the Phase 7 target channel order, for direct comparison."""
        return np.stack(
            [
                self.revealed_free.astype(np.float32),
                self.revealed_occupied.astype(np.float32),
                self.semantic.astype(np.float32),
            ]
        )


def depth_from_frames(frames: np.ndarray, estimator=None) -> np.ndarray:
    """Per-frame metric depth from generated RGB.

    `estimator` is any callable RGB -> depth in metres. Left injectable rather
    than hard-wired: which monocular model is used is a Phase 9 decision, and
    the conversion below must not depend on it.
    """
    if estimator is None:
        raise ValueError(
            "no depth estimator supplied; generated video carries no metric "
            "scale on its own, so one must be provided explicitly"
        )
    return np.stack([np.asarray(estimator(frame), dtype=np.float32) for frame in frames])


def accumulate_rollout(
    depths: np.ndarray,  # (T, H, W) metres
    rotations: np.ndarray,  # (T, 3, 3) sensor-to-world
    translations: np.ndarray,  # (T, 3) world
    frame,  # FrontierFrame from data.tensors
    grid_before: np.ndarray,
    map_origin: np.ndarray,
    map_resolution: float,
    hfov_deg: float,
    max_depth: float,
    floor_y: float,
    semantic_masks: np.ndarray | None = None,
    obstacle_band: tuple[float, float] = (0.20, 1.50),
) -> StructuredRollout:
    """Project a rollout's depths into the frontier frame and diff against
    what was already known.

    Uses the same OccupancyMap projection as the simulator path, so a rollout
    and a real branch are measured by identical geometry. Any difference in the
    numbers is then a difference in the *prediction*, not in how it was
    rasterised.
    """
    from frontierworld.data.tensors import resample_grid

    accumulator = OccupancyMap(
        resolution=map_resolution,
        size_m=float(grid_before.shape[0]) * map_resolution,
        obstacle_height_min=obstacle_band[0],
        obstacle_height_max=obstacle_band[1],
        center=(
            float(map_origin[0] + grid_before.shape[1] * map_resolution / 2),
            float(map_origin[1] + grid_before.shape[0] * map_resolution / 2),
        ),
        floor_y=floor_y,
    )

    for index in range(depths.shape[0]):
        accumulator.integrate(
            depth=np.clip(depths[index], 0.0, max_depth),
            rotation=rotations[index],
            translation=translations[index],
            agent_position=translations[index],
            hfov_deg=hfov_deg,
            max_depth=max_depth,
        )

    grid_after = accumulator.to_grid()

    # Revelation is a difference against the pre-rollout map, exactly as in
    # Phase 4 -- a rollout re-observing known space has revealed nothing.
    newly = (grid_before == UNKNOWN) & (grid_after != UNKNOWN)

    local_after, in_map = resample_grid(
        grid_after, map_origin, map_resolution, frame, fill=UNKNOWN
    )
    local_newly, _ = resample_grid(
        newly.astype(np.uint8), map_origin, map_resolution, frame, fill=0
    )
    revealed = local_newly.astype(bool) & in_map

    semantic_local = np.zeros_like(revealed)
    target_score = 0.0
    if semantic_masks is not None and semantic_masks.size:
        target_score = float(np.mean([m.mean() for m in semantic_masks]))
        # Without per-pixel correspondence from the generator, semantic
        # evidence is attributed to the revealed region rather than localised
        # inside it. Recorded so the limitation is visible in the numbers.
        semantic_local = revealed & (target_score > 0)

    return StructuredRollout(
        occupancy=local_after,
        revealed_free=revealed & (local_after == FREE),
        revealed_occupied=revealed & (local_after == OCCUPIED),
        semantic=semantic_local,
        valid=in_map,
        revealed_area_m2=float(newly.sum() * map_resolution**2),
        target_present_score=target_score,
        n_frames=int(depths.shape[0]),
    )


def compare_to_ground_truth(
    rollout: StructuredRollout, targets: np.ndarray, valid: np.ndarray
) -> dict:
    """Score a converted rollout against the Phase 4 targets.

    Same metric functions as the structured predictor, so Cosmos and the Phase
    8 model land in one table rather than two incomparable ones.
    """
    from frontierworld.models.metrics import masked_iou

    predicted = rollout.as_target_tensor()
    window = valid & rollout.valid
    return {
        "free_iou": masked_iou(predicted[0], targets[0], window),
        "occupied_iou": masked_iou(predicted[1], targets[1], window),
        "semantic_iou": masked_iou(predicted[2], targets[2], window),
        "revealed_area_m2": rollout.revealed_area_m2,
        "n_frames": rollout.n_frames,
    }
