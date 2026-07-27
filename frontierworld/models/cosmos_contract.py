"""The process boundary between habitat and Cosmos 3.

Cosmos 3 needs Python >= 3.10 (diffusers 0.37+); habitat-sim pins us to 3.9.
Generation therefore runs out of process, and the two sides communicate only
through files:

    habitat (3.9)                       cosmos (>= 3.10)
      writes conditioning manifest  -->  reads pending requests
      reads completed rollouts      <--  writes frames + metadata atomically

Nothing in this module imports Cosmos or torch. It defines the contract, and
both sides depend on the contract rather than on each other.

Two properties the contract has to guarantee:

  * a cache key covers *everything* that changes the output -- checkpoint hash,
    preprocessing version, scheduler, steps, resolution, frame count, seed and
    the action sequence. A key that omits any of these silently serves a
    rollout generated under different conditions, and the resulting numbers
    would be unreproducible in a way no test would catch.
  * writes are atomic. A generation interrupted halfway must never leave
    something that looks complete: the next run would read a truncated video as
    if it were a finished one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "cosmos-worker-1"

# Bump when anything about how conditioning frames are built changes -- crop,
# resize, colour handling, frame ordering. It is part of the cache key because
# the same request preprocessed differently is a different request.
PREPROCESSING_VERSION = "prep-1"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class GenerationRequest:
    """Everything that determines a generated rollout.

    Every field here enters the cache key. Adding a field that affects output
    without adding it here is the failure mode this dataclass exists to prevent.
    """

    # what is being predicted
    group_id: str
    frontier_id: int
    scene_id: str
    episode_id: str
    decision_timestep: int
    navigation_goal: str | None

    # conditioning. Cosmos consumes pose_deltas; action_sequence is retained
    # for provenance and for the alignment probe, which reasons about turns.
    action_sequence: list[int]  # omega_i, the executed discrete actions
    horizon: int
    conditioning_frames: list[str]  # paths, relative to the request directory
    pose_deltas: list = field(default_factory=list)  # 9D per transition
    action_mode: str = "fd"  # forward dynamics, per the action cookbook
    prompt: str = ""

    # generator settings
    checkpoint_hash: str = ""
    model_name: str = "cosmos3-nano"
    scheduler: str = "UniPCMultistepScheduler"
    inference_steps: int = 35
    guidance_scale: float = 7.0
    resolution: tuple[int, int] = (480, 640)
    n_frames: int = 17
    fps: int = 8
    seed: int = 0
    dtype: str = "bfloat16"

    # bookkeeping
    contract_version: str = CONTRACT_VERSION
    preprocessing_version: str = PREPROCESSING_VERSION
    extra: dict = field(default_factory=dict)

    def cache_key(self) -> str:
        """Hash of every field that changes the output."""
        payload = {
            "contract_version": self.contract_version,
            "preprocessing_version": self.preprocessing_version,
            "checkpoint_hash": self.checkpoint_hash,
            "model_name": self.model_name,
            "scheduler": self.scheduler,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "resolution": list(self.resolution),
            "n_frames": self.n_frames,
            "fps": self.fps,
            "seed": self.seed,
            "dtype": self.dtype,
            "action_sequence": list(self.action_sequence),
            # The deltas are what the generator consumes, so they belong in the
            # key: same discrete actions with different realised motion is a
            # different request.
            "pose_deltas": [[round(float(v), 6) for v in d] for d in self.pose_deltas],
            "action_mode": self.action_mode,
            "horizon": self.horizon,
            "prompt": self.prompt,
            # Identity of the conditioning, not the pixels: the frames are
            # reproducible from the simulator given these.
            "group_id": self.group_id,
            "frontier_id": self.frontier_id,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "decision_timestep": self.decision_timestep,
            "n_conditioning_frames": len(self.conditioning_frames),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:24]

    def to_dict(self) -> dict:
        return {**asdict(self), "cache_key": self.cache_key()}


def checkpoint_hash(checkpoint_dir: str | Path, sample_bytes: int = 1 << 20) -> str:
    """Identify a checkpoint without reading 33 GB.

    Hashes the index/config files in full plus the size and a prefix of each
    weight shard. Full content hashing would make every worker start slow
    enough to matter; this still changes if the weights are swapped.
    """
    directory = Path(checkpoint_dir)
    digest = hashlib.sha256()
    for name in sorted(["config.json", "model_index.json", "generation_config.json"]):
        path = directory / name
        if path.exists():
            digest.update(path.read_bytes())
    for path in sorted(directory.rglob("*.safetensors")):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()[:16]


class RolloutQueue:
    """File-backed request queue shared by the two processes.

    Deliberately filesystem-based rather than a socket or a server: the two
    sides run under different interpreters and may not be alive at the same
    time, and a crashed worker must leave the queue in a state the next one can
    pick up.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        for name in ("requests", "outputs", "failed"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    # -- habitat side ----------------------------------------------------

    def submit(self, request: GenerationRequest, frames: dict[str, Any] | None = None) -> str:
        """Enqueue a request. Returns its cache key.

        Idempotent: submitting a request whose output already exists is a
        no-op, which is what makes sequential generation with a cache safe to
        re-run after an interruption.
        """
        key = request.cache_key()
        if self.is_complete(key):
            return key

        directory = self.root / "requests" / key
        _atomic_write_dir(
            directory,
            lambda staging: _write_request(staging, request, frames),
        )
        return key

    def is_complete(self, key: str) -> bool:
        return (self.root / "outputs" / key / "COMPLETE").exists()

    def load_output(self, key: str) -> dict | None:
        """Read a finished rollout, or None if it is not finished.

        Keyed off the COMPLETE marker, which is written last, so a partially
        written directory is never mistaken for a usable one.
        """
        directory = self.root / "outputs" / key
        if not (directory / "COMPLETE").exists():
            return None
        import numpy as np

        payload = dict(np.load(directory / "frames.npz"))
        payload["manifest"] = json.loads((directory / "manifest.json").read_text())
        return payload

    def pending(self) -> list[str]:
        return sorted(
            path.name
            for path in (self.root / "requests").iterdir()
            if path.is_dir() and not self.is_complete(path.name)
        )

    # -- cosmos side -----------------------------------------------------

    def claim(self, key: str) -> dict | None:
        """Read a request and mark it running."""
        directory = self.root / "requests" / key
        manifest_path = directory / "request.json"
        if not manifest_path.exists():
            return None
        (directory / "STATUS").write_text(STATUS_RUNNING)
        return json.loads(manifest_path.read_text())

    def complete(self, key: str, frames, manifest: dict, **arrays) -> Path:
        """Write a finished rollout atomically.

        Everything lands in a staging directory that is renamed into place, and
        the COMPLETE marker is written last. An interrupted generation leaves
        either nothing or a directory without the marker; neither reads as
        valid.
        """
        import numpy as np

        target = self.root / "outputs" / key

        def populate(staging: Path) -> None:
            np.savez_compressed(staging / "frames.npz", frames=frames, **arrays)
            (staging / "manifest.json").write_text(
                json.dumps({**manifest, "cache_key": key}, indent=2, default=str)
            )
            # Written last, inside the staging directory, so it is only visible
            # once the rename makes the whole directory visible.
            (staging / "COMPLETE").write_text(key)

        _atomic_write_dir(target, populate)
        return target

    def fail(self, key: str, reason: str) -> None:
        directory = self.root / "failed" / key
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "error.txt").write_text(reason)

    def stats(self) -> dict:
        outputs = list((self.root / "outputs").glob("*/COMPLETE"))
        return {
            "requests": len(list((self.root / "requests").iterdir())),
            "completed": len(outputs),
            "failed": len(list((self.root / "failed").iterdir())),
            "pending": len(self.pending()),
        }


def _write_request(staging: Path, request: GenerationRequest, frames) -> None:
    (staging / "request.json").write_text(json.dumps(request.to_dict(), indent=2, default=str))
    (staging / "STATUS").write_text(STATUS_PENDING)
    if frames:
        import numpy as np

        np.savez_compressed(staging / "conditioning.npz", **frames)


def _atomic_write_dir(target: Path, populate) -> Path:
    """Populate a staging directory, then rename it into place.

    os.replace on a directory is atomic within a filesystem, so a reader either
    sees the previous state or the complete new one, never a half-written mix.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.tmp."))
    try:
        populate(staging)
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


# -- action / frame alignment ---------------------------------------------

FORWARD, TURN_LEFT, TURN_RIGHT = 1, 2, 3


def alignment_probe_sequence() -> list[int]:
    """A deliberately unambiguous action sequence for checking alignment.

    Forward, then a sustained turn, then forward again. If the generator's
    frames are offset by one, or the sequence is reversed, or the actions are
    ignored, the resulting camera motion looks obviously wrong -- unlike a
    generic trajectory, where a plausible-looking video can hide a
    misalignment.
    """
    return [FORWARD] * 4 + [TURN_LEFT] * 6 + [FORWARD] * 4


def check_action_frame_alignment(
    frames, action_sequence: list[int], tolerance: float = 0.35
) -> dict:
    """Do the generated frames move the way the actions say they should?

    Compares per-frame image displacement against what each action implies:
    turns produce large horizontal shifts, forward motion produces small ones.
    A crude test on purpose -- it is meant to catch off-by-one and
    action-ignored, not to measure odometry.
    """
    import numpy as np

    frames = np.asarray(frames)
    if frames.shape[0] < 2:
        return {"checked": False, "reason": "need at least two frames"}

    grey = frames.astype(np.float32).mean(axis=-1) if frames.ndim == 4 else frames.astype(np.float32)
    shifts = []
    for index in range(1, grey.shape[0]):
        previous, current = grey[index - 1], grey[index]
        columns_previous = previous.mean(axis=0)
        columns_current = current.mean(axis=0)
        correlation = np.correlate(
            columns_current - columns_current.mean(),
            columns_previous - columns_previous.mean(),
            mode="same",
        )
        shifts.append(float(np.argmax(correlation) - len(correlation) // 2))

    shifts = np.asarray(shifts)
    n = min(len(shifts), len(action_sequence) - 1) if action_sequence else 0
    if n <= 0:
        return {"checked": False, "reason": "no comparable actions"}

    actions = np.asarray(action_sequence[1 : n + 1])
    turning = np.isin(actions, [TURN_LEFT, TURN_RIGHT])
    forward = actions == FORWARD

    turn_shift = float(np.abs(shifts[:n][turning]).mean()) if turning.any() else float("nan")
    forward_shift = float(np.abs(shifts[:n][forward]).mean()) if forward.any() else float("nan")

    consistent = (
        np.isfinite(turn_shift)
        and np.isfinite(forward_shift)
        and turn_shift > forward_shift * (1.0 + tolerance)
    )
    return {
        "checked": True,
        "mean_turn_shift_px": turn_shift,
        "mean_forward_shift_px": forward_shift,
        "turns_shift_more_than_forward": bool(consistent),
        "per_frame_shift_px": shifts[:n].tolist(),
    }


# -- action representation -------------------------------------------------
#
# Cosmos 3 does NOT take discrete actions. Its unified action interface
# represents a transition between consecutive visual states as a 9D pose delta:
# 3D translation plus a 6D continuous rotation (first two columns of the
# rotation matrix). Passing habitat's discrete action indices would be silently
# meaningless -- they would be read as pose numbers.
#
# The listed embodiments are autonomous vehicle, DROID and UMI. There is no
# indoor-navigation embodiment, so an indoor agent uses the AV ego-pose
# convention out of distribution. Worth stating in any result.
#
# We can build exact deltas rather than approximating: every branch trajectory
# already records per-step position and rotation.

POSE_DELTA_DIM = 9


def rotation_matrix_to_6d(matrix) -> list[float]:
    """Continuous 6D rotation: the first two columns of the rotation matrix.

    Preferred over quaternions or Euler angles as network input because it is
    continuous -- quaternions double-cover and Euler angles gimbal-lock, both of
    which put discontinuities in a space the model has to interpolate across.
    """
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return matrix[:, :2].T.reshape(-1).tolist()


def pose_delta(position_from, rotation_from, position_to, rotation_to) -> list[float]:
    """9D delta between two poses, in the source pose's own frame.

    Ego-relative rather than world-relative: the generator sees ego-motion, and
    a world-frame delta would make identical physical motion look different
    depending on where in the scene it happened.
    """
    import numpy as np

    rotation_from = np.asarray(rotation_from, dtype=np.float64).reshape(3, 3)
    rotation_to = np.asarray(rotation_to, dtype=np.float64).reshape(3, 3)
    translation = rotation_from.T @ (
        np.asarray(position_to, dtype=np.float64)
        - np.asarray(position_from, dtype=np.float64)
    )
    return list(translation) + rotation_matrix_to_6d(rotation_from.T @ rotation_to)


def trajectory_to_pose_deltas(trajectory: list[dict]) -> list[list[float]]:
    """Recorded branch trajectory -> Cosmos 9D pose deltas.

    Entries carry position and a (w, x, y, z) rotation, as written by
    planning.branching. N poses give N-1 deltas; that off-by-one is exactly
    what the action/frame alignment check exists to catch.
    """
    import numpy as np
    import quaternion  # noqa: F401

    deltas = []
    for previous, current in zip(trajectory, trajectory[1:]):
        deltas.append(
            pose_delta(
                previous["position"],
                quaternion.as_rotation_matrix(np.quaternion(*previous["rotation"])),
                current["position"],
                quaternion.as_rotation_matrix(np.quaternion(*current["rotation"])),
            )
        )
    return deltas
