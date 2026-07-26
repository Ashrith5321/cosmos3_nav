"""Deterministic seeding.

Two runs of the same script with the same config must produce byte-identical
episode logs. That requires seeding Python, NumPy, torch and habitat, and
deriving every per-episode RNG from the run seed rather than from global state.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    """Seed all global RNGs used in this project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few ops have no deterministic kernel; we want the
        # warning rather than a hard crash mid-episode.
        torch.use_deterministic_algorithms(True, warn_only=True)


def episode_rng(seed: int, scene_id: str, episode_id: str | int) -> np.random.Generator:
    """A per-episode RNG derived from the run seed.

    Deriving from (seed, scene, episode) rather than advancing one global
    stream means an episode replays identically regardless of how many
    episodes ran before it -- needed for the Phase 5 branch protocol, where
    the same decision state is re-entered many times.
    """
    key = f"{seed}|{scene_id}|{episode_id}".encode("utf-8")
    digest = int.from_bytes(key, "big") % (2**63)
    return np.random.default_rng(np.random.SeedSequence([seed, digest]))


def rng_state_fingerprint() -> dict[str, Any]:
    """A hash of global RNG state, recorded in the episode log.

    Used to detect accidental nondeterminism: if two runs claim the same seed
    but end with different fingerprints, something consumed randomness that
    the seeding path does not control. Reading state does not advance it, so
    this is safe to call at any point in a run.
    """
    fingerprint: dict[str, Any] = {
        "python_random": _hash_obj(random.getstate()),
        "numpy_random": _hash_obj(np.random.get_state()),
    }
    try:
        import torch

        fingerprint["torch_random"] = _hash_obj(torch.random.get_rng_state().numpy())
    except ImportError:
        pass
    return fingerprint


def _hash_obj(state: Any) -> str:
    import hashlib

    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()[:16]
