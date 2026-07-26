"""Configuration loading for FrontierWorld.

A single YAML file (configs/base.yaml) holds every experiment knob. Habitat's
own hydra config is built from those values in frontierworld.habitat_env, so
there is one place to change a setting and one file to archive with a run.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

# Repository root: <root>/frontierworld/config.py -> <root>
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"

# Config keys holding filesystem paths that should be resolved against
# REPO_ROOT when given as relative paths.
_PATH_KEYS = (
    "experiment.output_dir",
    "data.scenes_dir",
    "data.scene_dataset_config",
    "data.episodes.train",
    "data.episodes.val",
    "data.episodes.val_mini",
)


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(
    config_path: str | os.PathLike | None = None,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    """Load the YAML config and apply dotlist overrides.

    Overrides use the usual ``key.subkey=value`` form, e.g.
    ``episode.max_steps=50``.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    for key in _PATH_KEYS:
        value = OmegaConf.select(cfg, key)
        if value is not None:
            OmegaConf.update(cfg, key, str(resolve_path(value)))

    return cfg  # type: ignore[return-value]


def episode_dataset_path(cfg: DictConfig) -> Path:
    """Path to the episode dataset for the configured split."""
    split = cfg.data.split
    if split not in cfg.data.episodes:
        raise KeyError(
            f"split {split!r} has no entry under data.episodes "
            f"(have: {list(cfg.data.episodes.keys())})"
        )
    return Path(cfg.data.episodes[split])


def config_hash(cfg: DictConfig) -> str:
    """Short stable hash of the resolved config, recorded with every run."""
    payload = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(OmegaConf.to_yaml(cfg, resolve=True))


def check_data_paths(cfg: DictConfig) -> list[str]:
    """Return a list of human-readable problems with the configured data paths.

    An empty list means everything the current split needs is present.
    """
    problems: list[str] = []

    scenes_dir = Path(cfg.data.scenes_dir)
    if not scenes_dir.exists():
        problems.append(f"scenes_dir does not exist: {scenes_dir}")

    scene_cfg = Path(cfg.data.scene_dataset_config)
    if not scene_cfg.exists():
        problems.append(
            f"scene_dataset_config does not exist: {scene_cfg} "
            "(semantic annotations will not load without it)"
        )

    try:
        episodes = episode_dataset_path(cfg)
    except KeyError as exc:
        problems.append(str(exc))
    else:
        if not episodes.exists():
            problems.append(f"episode dataset does not exist: {episodes}")

    return problems
