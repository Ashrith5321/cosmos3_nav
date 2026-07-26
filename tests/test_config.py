"""Config loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from frontierworld.config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    check_data_paths,
    config_hash,
    episode_dataset_path,
    load_config,
    resolve_path,
)


def test_default_config_exists():
    assert DEFAULT_CONFIG.exists(), f"missing {DEFAULT_CONFIG}"


def test_load_config_resolves_paths_to_absolute():
    cfg = load_config()
    assert Path(cfg.data.scenes_dir).is_absolute()
    assert Path(cfg.data.scene_dataset_config).is_absolute()
    assert Path(cfg.experiment.output_dir).is_absolute()


def test_overrides_apply():
    cfg = load_config(overrides=["episode.max_steps=7", "seed.value=42"])
    assert cfg.episode.max_steps == 7
    assert cfg.seed.value == 42


def test_config_hash_is_stable_and_sensitive():
    base = load_config()
    same = load_config()
    changed = load_config(overrides=["seed.value=999"])

    assert config_hash(base) == config_hash(same)
    assert config_hash(base) != config_hash(changed)


def test_resolve_path_leaves_absolute_paths_alone():
    assert resolve_path("/tmp/x") == Path("/tmp/x")
    assert resolve_path("configs/base.yaml") == REPO_ROOT / "configs" / "base.yaml"


def test_episode_dataset_path_follows_split():
    cfg = load_config(overrides=["data.split=val"])
    assert "val" in str(episode_dataset_path(cfg))

    cfg = load_config(overrides=["data.split=nonexistent"])
    with pytest.raises(KeyError):
        episode_dataset_path(cfg)


def test_configured_data_is_present():
    """The val split must be usable, or Phase 1 cannot run."""
    problems = check_data_paths(load_config())
    assert problems == [], "; ".join(problems)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
