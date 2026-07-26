"""Determinism tests.

Phase 5 forks the same decision state many times, so an episode must replay
identically no matter what ran before it. These tests pin that property down
now, while the seeding path is still small enough to reason about.
"""

from __future__ import annotations

import random

import numpy as np

from frontierworld.seeding import episode_rng, rng_state_fingerprint, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(1234, torch_deterministic=False)
    first = (random.random(), float(np.random.random()))

    seed_everything(1234, torch_deterministic=False)
    second = (random.random(), float(np.random.random()))

    assert first == second


def test_different_seeds_differ():
    seed_everything(0, torch_deterministic=False)
    first = float(np.random.random())
    seed_everything(1, torch_deterministic=False)
    second = float(np.random.random())
    assert first != second


def test_episode_rng_is_independent_of_call_order():
    """An episode's stream must not depend on how many episodes preceded it."""
    direct = episode_rng(0, "scene_a", 7).random(5)

    # Consume unrelated randomness, then draw the same episode again.
    episode_rng(0, "scene_b", 3).random(100)
    np.random.random(100)
    replayed = episode_rng(0, "scene_a", 7).random(5)

    np.testing.assert_array_equal(direct, replayed)


def test_episode_rng_separates_episodes_and_scenes():
    a = episode_rng(0, "scene_a", 1).random(5)
    b = episode_rng(0, "scene_a", 2).random(5)
    c = episode_rng(0, "scene_b", 1).random(5)

    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_fingerprint_does_not_advance_state():
    """Reading the fingerprint must not perturb the run it is describing."""
    seed_everything(7, torch_deterministic=False)
    rng_state_fingerprint()
    after_fingerprint = float(np.random.random())

    seed_everything(7, torch_deterministic=False)
    without_fingerprint = float(np.random.random())

    assert after_fingerprint == without_fingerprint


def test_fingerprint_matches_for_equal_seeds():
    seed_everything(99, torch_deterministic=False)
    first = rng_state_fingerprint()
    seed_everything(99, torch_deterministic=False)
    second = rng_state_fingerprint()
    assert first == second
