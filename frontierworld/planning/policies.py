"""Baseline frontier-selection policies.

These are the non-learning references the revelation model must beat. All four
score the same candidate set from the same decision state, so the only thing
that differs between them is the ranking rule -- which is what Phase 12's
regret metric measures.

Every policy is seeded: given the same map and the same episode RNG, it makes
the same choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from frontierworld.frontiers.extraction import Frontier


class FrontierPolicy(Protocol):
    name: str

    def select(
        self, frontiers: Sequence[Frontier], rng: np.random.Generator
    ) -> Frontier | None: ...


@dataclass
class RandomFrontier:
    """Uniform choice among reachable frontiers."""

    name: str = "random"

    def select(
        self, frontiers: Sequence[Frontier], rng: np.random.Generator
    ) -> Frontier | None:
        if not frontiers:
            return None
        return frontiers[int(rng.integers(len(frontiers)))]


@dataclass
class NearestFrontier:
    """Smallest geodesic travel distance."""

    name: str = "nearest"

    def select(
        self, frontiers: Sequence[Frontier], rng: np.random.Generator
    ) -> Frontier | None:
        if not frontiers:
            return None
        return min(frontiers, key=lambda f: _distance(f))


@dataclass
class MaxInformationGain:
    """Largest unknown area within the gain radius, ignoring travel cost."""

    name: str = "max_info_gain"

    def select(
        self, frontiers: Sequence[Frontier], rng: np.random.Generator
    ) -> Frontier | None:
        if not frontiers:
            return None
        return max(frontiers, key=lambda f: f.information_gain_m2)


@dataclass
class InformationGainMinusCost:
    """Information gain traded against travel distance.

    score = information_gain_m2 - cost_weight * geodesic_distance_m

    This is the strongest non-learning baseline and the one the paper's
    planner (Eq. 17) reduces to when the utility, risk and uncertainty terms
    are removed, so it is the honest reference for "does prediction help".
    """

    cost_weight: float = 1.0
    name: str = "info_gain_minus_cost"

    def select(
        self, frontiers: Sequence[Frontier], rng: np.random.Generator
    ) -> Frontier | None:
        if not frontiers:
            return None
        return max(frontiers, key=lambda f: self.score(f))

    def score(self, frontier: Frontier) -> float:
        distance = _distance(frontier)
        if not np.isfinite(distance):
            return -np.inf
        return float(frontier.information_gain_m2 - self.cost_weight * distance)


def _distance(frontier: Frontier) -> float:
    """Travel distance, falling back to infinity when unqueried."""
    if frontier.geodesic_distance_m is None:
        return float("inf")
    return float(frontier.geodesic_distance_m)


POLICIES = {
    "random": RandomFrontier,
    "nearest": NearestFrontier,
    "max_info_gain": MaxInformationGain,
    "info_gain_minus_cost": InformationGainMinusCost,
}


def make_policy(name: str, **kwargs) -> FrontierPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; have {sorted(POLICIES)}")
    policy_cls = POLICIES[name]
    if name == "info_gain_minus_cost":
        return policy_cls(cost_weight=float(kwargs.get("cost_weight", 1.0)))
    return policy_cls()
