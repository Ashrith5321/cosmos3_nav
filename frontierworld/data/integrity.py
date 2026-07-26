"""Phase 5.5 dataset integrity checks.

Every check here corresponds to a way the branch-complete protocol can be
violated without raising an exception. A dataset that fails silently would
produce a model that looks fine and is trained on leaked or inconsistent
labels, so these run over the whole dataset before it is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    n_checked: int = 0
    failures: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        line = f"[{status}] {self.name} ({self.n_checked} checked)"
        if self.detail:
            line += f" -- {self.detail}"
        for failure in self.failures[:5]:
            line += f"\n         {failure}"
        if len(self.failures) > 5:
            line += f"\n         ... and {len(self.failures) - 5} more"
        return line


def check_common_start_state(dataset) -> CheckResult:
    """Every branch in a group must record the same branch start position."""
    failures: list[str] = []
    checked = 0
    for group in dataset:
        starts = set()
        for example in group.examples:
            start = example.get("observation_history", {}).get("branch_start_position")
            if start is None:
                continue
            starts.add(tuple(np.round(np.asarray(start, dtype=float), 6)))
        checked += 1
        if len(starts) > 1:
            failures.append(f"{group.group_id}: {len(starts)} distinct start states")
    return CheckResult(
        "every branch starts from the same simulator state",
        not failures,
        f"{len(failures)} groups with divergent starts",
        checked,
        failures,
    )


def check_one_outcome_per_frontier(dataset) -> CheckResult:
    """Exactly one recorded outcome per valid candidate frontier."""
    failures: list[str] = []
    checked = 0
    for group in dataset:
        ids = group.frontier_ids()
        checked += len(ids)
        duplicates = [i for i, c in _counts(ids).items() if c > 1]
        if duplicates:
            failures.append(f"{group.group_id}: duplicate frontier ids {duplicates}")
        if len(ids) < 2:
            failures.append(f"{group.group_id}: only {len(ids)} branch(es)")
    return CheckResult(
        "one outcome exists for every valid frontier",
        not failures,
        f"{len(failures)} groups with duplicate or insufficient branches",
        checked,
        failures,
    )


def check_semantic_within_revealed(dataset) -> CheckResult:
    """Semantic revelation must be a subset of occupancy revelation.

    This is the defect manual verification caught in Phase 4: unrestricted, the
    semantic delta counted already-mapped cells finally receiving a label.
    """
    failures: list[str] = []
    checked = 0
    for group in dataset:
        for example in group.examples:
            revelation = example["future_revelation"]
            checked += 1
            semantic = float(revelation.get("newly_semantic_in_revealed_area_m2", 0.0))
            occupancy = float(revelation.get("newly_observed_area_m2", 0.0))
            if semantic > occupancy + 1e-6:
                failures.append(
                    f"{group.group_id} f{revelation.get('frontier_id')}: "
                    f"semantic {semantic:.2f} > revealed {occupancy:.2f}"
                )
    return CheckResult(
        "semantic changes restricted to newly observed cells",
        not failures,
        f"{len(failures)} branches violating the subset property",
        checked,
        failures,
    )


def check_no_cross_branch_leakage(dataset) -> CheckResult:
    """No branch's own future may appear in another branch's input.

    Inputs are shared per group and captured before any branch runs, so the
    test is that every branch in a group references the same pre-branch map and
    observation, and that none references another branch's trajectory.
    """
    failures: list[str] = []
    checked = 0
    for group in dataset:
        maps = {
            (
                example.get("current_map", {}).get("arrays"),
                example.get("current_map", {}).get("key"),
            )
            for example in group.examples
        }
        observations = {
            example.get("current_observation", {}).get("rgb")
            for example in group.examples
        }
        checked += len(group.examples)
        if len(maps) > 1:
            failures.append(f"{group.group_id}: {len(maps)} distinct input maps")
        if len(observations) > 1:
            failures.append(
                f"{group.group_id}: {len(observations)} distinct input observations"
            )
    return CheckResult(
        "no branch observation appears in another branch's input",
        not failures,
        f"{len(failures)} groups with divergent inputs",
        checked,
        failures,
    )


def check_scene_disjoint_splits(manifest) -> CheckResult:
    overlaps = manifest.check_disjoint()
    return CheckResult(
        "no scene overlaps across splits",
        not overlaps,
        f"{len(overlaps)} overlapping scenes",
        sum(len(v) for v in manifest.splits.values()),
        overlaps,
    )


def check_provenance(dataset) -> CheckResult:
    """Schema version and code commit recorded on every example."""
    failures: list[str] = []
    checked = 0
    for group in dataset:
        for example in group.examples:
            checked += 1
            provenance = example.get("privileged", {})
            if not provenance:
                failures.append(f"{group.group_id}: missing privileged block")
            if not example.get("schema_version"):
                failures.append(f"{group.group_id}: missing schema_version")
            if not example.get("git_commit"):
                failures.append(f"{group.group_id}: missing git_commit")
    return CheckResult(
        "schema version and code commit recorded",
        not failures,
        f"{len(failures)} examples missing provenance",
        checked,
        failures,
    )


def check_target_not_already_visible(dataset) -> CheckResult:
    """Decision states where the target was already detected must be excluded.

    Keeping them would let the model score a frontier for a target the agent
    could already see, which is not a prediction problem.
    """
    failures: list[str] = []
    checked = 0
    for group in dataset:
        for example in group.examples:
            checked += 1
            history = example.get("observation_history", {})
            if history.get("target_already_visible"):
                failures.append(f"{group.group_id}: target already visible at decision")
                break
    return CheckResult(
        "target not already detected at the decision state",
        not failures,
        f"{len(failures)} groups with the target already visible",
        checked,
        failures,
    )


def run_all(dataset, manifest=None) -> list[CheckResult]:
    results = [
        check_common_start_state(dataset),
        check_one_outcome_per_frontier(dataset),
        check_semantic_within_revealed(dataset),
        check_no_cross_branch_leakage(dataset),
        check_provenance(dataset),
        check_target_not_already_visible(dataset),
    ]
    if manifest is not None:
        results.append(check_scene_disjoint_splits(manifest))
    return results


def _counts(values) -> dict:
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
