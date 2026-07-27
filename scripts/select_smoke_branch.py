"""Predetermine the Phase 9D smoke-test branch, BEFORE any generation.

The point of writing this as a script with a fixed rule, rather than eyeballing
groups, is that the branch must not be chosen after seeing how Cosmos performs
on it. The rule below is the preregistration: it is evaluated once, its output
is archived, and the archived group is the one that gets generated.

Selection rule (fixed in advance, applied in this order):

  0. scene must be a development scene, defined here as a scene in the
     **train** split of manifests/full.json -- i.e. already consumed by Phase 8
     v0 training, so it carries no residual held-out value.

     Sealed-final scenes are never enumerated, read, or hashed. Two independent
     structural guards make that safe without opening the sealed manifest:
       (a) `manifests/SEALED_DO_NOT_TOUCH.md` records that sealed_final_v1 is
           drawn from HM3D **val-split** scenes, whereas full.json is drawn
           entirely from HM3D **train-split** scenes. The sets are disjoint by
           construction.
       (b) every selected scene path is asserted to lie under
           `hm3d_v0.2/train/`, which re-checks (a) on the actual path rather
           than trusting the manifest.
     Note that dev_pool_v1.json is NOT the right pool here: it is the pool of
     scenes still *untouched* and reserved for v1 development, and it is
     disjoint from every scene any dataset has been generated on.
  1. the branch must actually cross the frontier            crossed == True
  2. the crossing must contain BOTH rotation and translation, so a failure to
     follow the action can be attributed to one or the other:
         >= 2 turn actions (2 = LEFT, 3 = RIGHT) and >= 4 forward actions (1)
  3. the revelation must be nontrivial in BOTH classes, otherwise the converter
     comparison is dominated by one of them:
         newly_free_cells >= 200 and newly_occupied_cells >= 50
  4. the conditioning RGB must exist on disk
  5. tie-break: ascending (group_id, frontier_id). Purely lexical, so it cannot
     encode any preference for a branch that looks easy to generate.

Nothing here looks at the ObjectNav goal, and the goal never enters the
generation prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TURN_ACTIONS = {2, 3}
FORWARD_ACTION = 1

MIN_TURNS = 2
MIN_FORWARDS = 4
MIN_NEW_FREE_CELLS = 200
MIN_NEW_OCCUPIED_CELLS = 50


def scene_key(scene_id: str) -> str:
    return Path(str(scene_id)).name.replace(".basis.glb", "").replace(".glb", "")


HM3D_TRAIN_MARKER = "hm3d_v0.2/train/"


def development_keys(manifest_path: Path) -> set[str]:
    """Train-split scenes of the given manifest, guarded to HM3D train only."""
    manifest = json.loads(manifest_path.read_text())
    scenes = manifest["splits"]["train"]
    for scene in scenes:
        if HM3D_TRAIN_MARKER not in str(scene):
            raise SystemExit(
                f"refusing to proceed: {scene!r} is not under {HM3D_TRAIN_MARKER}; "
                "the sealed set is drawn from HM3D val-split scenes and this guard "
                "is what keeps them out without reading the sealed manifest"
            )
    return {scene_key(s) for s in scenes}


def candidates(dataset_root: Path, pool: set[str]) -> list[dict]:
    """Every (group, example) pair that satisfies the rule."""
    out: list[dict] = []
    index = dataset_root / "index.jsonl"
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if scene_key(record["scene_id"]) not in pool:
            continue
        if HM3D_TRAIN_MARKER not in record["scene_id"]:
            raise SystemExit(f"scene outside {HM3D_TRAIN_MARKER}: {record['scene_id']}")

        group_dir = dataset_root / record["path"]
        group = json.loads((group_dir / "group.json").read_text())

        for position, example in enumerate(group["examples"]):
            option = example["candidate_option"]
            revelation = example["future_revelation"]

            if not revelation.get("crossed"):
                continue

            actions = list(option.get("cross_actions") or [])
            n_turns = sum(1 for a in actions if a in TURN_ACTIONS)
            n_forwards = sum(1 for a in actions if a == FORWARD_ACTION)
            if n_turns < MIN_TURNS or n_forwards < MIN_FORWARDS:
                continue

            if revelation.get("newly_free_cells", 0) < MIN_NEW_FREE_CELLS:
                continue
            if revelation.get("newly_occupied_cells", 0) < MIN_NEW_OCCUPIED_CELLS:
                continue

            rgb = group_dir / example["current_observation"]["rgb"]
            if not rgb.exists():
                continue

            out.append(
                {
                    "group_id": group["group_id"],
                    "branch_index": position,
                    "frontier_id": option["frontier_id"],
                    "group_dir": str(group_dir),
                    "rgb_path": str(rgb),
                    "scene_id": example["scene_id"],
                    "episode_id": example["episode_id"],
                    "decision_timestep": example["decision_timestep"],
                    "n_turn_actions": n_turns,
                    "n_forward_actions": n_forwards,
                    "n_cross_actions": len(actions),
                    "n_approach_actions": option["n_approach_actions"],
                    "newly_free_cells": revelation["newly_free_cells"],
                    "newly_occupied_cells": revelation["newly_occupied_cells"],
                    "newly_observed_area_m2": revelation["newly_observed_area_m2"],
                    "n_examples_in_group": group["n_examples"],
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("outputs/phase5/full_train"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/full.json"))
    parser.add_argument("--out", type=Path, default=Path("archive/phase9d/smoke_selection.json"))
    args = parser.parse_args()

    pool = development_keys(args.manifest)
    surviving = candidates(args.dataset, pool)
    surviving.sort(key=lambda c: (c["group_id"], c["frontier_id"]))

    if not surviving:
        raise SystemExit("no branch satisfies the predetermined rule")

    payload = {
        "rule": {
            "description": "see module docstring of scripts/select_smoke_branch.py",
            "development_manifest": str(args.manifest),
            "development_pool": "full.json train split (consumed by Phase 8 v0)",
            "sealed_guard": "all paths asserted under hm3d_v0.2/train/; sealed set is HM3D val-split",
            "dataset": str(args.dataset),
            "require_crossed": True,
            "min_turn_actions": MIN_TURNS,
            "min_forward_actions": MIN_FORWARDS,
            "min_newly_free_cells": MIN_NEW_FREE_CELLS,
            "min_newly_occupied_cells": MIN_NEW_OCCUPIED_CELLS,
            "tie_break": "ascending (group_id, frontier_id)",
            "goal_used_in_selection": False,
        },
        "n_candidates_considered": len(surviving),
        "selected": surviving[0],
        "runners_up": surviving[1:4],
        "reason": (
            "First branch in lexical (group_id, frontier_id) order that crosses its "
            "frontier, executes both rotation and translation, and reveals a nontrivial "
            "amount of both free and occupied space. Chosen before any Cosmos "
            "generation was run, so it cannot have been selected for looking easy."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"n_candidates": len(surviving), "selected": surviving[0]}, indent=2))


if __name__ == "__main__":
    main()
