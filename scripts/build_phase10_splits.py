"""Phase 10.1 — freeze scene-disjoint calibration and validation manifests.

Calibration fits temperatures, Platt parameters and the conformal quantile;
validation is where every reported number comes from. They must be
scene-disjoint, not group-disjoint: two decision states in the same scene share
geometry, so a scene straddling the split would leak calibration information
into the evaluation.

Source is `dev_pool_v1.json` -- 103 HM3D train-split scenes that no dataset,
model or diagnostic has ever touched. Three guards run before anything is
written:

  1. no overlap with `archive/phase8_v0/consumed_scenes.json`
  2. no overlap between calibration and validation
  3. every path under `hm3d_v0.2/train/`, which keeps the sealed set out
     without reading it (sealed_final_v1 is drawn from HM3D **val**-split)

The split is a deterministic seeded shuffle of the sorted scene keys, so it is
reproducible and cannot have been chosen after seeing any result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

HM3D_TRAIN_MARKER = "hm3d_v0.2/train/"


def scene_key(scene_id: str) -> str:
    return Path(str(scene_id)).name.replace(".basis.glb", "").replace(".glb", "")


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()


def manifest(name: str, scenes: list[str], source: dict, reason: str) -> dict:
    payload = {
        "name": name,
        "schema_version": "frontierreveal-1",
        "pilot": False,
        "pilot_reason": None,
        "derived_from": "dev_pool_v1.json",
        "selection": reason,
        "git_commit": git_commit(),
        "splits": {"pool": sorted(scenes)},
        "scenes": {s: source["scenes"][s] for s in sorted(scenes) if s in source.get("scenes", {})},
    }
    payload["scene_set_hash"] = hashlib.sha256(
        json.dumps(sorted(scene_key(s) for s in scenes)).encode()
    ).hexdigest()[:16]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("manifests/dev_pool_v1.json"))
    parser.add_argument("--consumed", type=Path, default=Path("archive/phase8_v0/consumed_scenes.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--n-calibration", type=int, default=45)
    parser.add_argument("--n-validation", type=int, default=45)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    pool = json.loads(args.pool.read_text())
    scenes = sorted({s for v in pool["splits"].values() for s in v})

    for scene in scenes:
        if HM3D_TRAIN_MARKER not in str(scene):
            raise SystemExit(
                f"refusing: {scene!r} is not under {HM3D_TRAIN_MARKER}. The sealed set is "
                "HM3D val-split, and this guard is what keeps it out without reading it."
            )

    consumed = set(json.loads(args.consumed.read_text()))
    contaminated = sorted(s for s in scenes if scene_key(s) in consumed)
    if contaminated:
        raise SystemExit(f"refusing: {len(contaminated)} pool scenes already consumed: {contaminated[:5]}")

    if args.n_calibration + args.n_validation > len(scenes):
        raise SystemExit(
            f"requested {args.n_calibration}+{args.n_validation} scenes but the pool has {len(scenes)}"
        )

    # Deterministic seeded shuffle of the sorted keys.
    import random

    order = list(scenes)
    random.Random(args.seed).shuffle(order)
    calibration = order[: args.n_calibration]
    validation = order[args.n_calibration : args.n_calibration + args.n_validation]
    reserve = order[args.n_calibration + args.n_validation :]

    overlap = {scene_key(s) for s in calibration} & {scene_key(s) for s in validation}
    if overlap:
        raise SystemExit(f"refusing: calibration and validation share scenes: {sorted(overlap)}")

    written = []
    for name, subset, reason in [
        ("phase10_calibration", calibration,
         f"deterministic seeded shuffle (seed={args.seed}) of sorted dev_pool_v1 scene ids; first {args.n_calibration}"),
        ("phase10_validation", validation,
         f"deterministic seeded shuffle (seed={args.seed}) of sorted dev_pool_v1 scene ids; next {args.n_validation}"),
    ]:
        path = args.out_dir / f"{name}.json"
        if path.exists():
            raise SystemExit(f"refusing to overwrite frozen manifest {path}")
        path.write_text(json.dumps(manifest(name, subset, pool, reason), indent=2))
        written.append(path)

    summary = {
        "pool_scenes": len(scenes),
        "n_calibration": len(calibration),
        "n_validation": len(validation),
        "n_reserve": len(reserve),
        "reserve_note": "held for Phase 11; not used in Phase 10",
        "scene_disjoint": True,
        "overlap_with_consumed": 0,
        "sealed_guard": "all paths under hm3d_v0.2/train/; sealed_final_v1 is HM3D val-split and was not read",
        "shuffle_seed": args.seed,
        "calibration_hash": hashlib.sha256(
            json.dumps(sorted(scene_key(s) for s in calibration)).encode()
        ).hexdigest()[:16],
        "validation_hash": hashlib.sha256(
            json.dumps(sorted(scene_key(s) for s in validation)).encode()
        ).hexdigest()[:16],
        "reserve_scenes": sorted(scene_key(s) for s in reserve),
        "written": [str(p) for p in written],
    }
    (args.out_dir / "phase10_split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
