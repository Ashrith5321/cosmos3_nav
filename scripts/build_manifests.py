#!/usr/bin/env python
"""Phase 5.1: build scene-disjoint train/val/test manifests.

Audits every scene for semantic annotations, then partitions scenes (never
episodes) into splits.

    python scripts/build_manifests.py --name pilot_debug
    python scripts/build_manifests.py --name full --splits train,val,test
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontierworld.config import config_hash, load_config  # noqa: E402
from frontierworld.data.manifests import (  # noqa: E402
    SplitManifest,
    count_semantic_instances,
    stable_partition,
)
from frontierworld.evaluation.run_logger import _git  # noqa: E402

PILOT_REASON = (
    "HM3D train scene meshes are not downloaded, so this partition is carved "
    "out of the HM3D val split. Usable for pipeline development only; NOT for "
    "final training or reported evaluation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--name", default="pilot_debug")
    parser.add_argument(
        "--fractions",
        default="train=0.6,val=0.2,test=0.2",
        help="scene fractions per split",
    )
    parser.add_argument(
        "--audit-semantics",
        action="store_true",
        default=True,
        help="load each scene and count semantic instances",
    )
    parser.add_argument("--no-audit-semantics", dest="audit_semantics", action="store_false")
    parser.add_argument("--out", default="manifests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    fractions = {}
    for part in args.fractions.split(","):
        name, value = part.split("=")
        fractions[name.strip()] = float(value)

    import habitat

    from frontierworld.habitat_env import build_habitat_config

    habitat_cfg = build_habitat_config(cfg)
    dataset = habitat.datasets.make_dataset(
        habitat_cfg.habitat.dataset.type, config=habitat_cfg.habitat.dataset
    )

    episodes_per_scene = Counter(str(e.scene_id) for e in dataset.episodes)
    scene_ids = sorted(episodes_per_scene)
    print(f"scenes: {len(scene_ids)}  episodes: {len(dataset.episodes)}")

    source_split = str(cfg.data.split)
    pilot = source_split != "train"

    scenes: dict[str, dict] = {}
    without_semantics: list[str] = []
    for index, scene_id in enumerate(scene_ids, start=1):
        n_instances = -1
        if args.audit_semantics:
            n_instances = count_semantic_instances(
                scene_id, str(cfg.data.scene_dataset_config)
            )
            status = (
                "ok" if n_instances > 0 else ("FAILED TO LOAD" if n_instances < 0 else "NO SEMANTICS")
            )
            print(
                f"  [{index:3d}/{len(scene_ids)}] {Path(scene_id).stem:26s} "
                f"instances={n_instances:5d}  {status}",
                flush=True,
            )
            if n_instances <= 0:
                without_semantics.append(scene_id)

        scenes[scene_id] = {
            "scene_id": scene_id,
            "scene_name": Path(scene_id).stem.replace(".basis", ""),
            "split_source": source_split,
            "n_episodes": episodes_per_scene[scene_id],
            "n_semantic_instances": n_instances,
        }

    # Scenes without semantics cannot support the semantic revelation channel,
    # so they are excluded rather than silently producing empty labels.
    usable = [s for s in scene_ids if scenes[s]["n_semantic_instances"] != 0]
    if len(usable) != len(scene_ids):
        print(f"\nexcluding {len(scene_ids) - len(usable)} scenes without semantics")

    manifest = SplitManifest(
        name=args.name,
        pilot=pilot,
        pilot_reason=PILOT_REASON if pilot else None,
        config_hash=config_hash(cfg),
        git_commit=_git("rev-parse", "HEAD"),
        splits=stable_partition(usable, fractions, seed=int(cfg.seed.value)),
        scenes=scenes,
    )

    out = Path(args.out) / f"{args.name}.json"
    if not out.is_absolute():
        out = Path(cfg.experiment.output_dir).parent / out
    manifest.save(out)

    print()
    print(manifest.summary())
    print(f"\nwritten: {out}")

    overlaps = manifest.check_disjoint()
    if overlaps:
        print("SCENE OVERLAP DETECTED:", overlaps, file=sys.stderr)
        return 1
    if args.audit_semantics:
        with_semantics = sum(
            1 for s in usable if scenes[s]["n_semantic_instances"] > 0
        )
        print(f"scenes with nonzero semantics: {with_semantics}/{len(usable)}")
        if with_semantics < 5:
            print("FEWER THAN 5 SCENES WITH SEMANTICS", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
