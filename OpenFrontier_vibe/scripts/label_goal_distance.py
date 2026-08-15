#!/usr/bin/env python3
"""Attach ground-truth goal-distance labels to harvested beyond-frontier data.

WHY THIS EXISTS
---------------
The world model's goal-relevance score ranks frontiers at chance. The cause is
the training target. `object_labels` is produced by `_object_pseudo_labels`,
which marks a category present when ANY beyond-frontier frame clears a CLIP
similarity threshold. Over many frames that saturates: on held-out harvest
scenes 46,945 of 51,357 labels are positive (91%), and per-goal AUC of the
trained model against that target is 0.477-0.558, i.e. chance.

"Is there a chair somewhere beyond this frontier" is almost always yes in a
house, so the label cannot distinguish frontiers and no architecture trained on
it can either.

WHAT THIS PRODUCES
------------------
For every harvested frontier, the geodesic distance from the frontier to the
nearest ground-truth instance of each ObjectNav goal category, taken from the
scene's semantic annotations rather than from CLIP. That is a graded,
discriminative target: frontiers differ in it even when every one of them has
the category somewhere beyond.

Writes `goal_distance.npz` beside each `beyond_frontier.npz`:
    goal_geodesic  (N, 6)  metres to nearest instance, inf if none in scene
    goal_relevance (N, 6)  exp(-d / TAU), the training target
    categories     (6,)    category order

usage: label_goal_distance.py --harvest output/wm_harvest_demos --scenes 1 4
"""
import argparse, glob, json, os, sys

import numpy as np

GOALS = ["chair", "sofa", "bed", "toilet", "tv_monitor", "plant"]
# HM3D raw category names that map onto each ObjectNav goal
SYNONYMS = {
    "chair": {"chair", "stool", "armchair"},
    "sofa": {"sofa", "couch", "loveseat", "sectional"},
    "bed": {"bed", "mattress"},
    "toilet": {"toilet"},
    "tv_monitor": {"tv", "tv monitor", "television", "monitor", "tv screen"},
    "plant": {"plant", "potted plant", "houseplant", "flower", "flowerpot"},
}
TAU = 6.0  # metres; relevance = exp(-d/TAU)


def scene_glb_for(scene_name, root):
    hits = glob.glob(f"{root}/**/{scene_name}/*.basis.glb", recursive=True)
    hits += glob.glob(f"{root}/**/*{scene_name}*/*.glb", recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--scene-root", default="/home/ubuntu/work/data/scene_datasets/hm3d")
    ap.add_argument("--scenes", type=int, nargs=2, default=(1, 1),
                    help="shard as: index total (index in 1..total)")
    args = ap.parse_args()

    import habitat_sim

    dirs = sorted(d for d in glob.glob(os.path.join(args.harvest, "*"))
                  if os.path.exists(os.path.join(d, "beyond_frontier.npz")))
    idx, tot = args.scenes
    dirs = [d for i, d in enumerate(dirs) if i % tot == (idx - 1)]
    print(f"labeling {len(dirs)} scenes (shard {idx}/{tot})", flush=True)

    for d in dirs:
        scene = os.path.basename(d)
        out = os.path.join(d, "goal_distance.npz")
        if os.path.exists(out):
            continue
        npz = np.load(os.path.join(d, "beyond_frontier.npz"), allow_pickle=True)
        if "pos3d" not in npz:
            print(f"  {scene}: no pos3d (re-harvest needed), skipping", flush=True)
            continue
        pos = npz["pos3d"]
        glb = scene_glb_for(scene, args.scene_root)
        if glb is None:
            print(f"  {scene}: scene glb not found, skipping", flush=True)
            continue

        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = glb
        cfg.enable_physics = False
        # without the annotated scene-dataset config habitat loads the glb with
        # NO semantic annotations, and every category silently comes back empty
        scene_cfg = os.path.join(args.scene_root,
                                 "hm3d_annotated_basis.scene_dataset_config.json")
        if os.path.exists(scene_cfg):
            cfg.scene_dataset_config_file = scene_cfg
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.AgentConfiguration()]))
        try:
            pf = sim.pathfinder
            # ground-truth instance centroids per goal category
            centroids = {g: [] for g in GOALS}
            for obj in (sim.semantic_scene.objects or []):
                if obj is None or obj.category is None:
                    continue
                name = obj.category.name().lower().strip()
                for g in GOALS:
                    if name in SYNONYMS[g] or any(s in name for s in SYNONYMS[g]):
                        c = np.asarray(obj.aabb.center, dtype=np.float32)
                        if pf.is_navigable(c):
                            centroids[g].append(c)
                        else:
                            snap = pf.snap_point(c)
                            if np.all(np.isfinite(snap)):
                                centroids[g].append(np.asarray(snap, dtype=np.float32))
                        break

            D = np.full((len(pos), len(GOALS)), np.inf, dtype=np.float32)
            for gi, g in enumerate(GOALS):
                if not centroids[g]:
                    continue
                for pi, p in enumerate(pos):
                    sp_start = pf.snap_point(p)
                    if not np.all(np.isfinite(sp_start)):
                        continue
                    best = np.inf
                    for c in centroids[g]:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = sp_start
                        path.requested_end = c
                        if pf.find_path(path) and np.isfinite(path.geodesic_distance):
                            best = min(best, float(path.geodesic_distance))
                    D[pi, gi] = best

            rel = np.exp(-D / TAU).astype(np.float32)
            rel[~np.isfinite(D)] = 0.0
            np.savez_compressed(out, goal_geodesic=D, goal_relevance=rel,
                                categories=np.array(GOALS))
            finite = np.isfinite(D)
            print(f"  {scene}: n={len(pos)} labeled | "
                  f"cats present={sum(1 for g in GOALS if centroids[g])}/6 | "
                  f"median d={np.median(D[finite]) if finite.any() else float('nan'):.1f}m "
                  f"| spread(std within scene)={np.nanstd(np.where(finite, D, np.nan)):.2f}",
                  flush=True)
        finally:
            sim.close()


if __name__ == "__main__":
    main()
