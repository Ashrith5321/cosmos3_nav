#!/usr/bin/env python3
"""Counterfactual frontier rollout collector.

WHY
---
The previous world model was trained on p(object somewhere beyond frontier |
frontier crop). That target is 91% positive and semantic guessing, and every
downstream measurement came back at chance. This collects the data for the
action-conditioned formulation instead: execute a real trajectory THROUGH each
candidate frontier from the same simulator state, and record what actually
happened.

The action is the trajectory, not the frontier id.

WHAT IT PRODUCES
----------------
For each decision state, one group containing every candidate frontier's
realized branch, so within-decision counterfactual supervision is available:

  history          (K,  Hh, 3)   pose history before the decision
  cur_occ          (G, G)        occupancy at the decision (shared)
  goal_cat         str           goal category for this episode
  frontier_xyz     (N, 3)        candidate frontier positions
  actions          (N, H, 3)     executed dx,dy,dtheta per branch
  fut_rgb_emb      (N, S, 512)   CLIP embedding of future frames (subsampled)
  fut_occ          (N, S, G, G)  occupancy revealed along the branch
  target_visible   (N,)          did an instance of the goal enter view
  d_goal_delta     (N,)          geodesic distance to nearest goal, start->end
  new_area         (N,)          newly revealed free cells
  collision        (N,)          did the branch stall

Scenes are HM3D train only. Sharded and resumable per scene.

usage: collect_counterfactual.py --out output/cf --scenes 1 8 --states-per-scene 25
"""
import argparse, glob, json, os, sys
from pathlib import Path

import numpy as np

GOALS = ["chair", "sofa", "bed", "toilet", "tv_monitor", "plant"]
SYNONYMS = {
    "chair": {"chair", "stool", "armchair"},
    "sofa": {"sofa", "couch", "loveseat", "sectional"},
    "bed": {"bed", "mattress"},
    "toilet": {"toilet"},
    "tv_monitor": {"tv", "tv monitor", "television", "monitor", "tv screen"},
    "plant": {"plant", "potted plant", "houseplant", "flower", "flowerpot"},
}
GRID = 64          # local occupancy grid side
CELL = 0.25        # metres per cell
HORIZON = 32       # actions per branch
SUB = 4            # save every 4th frame


def make_sim(glb, scene_cfg, hfov=79, w=320, h=240):
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = glb
    cfg.enable_physics = False
    if scene_cfg and os.path.exists(scene_cfg):
        cfg.scene_dataset_config_file = scene_cfg
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid, rgb.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    rgb.resolution, rgb.hfov = [h, w], hfov
    rgb.position = [0.0, 0.88, 0.0]
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid, dep.sensor_type = "depth", habitat_sim.SensorType.DEPTH
    dep.resolution, dep.hfov = [h, w], hfov
    dep.position = [0.0, 0.88, 0.0]
    ag = habitat_sim.AgentConfiguration()
    ag.sensor_specifications = [rgb, dep]
    ag.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=30.0)),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=30.0)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))


def goal_centroids(sim):
    """Navigable centroids of every ground-truth instance, per goal category."""
    pf = sim.pathfinder
    out = {g: [] for g in GOALS}
    for obj in (sim.semantic_scene.objects or []):
        if obj is None or obj.category is None:
            continue
        name = obj.category.name().lower().strip()
        for g in GOALS:
            if name in SYNONYMS[g] or any(s in name for s in SYNONYMS[g]):
                c = np.asarray(obj.aabb.center, dtype=np.float32)
                p = c if pf.is_navigable(c) else pf.snap_point(c)
                if np.all(np.isfinite(p)):
                    out[g].append(np.asarray(p, dtype=np.float32))
                break
    return {g: v for g, v in out.items() if v}


def geo_to_nearest(pf, p, centroids):
    import habitat_sim
    best = np.inf
    s = pf.snap_point(p)
    if not np.all(np.isfinite(s)):
        return best
    for c in centroids:
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = s, c
        if pf.find_path(sp) and np.isfinite(sp.geodesic_distance):
            best = min(best, float(sp.geodesic_distance))
    return best


def depth_to_occ(depth, pose_xyz, yaw, occ, hfov=79.0):
    """Splat a depth frame into an ego-centred occupancy grid (free=1)."""
    h, w = depth.shape
    fx = (w / 2.0) / np.tan(np.deg2rad(hfov) / 2.0)
    row = depth[h // 2]
    for u in range(0, w, 4):
        d = float(row[u])
        if not np.isfinite(d) or d <= 0.1 or d > 8.0:
            continue
        ang = yaw + np.arctan2((u - w / 2.0), fx)
        for t in np.arange(0.3, d, CELL):
            gx = int(GRID // 2 + (t * np.cos(ang)) / CELL)
            gy = int(GRID // 2 + (t * np.sin(ang)) / CELL)
            if 0 <= gx < GRID and 0 <= gy < GRID:
                occ[gy, gx] = 1.0
    return occ


def yaw_of(state):
    q = state.rotation
    return float(2.0 * np.arctan2(q.y, q.w))


def rollout(sim, follower_target, steps, encoder, centroids, pf):
    """Execute toward a target, recording observations and consequences."""
    import habitat_sim
    ag = sim.get_agent(0)
    start = np.asarray(ag.get_state().position, dtype=np.float32)
    occ = np.zeros((GRID, GRID), dtype=np.float32)
    frames, acts = [], []
    stuck = 0
    from habitat_sim.nav import GreedyGeodesicFollower
    fol = GreedyGeodesicFollower(pf, ag, goal_radius=0.4,
                                 forward_key="move_forward",
                                 left_key="turn_left", right_key="turn_right")
    prev = np.asarray(ag.get_state().position, dtype=np.float32)
    for t in range(steps):
        try:
            a = fol.next_action_along(follower_target)
        except Exception:
            a = None
        if a is None:
            break
        obs = sim.step(a)
        st = ag.get_state()
        pos = np.asarray(st.position, dtype=np.float32)
        depth_to_occ(obs["depth"], pos, yaw_of(st), occ)
        if np.linalg.norm(pos - prev) < 1e-3:
            stuck += 1
        prev = pos
        acts.append([pos[0], pos[2], yaw_of(st)])
        if t % SUB == 0:
            frames.append(obs["rgb"][..., :3].copy())
    end = np.asarray(ag.get_state().position, dtype=np.float32)
    embs = encoder(frames) if frames else np.zeros((0, 512), np.float32)
    return {
        "occ": occ,
        "acts": np.asarray(acts, dtype=np.float32),
        "emb": embs,
        "stuck": stuck > steps // 2,
        "start": start,
        "end": end,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-root", required=True)
    ap.add_argument("--scenes", type=int, nargs=2, default=(1, 1))
    ap.add_argument("--states-per-scene", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()

    import torch
    from transformers import CLIPModel, CLIPProcessor
    import PIL.Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def encode(frames):
        if not frames:
            return np.zeros((0, 512), np.float32)
        ims = [PIL.Image.fromarray(f) for f in frames]
        with torch.no_grad():
            i = proc(images=ims, return_tensors="pt").to(dev)
            f = clip.get_image_features(**i)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    root = Path(args.scene_root)
    scene_cfg = str(root / "hm3d_annotated_basis.scene_dataset_config.json")
    dirs = sorted(d for d in (root / "train").iterdir() if d.is_dir()
                  and list(d.glob("*.semantic.glb")))
    i, n = args.scenes
    dirs = [d for k, d in enumerate(dirs) if k % n == (i - 1)]
    os.makedirs(args.out, exist_ok=True)
    print(f"scenes: {len(dirs)} (shard {i}/{n})", flush=True)

    rng = np.random.default_rng(1000 + i)
    for sd in dirs:
        name = sd.name
        outp = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(outp):
            continue
        glbs = list(sd.glob("*.basis.glb"))
        if not glbs:
            continue
        try:
            sim = make_sim(str(glbs[0]), scene_cfg)
        except Exception as e:
            print(f"  {name}: sim failed ({e})", flush=True)
            continue
        try:
            pf = sim.pathfinder
            cents = goal_centroids(sim)
            if not cents:
                print(f"  {name}: no goal instances", flush=True)
                continue
            groups = []
            rej = {'cands': 0, 'd0': 0, 'p0': 0}
            for _ in range(args.states_per_scene):
                goal = str(rng.choice(list(cents.keys())))
                p0 = pf.get_random_navigable_point()
                if not np.all(np.isfinite(p0)):
                    rej['p0'] += 1
                    continue
                # candidate "frontiers": navigable points 2-5 m away, spread in
                # bearing, standing in for the planner's frontier set
                cands = []
                for _try in range(300):
                    q = pf.get_random_navigable_point()
                    d = float(np.linalg.norm(q - p0))
                    if 1.5 < d < 7.0:
                        if all(np.linalg.norm(q - c) > 1.5 for c in cands):
                            cands.append(np.asarray(q, dtype=np.float32))
                    if len(cands) >= 4:
                        break
                if len(cands) < 2:
                    rej['cands'] += 1
                    continue
                d0 = geo_to_nearest(pf, p0, cents[goal])
                if not np.isfinite(d0):
                    rej['d0'] += 1
                    continue
                branches = []
                for c in cands:
                    st = sim.get_agent(0).get_state()
                    st.position = p0
                    yaw = rng.uniform(0, 2 * np.pi)
                    st.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
                    st.sensor_states = {}
                    sim.get_agent(0).set_state(st)
                    r = rollout(sim, c, args.horizon, encode, cents[goal], pf)
                    d1 = geo_to_nearest(pf, r["end"], cents[goal])
                    r["d_delta"] = float(d0 - d1) if np.isfinite(d1) else 0.0
                    r["new_area"] = float(r["occ"].sum())
                    branches.append(r)
                for b in branches:
                    a = b["acts"]
                    padded = np.zeros((HORIZON, 3), dtype=np.float32)
                    if len(a):
                        padded[:min(len(a), HORIZON)] = a[:HORIZON]
                    b["acts_pad"] = padded
                    b["n_act"] = min(len(a), HORIZON)
                if len(branches) >= 2:
                    groups.append({"goal": goal, "p0": p0,
                                   "cands": np.stack(cands), "br": branches})
            if groups:
                np.savez_compressed(
                    outp,
                    goal=np.array([g["goal"] for g in groups]),
                    p0=np.stack([g["p0"] for g in groups]),
                    n_branch=np.array([len(g["br"]) for g in groups]),
                    cands=np.concatenate([g["cands"] for g in groups]),
                    d_delta=np.concatenate(
                        [[b["d_delta"] for b in g["br"]] for g in groups]),
                    new_area=np.concatenate(
                        [[b["new_area"] for b in g["br"]] for g in groups]),
                    stuck=np.concatenate(
                        [[b["stuck"] for b in g["br"]] for g in groups]),
                    occ=np.concatenate(
                        [[b["occ"] for b in g["br"]] for g in groups]),
                    acts=np.concatenate(
                        [[b["acts_pad"] for b in g["br"]] for g in groups]),
                    n_act=np.concatenate(
                        [[b["n_act"] for b in g["br"]] for g in groups]),
                    emb_mean=np.concatenate(
                        [[b["emb"].mean(0) if len(b["emb"]) else np.zeros(512, np.float32)
                          for b in g["br"]] for g in groups]),
                )
                nb = sum(len(g["br"]) for g in groups)
                dd = np.concatenate([[b["d_delta"] for b in g["br"]] for g in groups])
                print(f"  {name}: {len(groups)} decisions, {nb} branches, "
                      f"d_delta spread={dd.std():.2f}m", flush=True)
            else:
                print(f"  {name}: no usable decisions  rejected={rej}", flush=True)
        finally:
            sim.close()


if __name__ == "__main__":
    main()
