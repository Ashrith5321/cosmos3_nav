"""Headless: spawn in a (random) HM3D val scene, build the GlobalMap by looking
around, detect frontiers, and imagine walking through each one via the Cosmos3
generator server(s). Frontiers are split across all --gen-ports so multiple GPUs
imagine in parallel. Videos + last-frames are written by the generator server(s)
to GENERATOR_SAVE_DIR; this driver writes the conditioning keyframes, the map,
and a manifest.

Run (habitat env), with generator server(s) already up:
    ~/miniconda3/envs/habitat033/bin/python src/imagine_spawn_headless.py \
        --gen-ports 8402,8403 --out eval/imagine_spawn_out
"""
import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

SPATIAL_ROOT = Path("/home/ashed/Documents/spatial_training")
os.chdir(SPATIAL_ROOT)
sys.path.insert(0, str(SPATIAL_ROOT / "src"))
sys.path.insert(0, "/home/ashed/Documents/cosmos3_nav/src")

import habitat  # noqa: E402
from habitat.config import read_write  # noqa: E402
from habitat.config.default import get_config  # noqa: E402
import longnav.utils.measures  # noqa: F401,E402
import longnav.utils.ovon.ovon_dataset  # noqa: F401,E402
import longnav.utils.ovon.ovon_nav  # noqa: F401,E402
from global_map import MultiFloorMap, pose_from_habitat_state  # noqa: E402

DEPTH_MAX = 5.0
FWD, LEFT = 1, 2   # habitat action ids
FKEY_GRID_M = 0.75

VAL_SCENES = ['4ok3usBNeis', '5cdEh9F2hJL', '6s7QHgap2fW', '7MXmsvcQjpJ', 'BAbdmeyTvMZ',
              'CrMo8WxCyVb', 'DYehNKdT76V', 'Dd4bFSTQ8gi', 'GLAQ4DNUx5U', 'HY1NcmCgn3n',
              'LT9Jq6dN3Ea', 'MHPLjHsuG27', 'Nfvxx8J5NCo', 'QaLdnwvtxbs', 'TEEsavR23oF',
              'VBzV5z6i1WS', 'XB4GS9ShBRE', 'a8BtkwhxdRV', 'bCPU9suPUw9', 'bxsVRursffK',
              'cvZr5TUy5C5', 'eF36g7L6Z9M', 'h1zeeAwLh9Z', 'k1cupFYWXJ6', 'mL8ThkuaVTM',
              'mv2HUxq3B53', 'p53SfW6mjZe', 'q3zU7Yy5E5s', 'q5QZSEeHe5g', 'qyAac8rV8Zk',
              'svBbv1Pavdk', 'wcojb4TFT35', 'y9hTuugGdiq', 'yr17PDCnDDW', 'ziup5kvtCCR',
              'zt1RVoi7PcG']


def build_config(scene, gpu):
    config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.data_path = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
        config.habitat.dataset.split = "val"
        config.habitat.dataset.content_scenes = [scene]
        ag = config.habitat.simulator.agents.main_agent
        for s in (ag.sim_sensors.rgb_sensor, ag.sim_sensors.depth_sensor):
            s.width, s.height, s.hfov = 640, 480, 79
            s.position = [0, 0.88, 0]
        ag.sim_sensors.depth_sensor.max_depth = DEPTH_MAX
        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu
        config.habitat.environment.max_episode_steps = 5000
    return config


def fkey(f):
    cx, cz = f["centroid_world"]
    return f"{round(cx / FKEY_GRID_M) * FKEY_GRID_M:.2f},{round(cz / FKEY_GRID_M) * FKEY_GRID_M:.2f}"


def best_keyframe(f, keyframes, cur_rgb):
    """Logged frame whose heading best faces this frontier."""
    cx, cz = f["centroid_world"]
    best, best_ang = cur_rgb, 1e9
    for pose, rgb in keyframes:
        R, t = pose[:3, :3], pose[:3, 3]
        dx, dz = cx - t[0], cz - t[2]
        if np.hypot(dx, dz) < 0.4:
            continue
        fwd = R @ np.array([0.0, 0.0, -1.0])
        ang = abs(np.degrees((np.arctan2(dx, -dz) - np.arctan2(fwd[0], -fwd[2])
                              + np.pi) % (2 * np.pi) - np.pi))
        if ang < best_ang:
            best, best_ang = rgb, ang
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="scene id (default: random val scene)")
    p.add_argument("--episode", type=int, default=None, help="episode index (default: random)")
    p.add_argument("--gpu", type=int, default=1, help="habitat render GPU")
    p.add_argument("--gen-ports", default="8402,8403", help="generator server ports")
    p.add_argument("--min-cells", type=int, default=15, help="min frontier cluster cells")
    p.add_argument("--max-frontiers", type=int, default=8)
    p.add_argument("--look-loops", type=int, default=2, help="360-deg look-around passes")
    p.add_argument("--forward-steps", type=int, default=4, help="forward steps between passes")
    p.add_argument("--out", default="/home/ashed/Documents/cosmos3_nav/eval/imagine_spawn_out")
    p.add_argument("--timeout", type=int, default=1800, help="secs to wait for imaginations")
    args = p.parse_args()

    rng = random.Random()
    scene = args.scene or rng.choice(VAL_SCENES)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ports = [int(x) for x in args.gen_ports.split(",")]
    print(f"[spawn] scene={scene} gpu={args.gpu} gen-ports={ports}", flush=True)

    env = habitat.Env(config=build_config(scene, args.gpu))
    n_ep = len(env.episodes)
    ep_idx = args.episode if args.episode is not None else rng.randrange(n_ep)
    for _ in range(ep_idx + 1):
        obs = env.reset()
    ep = env.current_episode
    goal = str(ep.object_category)
    print(f"[spawn] episode {ep.episode_id} goal={goal} ({n_ep} eps in scene)", flush=True)

    gm = MultiFloorMap()
    keyframes = []

    def integrate(obs):
        pose = pose_from_habitat_state(env.sim.get_agent(0).get_state())
        gm.update(obs["depth"], pose, rgb=obs["rgb"])
        keyframes.append((pose.copy(), np.asarray(obs["rgb"]).copy()))
        return pose

    # look around: rotate 360 (12x30deg), step forward, repeat -> reveals frontiers
    integrate(obs)
    for loop in range(args.look_loops):
        for _ in range(12):
            obs = env.step(LEFT); integrate(obs)
        for _ in range(args.forward_steps):
            if env.episode_over:
                break
            obs = env.step(FWD); integrate(obs)
        print(f"[look] pass {loop+1}/{args.look_loops} done, {len(keyframes)} frames", flush=True)

    fronts = gm.frontiers(min_cluster_cells=args.min_cells)
    fronts = sorted(fronts, key=lambda f: -f.get("size", len(f.get("cells", [[]])[0])))[:args.max_frontiers]
    print(f"[frontiers] {len(fronts)} frontiers detected", flush=True)
    if not fronts:
        print("[frontiers] none found; try lowering --min-cells or more look-loops", flush=True)
        return

    # save the map with frontiers + spawn FPV
    try:
        cv2.imwrite(str(out / "map.png"), gm.render(fronts))
    except Exception as e:
        print(f"[map] render failed: {e}", flush=True)
    cv2.imwrite(str(out / "spawn_fpv.png"), cv2.cvtColor(np.asarray(keyframes[0][1]), cv2.COLOR_RGB2BGR))

    # enqueue each frontier to a generator (round-robin across GPUs), save keyframe
    manifest = {"scene": scene, "episode": str(ep.episode_id), "goal": goal, "frontiers": []}
    keymap = {}  # key -> port
    for i, f in enumerate(fronts):
        label = chr(ord("A") + i)
        key = fkey(f)
        rgb = best_keyframe(f, keyframes, np.asarray(obs["rgb"]))
        cv2.imwrite(str(out / f"kf_{label}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        prompt = ("First-person indoor home walkthrough. Move forward through the opening "
                  f"ahead into the room beyond, exploring to look for a {goal}.")
        port = ports[i % len(ports)]
        keymap[key] = port
        try:
            requests.post(f"http://127.0.0.1:{port}/enqueue",
                          json={"frontiers": [{"key": key, "prompt": prompt,
                                               "image_jpg": base64.b64encode(buf).decode()}]},
                          timeout=15).raise_for_status()
        except Exception as e:
            print(f"[enqueue] frontier {label} -> :{port} FAILED: {e}", flush=True)
        manifest["frontiers"].append({"label": label, "key": key, "port": port,
                                      "size": int(f.get("size", 0)),
                                      "centroid_world": list(f["centroid_world"]),
                                      "keyframe": f"kf_{label}.png"})
        print(f"[enqueue] {label} key={key} -> GPU port {port}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # poll until every frontier is done/errored or timeout
    want = set(keymap)
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        done = {}
        for port in ports:
            try:
                res = requests.get(f"http://127.0.0.1:{port}/results", timeout=10).json()
                for k, v in res.items():
                    if k in want:
                        done[k] = v.get("status")
            except Exception:
                pass
        ndone = sum(1 for k in want if done.get(k) in ("done", "error"))
        print(f"[imagine] {ndone}/{len(want)} finished "
              f"({sum(1 for k in want if done.get(k)=='running')} running)", flush=True)
        if ndone >= len(want):
            break
        time.sleep(10)

    # final report: collect mp4 paths + timings
    print("\n=== IMAGINATION RESULTS ===", flush=True)
    for port in ports:
        try:
            res = requests.get(f"http://127.0.0.1:{port}/results", timeout=10).json()
        except Exception:
            continue
        for m in manifest["frontiers"]:
            v = res.get(m["key"])
            if v and m["port"] == port:
                st = v.get("status")
                extra = f"{v.get('seconds','?')}s -> {v.get('mp4')}" if st == "done" else v.get("error", "")
                print(f"  {m['label']} [{m['key']}] GPU:{port}: {st}  {extra}", flush=True)
    print(f"\nOutputs in: {out}", flush=True)


if __name__ == "__main__":
    main()
