"""Closed-loop HM3D v2 eval — PLANNING variant.

Architecture change vs run_habitat_map_eval.py:
  map + FrontierNet frontiers are the same, but Cosmos3 no longer emits a
  low-level action every step. Instead:
    1. Cosmos3 SELECTS which frontier to head toward (POST /select_frontier), or STOP.
    2. An external A* planner (src/map_planner.py) plans a path over the agent's
       *built* occupancy map to that frontier and returns the next discrete action.
    3. The planner drives until the frontier is reached / vanishes / a step cap
       hits, then Cosmos3 is queried again (sparse VLM calls).

Same results schema as the map eval, so numbers are matched-comparable.
Requires:  eval/cosmos3_server.py (:8399)  and  src/frontiernet_server.py (:12186, optional)

Run:  ~/miniconda3/envs/habitat033/bin/python eval/run_habitat_planning_eval.py --limit 1000
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SPATIAL_ROOT = Path("/home/ashed/Documents/spatial_training")
os.chdir(SPATIAL_ROOT)
sys.path.insert(0, str(SPATIAL_ROOT / "src"))

import habitat  # noqa: E402
from habitat.config import read_write  # noqa: E402
from habitat.config.default import get_config  # noqa: E402
import longnav.utils.measures  # noqa: F401, E402
import longnav.utils.ovon.ovon_dataset  # noqa: F401, E402
import longnav.utils.ovon.ovon_nav  # noqa: F401, E402

from global_map import MultiFloorMap, pose_from_habitat_state  # noqa: E402
from map_planner import plan_action  # noqa: E402

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
SERVER = os.environ.get("COSMOS3_SERVER", "http://127.0.0.1:8399")
FRONTIERNET_URL = os.environ.get("FRONTIERNET_URL", "http://localhost:12186/frontiernet")
ACTION_IDS = {"stop": 0, "forward": 1, "left": 2, "right": 3}
DEPTH_MAX = 5.0
FN_COLOR = (255, 0, 255)
FN_EVERY = 3
MIN_FRONTIER_CELLS = 25
REACH_M = 0.6              # frontier considered reached within this distance
MAX_SUBGOAL_STEPS = 60     # give up on a frontier after this many steps, re-pick


def build_config(scene):
    config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.data_path = DATASET_PATH
        config.habitat.dataset.split = "val"
        if scene:
            config.habitat.dataset.content_scenes = scene.split(",")
        agent = config.habitat.simulator.agents.main_agent
        for s in (agent.sim_sensors.rgb_sensor, agent.sim_sensors.depth_sensor):
            s.width, s.height, s.hfov = 640, 480, 79
            s.position = [0, 0.88, 0]
        agent.sim_sensors.depth_sensor.max_depth = DEPTH_MAX
        agent.height = 0.88
        agent.radius = 0.18
        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(os.environ.get("HABITAT_GPU", "1"))
        config.habitat.simulator.habitat_sim_v0.allow_sliding = True
        config.habitat.environment.max_episode_steps = 500
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.task.measurements.success.success_distance = 1.0
    return config


def b64_jpeg_arr(rgb_array, quality=85):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


class FrontierNetClient:
    def __init__(self):
        self.available = True
        self.warned = False

    def query(self, rgb, depth_m):
        if not self.available:
            return None
        try:
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
            ok, dbuf = cv2.imencode(".png", depth_mm)
            r = requests.post(FRONTIERNET_URL, json={
                "rgb_jpg": b64_jpeg_arr(rgb),
                "depth_png": base64.b64encode(dbuf).decode(),
            }, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if not self.warned:
                print(f"FrontierNet unavailable ({e}); continuing without learned frontiers",
                      flush=True)
                self.warned = True
            self.available = False
            return None


def project_learned(out, depth_m, pose, gm, fn_points):
    mask = np.array(out["ft_region"], np.uint8)
    gain = np.array(out["info_gain"], np.float32)
    s, (offx, offy) = out["scale"], out["offset"]
    H, W = depth_m.shape
    x0, y0 = int(round(offx / s)), int(round(offy / s))
    x1 = min(W, int(round((offx + mask.shape[1]) / s)))
    y1 = min(H, int(round((offy + mask.shape[0]) / s)))
    up = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    ug = cv2.resize(gain, (x1 - x0, y1 - y0))
    ov = np.zeros((H, W), np.float32)
    ov[y0:y1, x0:x1] = up * np.clip(ug, 0.2, 1.0)
    ys, xs = np.nonzero(ov > 0)
    if not len(ys):
        return
    sel = slice(None, None, max(1, len(ys) // 300))
    ys, xs = ys[sel], xs[sel]
    d = depth_m[ys, xs]
    ok = (d > 0.3) & (d < DEPTH_MAX - 1e-2)
    ys, xs, d = ys[ok], xs[ok], d[ok]
    pts = gm.active._rays[ys, xs] * d[:, None] @ pose[:3, :3].T + pose[:3, 3]
    lst = fn_points.setdefault(gm.floor_index, [])
    lst.extend((float(p[0]), float(p[2]), FN_COLOR) for p in pts)


def run_episode(env, index, max_steps, out_dir, fn_client, history=16, done=frozenset()):
    obs = env.reset()
    episode = env.current_episode
    key = (Path(episode.scene_id).name.replace(".basis.glb", ""), str(episode.episode_id))
    if key in done:
        print(f"[{index:02d}] skip (already ran): {key[0]}/{key[1]}", flush=True)
        return None
    goal = str(episode.object_category)

    gm = MultiFloorMap()
    fn_points = {}
    frames = [b64_jpeg_arr(obs["rgb"])]
    steps, stopped = 0, False
    n_select = 0                     # how many times Cosmos was queried
    n_greedy = 0                     # steps driven by greedy fallback (no A* path yet)
    subgoal = None                   # (label, (cx, cz))
    subgoal_steps = 0
    no_front_turns = 0
    started = time.time()

    while steps < max_steps and not env.episode_over:
        pose = pose_from_habitat_state(env.sim.get_agent(0).get_state())
        depth_m = np.asarray(obs["depth"], np.float32).squeeze() * DEPTH_MAX
        gm.update(obs["depth"], pose, rgb=obs["rgb"])
        if steps % FN_EVERY == 0:
            out = fn_client.query(obs["rgb"], depth_m)
            if out is not None:
                project_learned(out, depth_m, pose, gm, fn_points)

        fronts = gm.frontiers(min_cluster_cells=MIN_FRONTIER_CELLS)

        # no frontiers yet -> rotate in place to reveal them (bounded)
        if not fronts:
            no_front_turns += 1
            if no_front_turns > 14:          # ~full look-around, still nothing
                break
            obs = env.step(ACTION_IDS["left"]); steps += 1
            frames.append(b64_jpeg_arr(obs["rgb"]))
            continue
        no_front_turns = 0

        # decide whether to (re)query Cosmos for a frontier choice.
        # subgoal is a FIXED world target -> stay committed (sticky) until we
        # actually reach it or hit the per-subgoal step cap. This keeps VLM
        # calls sparse; the planner does the driving in between.
        agent_xz = (float(pose[0, 3]), float(pose[2, 3]))
        need_new = subgoal is None
        if subgoal is not None:
            dist_sub = np.hypot(subgoal[1][0] - agent_xz[0], subgoal[1][1] - agent_xz[1])
            if dist_sub < REACH_M or subgoal_steps >= MAX_SUBGOAL_STEPS:
                need_new = True

        if need_new:
            labels = [chr(ord("A") + i) for i in range(len(fronts))]
            map_bgr = gm.render(fronts, agent_pose=pose,
                                extra_points=fn_points.get(gm.floor_index))
            ftext = gm.frontier_text(pose, fronts)
            resp = requests.post(f"{SERVER}/select_frontier", json={
                "goal": goal,
                "images": frames[-history:],
                "map_image": b64_jpeg_arr(cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)),
                "frontier_text": ftext,
                "labels": labels,
            }, timeout=180)
            resp.raise_for_status()
            choice = resp.json()["choice"]
            n_select += 1
            if choice == "STOP":
                obs = env.step(ACTION_IDS["stop"]); steps += 1
                frames.append(b64_jpeg_arr(obs["rgb"]))
                stopped = True
                break
            idx = ord(choice) - ord("A")
            if not (0 <= idx < len(fronts)):
                idx = 0
            subgoal = (labels[idx], tuple(fronts[idx]["centroid_world"]))
            subgoal_steps = 0
            plan_fail = 0

        # plan on the built map toward the chosen frontier (always returns an
        # action; greedy fallback drives toward it when free space can't connect yet)
        plan = plan_action(gm, pose, subgoal[1])
        if not plan["planned"]:
            n_greedy += 1
        action = plan["action"]

        obs = env.step(ACTION_IDS[action])
        steps += 1
        subgoal_steps += 1
        frames.append(b64_jpeg_arr(obs["rgb"]))

    metrics = env.get_metrics()

    video_dir = out_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    clean = lambda v: "".join(c if c.isalnum() or c in "-_" else "_" for c in str(v))
    stem = (f"{index:04d}_{clean(key[0])}_{clean(episode.episode_id)}_{clean(goal)}"
            f"_{'succ' if metrics.get('success') else 'fail'}")
    video_path = video_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
    for b in frames:
        img = np.array(Image.open(io.BytesIO(base64.b64decode(b))))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()

    record = {
        "episode_index": index,
        "scene": key[0],
        "episode_id": key[1],
        "goal": goal,
        "geodesic_distance": episode.info.get("geodesic_distance", -1),
        "steps": steps,
        "stopped": stopped,
        "success": metrics.get("success"),
        "spl": metrics.get("spl"),
        "soft_spl": metrics.get("soft_spl"),
        "distance_to_goal": metrics.get("distance_to_goal"),
        "elapsed_seconds": round(time.time() - started, 1),
        "floors_visited": len(gm.main_floors()),
        "n_frontier_selections": n_select,
        "n_greedy_steps": n_greedy,
        "video": str(video_path),
    }
    with open(out_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{index:02d}] goal={goal:<10} steps={steps:3d} selects={n_select:2d} "
          f"success={record['success']} spl={record['spl']:.2f} "
          f"dtg={record['distance_to_goal']:.2f}m ({record['elapsed_seconds']}s)", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None,
                   help="Scene name or comma-separated list (default: all val scenes)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-results", default=None)
    p.add_argument("--output-dir",
                   default="/home/ashed/Documents/cosmos3_nav/eval/habitat_planning_out")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "driver.lock"
    if lock.exists():
        pid = lock.read_text().strip()
        if pid and Path(f"/proc/{pid}").exists():
            sys.exit(f"Another driver (pid {pid}) is writing to {out_dir}; refusing to start.")
    lock.write_text(str(os.getpid()))

    done = set()
    if args.skip_results:
        for line in open(args.skip_results):
            r = json.loads(line)
            done.add((r["scene"], r["episode_id"]))
        print(f"Skipping {len(done)} episodes from {args.skip_results}", flush=True)

    requests.get(f"{SERVER}/health", timeout=10).raise_for_status()
    print("Cosmos3 server is up.", flush=True)
    fn_client = FrontierNetClient()

    env = habitat.Env(config=build_config(args.scene))
    n = len(env.episodes) if args.limit is None else min(args.limit, len(env.episodes))
    print(f"Scenes: {args.scene or 'all val'}: running first {n}/{len(env.episodes)} episodes, "
          f"max_steps={args.max_steps}, history={args.history}, PLANNING=ON", flush=True)

    records = []
    try:
        for i in range(n):
            try:
                rec = run_episode(env, i, args.max_steps, out_dir, fn_client,
                                  history=args.history, done=done)
                if rec is not None:
                    records.append(rec)
            except requests.RequestException as e:
                print(f"[{i:02d}] ABORT: model server unreachable ({e})", flush=True)
                break
    finally:
        lock.unlink(missing_ok=True)

    if records:
        agg = {k: float(np.mean([r[k] for r in records]))
               for k in ["success", "spl", "soft_spl", "distance_to_goal", "steps"]}
        agg["episodes"] = len(records)
        print("\n=== AGGREGATE ===\n" + json.dumps(agg, indent=2), flush=True)
        (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    env.close()


if __name__ == "__main__":
    main()
