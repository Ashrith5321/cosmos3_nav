"""Closed-loop HM3D v2 eval: Cosmos3-Nano with multi-floor global map,
trajectory, and frontiers (geometric + FrontierNet).

Same protocol as the h16 baseline (16-frame history, max 500 steps) with one
change: each step also sends the rendered exploration map + frontier text.
Requires both servers:
    .venv/bin/python eval/cosmos3_server.py       (port 8399)
    .venv/bin/python src/frontiernet_server.py    (port 12186; optional)

Run in the habitat env:
  ~/miniconda3/envs/habitat033/bin/python eval/run_habitat_map_eval.py --limit 1000
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

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
SERVER = "http://127.0.0.1:8399"
FRONTIERNET_URL = "http://localhost:12186/frontiernet"
ACTION_IDS = {"stop": 0, "forward": 1, "left": 2, "right": 3}
DEPTH_MAX = 5.0
FN_COLOR = (255, 0, 255)
FN_EVERY = 3          # query FrontierNet every N steps
MIN_FRONTIER_CELLS = 25


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
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
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
    if len(lst) > 4000:
        del lst[:len(lst) - 4000]


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
    past_actions = []
    steps, stopped = 0, False
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
        map_bgr = gm.render(fronts, agent_pose=pose,
                            extra_points=fn_points.get(gm.floor_index))
        ftext = gm.frontier_text(pose, fronts)

        resp = requests.post(f"{SERVER}/act", json={
            "goal": goal,
            "images": frames[-history:],
            "past_actions": past_actions,
            "map_image": b64_jpeg_arr(cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)),
            "frontier_text": ftext,
        }, timeout=180)
        resp.raise_for_status()
        action = resp.json()["action"]
        past_actions.append(action)

        obs = env.step(ACTION_IDS[action])
        steps += 1
        frames.append(b64_jpeg_arr(obs["rgb"]))
        if action == "stop":
            stopped = True
            break

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
    # final map snapshot for post-hoc analysis
    cv2.imwrite(str(video_dir / f"{stem}_map.png"), map_bgr)

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
        "video": str(video_path),
    }
    with open(out_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{index:02d}] goal={goal:<10} steps={steps:3d} "
          f"success={record['success']} spl={record['spl']:.2f} "
          f"dtg={record['distance_to_goal']:.2f}m floors={record['floors_visited']} "
          f"({record['elapsed_seconds']}s)", flush=True)
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
                   default="/home/ashed/Documents/cosmos3_nav/eval/habitat_1000_map_out")
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
          f"max_steps={args.max_steps}, history={args.history}, map=ON", flush=True)

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
