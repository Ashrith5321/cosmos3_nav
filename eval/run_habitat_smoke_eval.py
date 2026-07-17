"""Closed-loop smoke eval: Cosmos3-Nano reasoner on HM3D v2 val, one scene.

Adapted from spatial_training/run_longnav_eval.py, with the in-process VLM
replaced by HTTP calls to eval/cosmos3_server.py. Runs all episodes of one
scene and logs habitat metrics (success / SPL / distance_to_goal) to JSONL.

Run in the habitat env, from anywhere:
  ~/miniconda3/envs/habitat033/bin/python eval/run_habitat_smoke_eval.py
"""
import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

SPATIAL_ROOT = Path("/home/ashed/Documents/spatial_training")
os.chdir(SPATIAL_ROOT)
sys.path.insert(0, str(SPATIAL_ROOT / "src"))

import habitat  # noqa: E402
from habitat.config import read_write  # noqa: E402
from habitat.config.default import get_config  # noqa: E402
import longnav.utils.measures  # noqa: F401, E402
import longnav.utils.ovon.ovon_dataset  # noqa: F401, E402
import longnav.utils.ovon.ovon_nav  # noqa: F401, E402

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
SERVER = "http://127.0.0.1:8399"
ACTION_IDS = {"stop": 0, "forward": 1, "left": 2, "right": 3}


def build_config(scene):
    config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.data_path = DATASET_PATH
        config.habitat.dataset.split = "val"
        if scene:
            config.habitat.dataset.content_scenes = [scene]

        agent = config.habitat.simulator.agents.main_agent
        agent.sim_sensors.rgb_sensor.width = 640
        agent.sim_sensors.rgb_sensor.height = 480
        agent.sim_sensors.rgb_sensor.hfov = 79
        agent.sim_sensors.rgb_sensor.position = [0, 0.88, 0]
        agent.sim_sensors.depth_sensor.width = 640
        agent.sim_sensors.depth_sensor.height = 480
        agent.sim_sensors.depth_sensor.hfov = 79
        agent.sim_sensors.depth_sensor.position = [0, 0.88, 0]
        agent.height = 0.88
        agent.radius = 0.18

        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        config.habitat.simulator.habitat_sim_v0.allow_sliding = True
        config.habitat.environment.max_episode_steps = 500
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.task.measurements.success.success_distance = 1.0
    return config


def b64_jpeg(rgb_array):
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def get_goal_name(env):
    return str(env.current_episode.object_category)


def run_episode(env, index, max_steps, out_dir, done=frozenset(), history=3):
    obs = env.reset()
    episode = env.current_episode
    key = (Path(episode.scene_id).name.replace(".basis.glb", ""), str(episode.episode_id))
    if key in done:
        print(f"[{index:02d}] skip (already ran): {key[0]}/{key[1]}", flush=True)
        return None
    goal = get_goal_name(env)
    frames = [b64_jpeg(obs["rgb"])]
    past_actions = []
    steps, stopped = 0, False
    started = time.time()

    while steps < max_steps and not env.episode_over:
        resp = requests.post(f"{SERVER}/act", json={
            "goal": goal,
            "images": frames[-history:],
            "past_actions": past_actions,
        }, timeout=120)
        resp.raise_for_status()
        action = resp.json()["action"]
        past_actions.append(action)

        obs = env.step(ACTION_IDS[action])
        steps += 1
        frames.append(b64_jpeg(obs["rgb"]))
        if action == "stop":
            stopped = True
            break

    metrics = env.get_metrics()

    # Save the episode rollout as an mp4 (frames are already collected as JPEGs)
    video_dir = out_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    clean = lambda v: "".join(c if c.isalnum() or c in "-_" else "_" for c in str(v))
    stem = (f"{index:04d}_{clean(key[0])}_{clean(episode.episode_id)}_{clean(goal)}"
            f"_{'succ' if metrics.get('success') else 'fail'}")
    video_path = video_dir / f"{stem}.mp4"
    import cv2
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
    for b in frames:
        img = np.array(Image.open(io.BytesIO(base64.b64decode(b))))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()

    record = {
        "episode_index": index,
        "scene": Path(episode.scene_id).name.replace(".basis.glb", ""),
        "episode_id": str(episode.episode_id),
        "goal": goal,
        "geodesic_distance": episode.info.get("geodesic_distance", -1),
        "steps": steps,
        "stopped": stopped,
        "success": metrics.get("success"),
        "spl": metrics.get("spl"),
        "soft_spl": metrics.get("soft_spl"),
        "distance_to_goal": metrics.get("distance_to_goal"),
        "elapsed_seconds": round(time.time() - started, 1),
        "video": str(video_path),
    }
    with open(out_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[{index:02d}] goal={goal:<10} steps={steps:3d} "
          f"success={record['success']} spl={record['spl']:.2f} "
          f"dtg={record['distance_to_goal']:.2f}m ({record['elapsed_seconds']}s)", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="Restrict to one scene (default: all val scenes)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--history", type=int, default=3, help="Number of recent frames sent per step")
    p.add_argument("--limit", type=int, default=None, help="Cap episode count")
    p.add_argument("--skip-results", default=None,
                   help="results.jsonl of a previous run; matching episodes are skipped")
    p.add_argument("--output-dir",
                   default="/home/ashed/Documents/cosmos3_nav/eval/habitat_smoke_out")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.skip_results:
        for line in open(args.skip_results):
            r = json.loads(line)
            done.add((r["scene"], r["episode_id"]))
        print(f"Skipping {len(done)} episodes from {args.skip_results}", flush=True)

    requests.get(f"{SERVER}/health", timeout=10).raise_for_status()
    print("Cosmos3 server is up.", flush=True)

    env = habitat.Env(config=build_config(args.scene))
    n = len(env.episodes) if args.limit is None else min(args.limit, len(env.episodes))
    print(f"Scenes: {args.scene or 'all val'}: running first {n}/{len(env.episodes)} episodes, "
          f"max_steps={args.max_steps}", flush=True)

    records = []
    for i in range(n):
        try:
            rec = run_episode(env, i, args.max_steps, out_dir, done=done, history=args.history)
            if rec is not None:
                records.append(rec)
        except requests.RequestException as e:
            print(f"[{i:02d}] ABORT: model server unreachable ({e}) — saving partials", flush=True)
            break

    if records:
        agg = {k: float(np.mean([r[k] for r in records]))
               for k in ["success", "spl", "soft_spl", "distance_to_goal", "steps"]}
        agg["episodes"] = len(records)
        print("\n=== AGGREGATE ===\n" + json.dumps(agg, indent=2), flush=True)
        (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    env.close()


if __name__ == "__main__":
    main()
