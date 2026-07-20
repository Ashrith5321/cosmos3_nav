"""Spawn in an HM3D val scene, DRIVE the agent for N steps, and at EVERY step
imagine the future from the current first-person view via the Cosmos3 generator
(fast resident config). Imaginations are round-robined across --gen-ports so both
GPUs pipeline while the agent keeps driving. Each step saves the real FPV; the
generator server writes the imagined mp4 (key = step_XX).

Run (habitat env), resident generator server(s) already up:
    ~/miniconda3/envs/habitat033/bin/python src/drive_and_imagine.py \
        --steps 15 --gen-ports 8402,8403 --out eval/drive_imagine_out
"""
import argparse, base64, json, os, random, shutil, sys, time
from pathlib import Path
import cv2, numpy as np, requests

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

FWD, LEFT, RIGHT = 1, 2, 3
VAL_SCENES = ['4ok3usBNeis','5cdEh9F2hJL','6s7QHgap2fW','7MXmsvcQjpJ','BAbdmeyTvMZ','CrMo8WxCyVb',
              'DYehNKdT76V','Dd4bFSTQ8gi','GLAQ4DNUx5U','HY1NcmCgn3n','LT9Jq6dN3Ea','MHPLjHsuG27',
              'Nfvxx8J5NCo','QaLdnwvtxbs','TEEsavR23oF','VBzV5z6i1WS','XB4GS9ShBRE','a8BtkwhxdRV',
              'bCPU9suPUw9','bxsVRursffK','cvZr5TUy5C5','eF36g7L6Z9M','h1zeeAwLh9Z','k1cupFYWXJ6',
              'mL8ThkuaVTM','mv2HUxq3B53','p53SfW6mjZe','q3zU7Yy5E5s','q5QZSEeHe5g','qyAac8rV8Zk',
              'svBbv1Pavdk','wcojb4TFT35','y9hTuugGdiq','yr17PDCnDDW','ziup5kvtCCR','zt1RVoi7PcG']


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
        ag.sim_sensors.depth_sensor.max_depth = 5.0
        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu
        config.habitat.environment.max_episode_steps = 5000
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None)
    p.add_argument("--episode", type=int, default=None)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--gpu", type=int, default=0, help="habitat render GPU")
    p.add_argument("--gen-ports", default="8402,8403")
    p.add_argument("--size", default="256x320")
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--gen-steps", type=int, default=6)
    p.add_argument("--out", default="/home/ashed/Documents/cosmos3_nav/eval/drive_imagine_out")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    rng = random.Random()
    scene = args.scene or rng.choice(VAL_SCENES)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ports = [int(x) for x in args.gen_ports.split(",")]
    h, w = (int(x) for x in args.size.lower().split("x"))
    print(f"[spawn] scene={scene} render-gpu={args.gpu} gen-ports={ports} "
          f"imagine={args.size}/{args.frames}f/{args.gen_steps}steps", flush=True)

    env = habitat.Env(config=build_config(scene, args.gpu))
    n_ep = len(env.episodes)
    ep_idx = args.episode if args.episode is not None else rng.randrange(n_ep)
    for _ in range(ep_idx + 1):
        obs = env.reset()
    ep = env.current_episode
    goal = str(ep.object_category)
    # per-run unique key prefix so the resident server never dedups against a prior run
    tag = f"{scene}_{ep.episode_id}"
    print(f"[spawn] episode {ep.episode_id} goal={goal} tag={tag}", flush=True)

    def agent_pos():
        return np.asarray(env.sim.get_agent(0).get_state().position, dtype=np.float32)

    def enqueue(step, rgb, port):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        prompt = ("First-person indoor home walkthrough, moving forward through the "
                  f"space ahead, looking for a {goal}.")
        requests.post(f"http://127.0.0.1:{port}/enqueue",
                      json={"frontiers": [{"key": f"{tag}_step_{step:02d}", "prompt": prompt,
                                           "image_jpg": base64.b64encode(buf).decode()}],
                            "height": h, "width": w, "frames": args.frames, "steps": args.gen_steps},
                      timeout=15).raise_for_status()

    manifest = {"scene": scene, "episode": str(ep.episode_id), "goal": goal, "steps": []}
    prev = agent_pos()
    action = FWD
    t_start = time.time()
    for step in range(args.steps):
        rgb = np.asarray(obs["rgb"]).copy()
        cv2.imwrite(str(out / f"step_{step:02d}_fpv.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        port = ports[step % len(ports)]
        try:
            enqueue(step, rgb, port)
            st = "queued"
        except Exception as e:
            st = f"enqueue-failed:{e}"
        act_name = {FWD: "forward", LEFT: "left", RIGHT: "right"}[action]
        manifest["steps"].append({"step": step, "action_taken": act_name,
                                  "imagine_port": port, "key": f"{tag}_step_{step:02d}"})
        print(f"[step {step:02d}] action={act_name} -> imagine on GPU port {port} ({st})", flush=True)

        # advance the agent; forward-biased, turn when blocked or periodically
        obs = env.step(action)
        moved = np.linalg.norm(agent_pos() - prev)
        prev = agent_pos()
        blocked = (action == FWD and moved < 0.05)
        if blocked:
            action = rng.choice([LEFT, RIGHT])           # bumped a wall -> turn
        elif step % 4 == 3:
            action = rng.choice([LEFT, RIGHT, FWD, FWD])  # occasional look-around
        else:
            action = FWD
        if env.episode_over:
            print(f"[drive] episode ended early at step {step}", flush=True)
            break

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    drive_s = time.time() - t_start
    print(f"[drive] {len(manifest['steps'])} steps driven+enqueued in {drive_s:.1f}s", flush=True)

    # wait for every step's imagination
    want = {s["key"] for s in manifest["steps"]}
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        got = {}
        for port in ports:
            try:
                res = requests.get(f"http://127.0.0.1:{port}/results", timeout=10).json()
                for k, v in res.items():
                    if k in want:
                        got[k] = v
            except Exception:
                pass
        ndone = sum(1 for k in want if got.get(k, {}).get("status") in ("done", "error"))
        print(f"[imagine] {ndone}/{len(want)} finished", flush=True)
        if ndone >= len(want):
            break
        time.sleep(3)

    print("\n=== PER-STEP IMAGINATION ===", flush=True)
    total = 0.0
    for port in ports:
        try:
            res = requests.get(f"http://127.0.0.1:{port}/results", timeout=10).json()
        except Exception:
            continue
        for s in manifest["steps"]:
            if s["imagine_port"] != port:
                continue
            v = res.get(s["key"], {})
            if v.get("status") == "done":
                total += float(v.get("seconds", 0))
                # copy the imagined mp4 into this run's own dir as step_XX.mp4
                dst = out / f"step_{s['step']:02d}.mp4"
                try:
                    if v.get("mp4") and os.path.exists(v["mp4"]):
                        shutil.copy(v["mp4"], dst)
                except Exception as e:
                    print(f"  [copy warn] {s['key']}: {e}", flush=True)
                print(f"  {s['key']} (act={s['action_taken']:7s} GPU:{port}): "
                      f"{v.get('seconds')}s -> {dst}", flush=True)
            else:
                print(f"  {s['key']}: {v.get('status')} {v.get('error','')}", flush=True)
    print(f"\nsum of per-imagine gen time: {total:.1f}s (ran in parallel across {len(ports)} GPUs)")
    print(f"Outputs in: {out}", flush=True)


if __name__ == "__main__":
    main()
