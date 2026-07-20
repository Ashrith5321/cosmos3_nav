"""Teleoperate a habitat agent and, at EVERY step, imagine the future straight
ahead with the Cosmos3 generator. Left panel = real first-person view; right
panel = the generator's imagined roll-out from the current view. The whole
session (both panels) is recorded to an mp4.

Start a resident generator server first (cosmos3_nav venv), fast config:
    .venv/bin/python src/generator_server.py --offload none --size 256x320 \
        --frames 17 --steps 6 --port 8402

Then, in the habitat env (on a machine with a display, e.g. NoMachine):
    ~/miniconda3/envs/habitat033/bin/python src/teleop_imagine.py \
        --scene DYehNKdT76V --gen-port 8402 --out eval/teleop_session

Keys (focus the window):  w=forward  a=turn left  d=turn right
                          n=next episode   q=quit (finalizes the recording)
"""
import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import quaternion  # numpy-quaternion (free-fly rotation math)
import requests

SPATIAL_ROOT = Path("/home/ashed/Documents/spatial_training")
os.chdir(SPATIAL_ROOT)
sys.path.insert(0, str(SPATIAL_ROOT / "src"))
sys.path.insert(0, "/home/ashed/Documents/cosmos3_nav/src")
import habitat  # noqa: E402
from habitat.config import read_write  # noqa: E402
from habitat.config.default import get_config  # noqa: E402
import longnav.utils.ovon.ovon_dataset  # noqa: F401,E402
import longnav.utils.ovon.ovon_nav  # noqa: F401,E402

ACTIONS = {"w": 1, "a": 2, "d": 3}   # forward / turn-left / turn-right


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


class TeleopImagine:
    def __init__(self, args):
        self.args = args
        self.url = f"http://127.0.0.1:{args.gen_port}"
        self.h, self.w = (int(x) for x in args.size.lower().split("x"))
        self.out = Path(args.out); self.out.mkdir(parents=True, exist_ok=True)
        self.env = habitat.Env(config=build_config(args.scene, args.gpu))
        self.env.reset()
        self.goal = str(self.env.current_episode.object_category)
        self.step = 0
        self.results = {}
        self.last_key = None
        # free-fly point mass: lock height, no collisions
        self.step_size = args.step_size
        self.turn = args.turn
        self.spawn_y = float(np.array(self.env.sim.get_agent(0).get_state().position)[1])
        self.obs = self._obs_from_sim()

        self.fig, (self.ax_fpv, self.ax_imag) = plt.subplots(1, 2, figsize=(13, 5))
        self.ax_fpv.set_title("REAL first-person view"); self.ax_fpv.axis("off")
        self.ax_imag.set_title("IMAGINED future (generator)"); self.ax_imag.axis("off")
        self.im_fpv = self.ax_fpv.imshow(np.asarray(self.obs["rgb"]))
        self.im_imag = self.ax_imag.imshow(np.full((256, 320, 3), 40, np.uint8))
        self.title = self.fig.suptitle("", fontsize=12)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._stop = threading.Event()
        threading.Thread(target=self._poll_loop, daemon=True).start()

        # recording: grab the canvas on a timer -> mp4
        self.writer = None
        self.rec_path = str(self.out / "teleop_session.mp4")
        self.timer = self.fig.canvas.new_timer(int(1000 / args.record_fps))
        self.timer.add_callback(self._tick)
        self.timer.start()
        self.imagine()          # imagine the spawn view immediately
        self._refresh()

    # ---------- free-fly point-mass control ----------
    def _obs_from_sim(self):
        raw = self.env.sim.get_sensor_observations()
        rgb = raw["rgb"]
        if rgb.shape[-1] == 4:            # RGBA -> RGB
            rgb = rgb[..., :3]
        return {"rgb": np.ascontiguousarray(rgb)}

    def free_move(self, k):
        """Move as a point mass: no collision, height locked to spawn height."""
        agent = self.env.sim.get_agent(0)
        st = agent.get_state()
        q = st.rotation
        pos = np.array(st.position, dtype=np.float32)
        if k == "w":
            fwd = quaternion.rotate_vectors(q, np.array([0.0, 0.0, -1.0]))
            fwd[1] = 0.0
            n = np.linalg.norm(fwd)
            if n > 1e-6:
                pos = pos + (fwd / n) * self.step_size
        elif k in ("a", "d"):
            yaw = np.deg2rad(self.turn) * (1.0 if k == "a" else -1.0)
            st.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0]) * q
        pos[1] = self.spawn_y            # keep the same height
        st.position = pos
        agent.set_state(st)
        self.obs = self._obs_from_sim()

    # ---------- imagination ----------
    def imagine(self):
        key = f"ep{self.env.current_episode.episode_id}_s{self.step:03d}"
        self.last_key = key
        rgb = np.asarray(self.obs["rgb"])
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        prompt = ("First-person indoor home walkthrough, moving forward through the "
                  f"space ahead, looking for a {self.goal}.")
        try:
            requests.post(f"{self.url}/enqueue", timeout=8, json={
                "frontiers": [{"key": key, "prompt": prompt,
                               "image_jpg": base64.b64encode(buf).decode()}],
                "height": self.h, "width": self.w,
                "frames": self.args.frames, "steps": self.args.gen_steps})
        except Exception as e:
            print(f"[imagine] generator unavailable ({e}); start generator_server.py", flush=True)

    def _poll_loop(self):
        while not self._stop.is_set():
            try:
                self.results = requests.get(f"{self.url}/results", timeout=8).json()
            except Exception:
                pass
            self._stop.wait(1.0)

    def _imag_panel(self):
        v = self.results.get(self.last_key, {})
        st = v.get("status")
        if st == "done" and v.get("last_jpg"):
            last = cv2.imdecode(np.frombuffer(base64.b64decode(v["last_jpg"]), np.uint8),
                                cv2.IMREAD_COLOR)[..., ::-1]
            return last, f"imagined last frame ({v.get('seconds','?')}s)"
        return np.full((self.h, self.w, 3), 40, np.uint8), (st or "imagining...")

    # ---------- display + recording ----------
    def _refresh(self):
        self.im_fpv.set_data(np.asarray(self.obs["rgb"]))
        panel, cap = self._imag_panel()
        self.im_imag.set_data(panel)
        self.ax_imag.set_title(f"IMAGINED future — {cap}")
        self.title.set_text(f"scene={self.args.scene}  goal={self.goal}  step={self.step}  "
                            f"[w/a/d move · n next ep · q quit]")
        self.fig.canvas.draw_idle()

    def _tick(self):
        self._refresh()
        # append current canvas to the session video
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        frame = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.rec_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                          self.args.record_fps, (w, h))
        self.writer.write(frame)

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            self._stop.set()
            if self.writer is not None:
                self.writer.release()
                print(f"[record] saved {self.rec_path}", flush=True)
            plt.close(self.fig)
            return
        if k == "n":
            self.env.reset()
            self.goal = str(self.env.current_episode.object_category)
            self.spawn_y = float(np.array(self.env.sim.get_agent(0).get_state().position)[1])
            self.obs = self._obs_from_sim()
            self.step = 0
            self.imagine()
        elif k in ("w", "a", "d"):
            self.free_move(k)          # point-mass move (no collision, fixed height)
            self.step += 1
            self.imagine()             # imagine the future at EVERY step
        else:
            return
        self._refresh()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="DYehNKdT76V")
    p.add_argument("--gpu", type=int, default=0, help="habitat render GPU")
    p.add_argument("--gen-port", type=int, default=8402)
    p.add_argument("--size", default="256x320")
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--gen-steps", type=int, default=6)
    p.add_argument("--record-fps", type=int, default=5)
    p.add_argument("--step-size", type=float, default=0.25, help="meters moved per forward press")
    p.add_argument("--turn", type=float, default=30.0, help="degrees per turn press")
    p.add_argument("--out", default="/home/ashed/Documents/cosmos3_nav/eval/teleop_session")
    args = p.parse_args()
    t = TeleopImagine(args)
    print("Teleop ready. Focus the window. Keys: w=forward a=left d=right n=next-ep q=quit", flush=True)
    plt.show()
    t._stop.set()
    if t.writer is not None:
        t.writer.release()
    t.env.close()


if __name__ == "__main__":
    main()
