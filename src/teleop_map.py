"""Teleoperate a habitat agent and watch the GlobalMap2D build live, with
world-model frontier imagination running asynchronously in a side panel.

As you move, each frontier on the map is queued to the generator server
(src/generator_server.py); the server imagines walking into it and returns a
preview strip, which is painted next to that frontier's A/B/C label. Imagination
is slow (~15-30 s/frontier) so it never blocks driving -- previews fill in as
they finish.

Run (needs a display). Start the generator server first (cosmos3_nav venv):
    .venv/bin/python src/generator_server.py --offload model
Then, in the habitat env:
    ~/miniconda3/envs/habitat033/bin/python src/teleop_map.py --scene 4ok3usBNeis

Keys (focus the matplotlib window):
    w = forward 0.25m   a = turn left 30   d = turn right 30
    i = imagine current frontiers now   n = next episode   q = quit

Uses matplotlib TkAgg for display because the habitat env ships headless
OpenCV (cv2.imshow unavailable).
"""
import argparse
import base64
import io
import os
import sys
import threading
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, "/home/ashed/Documents/spatial_training/src")
os.chdir("/home/ashed/Documents/spatial_training")

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import habitat
from habitat.config import read_write
from habitat.config.default import get_config
import longnav.utils.ovon.ovon_dataset  # noqa: F401
import longnav.utils.ovon.ovon_nav  # noqa: F401

from global_map import MultiFloorMap, pose_from_habitat_state

ACTIONS = {"w": 1, "a": 2, "d": 3}  # forward / left / right
FRONTIERNET_URL = "http://localhost:12186/frontiernet"
FN_COLOR = (255, 0, 255)   # magenta (BGR) for learned frontiers on the map
DEPTH_MAX = 5.0

GEN_URL = "http://localhost:8402"     # generator_server.py
IMAGINE_EVERY = 4                     # auto-enqueue frontiers every N steps
MAX_KEYFRAMES = 200                   # bounded (pose, rgb) log for conditioning
FKEY_GRID_M = 0.75                    # frontier identity = centroid snapped to this grid


def build_config(scene):
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
        config.habitat.environment.max_episode_steps = 5000
    return config


class Teleop:
    def __init__(self, scene, min_frontier_cells):
        self.env = habitat.Env(config=build_config(scene))
        self.min_cells = min_frontier_cells
        self.obs = self.env.reset()
        self.gm = MultiFloorMap()
        self.step = 0
        self.fn_points = {}          # floor idx -> [(x, z, color), ...]
        self.fn_overlay = None       # full-res FPV frontier mask (float alpha)
        self.fn_available = True
        self.keyframes = []          # [(pose 4x4, rgb HxWx3), ...] for conditioning
        self.imag_results = {}       # fkey -> server result dict
        self.gen_available = True
        self.pose = self._integrate()

        self.fig, (self.ax_fpv, self.ax_map, self.ax_imag) = plt.subplots(
            1, 3, figsize=(19, 6), gridspec_kw={"width_ratios": [1.1, 1, 0.9]})
        self.fig.canvas.manager.set_window_title(
            "cosmos3_nav teleop | FPV + global map + imagined frontiers")
        for ax in (self.ax_fpv, self.ax_map, self.ax_imag):
            ax.set_axis_off()
        self.ax_imag.set_title("imagined frontiers", fontsize=10)
        self.im_fpv = self.ax_fpv.imshow(self.obs["rgb"])
        self.im_map = self.ax_map.imshow(self._map_rgb())
        self.im_imag = self.ax_imag.imshow(np.full((400, 300, 3), 40, np.uint8))
        self.title = self.fig.suptitle("", fontsize=11)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # background poller: pull finished imaginations from the server so the
        # panel fills in while you drive (matplotlib redraw stays on main thread
        # via the timer below).
        self._stop = threading.Event()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.timer = self.fig.canvas.new_timer(1500)
        self.timer.add_callback(self._refresh)
        self.timer.start()
        self._refresh()

    def _integrate(self):
        pose = pose_from_habitat_state(self.env.sim.get_agent(0).get_state())
        self.gm.update(self.obs["depth"], pose, rgb=self.obs["rgb"])
        self.keyframes.append((pose.copy(), np.asarray(self.obs["rgb"]).copy()))
        if len(self.keyframes) > MAX_KEYFRAMES:
            del self.keyframes[:len(self.keyframes) - MAX_KEYFRAMES]
        self._query_frontiernet(pose)
        return pose

    def _query_frontiernet(self, pose):
        """Run the learned frontier detector on the current frame; build the
        FPV overlay and project detections onto the active floor map."""
        if not self.fn_available:
            return
        depth_m = np.asarray(self.obs["depth"], dtype=np.float32).squeeze() * DEPTH_MAX
        try:
            r = requests.post(FRONTIERNET_URL, json={
                "rgb": self.obs["rgb"].tolist(),
                "depth": depth_m.tolist(),
            }, timeout=30)
            r.raise_for_status()
            out = r.json()
        except Exception as e:
            print(f"FrontierNet server unavailable ({e}) — learned frontiers off. "
                  f"Start it with: .venv/bin/python src/frontiernet_server.py")
            self.fn_available = False
            return

        mask = np.array(out["ft_region"], dtype=np.uint8)
        gain = np.array(out["info_gain"], dtype=np.float32)
        s = out["scale"]
        offx, offy = out["offset"]
        H, W = depth_m.shape

        # inverse of resize(scale) + center-crop: model px -> original px
        x0, y0 = int(round(offx / s)), int(round(offy / s))
        x1 = min(W, int(round((offx + mask.shape[1]) / s)))
        y1 = min(H, int(round((offy + mask.shape[0]) / s)))
        import cv2
        up_mask = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        up_gain = cv2.resize(gain, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        overlay = np.zeros((H, W), np.float32)
        overlay[y0:y1, x0:x1] = up_mask * np.clip(up_gain, 0.2, 1.0)
        self.fn_overlay = overlay

        # project frontier pixels to the world via depth + pose
        ys, xs = np.nonzero(overlay > 0)
        if len(ys) == 0:
            return
        sel = slice(None, None, max(1, len(ys) // 400))   # subsample for speed
        ys, xs = ys[sel], xs[sel]
        d = depth_m[ys, xs]
        ok = (d > 0.3) & (d < DEPTH_MAX - 1e-3)
        ys, xs, d = ys[ok], xs[ok], d[ok]
        rays = self.gm.active._rays[ys, xs]           # (N, 3) camera rays
        pts_cam = rays * d[:, None]
        R, t = pose[:3, :3], pose[:3, 3]
        pts_world = pts_cam @ R.T + t
        floor = self.gm.floor_index
        pts = self.fn_points.setdefault(floor, [])
        pts.extend((float(p[0]), float(p[2]), FN_COLOR) for p in pts_world)
        if len(pts) > 4000:                           # keep the map readable
            del pts[:len(pts) - 4000]

    def _map_rgb(self):
        self.fronts = self.gm.frontiers(min_cluster_cells=self.min_cells)
        bgr = self.gm.render(self.fronts, agent_pose=self.pose,
                             extra_points=self.fn_points.get(self.gm.floor_index))
        return bgr[..., ::-1]  # BGR -> RGB for matplotlib

    def _fpv_with_overlay(self):
        fpv = self.obs["rgb"].astype(np.float32)
        if self.fn_overlay is not None:
            a = self.fn_overlay[..., None] * 0.55        # gain-weighted alpha
            magenta = np.array([255.0, 0.0, 255.0])
            fpv = fpv * (1 - a) + magenta * a
        return fpv.astype(np.uint8)

    # ---------- imagination ----------
    @staticmethod
    def _fkey(frontier):
        """Stable id for a frontier across steps: centroid snapped to a grid so
        the same opening is imagined once, not re-queued every frame."""
        cx, cz = frontier["centroid_world"]
        return f"{round(cx / FKEY_GRID_M) * FKEY_GRID_M:.2f},{round(cz / FKEY_GRID_M) * FKEY_GRID_M:.2f}"

    def _best_keyframe(self, frontier):
        """Logged frame whose camera heading best faces this frontier (design
        doc: condition on the keyframe that looks toward the opening)."""
        cx, cz = frontier["centroid_world"]
        best, best_ang = self.obs["rgb"], 1e9
        for pose, rgb in self.keyframes:
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

    def _enqueue_frontiers(self):
        """Queue every current frontier's best-facing keyframe to the server."""
        if not self.gen_available or not self.fronts:
            return
        goal = str(self.env.current_episode.object_category)
        jobs = []
        for f in self.fronts:
            rgb = self._best_keyframe(f)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            jobs.append({
                "key": self._fkey(f),
                "prompt": ("First-person indoor home view. Move forward through the "
                           f"opening ahead to explore the room beyond, searching for a {goal}."),
                "image_jpg": base64.b64encode(buf).decode(),
            })

        def _post():
            try:
                requests.post(f"{GEN_URL}/enqueue", json={"frontiers": jobs}, timeout=10)
            except Exception as e:  # noqa: BLE001
                if self.gen_available:
                    print(f"generator server unavailable ({e}) — imagination off. "
                          f"Start it: .venv/bin/python src/generator_server.py")
                self.gen_available = False
        threading.Thread(target=_post, daemon=True).start()

    def _poll_loop(self):
        while not self._stop.is_set():
            if self.gen_available:
                try:
                    r = requests.get(f"{GEN_URL}/results", timeout=10)
                    self.imag_results = r.json()
                except Exception:
                    pass
            self._stop.wait(1.5)

    def _imag_panel(self):
        """Vertical stack of the current frontiers' imagined preview strips,
        labelled A/B/C to match the map."""
        tiles, W = [], 300
        for i, f in enumerate(self.fronts[:5]):
            label = chr(ord("A") + i)
            res = self.imag_results.get(self._fkey(f), {})
            status = res.get("status")
            if status == "done":
                strip = cv2.imdecode(np.frombuffer(base64.b64decode(res["strip_jpg"]),
                                                   np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                tile = cv2.resize(strip, (W, int(strip.shape[0] * W / strip.shape[1])))
                txt = f"{label}: imagined ({res.get('seconds', '?')}s)"
            else:
                tile = np.full((70, W, 3), 55, np.uint8)
                txt = f"{label}: {'imagining...' if status == 'running' else 'queued...'}"
            tile = np.ascontiguousarray(tile)
            cv2.putText(tile, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 0) if status == "done" else (200, 200, 200), 1)
            tiles.append(tile)
            tiles.append(np.full((4, W, 3), 40, np.uint8))
        if not tiles:
            return np.full((400, W, 3), 40, np.uint8)
        return np.concatenate(tiles, axis=0)

    def _refresh(self):
        self.im_fpv.set_data(self._fpv_with_overlay())
        m = self._map_rgb()
        self.im_map.set_data(m)
        self.im_map.set_extent((0, m.shape[1], m.shape[0], 0))
        panel = self._imag_panel()
        self.im_imag.set_data(panel)
        self.im_imag.set_extent((0, panel.shape[1], panel.shape[0], 0))
        goal = str(self.env.current_episode.object_category)
        ndone = sum(1 for f in self.fronts
                    if self.imag_results.get(self._fkey(f), {}).get("status") == "done")
        ftxt = self.gm.frontier_text(self.pose, self.fronts).splitlines()
        self.title.set_text(
            f"goal: {goal}   step: {self.step}   floor: {self.gm.floor_index}   "
            f"frontiers: {len(self.fronts)}   imagined: {ndone}/{len(self.fronts)}\n"
            + "  |  ".join(ftxt[:4]))
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            self._stop.set()
            plt.close(self.fig)
            return
        if k == "n":
            self.obs = self.env.reset()
            self.gm = MultiFloorMap()
            self.step = 0
            self.fn_points = {}
            self.fn_overlay = None
            self.keyframes = []
            self.imag_results = {}
            self.pose = self._integrate()
        elif k == "i":                       # imagine current frontiers now
            self._map_rgb()                  # refresh self.fronts
            self._enqueue_frontiers()
        elif k in ACTIONS and not self.env.episode_over:
            self.obs = self.env.step(ACTIONS[k])
            self.step += 1
            self.pose = self._integrate()
            if self.step % IMAGINE_EVERY == 0:
                self._map_rgb()
                self._enqueue_frontiers()
        else:
            return
        self._refresh()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="4ok3usBNeis")
    p.add_argument("--min-frontier-cells", type=int, default=25)
    args = p.parse_args()

    t = Teleop(args.scene, args.min_frontier_cells)
    print("Teleop ready. Keys: w=forward  a=left  d=right  "
          "i=imagine frontiers  n=next episode  q=quit")
    plt.show()
    t.env.close()


if __name__ == "__main__":
    main()
