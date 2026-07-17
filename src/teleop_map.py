"""Teleoperate a habitat agent and watch the GlobalMap2D build live.

Run (needs a display):
    ~/miniconda3/envs/habitat033/bin/python src/teleop_map.py --scene 4ok3usBNeis

Keys (focus the matplotlib window):
    w = forward 0.25m   a = turn left 30   d = turn right 30
    n = next episode    q = quit

Uses matplotlib TkAgg for display because the habitat env ships headless
OpenCV (cv2.imshow unavailable).
"""
import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, "/home/ashed/Documents/spatial_training/src")
os.chdir("/home/ashed/Documents/spatial_training")

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
        self.pose = self._integrate()

        self.fig, (self.ax_fpv, self.ax_map) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]})
        self.fig.canvas.manager.set_window_title("cosmos3_nav teleop | FPV + global map")
        for ax in (self.ax_fpv, self.ax_map):
            ax.set_axis_off()
        self.im_fpv = self.ax_fpv.imshow(self.obs["rgb"])
        self.im_map = self.ax_map.imshow(self._map_rgb())
        self.title = self.fig.suptitle("", fontsize=11)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._refresh()

    def _integrate(self):
        pose = pose_from_habitat_state(self.env.sim.get_agent(0).get_state())
        self.gm.update(self.obs["depth"], pose, rgb=self.obs["rgb"])
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

    def _refresh(self):
        self.im_fpv.set_data(self._fpv_with_overlay())
        m = self._map_rgb()
        self.im_map.set_data(m)
        self.im_map.set_extent((0, m.shape[1], m.shape[0], 0))
        goal = str(self.env.current_episode.object_category)
        ftxt = self.gm.frontier_text(self.pose, self.fronts).splitlines()
        self.title.set_text(
            f"goal: {goal}   step: {self.step}   floor: {self.gm.floor_index}   "
            f"frontiers: {len(self.fronts)}\n" + "  |  ".join(ftxt[:4]))
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            plt.close(self.fig)
            return
        if k == "n":
            self.obs = self.env.reset()
            self.gm = MultiFloorMap()
            self.step = 0
            self.fn_points = {}
            self.fn_overlay = None
            self.pose = self._integrate()
        elif k in ACTIONS and not self.env.episode_over:
            self.obs = self.env.step(ACTIONS[k])
            self.step += 1
            self.pose = self._integrate()
        else:
            return
        self._refresh()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="4ok3usBNeis")
    p.add_argument("--min-frontier-cells", type=int, default=25)
    args = p.parse_args()

    t = Teleop(args.scene, args.min_frontier_cells)
    print("Teleop ready. Keys: w=forward  a=left  d=right  n=next episode  q=quit")
    plt.show()
    t.env.close()


if __name__ == "__main__":
    main()
