"""Incremental 2D global map from posed RGB-D, with frontier extraction.

Built for the habitat setup used in this repo (eval/run_habitat_smoke_eval.py):
640x480 depth, hfov 79, camera at [0, 0.88, 0] on the agent, max depth 5 m.

Habitat world frame: x right, y up, camera looks along -z.
Map grid: rows = world z, cols = world x, agent starts at the center.

Usage:
    gm = GlobalMap2D()
    gm.update(depth, pose_from_habitat_state(env.sim.get_agent(0).get_state()))
    fronts = gm.frontiers()              # clustered frontier regions
    bgr = gm.render(fronts)              # map image with frontiers marked
"""
import numpy as np
import cv2

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def pose_from_habitat_state(state, camera_height=0.88):
    """4x4 world_T_camera from a habitat agent state (position + rotation quat)."""
    q = state.rotation  # np.quaternion (w, x, y, z)
    w, x, y, z = q.w, q.x, q.y, q.z
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(state.position) + np.array([0.0, camera_height, 0.0])
    return T


class GlobalMap2D:
    def __init__(self, map_size_m=40.0, resolution=0.05, hfov_deg=79.0,
                 depth_max=5.0, depth_normalized=True,
                 floor_band=(-1.2, -0.5), obstacle_band=(-0.5, 0.8),
                 image_hw=(480, 640)):
        """
        floor_band / obstacle_band: height ranges in meters RELATIVE TO THE
        CAMERA (camera is 0.88 m above the floor, so the floor sits near -0.88).
        Points in floor_band mark FREE cells; points in obstacle_band mark
        OCCUPIED cells (walls and furniture the agent cannot cross).
        """
        self.res = resolution
        self.n = int(map_size_m / resolution)
        self.origin = self.n // 2  # grid index of world (0, 0)
        self.grid = np.full((self.n, self.n), UNKNOWN, dtype=np.uint8)
        self.color = np.zeros((self.n, self.n, 3), dtype=np.uint8)  # optional RGB paint
        self.depth_max = depth_max
        self.depth_normalized = depth_normalized
        self.floor_band = floor_band
        self.obstacle_band = obstacle_band
        self.trajectory = []  # [(row, col), ...] of agent positions

        h, w = image_hw
        f = (w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        u = np.arange(w) - (w - 1) / 2.0
        v = np.arange(h) - (h - 1) / 2.0
        uu, vv = np.meshgrid(u, v)
        # camera frame: x right, y up, looking along -z  ->  ray per pixel
        self._rays = np.stack([uu / f, -vv / f, -np.ones_like(uu)], axis=-1)

    # ---------- coordinates ----------
    def world_to_grid(self, xz):
        """(..., 2) world (x, z) -> (row, col) int arrays."""
        xz = np.asarray(xz)
        col = np.round(xz[..., 0] / self.res).astype(int) + self.origin
        row = np.round(xz[..., 1] / self.res).astype(int) + self.origin
        return row, col

    def grid_to_world(self, row, col):
        return np.stack([(np.asarray(col) - self.origin) * self.res,
                         (np.asarray(row) - self.origin) * self.res], axis=-1)

    # ---------- mapping ----------
    def update(self, depth, world_T_camera, rgb=None):
        """Integrate one frame. depth: (H, W) or (H, W, 1); pose: 4x4."""
        d = np.asarray(depth, dtype=np.float32).squeeze()
        if self.depth_normalized:
            d = d * self.depth_max
        valid = (d > 0.2) & (d < self.depth_max - 1e-3)

        pts_cam = self._rays * d[..., None]          # (H, W, 3), z = -depth
        R, t = world_T_camera[:3, :3], world_T_camera[:3, 3]
        pts_rel_h = pts_cam[..., 1]                  # height relative to camera
        pts_world = pts_cam[valid] @ R.T + t
        rel_h = pts_rel_h[valid]

        row, col = self.world_to_grid(pts_world[:, [0, 2]])
        inb = (row >= 0) & (row < self.n) & (col >= 0) & (col < self.n)
        row, col, rel_h = row[inb], col[inb], rel_h[inb]

        floor = (rel_h >= self.floor_band[0]) & (rel_h < self.floor_band[1])
        obst = (rel_h >= self.obstacle_band[0]) & (rel_h < self.obstacle_band[1])

        # free floor first, then obstacles overwrite (occupied is sticky)
        fr, fc = row[floor], col[floor]
        keep = self.grid[fr, fc] != OCCUPIED
        self.grid[fr[keep], fc[keep]] = FREE
        self.grid[row[obst], col[obst]] = OCCUPIED

        if rgb is not None:
            rgbv = np.asarray(rgb)[valid][inb]
            self.color[row, col] = rgbv[:, :3]

        # agent cell is free by definition; extend the trajectory
        ar, ac = self.world_to_grid(t[[0, 2]])
        if 0 <= ar < self.n and 0 <= ac < self.n:
            self.grid[ar, ac] = FREE
            self.trajectory.append((int(ar), int(ac)))

    # ---------- frontiers ----------
    def frontiers(self, min_cluster_cells=10):
        """Frontier = FREE cell with an UNKNOWN 8-neighbour and no OCCUPIED one.
        Returns clusters: [{'cells': (rows, cols), 'centroid_world': (x, z),
                            'centroid_grid': (row, col), 'size': int}, ...]
        sorted largest-first."""
        free = self.grid == FREE
        unknown = (self.grid == UNKNOWN).astype(np.uint8)
        occ = (self.grid == OCCUPIED).astype(np.uint8)
        k = np.ones((3, 3), np.uint8)
        near_unknown = cv2.dilate(unknown, k).astype(bool)
        near_occ = cv2.dilate(occ, k).astype(bool)
        frontier = free & near_unknown & ~near_occ

        num, labels = cv2.connectedComponents(frontier.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, num):
            rows, cols = np.nonzero(labels == i)
            if len(rows) < min_cluster_cells:
                continue
            cr, cc = rows.mean(), cols.mean()
            out.append({
                "cells": (rows, cols),
                "centroid_grid": (int(round(cr)), int(round(cc))),
                "centroid_world": tuple(self.grid_to_world(cr, cc).tolist()),
                "size": len(rows),
            })
        out.sort(key=lambda f: -f["size"])
        return out

    # ---------- rendering ----------
    def render(self, frontiers=None, agent_pose=None, crop=True, pad_cells=20,
               label_frontiers=True, scale=2, extra_points=None):
        """BGR image: unknown=gray, free=white, occupied=black, frontier=red,
        trajectory=blue, agent=green arrow, frontier centroids labelled A, B, ...
        extra_points: [(world_x, world_z, (b, g, r)), ...] extra markers
        (e.g. learned FrontierNet detections)."""
        img = np.full((self.n, self.n, 3), 128, np.uint8)   # unknown: gray
        img[self.grid == FREE] = (255, 255, 255)
        img[self.grid == OCCUPIED] = (30, 30, 30)

        if frontiers is None:
            frontiers = self.frontiers()
        for f in frontiers:
            img[f["cells"]] = (0, 0, 255)

        for (r, c) in self.trajectory:
            img[r, c] = (255, 120, 0)

        if extra_points:
            for (wx, wz, color) in extra_points:
                pr, pc = self.world_to_grid(np.array([wx, wz]))
                if 0 <= pr < self.n and 0 <= pc < self.n:
                    cv2.circle(img, (int(pc), int(pr)), 2, color, -1)

        if agent_pose is not None:
            R, t = agent_pose[:3, :3], agent_pose[:3, 3]
            ar, ac = self.world_to_grid(t[[0, 2]])
            fwd = R @ np.array([0.0, 0.0, -1.0])            # camera forward
            tip = (int(ac + 12 * fwd[0]), int(ar + 12 * fwd[2]))
            cv2.arrowedLine(img, (int(ac), int(ar)), tip, (0, 200, 0), 2, tipLength=0.5)

        if crop:
            known = np.argwhere(self.grid != UNKNOWN)
            if len(known):
                r0, c0 = np.maximum(known.min(0) - pad_cells, 0)
                r1, c1 = np.minimum(known.max(0) + pad_cells, self.n - 1)
                img = img[r0:r1 + 1, c0:c1 + 1]
                off = (r0, c0)
            else:
                off = (0, 0)
        else:
            off = (0, 0)

        if scale != 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        if label_frontiers:
            for i, f in enumerate(frontiers):
                r, c = f["centroid_grid"]
                p = ((c - off[1]) * scale, (r - off[0]) * scale)
                cv2.circle(img, p, 6, (0, 0, 255), -1)
                cv2.putText(img, chr(ord("A") + i), (p[0] + 6, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        return img

    def frontier_text(self, agent_pose, frontiers=None):
        """Compact text description of frontiers relative to the agent —
        ready to drop into a VLM prompt."""
        if frontiers is None:
            frontiers = self.frontiers()
        R, t = agent_pose[:3, :3], agent_pose[:3, 3]
        fwd = R @ np.array([0.0, 0.0, -1.0])
        heading = np.arctan2(fwd[0], -fwd[2])
        lines = []
        for i, f in enumerate(frontiers):
            dx = f["centroid_world"][0] - t[0]
            dz = f["centroid_world"][1] - t[2]
            dist = float(np.hypot(dx, dz))
            ang = np.degrees((np.arctan2(dx, -dz) - heading + np.pi) % (2 * np.pi) - np.pi)
            side = "ahead" if abs(ang) < 30 else ("left" if ang < 0 else "right")
            if abs(ang) > 120:
                side = "behind"
            lines.append(f"Frontier {chr(ord('A') + i)}: {dist:.1f}m {side} "
                         f"({ang:+.0f} deg), {f['size']} cells unexplored beyond")
        return "\n".join(lines) if lines else "No frontiers: environment fully explored."


class MultiFloorMap:
    """Set of GlobalMap2D layers, one per building floor.

    Floors are discovered from the camera height: each update is assigned to
    the floor whose reference height is within `floor_thresh` meters of the
    current camera y; if none matches (agent climbed stairs), a new floor map
    is created. Mid-staircase frames create a short-lived transition floor,
    which is fine: it stays tiny and is ignored by `main_floors()`.

    Delegates update / frontiers / render / frontier_text to the ACTIVE floor.
    """

    def __init__(self, floor_thresh=1.2, **map_kwargs):
        self.floor_thresh = floor_thresh
        self.map_kwargs = map_kwargs
        self.floors = []          # [{'ref_y': float, 'map': GlobalMap2D, 'updates': int}]
        self.active_idx = None

    def _select_floor(self, cam_y):
        best, best_d = None, self.floor_thresh
        for i, f in enumerate(self.floors):
            d = abs(f["ref_y"] - cam_y)
            if d < best_d:
                best, best_d = i, d
        if best is None:
            self.floors.append({"ref_y": cam_y, "map": GlobalMap2D(**self.map_kwargs),
                                "updates": 0})
            best = len(self.floors) - 1
        return best

    def update(self, depth, world_T_camera, rgb=None):
        cam_y = float(world_T_camera[1, 3])
        self.active_idx = self._select_floor(cam_y)
        f = self.floors[self.active_idx]
        # slowly track the reference height so a floor's ref settles on the
        # true walking height rather than the first (possibly mid-stair) frame
        f["ref_y"] = 0.95 * f["ref_y"] + 0.05 * cam_y
        f["updates"] += 1
        f["map"].update(depth, world_T_camera, rgb=rgb)

    @property
    def active(self):
        return self.floors[self.active_idx]["map"]

    @property
    def floor_index(self):
        return self.active_idx

    def main_floors(self, min_updates=10):
        """Floors with real dwell time (filters out stair-transition layers)."""
        return [i for i, f in enumerate(self.floors) if f["updates"] >= min_updates]

    def frontiers(self, **kw):
        return self.active.frontiers(**kw)

    def render(self, *a, **kw):
        return self.active.render(*a, **kw)

    def frontier_text(self, agent_pose, frontiers=None):
        txt = self.active.frontier_text(agent_pose, frontiers)
        others = [i for i in self.main_floors() if i != self.active_idx]
        if others:
            txt += f"\n(Other explored floors: {len(others)} — stairs connect them.)"
        return txt
