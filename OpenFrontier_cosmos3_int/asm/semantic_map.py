"""The annotated semantic map itself: a top-down, vote-accumulating grid.

World frame follows utils/transform.py: `from_habitat_position` maps habitat
(x, y, z) to (x, -z, y), so the world is **Z-up** and the ground plane is XY.
That is also why the yaml's `filter_bbox` z-range is 0.4-1.8 m.

Difference from MapNav's pipeline (huatu3.py): MapNav renders the map to a PNG
and then recovers object identity by matching RGB values back to a palette with
a +/-5 tolerance. Here the label grid is kept as integers throughout and the
annotation is computed from it directly, so nothing is lost to colour collisions
and the rendered image is a pure output rather than an intermediate.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import mapnav
from .config import ASMConfig

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
N_BASE_CLASSES = 3  # unknown / free / occupied occupy the first label slots

# Distinct, colour-blind-friendly-ish hues for the semantic classes. Base
# classes use greys so labels stay legible on top of them.
_BASE_COLORS = [
    (255, 255, 255),   # unknown
    (214, 216, 212),   # free
    (110, 116, 120),   # occupied
]
_CLASS_COLORS = [
    (230, 126, 90), (232, 178, 74), (150, 190, 90), (86, 180, 152),
    (84, 158, 214), (140, 130, 210), (206, 120, 180), (170, 150, 110),
    (120, 200, 200), (200, 110, 110), (130, 170, 60), (100, 140, 190),
]


@dataclass
class LabeledObject:
    """One annotated blob on the map."""
    label: str
    centroid_world: Tuple[float, float]
    centroid_cell: Tuple[int, int]
    area_m2: float
    votes: int

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "centroid_world_xy": [round(float(c), 3) for c in self.centroid_world],
            "centroid_cell_xy": [int(c) for c in self.centroid_cell],
            "area_m2": round(float(self.area_m2), 3),
            "votes": int(self.votes),
        }


class AnnotatedSemanticMap:
    """Accumulates geometry and semantics into one top-down grid."""

    def __init__(self, cfg: ASMConfig, intrinsic_matrix: np.ndarray):
        self.cfg = cfg
        self.K = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
        self.n = cfg.grid_size

        self.categories: List[str] = list(cfg.categories)
        self.sem_votes = np.zeros((len(self.categories), self.n, self.n), np.uint16)
        self.free_votes = np.zeros((self.n, self.n), np.uint16)
        self.occ_votes = np.zeros((self.n, self.n), np.uint16)

        self.origin_xy: Optional[np.ndarray] = None  # world XY of cell (0, 0)
        self.trajectory: List[Tuple[float, float]] = []
        self.last_pose: Optional[np.ndarray] = None
        self.n_geometry_updates = 0
        self.n_semantic_updates = 0

    # ------------------------------------------------------------------ setup

    def _anchor(self, world_xy: np.ndarray) -> None:
        half = 0.5 * self.cfg.extent_m
        self.origin_xy = np.asarray(world_xy, dtype=np.float64) - half

    def _to_cells(self, pts_xy: np.ndarray) -> np.ndarray:
        """World XY (N, 2) -> integer cell indices (N, 2) as (ix, iy)."""
        return np.floor((pts_xy - self.origin_xy) / self.cfg.resolution_m).astype(np.int32)

    def _in_bounds(self, cells: np.ndarray) -> np.ndarray:
        return (
            (cells[:, 0] >= 0) & (cells[:, 0] < self.n)
            & (cells[:, 1] >= 0) & (cells[:, 1] < self.n)
        )

    # ------------------------------------------------------------- projection

    def backproject(self, depth: np.ndarray, W_T_C2: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Depth image -> world points.

        Returns (points (H, W, 3) in world coordinates, valid mask (H, W)).
        Mirrors the pinhole convention used at frontier/detector.py:344-349.
        """
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        h, w = depth.shape

        valid = np.isfinite(depth) & (depth > self.cfg.min_depth_m) \
            & (depth < self.cfg.max_depth_m)

        us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        z = depth
        x = (us - cx) * z / fx
        y = (vs - cy) * z / fy
        cam = np.stack([x, y, z], axis=-1)                      # (H, W, 3)

        R = W_T_C2[:3, :3]
        t = W_T_C2[:3, 3]
        world = cam.reshape(-1, 3) @ R.T + t
        return world.reshape(h, w, 3), valid

    # ------------------------------------------------------------- integration

    def integrate_geometry(self, depth: np.ndarray, W_T_C2: np.ndarray,
                           floor_z: float) -> None:
        """Fold one depth frame into the free/occupied layers."""
        if self.origin_xy is None:
            self._anchor(W_T_C2[:2, 3])

        world, valid = self.backproject(depth, W_T_C2)
        pts = world[valid]
        if pts.size == 0:
            return

        rel_z = pts[:, 2] - floor_z
        cells = self._to_cells(pts[:, :2])
        ok = self._in_bounds(cells)

        floor_sel = ok & (rel_z < self.cfg.floor_band_m)
        obst_sel = ok & (rel_z >= self.cfg.obstacle_band_lo_m) \
            & (rel_z <= self.cfg.obstacle_band_hi_m)

        self._accumulate(self.free_votes, cells[floor_sel])
        self._accumulate(self.occ_votes, cells[obst_sel])

        self.trajectory.append((float(W_T_C2[0, 3]), float(W_T_C2[1, 3])))
        self.last_pose = W_T_C2.copy()
        self.n_geometry_updates += 1

    def integrate_semantics(self, depth: np.ndarray, W_T_C2: np.ndarray,
                            floor_z: float, masks: Dict[str, np.ndarray]) -> None:
        """Fold per-category masks into the semantic vote layers."""
        if not masks:
            return
        if self.origin_xy is None:
            self._anchor(W_T_C2[:2, 3])

        world, valid = self.backproject(depth, W_T_C2)
        rel_z_full = world[:, :, 2] - floor_z
        # Objects live above the floor and below the ceiling band.
        band = (rel_z_full >= 0.0) & (rel_z_full <= self.cfg.obstacle_band_hi_m)

        for category, mask in masks.items():
            if category not in self.categories:
                continue
            idx = self.categories.index(category)
            sel = valid & band & np.asarray(mask, dtype=bool)
            if not sel.any():
                continue
            cells = self._to_cells(world[sel][:, :2])
            cells = cells[self._in_bounds(cells)]
            self._accumulate(self.sem_votes[idx], cells)

        self.n_semantic_updates += 1

    @staticmethod
    def _accumulate(layer: np.ndarray, cells: np.ndarray) -> None:
        """Saturating += 1 at the given (ix, iy) cells."""
        if cells.size == 0:
            return
        flat = np.ravel_multi_index((cells[:, 1], cells[:, 0]), layer.shape)
        counts = np.bincount(flat, minlength=layer.size).astype(np.uint32)
        updated = layer.reshape(-1).astype(np.uint32) + counts
        np.clip(updated, 0, np.iinfo(np.uint16).max, out=updated)
        layer[:] = updated.reshape(layer.shape).astype(np.uint16)

    # ------------------------------------------------------------------ output

    def label_grid(self) -> np.ndarray:
        """Integer label per cell: 0 unknown, 1 free, 2 occupied, 3+ semantic."""
        grid = np.full((self.n, self.n), UNKNOWN, dtype=np.uint8)
        grid[self.free_votes > 0] = FREE
        grid[self.occ_votes >= self.free_votes] = OCCUPIED
        grid[(self.free_votes == 0) & (self.occ_votes == 0)] = UNKNOWN

        if self.sem_votes.size:
            best = self.sem_votes.argmax(axis=0)
            best_votes = self.sem_votes.max(axis=0)
            strong = best_votes >= self.cfg.min_votes_per_cell
            grid[strong] = (N_BASE_CLASSES + best[strong]).astype(np.uint8)
        return grid

    def _palette(self) -> np.ndarray:
        colors = list(_BASE_COLORS)
        for i in range(len(self.categories)):
            colors.append(_CLASS_COLORS[i % len(_CLASS_COLORS)])
        return np.asarray(colors, dtype=np.uint8)

    def find_objects(self, grid: Optional[np.ndarray] = None) -> List[LabeledObject]:
        """Connected components per semantic class -> labelled blobs.

        Same idea as MapNav's `process_semantic_map`, but run on the label grid
        rather than on a re-decoded PNG.
        """
        if grid is None:
            grid = self.label_grid()
        objects: List[LabeledObject] = []
        cell_area = self.cfg.resolution_m ** 2

        for i, category in enumerate(self.categories):
            mask = (grid == (N_BASE_CLASSES + i)).astype(np.uint8)
            if not mask.any():
                continue
            n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for k in range(1, n_labels):
                area_cells = int(stats[k, cv2.CC_STAT_AREA])
                if area_cells < self.cfg.min_blob_area_cells:
                    continue
                cx, cy = centroids[k]
                world_x = self.origin_xy[0] + (cx + 0.5) * self.cfg.resolution_m
                world_y = self.origin_xy[1] + (cy + 0.5) * self.cfg.resolution_m
                objects.append(LabeledObject(
                    label=category,
                    centroid_world=(float(world_x), float(world_y)),
                    centroid_cell=(int(round(cx)), int(round(cy))),
                    area_m2=area_cells * cell_area,
                    votes=int(self.sem_votes[i].sum()),
                ))

        objects.sort(key=lambda o: o.area_m2, reverse=True)
        return objects

    def render(self, grid: Optional[np.ndarray] = None,
               crop: bool = True) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Render the label grid to RGB. Returns (image, (x0, y0) crop offset)."""
        if grid is None:
            grid = self.label_grid()
        rgb = self._palette()[grid]

        x0 = y0 = 0
        if crop:
            observed = np.argwhere(grid != UNKNOWN)
            if observed.size:
                pad = 10
                y0 = max(0, int(observed[:, 0].min()) - pad)
                y1 = min(self.n, int(observed[:, 0].max()) + pad + 1)
                x0 = max(0, int(observed[:, 1].min()) - pad)
                x1 = min(self.n, int(observed[:, 1].max()) + pad + 1)
                rgb = rgb[y0:y1, x0:x1]
        return rgb, (x0, y0)

    # ------------------------------------------------- MapNav-faithful path

    def mapnav_label_grid(self) -> np.ndarray:
        """Label grid in MapNav's indexing (r2rnav_agent_nohis.py:495-511).

        0 unexplored, 1 obstacle, 2 explored, 3 visited/edge, 5+ semantic --
        the palette slots, so `book` (15) is skipped exactly as MapNav skips it.
        """
        grid = np.full((self.n, self.n), mapnav.UNEXPLORED, dtype=np.uint8)
        grid[self.free_votes > 0] = mapnav.EXPLORED
        grid[self.occ_votes >= self.free_votes] = mapnav.OBSTACLE
        grid[(self.free_votes == 0) & (self.occ_votes == 0)] = mapnav.UNEXPLORED

        if self.sem_votes.size:
            best = self.sem_votes.argmax(axis=0)
            best_votes = self.sem_votes.max(axis=0)
            strong = best_votes >= self.cfg.min_votes_per_cell
            slots = np.array(
                [mapnav.CATEGORY_PALETTE_INDEX.get(c, 0) for c in self.categories],
                dtype=np.uint8,
            )
            if slots.size:
                grid[strong] = slots[best[strong]]

        # MapNav paints the trajectory into the map itself, not over it, and its
        # `visited_vis` is a continuously filled path -- so connect consecutive
        # poses rather than stamping isolated cells (they can be metres apart).
        if self.trajectory:
            cells = self._to_cells(np.asarray(self.trajectory, dtype=np.float64))
            visited = np.zeros((self.n, self.n), np.uint8)
            if len(cells) > 1:
                cv2.polylines(visited, [cells.reshape(-1, 1, 2).astype(np.int32)],
                              False, 1, thickness=1)
            else:
                c = cells[0]
                if 0 <= c[0] < self.n and 0 <= c[1] < self.n:
                    visited[c[1], c[0]] = 1
            grid[visited == 1] = mapnav.VISITED
        return grid

    def _local_window(self, grid: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Crop MapNav's local window, centred on the agent."""
        half = self.cfg.local_window_cells // 2
        if self.last_pose is not None:
            c = self._to_cells(self.last_pose[:2, 3].reshape(1, 2))[0]
            cx, cy = int(c[0]), int(c[1])
        else:
            cx = cy = self.n // 2
        x0 = int(np.clip(cx - half, 0, max(0, self.n - 2 * half)))
        y0 = int(np.clip(cy - half, 0, max(0, self.n - 2 * half)))
        return grid[y0:y0 + 2 * half, x0:x0 + 2 * half], x0, y0

    def annotate_mapnav(self) -> Tuple[np.ndarray, List[LabeledObject], List[str]]:
        """MapNav's exact ASM: local window -> palette render -> colour decode.

        Returns the annotated image, the blobs with world coordinates recovered,
        and MapNav's duplicate-preserving name list (what its prompt joins).
        """
        grid = self.mapnav_label_grid()
        window, x0, y0 = self._local_window(grid)
        if window.size == 0:
            size = self.cfg.render_size
            return np.full((size, size, 3), 255, np.uint8), [], []

        size = self.cfg.render_size
        rgb = mapnav.render_label_grid(window, size)

        # MapNav's saved map already carries the agent marker, and huatu3
        # ignores it because red is not a decodable class colour.
        if self.last_pose is not None:
            wh = window.shape[0]
            scale = size / float(wh)
            c = self._to_cells(self.last_pose[:2, 3].reshape(1, 2))[0]
            px = int((c[0] - x0) * scale)
            py = int((wh - 1 - (c[1] - y0)) * scale)   # flipud
            heading = self.last_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
            tip = (int(px + heading[0] * 20), int(py - heading[1] * 20))
            cv2.arrowedLine(rgb, (px, py), tip, (245, 92, 66), 2, tipLength=0.4)

        annotated, labels, objects = mapnav.process_semantic_map(
            rgb, font_size=self.cfg.font_size
        )

        # Recover world coordinates for the blobs MapNav accepted.
        wh = window.shape[0]
        scale = size / float(wh)
        cell_area = self.cfg.resolution_m ** 2
        found: List[LabeledObject] = []
        for lab in labels:
            ux, uy = lab["centroid_px"]
            gx = x0 + ux / scale
            gy = y0 + (wh - 1 - uy / scale)
            found.append(LabeledObject(
                label=lab["object"],
                centroid_world=(
                    float(self.origin_xy[0] + (gx + 0.5) * self.cfg.resolution_m),
                    float(self.origin_xy[1] + (gy + 0.5) * self.cfg.resolution_m),
                ),
                centroid_cell=(int(gx), int(gy)),
                area_m2=lab["area_px"] / (scale ** 2) * cell_area,
                votes=int(lab["area_px"]),
            ))
        return annotated, found, objects

    def annotate(self, target_size: int = 640
                 ) -> Tuple[np.ndarray, List[LabeledObject]]:
        """Render, draw the trajectory and pose, and stamp object name labels."""
        if self.cfg.mapnav_faithful:
            image, found, _ = self.annotate_mapnav()
            return image, found

        grid = self.label_grid()
        objects = self.find_objects(grid)
        rgb, (x0, y0) = self.render(grid)

        if rgb.size == 0:
            return np.zeros((target_size, target_size, 3), np.uint8), objects

        h, w = rgb.shape[:2]
        scale = max(1.0, target_size / max(h, w))
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_NEAREST)

        # The grid is built with +y northward, so flip to put north up. This
        # has to happen BEFORE anything is drawn, otherwise the text labels
        # come out mirrored.
        rgb = np.flipud(rgb).copy()
        out_h = rgb.shape[0]

        def to_px(cell_x: float, cell_y: float) -> Tuple[int, int]:
            px = int((cell_x - x0) * scale)
            py = out_h - 1 - int((cell_y - y0) * scale)
            return px, py

        # Trajectory, then current pose as a heading arrow.
        if len(self.trajectory) > 1:
            pts = []
            for wx, wy in self.trajectory:
                c = (np.array([wx, wy]) - self.origin_xy) / self.cfg.resolution_m
                pts.append(to_px(c[0], c[1]))
            cv2.polylines(rgb, [np.asarray(pts, np.int32)], False, (200, 40, 40), 2)

        if self.last_pose is not None:
            c = (self.last_pose[:2, 3] - self.origin_xy) / self.cfg.resolution_m
            px, py = to_px(c[0], c[1])
            heading = self.last_pose[:3, :3] @ np.array([0.0, 0.0, 1.0])
            # y is inverted on screen after the flip.
            tip = (int(px + heading[0] * 18), int(py - heading[1] * 18))
            cv2.arrowedLine(rgb, (px, py), tip, (220, 30, 30), 2, tipLength=0.4)

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", self.cfg.font_size)
        except Exception:
            font = ImageFont.load_default()

        for obj in objects:
            px, py = to_px(*obj.centroid_cell)
            box = draw.textbbox((0, 0), obj.label, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
            x = int(np.clip(px - tw // 2, 0, image.width - tw - 1))
            y = int(np.clip(py - th // 2, 0, image.height - th - 1))
            pad = 3
            draw.rounded_rectangle(
                (x - pad, y - pad, x + tw + pad, y + th + pad),
                radius=4, fill=(255, 165, 0),
            )
            draw.text((x, y), obj.label, fill=(0, 0, 0), font=font)

        return np.array(image), objects

    # ---------------------------------------------------------------- summary

    def prompt_text(self, objects: Optional[List[LabeledObject]] = None) -> str:
        """The text half of MapNav's ASM: what is on the map, in words.

        This is the piece worth having even when the image is not consumed --
        the frontier utility at frontier/manager.py:905-944 currently runs with
        a constant probability, so a textual object inventory is the cheapest
        real semantic signal available to a frontier scorer.
        """
        if objects is None:
            _, objects = self.annotate()
        if not objects:
            return ""

        if self.cfg.mapnav_faithful:
            # MapNav joins the raw per-blob list, duplicates and all.
            return mapnav.prompt_sentence([o.label for o in objects]).strip()

        counts: Dict[str, int] = {}
        for obj in objects:
            counts[obj.label] = counts.get(obj.label, 0) + 1
        parts = [f"{n} {name}" if n > 1 else name for name, n in counts.items()]
        return (
            "As shown, this semantic map includes objects such as "
            + ", ".join(parts) + "."
        )
