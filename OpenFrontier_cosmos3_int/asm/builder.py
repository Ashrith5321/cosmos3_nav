"""Background worker that keeps the annotated semantic map up to date.

Contract with the navigation loop: `submit()` copies its inputs into a bounded
queue and returns immediately. When the worker falls behind, frames are dropped
rather than queued, so enabling the ASM can never stall navigation. Geometry is
folded in for every frame that survives the queue; semantics (the expensive
SAM3 calls) only every Nth frame and only after the agent has actually moved.
"""
import json
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import mapnav
from .config import ASMConfig
from .segmenter import Sam3MultiSegmenter
from .semantic_map import AnnotatedSemanticMap, LabeledObject


class ASMBuilder:
    """Owns the map, the worker thread and the on-disk snapshots."""

    def __init__(self, cfg: ASMConfig, intrinsic_matrix: np.ndarray,
                 out_dir: Path, logger=None):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger

        self.map = AnnotatedSemanticMap(cfg, intrinsic_matrix)
        self.segmenter = Sam3MultiSegmenter(
            cfg.categories, cfg.sam3_port, cfg.sam3_score_threshold, logger=logger
        )

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=cfg.queue_size)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="asm-builder",
                                        daemon=True)

        self.frames_submitted = 0
        self.frames_dropped = 0
        self.frames_processed = 0
        self.segmentation_calls = 0
        self.seconds_in_segmentation = 0.0

        self._last_seg_pose: Optional[np.ndarray] = None
        self._frames_since_seg = 0
        self._updates_since_write = 0
        self._latest_objects: List[LabeledObject] = []

        self._thread.start()

    def log(self, msg: str) -> None:
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:
                pass

    # ------------------------------------------------------------- producer

    def submit(self, rgb: np.ndarray, depth: np.ndarray, W_T_C2: np.ndarray,
               floor_z: float, step: int) -> None:
        """Non-blocking. Called from the navigation thread."""
        if self._stop.is_set():
            return
        self.frames_submitted += 1
        try:
            # Copy: the caller reuses/overwrites these buffers.
            item = (np.array(rgb, copy=True),
                    np.array(depth, copy=True),
                    np.array(W_T_C2, copy=True),
                    float(floor_z), int(step))
            self._queue.put_nowait(item)
        except queue.Full:
            self.frames_dropped += 1

    # ------------------------------------------------------------- consumer

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break

            # Drain everything else that arrived while we were busy. Geometry is
            # cheap, so it is integrated for every frame that made it into the
            # queue; only the freshest frame is considered for segmentation.
            batch = [item]
            stop_after = False
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    stop_after = True
                    self._queue.task_done()
                    break
                batch.append(nxt)

            try:
                self._process_batch(batch)
            except Exception as exc:
                self.log(f"ASM: worker error: {exc!r}")
            finally:
                for _ in batch:
                    self._queue.task_done()
            if stop_after:
                break

    def _process_batch(self, batch: List[tuple]) -> None:
        with self._lock:
            for rgb, depth, W_T_C2, floor_z, step in batch:
                self.map.integrate_geometry(depth, W_T_C2, floor_z)
        self.frames_processed += len(batch)
        self._frames_since_seg += len(batch)

        rgb, depth, W_T_C2, floor_z, step = batch[-1]
        if self._should_segment(W_T_C2):
            t0 = time.time()
            masks = self.segmenter.segment(rgb)
            self.seconds_in_segmentation += time.time() - t0
            self.segmentation_calls += 1
            self._frames_since_seg = 0
            self._last_seg_pose = W_T_C2.copy()
            if masks:
                with self._lock:
                    self.map.integrate_semantics(depth, W_T_C2, floor_z, masks)

        self._updates_since_write += 1
        if self._updates_since_write >= self.cfg.write_every_n_updates:
            self._updates_since_write = 0
            self._write(step)

    def _should_segment(self, W_T_C2: np.ndarray) -> bool:
        if self._frames_since_seg < self.cfg.segment_every_n_frames:
            return False
        if self._last_seg_pose is None:
            return True
        moved = float(np.linalg.norm(W_T_C2[:2, 3] - self._last_seg_pose[:2, 3]))
        if moved >= self.cfg.min_translation_m:
            return True
        # Rotation about the world Z axis, from the camera forward vector.
        def yaw(T: np.ndarray) -> float:
            f = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
            return float(np.degrees(np.arctan2(f[1], f[0])))
        delta = abs(yaw(W_T_C2) - yaw(self._last_seg_pose))
        delta = min(delta, 360.0 - delta)
        return delta >= self.cfg.min_rotation_deg

    # ---------------------------------------------------------------- output

    def _write(self, step: int) -> None:
        with self._lock:
            if self.cfg.mapnav_faithful:
                image, objects, names = self.map.annotate_mapnav()
            else:
                image, objects = self.map.annotate()
                names = [o.label for o in objects]
            summary = self.map.prompt_text(objects)
            block = mapnav.observation_block("", names)
            stats = self.stats()
        self._latest_objects = objects

        payload = {
            "step": int(step),
            "mapnav_faithful": bool(self.cfg.mapnav_faithful),
            # MapNav's own output: the per-blob name list its prompt joins.
            "mapnav_objects": names,
            "prompt_text": summary,
            "semantic_map_block": block,
            # Additive: world coordinates for each accepted blob.
            "objects": [o.to_dict() for o in objects],
            "stats": stats,
        }

        cv2.imwrite(str(self.out_dir / "latest_asm.png"), image[:, :, ::-1])
        (self.out_dir / "latest_asm.json").write_text(json.dumps(payload, indent=2))
        if self.cfg.keep_snapshots:
            cv2.imwrite(str(self.out_dir / f"{step:06d}_asm.png"), image[:, :, ::-1])
            (self.out_dir / f"{step:06d}_asm.json").write_text(
                json.dumps(payload, indent=2)
            )

    # ------------------------------------------------------------- accessors

    def latest(self) -> Tuple[np.ndarray, List[LabeledObject], str]:
        """Current annotated image, object list and prompt sentence."""
        with self._lock:
            image, objects = self.map.annotate()
            return image, objects, self.map.prompt_text(objects)

    def latest_objects(self) -> List[LabeledObject]:
        return list(self._latest_objects)

    def stats(self) -> dict:
        return {
            "frames_submitted": self.frames_submitted,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "segmentation_calls": self.segmentation_calls,
            "seconds_in_segmentation": round(self.seconds_in_segmentation, 2),
            "geometry_updates": self.map.n_geometry_updates,
            "semantic_updates": self.map.n_semantic_updates,
        }

    def close(self, final_step: int = 0, timeout: float = 20.0) -> None:
        """Drain what is queued, write a final snapshot, stop the thread."""
        if self._stop.is_set():
            return
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)
        try:
            self._write(final_step)
            (self.out_dir / "asm_stats.json").write_text(
                json.dumps(self.stats(), indent=2)
            )
        except Exception as exc:
            self.log(f"ASM: final write failed: {exc!r}")
