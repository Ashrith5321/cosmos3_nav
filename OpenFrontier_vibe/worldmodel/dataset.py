"""
Self-supervised beyond-frontier dataset generation (design sections 23-25).

During an episode, WMDataRecorder captures (a) each frontier's goal-agnostic
context (crop/scene embeddings + geometry) at detection time and (b) every
observation embedding with its camera pose. At episode end, each frontier's
"beyond-region" label is built from the observations that were captured
LATER than the frontier's detection, from within/behind the frontier region
(camera within `reveal_radius`, or beyond the frontier along its view
direction within `beyond_range`). Labels:

  - future embedding: mean encoder embedding of beyond-region keyframes
  - room pseudo-label: zero-shot room classification of those keyframes
  - object pseudo-labels: zero-shot multi-label over OBJECT_VOCAB
  - realized information gain: from the frontier's own gain trajectory

Counterfactual coverage (section 24) comes for free: every detected
frontier of a state contributes a (context -> outcome) sample; frontiers
never revealed yield no label and are dropped.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from worldmodel.vocab import OBJECT_VOCAB, ROOM_SCENE_PROMPTS, ROOM_TYPES

logger = logging.getLogger(__name__)


@dataclass
class _FrontierSample:
    uid: str
    step: int
    crop_embedding: Optional[np.ndarray]
    scene_embedding: Optional[np.ndarray]
    geom: np.ndarray
    pos3d: np.ndarray
    view_direction: np.ndarray


@dataclass
class _Observation:
    step: int
    position: np.ndarray
    forward: np.ndarray
    embedding: np.ndarray


class WMDataRecorder:
    def __init__(
        self,
        out_dir: str,
        encoder,
        reveal_radius: float = 1.5,
        beyond_range: float = 4.0,
        obs_stride: int = 2,
        object_sim_threshold: float = 0.23,
    ):
        self.out_dir = out_dir
        self.encoder = encoder
        self.reveal_radius = reveal_radius
        self.beyond_range = beyond_range
        self.obs_stride = obs_stride
        self.object_sim_threshold = object_sim_threshold

        os.makedirs(out_dir, exist_ok=True)
        self._frontiers: dict = {}
        self._observations: List[_Observation] = []
        self._obs_counter = 0

        self._room_text_emb = None
        self._object_text_emb = None

    # ---------------- online recording ----------------

    def record_frontier(self, rec, step: int) -> None:
        """Snapshot the frontier's context the first time full context exists."""
        if rec.uid in self._frontiers:
            return
        if rec.pos3d is None or rec.geom_features is None:
            return
        self._frontiers[rec.uid] = _FrontierSample(
            uid=rec.uid,
            step=step,
            crop_embedding=(
                None
                if rec.context_embedding is None
                else rec.context_embedding.copy()
            ),
            scene_embedding=(
                None if rec.scene_embedding is None else rec.scene_embedding.copy()
            ),
            geom=rec.geom_features.copy(),
            pos3d=rec.pos3d.copy(),
            view_direction=(
                rec.view_direction.copy()
                if rec.view_direction is not None
                else np.zeros(3)
            ),
        )

    def record_observation(self, rgb: np.ndarray, W_T_C2: np.ndarray, step: int) -> None:
        self._obs_counter += 1
        if self._obs_counter % self.obs_stride != 0:
            return
        emb = self.encoder.encode_image(rgb[..., :3])
        self._observations.append(
            _Observation(
                step=step,
                position=W_T_C2[:3, 3].copy(),
                forward=W_T_C2[:3, 2].copy(),
                embedding=emb,
            )
        )

    # ---------------- label construction ----------------

    def _beyond_observations(self, fs: _FrontierSample) -> List[_Observation]:
        """Observations that reveal the region beyond frontier ``fs``."""
        vd = fs.view_direction
        vd_norm = np.linalg.norm(vd)
        vd = vd / vd_norm if vd_norm > 1e-8 else vd
        out = []
        for obs in self._observations:
            if obs.step <= fs.step:
                continue
            rel = obs.position - fs.pos3d
            d = float(np.linalg.norm(rel))
            along = float(rel @ vd)
            # inside the portal, or beyond it along the view direction
            if d < self.reveal_radius or (0.0 < along < self.beyond_range and d < self.beyond_range):
                out.append(obs)
        return out

    def _room_pseudo_label(self, emb: np.ndarray) -> np.ndarray:
        if self._room_text_emb is None:
            prompts = [ROOM_SCENE_PROMPTS[r] for r in ROOM_TYPES]
            self._room_text_emb = self.encoder.encode_texts(prompts)
        sims = self._room_text_emb @ emb
        logits = sims / 0.02
        logits -= logits.max()
        p = np.exp(logits)
        return (p / p.sum()).astype(np.float32)

    def _object_pseudo_labels(self, embs: np.ndarray) -> np.ndarray:
        if self._object_text_emb is None:
            prompts = [f"a photo of a {o.replace('_', ' ')}" for o in OBJECT_VOCAB]
            self._object_text_emb = self.encoder.encode_texts(prompts)
        sims = embs @ self._object_text_emb.T  # (N_frames, O)
        present = (sims > self.object_sim_threshold).any(axis=0)
        return present.astype(np.float32)

    # ---------------- finalization ----------------

    def close(self, memory=None) -> Optional[str]:
        """Build labels and write one npz per episode. Returns the file path."""
        samples = []
        for fs in self._frontiers.values():
            beyond = self._beyond_observations(fs)
            if len(beyond) == 0:
                continue
            frame_embs = np.stack([o.embedding for o in beyond], axis=0)
            z_future = frame_embs.mean(axis=0)
            z_future = z_future / max(np.linalg.norm(z_future), 1e-8)

            d = self.encoder.embed_dim
            samples.append(
                {
                    "uid": fs.uid,
                    "crop_embedding": (
                        fs.crop_embedding
                        if fs.crop_embedding is not None
                        else np.zeros(d, dtype=np.float32)
                    ),
                    "scene_embedding": (
                        fs.scene_embedding
                        if fs.scene_embedding is not None
                        else np.zeros(d, dtype=np.float32)
                    ),
                    "geom": fs.geom,
                    "future_embedding": z_future.astype(np.float32),
                    "room_label": self._room_pseudo_label(z_future),
                    "object_labels": self._object_pseudo_labels(frame_embs),
                    "gain_label": float(np.expm1(fs.geom[0])),
                    "num_beyond_frames": len(beyond),
                }
            )

        if not samples:
            logger.info("WMDataRecorder: no revealed frontiers, nothing written")
            return None

        path = os.path.join(self.out_dir, "beyond_frontier.npz")
        np.savez_compressed(
            path,
            uid=np.array([s["uid"] for s in samples]),
            crop_embedding=np.stack([s["crop_embedding"] for s in samples]),
            scene_embedding=np.stack([s["scene_embedding"] for s in samples]),
            geom=np.stack([s["geom"] for s in samples]),
            future_embedding=np.stack([s["future_embedding"] for s in samples]),
            room_label=np.stack([s["room_label"] for s in samples]),
            object_labels=np.stack([s["object_labels"] for s in samples]),
            gain_label=np.array([s["gain_label"] for s in samples], dtype=np.float32),
            num_beyond_frames=np.array(
                [s["num_beyond_frames"] for s in samples], dtype=np.int64
            ),
        )
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "num_samples": len(samples),
                    "num_frontiers_seen": len(self._frontiers),
                    "num_observations": len(self._observations),
                    "embed_dim": self.encoder.embed_dim,
                },
                f,
                indent=1,
            )
        logger.info("WMDataRecorder: wrote %d samples to %s", len(samples), path)
        return path


class BeyondFrontierDataset:
    """Torch dataset over recorded beyond_frontier.npz files."""

    def __init__(self, npz_paths: List[str]):
        arrays = {
            "crop_embedding": [],
            "scene_embedding": [],
            "geom": [],
            "future_embedding": [],
            "room_label": [],
            "object_labels": [],
            "gain_label": [],
        }
        for path in npz_paths:
            data = np.load(path, allow_pickle=False)
            for key in arrays:
                arrays[key].append(data[key])
        if not arrays["geom"]:
            raise ValueError("No dataset files provided")
        self.data = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}

    def __len__(self) -> int:
        return self.data["geom"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.data.items()}
