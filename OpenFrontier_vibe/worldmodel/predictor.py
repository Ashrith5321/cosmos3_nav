"""
Frontier-conditioned world-model predictors (design sections 4-6, 22).

Prediction is goal-AGNOSTIC (section 7.2): backends see only the frontier
context, never the language goal. Two backends:

- ZeroShotClipPredictor: no training. Classifies what the frontier's local
  view opens into using CLIP room prompts, then emits one hypothesis per
  plausible room: language-event text + its embedding, and objects from the
  room->object co-occurrence prior. Multi-hypothesis weights come from the
  room posterior, so ambiguity produces genuine hypothesis spread.

- LearnedPredictor: a small frontier-conditioned transformer over frozen
  encoder features with K hypothesis queries and multi-head outputs
  (embedding / room / objects / gain), trained with winner-takes-all
  multi-hypothesis losses (worldmodel/train.py).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from worldmodel.encoder import BaseEncoder
from worldmodel.records import FutureHypothesis, WMPrediction
from worldmodel.vocab import (
    AFFORDANCES,
    OBJECT_VOCAB,
    ROOM_CONTEXT_PROMPTS,
    ROOM_TYPES,
    object_prior_vector,
    room_event_text,
)

logger = logging.getLogger(__name__)

GEOM_DIM = 8


@dataclass
class FrontierContext:
    """Goal-agnostic evidence bundle for one frontier (design section 13.1)."""

    uid: str
    crop_embedding: Optional[np.ndarray]   # (D,) local crop around frontier pixel
    scene_embedding: Optional[np.ndarray]  # (D,) full keyframe
    geom: np.ndarray                       # (GEOM_DIM,) geometric features
    crop: Optional[np.ndarray] = None      # raw crop (dataset recording only)


def make_geom_features(
    gain: float,
    u_gain: float,
    view_direction: np.ndarray,
    rel_distance: float,
    depth_at_frontier: float,
    n_parents: int,
) -> np.ndarray:
    vd = np.asarray(view_direction, dtype=np.float32).reshape(3)
    return np.array(
        [
            np.log1p(max(gain, 0.0)),
            np.log1p(max(u_gain, 0.0)),
            vd[0],
            vd[1],
            vd[2],
            min(rel_distance / 10.0, 2.0),
            min(depth_at_frontier / 10.0, 2.0),
            min(n_parents / 10.0, 2.0),
        ],
        dtype=np.float32,
    )


class BasePredictor:
    backend_name = "base"

    def predict(
        self, ctx: FrontierContext, num_hypotheses: int, step: int, context_hash: str
    ) -> WMPrediction:
        raise NotImplementedError


class ZeroShotClipPredictor(BasePredictor):
    backend_name = "zero_shot"

    def __init__(self, encoder: BaseEncoder, params: Optional[dict] = None):
        p = params or {}
        self.encoder = encoder
        self.temperature = float(p.get("room_temperature", 0.02))
        self.context_blend = float(p.get("context_blend", 0.65))  # crop vs scene
        self.event_blend = float(p.get("event_blend", 0.6))  # event text vs context
        room_prompts = [ROOM_CONTEXT_PROMPTS[r] for r in ROOM_TYPES]
        self._room_text_emb = self.encoder.encode_texts(room_prompts)  # (R, D)
        self._event_emb_cache = {}

    def _context_embedding(self, ctx: FrontierContext) -> Optional[np.ndarray]:
        crop, scene = ctx.crop_embedding, ctx.scene_embedding
        if crop is None and scene is None:
            return None
        if crop is None:
            return scene
        if scene is None:
            return crop
        z = self.context_blend * crop + (1.0 - self.context_blend) * scene
        n = np.linalg.norm(z)
        return z / n if n > 1e-8 else z

    def _room_posterior(self, z_ctx: Optional[np.ndarray]) -> np.ndarray:
        if z_ctx is None:
            return np.full(len(ROOM_TYPES), 1.0 / len(ROOM_TYPES), dtype=np.float64)
        sims = self._room_text_emb @ z_ctx  # (R,)
        logits = sims / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def _event_embedding(self, room: str) -> np.ndarray:
        if room not in self._event_emb_cache:
            self._event_emb_cache[room] = self.encoder.encode_text(
                room_event_text(room)
            )
        return self._event_emb_cache[room]

    def predict(
        self, ctx: FrontierContext, num_hypotheses: int, step: int, context_hash: str
    ) -> WMPrediction:
        z_ctx = self._context_embedding(ctx)
        room_probs = self._room_posterior(z_ctx)

        k = min(num_hypotheses, len(ROOM_TYPES))
        top_rooms = np.argsort(-room_probs)[:k]
        weights = room_probs[top_rooms]
        weights = weights / weights.sum()

        # Affordance heuristic from geometry: high gain -> open space,
        # low gain -> corridor/doorway-like.
        gain = float(np.expm1(ctx.geom[0])) if ctx.geom is not None else 1.0
        open_prob = float(np.clip(gain / 8.0, 0.05, 0.95))
        affordances = np.array(
            [
                0.9,               # traversable
                open_prob,         # enterable_room
                1.0 - open_prob,   # corridor
                0.5,               # doorway
                0.15,              # dead_end
                0.05,              # stairs
                open_prob,         # open_space
                0.3,               # cluttered
            ],
            dtype=np.float32,
        )
        assert len(affordances) == len(AFFORDANCES)

        hypotheses: List[FutureHypothesis] = []
        for room_idx, w in zip(top_rooms, weights):
            room = ROOM_TYPES[int(room_idx)]
            # per-hypothesis room distribution: committed to this room but
            # keeping some posterior mass structure
            h_room = 0.15 * room_probs.astype(np.float32)
            h_room[room_idx] += 0.85
            h_room = h_room / h_room.sum()

            z_event = self._event_embedding(room)
            if z_ctx is not None:
                z = self.event_blend * z_event + (1.0 - self.event_blend) * z_ctx
            else:
                z = z_event
            n = np.linalg.norm(z)
            z = z / n if n > 1e-8 else z

            hypotheses.append(
                FutureHypothesis(
                    embedding=z.astype(np.float32),
                    room_probs=h_room,
                    object_probs=object_prior_vector(room),
                    affordance_probs=affordances,
                    events=[room_event_text(room)],
                    info_gain=gain,
                    weight=float(w),
                )
            )

        return WMPrediction(
            uid=ctx.uid,
            hypotheses=hypotheses,
            created_step=step,
            context_hash=context_hash,
            backend=self.backend_name,
        )


class LearnedPredictor(BasePredictor):
    """Wrapper around the trained FrontierWorldModelNet checkpoint."""

    backend_name = "learned"

    def __init__(
        self,
        encoder: BaseEncoder,
        checkpoint: str,
        params: Optional[dict] = None,
    ):
        import torch

        from worldmodel.net import FrontierWorldModelNet

        p = params or {}
        self.torch = torch
        self.encoder = encoder
        self.device = p.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.net = FrontierWorldModelNet(**ckpt["net_kwargs"])
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()
        if self.net.embed_dim != encoder.embed_dim:
            raise ValueError(
                f"Checkpoint embed_dim {self.net.embed_dim} does not match "
                f"encoder embed_dim {encoder.embed_dim}"
            )
        logger.info("Loaded learned world model from %s", checkpoint)

    def predict(
        self, ctx: FrontierContext, num_hypotheses: int, step: int, context_hash: str
    ) -> WMPrediction:
        torch = self.torch
        d = self.encoder.embed_dim
        crop = ctx.crop_embedding if ctx.crop_embedding is not None else np.zeros(d)
        scene = ctx.scene_embedding if ctx.scene_embedding is not None else np.zeros(d)
        with torch.no_grad():
            out = self.net(
                torch.as_tensor(crop, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(scene, dtype=torch.float32, device=self.device)[None],
                torch.as_tensor(ctx.geom, dtype=torch.float32, device=self.device)[
                    None
                ],
            )
        emb = out["embedding"][0].cpu().numpy()          # (K, D)
        rooms = out["room_probs"][0].cpu().numpy()       # (K, R)
        objects = out["object_probs"][0].cpu().numpy()   # (K, O)
        gains = out["gain"][0].cpu().numpy()             # (K,)
        weights = out["weights"][0].cpu().numpy()        # (K,)

        k = min(num_hypotheses, emb.shape[0])
        order = np.argsort(-weights)[:k]
        hypotheses = []
        for i in order:
            room = ROOM_TYPES[int(np.argmax(rooms[i]))]
            hypotheses.append(
                FutureHypothesis(
                    embedding=emb[i],
                    room_probs=rooms[i],
                    object_probs=objects[i],
                    affordance_probs=None,
                    events=[room_event_text(room)],
                    info_gain=float(gains[i]),
                    weight=float(weights[i]),
                )
            )
        return WMPrediction(
            uid=ctx.uid,
            hypotheses=hypotheses,
            created_step=step,
            context_hash=context_hash,
            backend=self.backend_name,
        )


def build_predictor(params: dict, encoder: BaseEncoder) -> BasePredictor:
    backend = params.get("backend", "zero_shot")
    if backend == "zero_shot":
        return ZeroShotClipPredictor(encoder, params)
    if backend == "learned":
        ckpt = params.get("checkpoint")
        if not ckpt:
            raise ValueError("learned backend requires world_model.checkpoint")
        return LearnedPredictor(encoder, ckpt, params)
    if backend == "cosmos_gen":
        from worldmodel.generative import CosmosGenerativePredictor

        return CosmosGenerativePredictor(encoder, params)
    raise ValueError(f"Unknown world model backend: {backend}")
