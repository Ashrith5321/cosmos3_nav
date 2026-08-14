"""
Generative (pixel-space) world-model backend using Cosmos3 image2image
(design section 6.1 / Phase 3, section 42).

For each frontier, the local view is pushed through a Cosmos3 "move the
camera forward through the opening" edit with K seeds; the K generated
images are the future hypotheses, encoded into the shared CLIP space and
classified into rooms/objects zero-shot. Generation stays goal-agnostic
(the prompt never mentions the goal), evaluation stays goal-conditioned.

Requires worldmodel/cosmos_gen_server.py running under the cosmos venv
(the benchmark env cannot import cosmos_framework).

Caveat (recorded in this repo's prior experiments): Cosmos3 image2image
tends to re-render its conditioning image rather than actually travel
(frontierworld Phase 9D; eval/cosmos3_nav_test 20260811_231249). This
backend exists for the design's Phase-3 comparison and qualitative
figures - the zero-shot / learned latent backends are the defaults.
"""

import base64
import io
import json
import logging
import urllib.request
from typing import List, Optional

import numpy as np

from worldmodel.encoder import BaseEncoder
from worldmodel.predictor import BasePredictor, FrontierContext
from worldmodel.records import FutureHypothesis, WMPrediction
from worldmodel.vocab import (
    ROOM_SCENE_PROMPTS,
    ROOM_TYPES,
    object_prior_vector,
    room_event_text,
)

logger = logging.getLogger(__name__)

GEN_PROMPT = (
    "Move the camera straight forward {advance:.1f} metres, through the "
    "opening ahead, and render the view from the new position inside the "
    "next room or corridor."
)


class CosmosGenerativePredictor(BasePredictor):
    backend_name = "cosmos_gen"

    def __init__(self, encoder: BaseEncoder, params: Optional[dict] = None):
        p = dict(params or {})
        gen = dict(p.get("cosmos") or {})
        self.encoder = encoder
        self.port = int(gen.get("port", 12186))
        self.num_steps = int(gen.get("num_steps", 12))
        self.resolution = str(gen.get("resolution", "480"))
        self.advance_m = float(gen.get("advance_m", 2.5))
        self.timeout_s = float(gen.get("timeout_s", 600.0))
        self.url = f"http://127.0.0.1:{self.port}/generate"

        prompts = [ROOM_SCENE_PROMPTS[r] for r in ROOM_TYPES]
        self._room_text_emb = self.encoder.encode_texts(prompts)

    # ---------------- server I/O ----------------

    def _request_rollouts(self, crop: np.ndarray, seeds: List[int]) -> List[np.ndarray]:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(crop[..., :3]).save(buf, format="PNG")
        payload = json.dumps(
            {
                "image": base64.b64encode(buf.getvalue()).decode(),
                "prompt": GEN_PROMPT.format(advance=self.advance_m),
                "seeds": seeds,
                "num_steps": self.num_steps,
                "resolution": self.resolution,
            }
        ).encode()
        req = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            out = json.loads(resp.read())
        if out.get("result") != "success":
            raise RuntimeError(out.get("message", "generation failed"))
        frames = []
        for img_b64 in out["images"]:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            frames.append(np.asarray(img))
        return frames

    # ---------------- prediction ----------------

    def _room_posterior(self, z: np.ndarray) -> np.ndarray:
        logits = (self._room_text_emb @ z) / 0.02
        logits -= logits.max()
        probs = np.exp(logits)
        return (probs / probs.sum()).astype(np.float32)

    def predict(
        self, ctx: FrontierContext, num_hypotheses: int, step: int, context_hash: str
    ) -> WMPrediction:
        if ctx.crop is None:
            raise ValueError(
                "cosmos_gen backend needs the raw frontier crop "
                "(record kept only embeddings) - cannot generate"
            )
        seeds = list(range(num_hypotheses))
        frames = self._request_rollouts(ctx.crop, seeds)
        if not frames:
            raise RuntimeError("Cosmos3 returned no rollouts")

        embs = self.encoder.encode_images(frames)
        gain = float(np.expm1(ctx.geom[0])) if ctx.geom is not None else 1.0

        hypotheses = []
        for z in embs:
            room_probs = self._room_posterior(z)
            room = ROOM_TYPES[int(np.argmax(room_probs))]
            hypotheses.append(
                FutureHypothesis(
                    embedding=z.astype(np.float32),
                    room_probs=room_probs,
                    object_probs=object_prior_vector(room),
                    events=[room_event_text(room)],
                    info_gain=gain,
                    weight=1.0 / len(embs),
                )
            )
        return WMPrediction(
            uid=ctx.uid,
            hypotheses=hypotheses,
            created_step=step,
            context_hash=context_hash,
            backend=self.backend_name,
        )
