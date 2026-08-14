"""
Goal-conditioned evaluation of predicted futures (design sections 7.2, 8, 9).

Prediction stays goal-agnostic; this module answers "does that predicted
future help satisfy the current language goal?". Channels:

- embedding channel: contrast-normalized cosine between the hypothesis
  embedding and the goal text embedding (raw CLIP cosines live in a
  narrow band, so scores are normalized against distractor texts).
- object channel: predicted P(goal object | hypothesis) when the goal
  maps onto the object vocabulary.
- optional VLM channel (section 8.2/8.3): re-scores predicted language
  events with the OpenFrontier VLM server for relational goals or
  ambiguous rankings. Disabled by default.
"""

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from worldmodel.encoder import BaseEncoder
from worldmodel.records import FutureHypothesis, GoalScore, WMPrediction
from worldmodel.vocab import DISTRACTOR_TEXTS, match_goal_to_vocab

logger = logging.getLogger(__name__)

RELATIONAL_PATTERN = re.compile(
    r"\b(next to|near|beside|on top of|on the|in the|under|above|between|close to)\b",
    re.IGNORECASE,
)


def is_relational_goal(goal: str) -> bool:
    return RELATIONAL_PATTERN.search(goal) is not None


class GoalEvaluator:
    def __init__(self, encoder: BaseEncoder, params: Optional[dict] = None):
        p = params or {}
        self.encoder = encoder
        self.temperature = float(p.get("temperature", 0.07))
        self.embed_weight = float(p.get("embed_weight", 0.5))
        self.object_weight = float(p.get("object_weight", 0.5))
        self._goal_cache = {}
        self._distractor_emb = self.encoder.encode_texts(DISTRACTOR_TEXTS)

    # ---------------- embeddings ----------------

    def goal_embedding(self, goal: str) -> np.ndarray:
        if goal not in self._goal_cache:
            texts = [
                f"a {goal}",
                f"a room containing a {goal}",
            ]
            embs = self.encoder.encode_texts(texts)
            z = embs.mean(axis=0)
            self._goal_cache[goal] = z / max(np.linalg.norm(z), 1e-8)
        return self._goal_cache[goal]

    def _normalized_cosine(self, z: np.ndarray, z_goal: np.ndarray) -> float:
        """Softmax of goal similarity against distractor similarities -> [0,1]."""
        t = max(self.temperature, 1e-6)
        s_goal = float(z @ z_goal) / t
        s_dis = (self._distractor_emb @ z) / t
        m = max(s_goal, float(s_dis.max()))
        e_goal = np.exp(s_goal - m)
        denom = e_goal + np.exp(s_dis - m).sum()
        return float(e_goal / denom)

    # ---------------- scoring ----------------

    def score_hypothesis(self, hyp: FutureHypothesis, goal: str) -> float:
        z_goal = self.goal_embedding(goal)
        s_embed = self._normalized_cosine(hyp.embedding, z_goal)

        obj_idx = match_goal_to_vocab(goal)
        if obj_idx is not None:
            s_obj = float(hyp.object_probs[obj_idx])
            return self.embed_weight * s_embed + self.object_weight * s_obj
        return s_embed

    def score_prediction(self, pred: WMPrediction, goal: str) -> GoalScore:
        """Expected relevance mu and hypothesis spread sigma (design section 9)."""
        scores = [self.score_hypothesis(h, goal) for h in pred.hypotheses]
        w = pred.weights
        s = np.asarray(scores, dtype=np.float64)
        mu = float((w * s).sum())
        sigma = float(np.sqrt((w * (s - mu) ** 2).sum()))
        return GoalScore(
            mu=mu, sigma=sigma, per_hypothesis=scores, prediction_version=pred.version
        )


class VLMEventEvaluator:
    """Optional rich evaluator: asks the OpenFrontier VLM to judge whether a
    predicted future satisfies the goal (design section 8.2).

    Talks to the same VLM server / API the agent already uses; failures are
    swallowed and reported as None so the cheap channel always remains the
    fallback.
    """

    def __init__(self, vlm_model: str, api_key: Optional[str] = None):
        self.vlm_model = vlm_model
        self.api_key = api_key

    def score_events(self, events: List[str], goal: str) -> Optional[float]:
        prompt = (
            "A robot is exploring an unknown indoor environment looking for: "
            f"{goal}.\nA world model predicts that exploring through a candidate "
            "frontier leads to the following:\n"
            + "\n".join(f"- {e}" for e in events[:6])
            + "\nEstimate the probability (0 to 1) that exploring this frontier "
            'leads to finding the target. Reply with JSON: {"probability": p}'
        )
        try:
            import json

            from vlm.models import VLMModel, is_google_api
            from vlm.utils import GEMINI_CONFIG

            if is_google_api(self.vlm_model):
                from google import genai

                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=(
                        self.vlm_model.value
                        if isinstance(self.vlm_model, VLMModel)
                        else self.vlm_model
                    ),
                    contents=[prompt],
                    config=GEMINI_CONFIG,
                )
                text = response.text
            else:
                import os

                from vlm.client import VLMClient

                client = VLMClient("vlm", port=int(os.environ.get("OF_VLM_PORT", 12185)))
                text = client.send_request(prompt=prompt)["response"]
            text = text.strip().removeprefix("```json").removesuffix("```").strip()
            return float(np.clip(json.loads(text)["probability"], 0.0, 1.0))
        except Exception as e:  # noqa: BLE001 - any failure falls back to cheap score
            logger.warning("VLM event evaluator failed: %s", e)
            return None


class HybridEvaluator:
    """Cheap embedding scores everywhere; expensive VLM only when it matters
    (design section 8.3 + section 21 adaptive invocation)."""

    def __init__(
        self,
        goal_evaluator: GoalEvaluator,
        vlm_evaluator: Optional[VLMEventEvaluator] = None,
        params: Optional[dict] = None,
    ):
        p = params or {}
        self.cheap = goal_evaluator
        self.vlm = vlm_evaluator
        self.beta = float(p.get("beta", 0.5))  # weight on cheap score when mixing

    def score_prediction(self, pred: WMPrediction, goal: str) -> GoalScore:
        return self.cheap.score_prediction(pred, goal)

    def refine(
        self, pred: WMPrediction, goal: str, base: GoalScore
    ) -> Tuple[GoalScore, bool]:
        """Blend in a VLM judgment; returns (score, used_vlm)."""
        if self.vlm is None:
            return base, False
        p_vlm = self.vlm.score_events(pred.events(), goal)
        if p_vlm is None:
            return base, False
        mu = self.beta * base.mu + (1.0 - self.beta) * p_vlm
        refined = GoalScore(
            mu=mu,
            sigma=base.sigma,
            per_hypothesis=base.per_hypothesis,
            prediction_version=base.prediction_version,
        )
        return refined, True
