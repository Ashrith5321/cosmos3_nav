"""
Persistent per-frontier world-model records (design doc sections 10-13).

A ``FrontierWMRecord`` outlives the OpenFrontier ``Frontier`` objects:
FrontierManager reallocates integer ids on every merge, so records are
keyed by a stable ``uid`` carried through merges. Each record stores the
cached prediction, goal scores, residual history, and lifecycle status.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Lifecycle statuses (design section 10)
ACTIVE = "ACTIVE"
SELECTED = "SELECTED"
VISITED = "VISITED"
CONSUMED = "CONSUMED"
FAILED = "FAILED"
MERGED = "MERGED"
ARCHIVED = "ARCHIVED"


def new_uid() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class FutureHypothesis:
    """One sampled future outcome for a frontier (design section 9)."""

    embedding: np.ndarray            # (D,) unit-norm semantic latent
    room_probs: np.ndarray           # (R,) over vocab.ROOM_TYPES
    object_probs: np.ndarray         # (O,) over vocab.OBJECT_VOCAB
    affordance_probs: Optional[np.ndarray] = None  # (A,) over vocab.AFFORDANCES
    events: List[str] = field(default_factory=list)  # language propositions
    info_gain: float = 0.0
    weight: float = 1.0              # hypothesis probability mass

    def top_room(self, room_types: List[str]) -> str:
        return room_types[int(np.argmax(self.room_probs))]


@dataclass
class WMPrediction:
    """Cached multi-hypothesis prediction for one persistent frontier."""

    uid: str
    hypotheses: List[FutureHypothesis]
    created_step: int
    context_hash: str
    backend: str = "zero_shot"
    version: int = 0

    @property
    def weights(self) -> np.ndarray:
        w = np.array([h.weight for h in self.hypotheses], dtype=np.float64)
        s = w.sum()
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))

    def mean_embedding(self) -> np.ndarray:
        w = self.weights
        emb = np.stack([h.embedding for h in self.hypotheses], axis=0)
        mean = (w[:, None] * emb).sum(axis=0)
        n = np.linalg.norm(mean)
        return mean / n if n > 1e-8 else mean

    def room_dist(self) -> np.ndarray:
        w = self.weights
        rooms = np.stack([h.room_probs for h in self.hypotheses], axis=0)
        return (w[:, None] * rooms).sum(axis=0)

    def object_dist(self) -> np.ndarray:
        w = self.weights
        objs = np.stack([h.object_probs for h in self.hypotheses], axis=0)
        return (w[:, None] * objs).sum(axis=0)

    def events(self) -> List[str]:
        out = []
        for h in self.hypotheses:
            out.extend(h.events)
        return out


@dataclass
class GoalScore:
    """Goal-conditioned evaluation of a prediction (mu/sigma, design section 9)."""

    mu: float
    sigma: float
    per_hypothesis: List[float]
    calibrated_mu: Optional[float] = None
    calibrated_sigma: Optional[float] = None
    prediction_version: int = 0

    @property
    def value(self) -> float:
        return self.calibrated_mu if self.calibrated_mu is not None else self.mu

    @property
    def spread(self) -> float:
        return (
            self.calibrated_sigma if self.calibrated_sigma is not None else self.sigma
        )


@dataclass
class FrontierWMRecord:
    """Persistent state for one physical frontier (design section 10)."""

    uid: str
    pos3d: Optional[np.ndarray] = None
    view_direction: Optional[np.ndarray] = None
    status: str = ACTIVE

    # context
    context_embedding: Optional[np.ndarray] = None  # local crop embedding
    scene_embedding: Optional[np.ndarray] = None    # full-frame embedding
    geom_features: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None               # HxWx3 uint8 local crop

    # prediction + evaluation
    prediction: Optional[WMPrediction] = None
    goal_scores: Dict[str, GoalScore] = field(default_factory=dict)

    # navigation
    geodesic_cost: Optional[float] = None
    geodesic_cost_key: Optional[str] = None

    # calibration
    residuals: List[float] = field(default_factory=list)
    observed_embedding: Optional[np.ndarray] = None

    # bookkeeping
    first_seen_step: int = 0
    last_seen_step: int = 0
    last_prediction_step: int = -1
    last_observation_step: int = -1
    times_selected: int = 0
    merged_from: List[str] = field(default_factory=list)
    created_wall_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe summary (arrays reduced, embeddings omitted)."""
        d: Dict[str, Any] = {
            "uid": self.uid,
            "status": self.status,
            "pos3d": None if self.pos3d is None else np.asarray(self.pos3d).tolist(),
            "first_seen_step": self.first_seen_step,
            "last_seen_step": self.last_seen_step,
            "last_prediction_step": self.last_prediction_step,
            "last_observation_step": self.last_observation_step,
            "times_selected": self.times_selected,
            "geodesic_cost": self.geodesic_cost,
            "residuals": [float(r) for r in self.residuals],
            "merged_from": list(self.merged_from),
        }
        if self.prediction is not None:
            from worldmodel.vocab import ROOM_TYPES

            room = self.prediction.room_dist()
            top = int(np.argmax(room))
            d["prediction"] = {
                "backend": self.prediction.backend,
                "version": self.prediction.version,
                "created_step": self.prediction.created_step,
                "num_hypotheses": len(self.prediction.hypotheses),
                "top_room": ROOM_TYPES[top],
                "top_room_prob": float(room[top]),
                "events": self.prediction.events()[:6],
            }
        if self.goal_scores:
            d["goal_scores"] = {
                goal: {
                    "mu": gs.mu,
                    "sigma": gs.sigma,
                    "calibrated_mu": gs.calibrated_mu,
                    "calibrated_sigma": gs.calibrated_sigma,
                }
                for goal, gs in self.goal_scores.items()
            }
        return d
