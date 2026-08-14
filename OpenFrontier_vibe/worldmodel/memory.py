"""
Persistent frontier memory with prediction caching (design sections 10-13).

Keyed by stable frontier ``uid`` (survives FrontierManager merges).
Implements the cache-staleness rules of section 12 and the two-level
memory (per-frontier local records + episode-global stats) of section 13.
"""

import hashlib
import logging
from typing import Dict, List, Optional

import numpy as np

from worldmodel.records import (
    ACTIVE,
    MERGED,
    SELECTED,
    VISITED,
    FrontierWMRecord,
    GoalScore,
    WMPrediction,
)

logger = logging.getLogger(__name__)


class FrontierWorldMemory:
    def __init__(self, params: Optional[dict] = None):
        p = params or {}
        self.records: Dict[str, FrontierWMRecord] = {}
        self.max_age_steps = int(p.get("max_age_steps", 24))
        self.move_thresh = float(p.get("move_thresh", 0.6))
        self.min_context_gain = int(p.get("min_context_gain", 2))

        # episode-global stats (design section 37 efficiency metrics)
        self.stats = {
            "wm_calls": 0,
            "cache_hits": 0,
            "evaluator_calls": 0,
            "vlm_evaluator_calls": 0,
            "residual_updates": 0,
            "merges": 0,
        }

    # ---------------- lifecycle ----------------

    def touch(
        self,
        uid: str,
        step: int,
        pos3d: Optional[np.ndarray] = None,
        view_direction: Optional[np.ndarray] = None,
    ) -> FrontierWMRecord:
        """Get-or-create the record for a frontier and refresh its pose."""
        rec = self.records.get(uid)
        if rec is None:
            rec = FrontierWMRecord(uid=uid, first_seen_step=step)
            self.records[uid] = rec
        rec.last_seen_step = step
        if pos3d is not None:
            rec.pos3d = np.asarray(pos3d, dtype=float).copy()
        if view_direction is not None:
            rec.view_direction = np.asarray(view_direction, dtype=float).copy()
        return rec

    def get(self, uid: str) -> Optional[FrontierWMRecord]:
        return self.records.get(uid)

    def merge(self, kept_uid: str, absorbed_uids: List[str], step: int) -> None:
        """Record identity merge: absorbed frontiers fold into kept_uid.

        The kept record inherits the best available context/prediction
        (from the absorbed ones only if it has none of its own).
        """
        kept = self.touch(kept_uid, step)
        for uid in absorbed_uids:
            if uid == kept_uid:
                continue
            rec = self.records.get(uid)
            if rec is None:
                continue
            rec.status = MERGED
            kept.merged_from.append(uid)
            kept.residuals.extend(rec.residuals)
            if kept.context_embedding is None and rec.context_embedding is not None:
                kept.context_embedding = rec.context_embedding
                kept.scene_embedding = rec.scene_embedding
                kept.geom_features = rec.geom_features
                kept.crop = rec.crop
            if kept.prediction is None and rec.prediction is not None:
                kept.prediction = rec.prediction
                kept.last_prediction_step = rec.last_prediction_step
                kept.goal_scores.update(rec.goal_scores)
            self.stats["merges"] += 1

    def mark_status(self, uid: str, status: str) -> None:
        rec = self.records.get(uid)
        if rec is not None:
            rec.status = status
            if status == SELECTED:
                rec.times_selected += 1

    # ---------------- prediction cache ----------------

    @staticmethod
    def context_hash(
        pos3d: np.ndarray,
        n_parents: int,
        has_context: bool,
        quant: float = 0.5,
    ) -> str:
        """Hash of the evidence context a prediction was generated from."""
        q = np.round(np.asarray(pos3d, dtype=float) / quant).astype(int)
        key = f"{q.tolist()}|{n_parents}|{int(has_context)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def prediction_is_stale(
        self, uid: str, context_hash: str, step: int
    ) -> bool:
        """Section 12: only regenerate when relevant evidence changed."""
        rec = self.records.get(uid)
        if rec is None or rec.prediction is None:
            return True
        pred = rec.prediction
        if step - pred.created_step > self.max_age_steps:
            return True
        if pred.context_hash != context_hash:
            return True
        return False

    def store_prediction(self, uid: str, prediction: WMPrediction, step: int) -> None:
        rec = self.touch(uid, step)
        prev = rec.prediction
        prediction.version = (prev.version + 1) if prev is not None else 0
        rec.prediction = prediction
        rec.last_prediction_step = step
        rec.goal_scores = {}  # scores tied to the old prediction are invalid
        self.stats["wm_calls"] += 1

    def get_prediction(self, uid: str) -> Optional[WMPrediction]:
        rec = self.records.get(uid)
        return rec.prediction if rec is not None else None

    def store_goal_score(self, uid: str, goal: str, score: GoalScore) -> None:
        rec = self.records.get(uid)
        if rec is not None:
            rec.goal_scores[goal] = score

    def get_goal_score(self, uid: str, goal: str) -> Optional[GoalScore]:
        rec = self.records.get(uid)
        if rec is None or rec.prediction is None:
            return None
        gs = rec.goal_scores.get(goal)
        if gs is not None and gs.prediction_version == rec.prediction.version:
            return gs
        return None

    # ---------------- residuals (design section 17) ----------------

    def update_residual(
        self,
        uid: str,
        residual: float,
        observed_embedding: Optional[np.ndarray] = None,
        step: int = -1,
    ) -> None:
        rec = self.records.get(uid)
        if rec is None:
            return
        rec.residuals.append(float(residual))
        if observed_embedding is not None:
            rec.observed_embedding = observed_embedding
        rec.last_observation_step = step
        self.stats["residual_updates"] += 1

    # ---------------- queries ----------------

    def active_records(self) -> List[FrontierWMRecord]:
        return [r for r in self.records.values() if r.status in (ACTIVE, SELECTED)]

    def pending_verification(self) -> List[FrontierWMRecord]:
        """Records with a prediction not yet compared against observation."""
        return [
            r
            for r in self.records.values()
            if r.prediction is not None
            and r.status in (ACTIVE, SELECTED, VISITED)
            and r.last_observation_step < r.last_prediction_step
        ]

    def snapshot(self) -> dict:
        return {
            "stats": dict(self.stats),
            "num_records": len(self.records),
            "records": {uid: rec.to_dict() for uid, rec in self.records.items()},
        }
