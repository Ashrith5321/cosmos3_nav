"""
WorldModelPipeline: orchestrates the frontier-conditioned world model
inside the OpenFrontier navigation loop (design sections 20, 21, 28, 29).

Per navigation step:
  1. attach_context(): local crops + frozen-encoder embeddings for newly
     detected frontiers (goal-agnostic evidence).
  2. step(): coarse-to-fine cascade - cheap prescore -> top-M candidates
     -> cached or fresh world-model predictions -> goal evaluation ->
     episodic calibration -> results written into frontier features; the
     ranker then computes Q_i when FrontierManager.update_utility runs.
  3. observe(): closed-loop verification - when the robot gets close
     enough to see beyond a predicted frontier, the actual observation
     embedding is compared with the prediction and the residual feeds
     the calibrator (sections 17, 30).
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np

from worldmodel.calibration import EpisodicCalibrator
from worldmodel.cost import PathCostEstimator
from worldmodel.encoder import build_encoder
from worldmodel.evaluator import (
    GoalEvaluator,
    HybridEvaluator,
    VLMEventEvaluator,
    is_relational_goal,
)
from worldmodel.memory import FrontierWorldMemory
from worldmodel.predictor import (
    FrontierContext,
    build_predictor,
    make_geom_features,
)
from worldmodel.ranker import FrontierRanker
from worldmodel.records import CONSUMED, SELECTED, VISITED, new_uid
from worldmodel.vocab import ROOM_TYPES

logger = logging.getLogger(__name__)


class WorldModelPipeline:
    def __init__(
        self,
        cfg: dict,
        manager,
        goal: str,
        save_dir: Optional[str] = None,
        encoder=None,
    ):
        self.cfg = cfg
        self.manager = manager
        self.goal = goal
        self.save_dir = save_dir

        self.encoder = encoder or build_encoder(cfg.get("encoder"))
        self.predictor = build_predictor(cfg, self.encoder)
        self.memory = FrontierWorldMemory(cfg.get("cache"))
        self.calibrator = EpisodicCalibrator(cfg.get("calibration"))
        self.cost_estimator = PathCostEstimator(manager, self.memory, cfg.get("cost"))
        self.ranker = FrontierRanker(self.memory, self.cost_estimator, cfg.get("ranker"))

        eval_cfg = dict(cfg.get("evaluator") or {})
        cheap = GoalEvaluator(self.encoder, eval_cfg)
        vlm_eval = None
        if eval_cfg.get("use_vlm", False):
            vlm_eval = VLMEventEvaluator(
                vlm_model=eval_cfg.get("vlm_model", "gemini-2.5-flash"),
                api_key=eval_cfg.get("api_key"),
            )
        self.evaluator = HybridEvaluator(cheap, vlm_eval, eval_cfg)
        self.vlm_ambiguity_threshold = float(
            eval_cfg.get("vlm_ambiguity_threshold", 0.05)
        )

        self.top_m = int(cfg.get("top_m", 4))
        self.num_hypotheses = int(cfg.get("num_hypotheses", 4))
        self.crop_scale = float(cfg.get("crop_scale", 0.45))
        cal_cfg = cfg.get("calibration") or {}
        self.reveal_radius = float(cal_cfg.get("reveal_radius", 1.2))
        self.visit_radius = float(cal_cfg.get("visit_radius", 0.8))

        self.detector = None  # set via set_detector() for pixel unprojection
        self.oracle = None    # optional OracleFrontierEvaluator (eval only)
        self.recorder = None  # optional WMDataRecorder

        self.log_decisions = bool(cfg.get("log_decisions", True))
        self._log_path = (
            os.path.join(save_dir, "wm_state.jsonl") if save_dir else None
        )
        if self._log_path and os.path.exists(self._log_path):
            os.remove(self._log_path)

        self._step = 0
        self._scene_embedding = None

        # wire manager hooks: merged identity + external utility
        manager.on_frontiers_merged = self._on_frontiers_merged
        manager.external_utility_fn = self._external_utility

        if cfg.get("record_dataset", False):
            from worldmodel.dataset import WMDataRecorder

            self.recorder = WMDataRecorder(
                out_dir=cfg.get("dataset_dir")
                or (os.path.join(save_dir, "wm_dataset") if save_dir else "wm_dataset"),
                encoder=self.encoder,
            )

    # ------------------------------------------------------------------
    # manager hooks
    # ------------------------------------------------------------------

    def _on_frontiers_merged(self, kept_uid: str, absorbed_uids: List[str]) -> None:
        self.memory.merge(kept_uid, absorbed_uids, self._step)

    def _external_utility(self, valid_frontiers: List, current_pos: np.ndarray) -> None:
        """Hook FrontierManager.update_utility routes through."""
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(current_pos, dtype=float).reshape(3)
        self.ranker.update_utilities(
            valid_frontiers, pose, manager=self.manager, goal=self.goal
        )

    # ------------------------------------------------------------------
    # 1) context attachment
    # ------------------------------------------------------------------

    def set_detector(self, detector) -> None:
        self.detector = detector

    def _norm_to_raw_pixel(self, pixel_pos, raw_shape) -> tuple:
        """Map detector-normalized (x, y) back to raw-image pixels."""
        H, W = raw_shape[:2]
        if self.detector is not None and self.detector.scale_factor:
            s = float(self.detector.scale_factor)
            mw, mh = self.detector.img_size_model
            scaled_w, scaled_h = s * W, s * H
            off_x = (scaled_w - mw) / 2.0
            off_y = (scaled_h - mh) / 2.0
            x = (float(pixel_pos[0]) * mw + off_x) / s
            y = (float(pixel_pos[1]) * mh + off_y) / s
        else:
            x = float(pixel_pos[0]) * W
            y = float(pixel_pos[1]) * H
        return int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))

    def _crop(self, rgb: np.ndarray, ft) -> Optional[np.ndarray]:
        if ft.pixel_pos is None:
            return None
        x, y = self._norm_to_raw_pixel(ft.pixel_pos, rgb.shape)
        H, W = rgb.shape[:2]
        half = int(0.5 * self.crop_scale * min(H, W))
        x0, x1 = max(x - half, 0), min(x + half, W)
        y0, y1 = max(y - half, 0), min(y + half, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return np.ascontiguousarray(rgb[y0:y1, x0:x1, :3])

    def attach_context(
        self,
        ft_list: List,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        W_T_C2: np.ndarray,
        step: int,
    ) -> None:
        """Compute and store goal-agnostic context for new frontiers."""
        self._step = step
        if not ft_list:
            return

        # full-frame embedding shared by all frontiers of this keyframe
        scene_emb = self.encoder.encode_image(rgb[..., :3])
        self._scene_embedding = scene_emb

        crops, crop_owners = [], []
        for ft in ft_list:
            uid = ft.features.get("uid") or new_uid()
            ft.features["uid"] = uid
            crop = self._crop(rgb, ft)
            if crop is not None:
                crops.append(crop)
                crop_owners.append(ft)

        crop_embs = self.encoder.encode_images(crops) if crops else []

        crop_by_id = {id(ft): e for ft, e in zip(crop_owners, crop_embs)}
        crop_img_by_id = {id(ft): c for ft, c in zip(crop_owners, crops)}
        robot_pos = W_T_C2[:3, 3]

        for ft in ft_list:
            uid = ft.features["uid"]
            rec = self.memory.touch(
                uid, step, pos3d=ft.pos3d, view_direction=ft.view_direction
            )
            emb = crop_by_id.get(id(ft))
            if emb is not None:
                rec.context_embedding = emb
                rec.crop = crop_img_by_id.get(id(ft))
            rec.scene_embedding = scene_emb

            depth_at = 0.0
            if depth is not None and ft.pixel_pos is not None:
                x, y = self._norm_to_raw_pixel(ft.pixel_pos, depth.shape)
                depth_at = float(depth[y, x])
            rec.geom_features = make_geom_features(
                gain=float(ft.gain or 0.0),
                u_gain=float(ft.u_gain or 0.0),
                view_direction=np.asarray(ft.view_direction, dtype=float),
                rel_distance=float(np.linalg.norm(np.asarray(ft.pos3d) - robot_pos)),
                depth_at_frontier=depth_at,
                n_parents=len(ft.parent_ids),
            )

            if self.recorder is not None:
                self.recorder.record_frontier(rec, step)

    # ------------------------------------------------------------------
    # 2) predictive cascade
    # ------------------------------------------------------------------

    def _cheap_prescore(self, ft, current_pos: np.ndarray) -> float:
        p = float(ft.probability if ft.probability is not None else 0.5)
        gain = float(ft.u_gain if ft.u_gain is not None else (ft.gain or 0.0))
        d = max(float(np.linalg.norm(np.asarray(ft.pos3d) - current_pos)), 1e-6)
        return p * np.log1p(max(gain, 0.0)) / d

    def _context_for(self, uid: str) -> Optional[FrontierContext]:
        rec = self.memory.get(uid)
        if rec is None:
            return None
        if rec.context_embedding is None and rec.scene_embedding is None:
            return None
        geom = (
            rec.geom_features
            if rec.geom_features is not None
            else np.zeros(8, dtype=np.float32)
        )
        return FrontierContext(
            uid=uid,
            crop_embedding=rec.context_embedding,
            scene_embedding=rec.scene_embedding,
            geom=geom,
            crop=rec.crop,
        )

    def step(self, current_pose: np.ndarray, step: int) -> None:
        """Run the coarse-to-fine world-model cascade (design section 20)."""
        self._step = step
        current_pos = np.asarray(current_pose[:3, 3], dtype=float).reshape(3)

        candidates = [
            ft
            for ft in self.manager.valid_frontiers
            if not ft.is_object
            and ft.justification != "Rotation Frontier"
            and ft.features.get("uid") is not None
        ]
        if not candidates:
            self.ranker.allow_geodesic_for([])
            return

        for ft in candidates:
            self.memory.touch(
                ft.features["uid"], step, pos3d=ft.pos3d,
                view_direction=ft.view_direction,
            )

        candidates.sort(key=lambda f: -self._cheap_prescore(f, current_pos))
        top = candidates[: self.top_m]
        top_uids = [ft.features["uid"] for ft in top]
        self.ranker.allow_geodesic_for(top_uids)

        relational = is_relational_goal(self.goal)

        for ft in top:
            uid = ft.features["uid"]
            ctx = self._context_for(uid)
            if ctx is None:
                continue

            chash = self.memory.context_hash(
                ft.pos3d,
                n_parents=len(ft.parent_ids),
                has_context=ctx.crop_embedding is not None,
            )
            if self.memory.prediction_is_stale(uid, chash, step):
                pred = self.predictor.predict(
                    ctx,
                    num_hypotheses=self.num_hypotheses,
                    step=step,
                    context_hash=chash,
                )
                self.memory.store_prediction(uid, pred, step)
            else:
                pred = self.memory.get_prediction(uid)
                self.memory.stats["cache_hits"] += 1

            score = self.memory.get_goal_score(uid, self.goal)
            if score is None:
                score = self.evaluator.score_prediction(pred, self.goal)
                self.memory.stats["evaluator_calls"] += 1
                if relational:
                    score, used = self.evaluator.refine(pred, self.goal, score)
                    if used:
                        self.memory.stats["vlm_evaluator_calls"] += 1
                room_top = ROOM_TYPES[int(np.argmax(pred.room_dist()))]
                mu_c, sigma_c = self.calibrator.calibrate(
                    score.mu, score.sigma, room_top
                )
                score.calibrated_mu = mu_c
                score.calibrated_sigma = sigma_c
                self.memory.store_goal_score(uid, self.goal, score)

            # surface onto the frontier for logging/serialization
            room_dist = pred.room_dist()
            ft.features["wm_mean"] = float(score.value)
            ft.features["wm_sigma"] = float(score.spread)
            ft.features["wm_room_top"] = ROOM_TYPES[int(np.argmax(room_dist))]
            ft.features["wm_events"] = pred.events()[:4]

        # Adaptive VLM refinement when the top-2 gap is ambiguous (section 21)
        if (
            self.evaluator.vlm is not None
            and not relational
            and len(top) >= 2
        ):
            scored = [
                (ft, self.memory.get_goal_score(ft.features["uid"], self.goal))
                for ft in top
            ]
            scored = [(ft, s) for ft, s in scored if s is not None]
            scored.sort(key=lambda x: -x[1].value)
            if (
                len(scored) >= 2
                and scored[0][1].value - scored[1][1].value
                < self.vlm_ambiguity_threshold
            ):
                for ft, s in scored[:2]:
                    uid = ft.features["uid"]
                    pred = self.memory.get_prediction(uid)
                    refined, used = self.evaluator.refine(pred, self.goal, s)
                    if used:
                        self.memory.stats["vlm_evaluator_calls"] += 1
                        refined.calibrated_mu, refined.calibrated_sigma = (
                            self.calibrator.calibrate(refined.mu, refined.sigma)
                        )
                        self.memory.store_goal_score(uid, self.goal, refined)
                        ft.features["wm_mean"] = float(refined.value)

    # ------------------------------------------------------------------
    # 3) closed-loop verification
    # ------------------------------------------------------------------

    def observe(self, rgb: np.ndarray, W_T_C2: np.ndarray, step: int) -> None:
        """Compare predictions with reality once frontier regions are seen."""
        self._step = step
        robot_pos = W_T_C2[:3, 3]
        forward = W_T_C2[:3, 2]

        pending = self.memory.pending_verification()
        if self.recorder is not None:
            self.recorder.record_observation(rgb, W_T_C2, step)
        if not pending:
            return

        obs_emb = None
        for rec in pending:
            if rec.pos3d is None:
                continue
            d = float(np.linalg.norm(rec.pos3d - robot_pos))
            if d > self.reveal_radius:
                continue
            # require roughly facing through the frontier region
            if rec.view_direction is not None and d > 0.2:
                to_ft = (rec.pos3d - robot_pos) / d
                if float(np.dot(forward, to_ft)) < 0.0 and d > self.visit_radius:
                    continue

            if obs_emb is None:
                obs_emb = self.encoder.encode_image(rgb[..., :3])

            z_pred = rec.prediction.mean_embedding()
            residual = self.calibrator.residual(z_pred, obs_emb)
            room_top = ROOM_TYPES[int(np.argmax(rec.prediction.room_dist()))]
            self.calibrator.record(residual, room_top)
            self.memory.update_residual(rec.uid, residual, obs_emb, step)

            if d < self.visit_radius:
                self.memory.mark_status(rec.uid, CONSUMED)
            else:
                self.memory.mark_status(rec.uid, VISITED)

            logger.debug(
                "Residual for frontier %s (%s): %.3f", rec.uid, room_top, residual
            )

    def notify_goal_selected(self, ft) -> None:
        if ft is not None and ft.features.get("uid"):
            self.memory.mark_status(ft.features["uid"], SELECTED)

    # ------------------------------------------------------------------
    # logging / lifecycle
    # ------------------------------------------------------------------

    def log_step(self, step: int, current_pose: np.ndarray) -> None:
        if not self.log_decisions or self._log_path is None:
            return
        fts = self.manager.valid_frontiers
        entry = {
            "step": step,
            "goal": self.goal,
            "stats": dict(self.memory.stats),
            "calibrator": self.calibrator.snapshot(),
            "goal_ft_id": self.manager.current_goal_ft_id,
            "frontiers": [
                {
                    "id": ft.id,
                    "uid": ft.features.get("uid"),
                    "label": ft.label,
                    "is_object": ft.is_object,
                    "utility": None if ft.utility is None else float(ft.utility),
                    "terms": ft.features.get("wm_terms"),
                    "wm_room_top": ft.features.get("wm_room_top"),
                    "wm_events": ft.features.get("wm_events"),
                }
                for ft in fts
            ],
        }
        if self.oracle is not None:
            try:
                entry["oracle"] = self.oracle.evaluate(fts, current_pose)
            except Exception as e:  # noqa: BLE001
                logger.debug("Oracle evaluation failed: %s", e)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close(self.memory)
        if self.save_dir:
            with open(os.path.join(self.save_dir, "wm_memory.json"), "w") as f:
                json.dump(self.memory.snapshot(), f, indent=1)

    @property
    def stats(self) -> dict:
        return dict(self.memory.stats)


def build_pipeline(
    cfg: dict, manager, goal: str, save_dir: Optional[str] = None, encoder=None
) -> WorldModelPipeline:
    return WorldModelPipeline(
        cfg=cfg, manager=manager, goal=goal, save_dir=save_dir, encoder=encoder
    )
