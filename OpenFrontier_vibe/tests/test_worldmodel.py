"""
Dependency-light smoke tests for the worldmodel package.

Uses the DummyEncoder (no torch/transformers/CLIP needed) plus stub
manager/planner objects, so this runs in any Python env with numpy:

    python tests/test_worldmodel.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontier.frontier import Frontier
from worldmodel.calibration import EpisodicCalibrator
from worldmodel.encoder import DummyEncoder
from worldmodel.evaluator import GoalEvaluator, is_relational_goal
from worldmodel.memory import FrontierWorldMemory
from worldmodel.pipeline import WorldModelPipeline
from worldmodel.predictor import FrontierContext, ZeroShotClipPredictor, make_geom_features
from worldmodel.records import CONSUMED, MERGED, SELECTED
from worldmodel.vocab import OBJECT_VOCAB, ROOM_TYPES, match_goal_to_vocab


def make_frontier(pos, vd=(1, 0, 0), gain=5.0, prob=0.5, pixel=(0.5, 0.5)):
    ft = Frontier()
    ft.pos3d = np.asarray(pos, dtype=float)
    ft.view_direction = np.asarray(vd, dtype=float)
    ft.gain = ft.u_gain = gain
    ft.probability = prob
    ft.pixel_pos = pixel
    ft.direct_angle = 0.0
    pose = np.eye(4)
    pose[:3, 3] = ft.pos3d
    ft.pose6d = pose
    ft.set_valid()
    return ft


class StubManager:
    """Minimal FrontierManager stand-in for pipeline tests."""

    def __init__(self, frontiers):
        self.frontiers = {i: ft for i, ft in enumerate(frontiers)}
        for i, ft in self.frontiers.items():
            ft.id = i
        self.robot_poses = {0: np.eye(4)}
        self.current_goal_ft_id = None
        self._unreachable_positions = []
        self.external_utility_fn = None
        self.on_frontiers_merged = None

    @property
    def valid_frontiers(self):
        return [ft for ft in self.frontiers.values() if ft.is_valid]

    def get_optimal_path_length(self, start_pose, goal_pose):
        return float(
            np.linalg.norm(goal_pose[:3, 3] - start_pose[:3, 3]) * 1.3
        )


def test_vocab():
    assert match_goal_to_vocab("tv") == OBJECT_VOCAB.index("tv_monitor")
    assert match_goal_to_vocab("couch") == OBJECT_VOCAB.index("sofa")
    assert match_goal_to_vocab("armchair") == OBJECT_VOCAB.index("chair")
    assert match_goal_to_vocab("xyzzy") is None
    assert is_relational_goal("chair next to the monitor")
    assert not is_relational_goal("microwave")
    print("test_vocab OK")


def test_memory_merge_and_cache():
    mem = FrontierWorldMemory({"max_age_steps": 10})
    rec_a = mem.touch("aaa", step=1, pos3d=np.array([1.0, 0, 0]))
    rec_a.context_embedding = np.ones(8)
    mem.touch("bbb", step=1, pos3d=np.array([1.2, 0, 0]))

    # cache staleness
    h1 = mem.context_hash(np.array([1.0, 0, 0]), n_parents=1, has_context=True)
    assert mem.prediction_is_stale("aaa", h1, step=2)  # no prediction yet

    from worldmodel.records import FutureHypothesis, WMPrediction

    hyp = FutureHypothesis(
        embedding=np.ones(8) / np.sqrt(8),
        room_probs=np.full(len(ROOM_TYPES), 1 / len(ROOM_TYPES)),
        object_probs=np.zeros(len(OBJECT_VOCAB)),
    )
    mem.store_prediction(
        "aaa",
        WMPrediction(uid="aaa", hypotheses=[hyp], created_step=2, context_hash=h1),
        step=2,
    )
    assert not mem.prediction_is_stale("aaa", h1, step=3)
    assert mem.prediction_is_stale("aaa", h1, step=20)  # too old
    h2 = mem.context_hash(np.array([5.0, 0, 0]), n_parents=1, has_context=True)
    assert mem.prediction_is_stale("aaa", h2, step=3)  # moved

    # merge: bbb absorbed into aaa
    mem.merge("aaa", ["bbb"], step=3)
    assert mem.get("bbb").status == MERGED
    assert "bbb" in mem.get("aaa").merged_from
    print("test_memory_merge_and_cache OK")


def test_zero_shot_predictor_and_evaluator():
    enc = DummyEncoder(embed_dim=64)
    pred = ZeroShotClipPredictor(enc, {"room_temperature": 0.05})
    geom = make_geom_features(5.0, 4.0, np.array([1.0, 0, 0]), 3.0, 2.5, 2)
    rng = np.random.default_rng(0)
    crop = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    ctx = FrontierContext(
        uid="u1",
        crop_embedding=enc.encode_image(crop),
        scene_embedding=enc.encode_image(crop[::2, ::2]),
        geom=geom,
    )
    p = pred.predict(ctx, num_hypotheses=4, step=0, context_hash="h")
    assert len(p.hypotheses) == 4
    w = p.weights
    assert abs(w.sum() - 1.0) < 1e-6
    assert p.mean_embedding().shape == (64,)
    assert p.room_dist().shape == (len(ROOM_TYPES),)
    assert all(len(h.events) >= 1 for h in p.hypotheses)

    ev = GoalEvaluator(enc, {"temperature": 0.07})
    gs = ev.score_prediction(p, "microwave")
    assert 0.0 <= gs.mu <= 1.0
    assert gs.sigma >= 0.0
    assert len(gs.per_hypothesis) == 4

    # object channel: a hypothesis with high microwave prob must outscore
    # the same hypothesis with zero microwave prob
    h_hi = p.hypotheses[0]
    h_lo_objects = h_hi.object_probs.copy()
    h_lo_objects[OBJECT_VOCAB.index("microwave")] = 0.0
    h_hi.object_probs[OBJECT_VOCAB.index("microwave")] = 0.9
    from worldmodel.records import FutureHypothesis

    h_lo = FutureHypothesis(
        embedding=h_hi.embedding,
        room_probs=h_hi.room_probs,
        object_probs=h_lo_objects,
    )
    assert ev.score_hypothesis(h_hi, "microwave") > ev.score_hypothesis(
        h_lo, "microwave"
    )
    print("test_zero_shot_predictor_and_evaluator OK")


def test_calibrator():
    cal = EpisodicCalibrator({"min_updates": 2, "error_gain": 2.0})
    mu, sigma = cal.calibrate(0.9, 0.05, "kitchen")
    assert mu == 0.9  # no residuals yet -> no shrinkage
    for _ in range(4):
        cal.record(1.6, "kitchen")  # consistently bad kitchen predictions
    mu2, sigma2 = cal.calibrate(0.9, 0.05, "kitchen")
    assert mu2 < 0.9  # shrunk toward 0.5
    assert sigma2 > 0.05
    # residual math
    z = np.array([1.0, 0.0])
    assert abs(cal.residual(z, z)) < 1e-9
    assert abs(cal.residual(z, -z) - 2.0) < 1e-9
    print("test_calibrator OK")


def test_pipeline_end_to_end():
    fts = [
        make_frontier([2.0, 0, 1], prob=0.7, gain=6.0, pixel=(0.3, 0.5)),
        make_frontier([-2.0, 1, 1], prob=0.3, gain=3.0, pixel=(0.7, 0.5)),
        make_frontier([0.5, 2, 1], prob=0.5, gain=9.0, pixel=(0.5, 0.4)),
    ]
    mgr = StubManager(fts)
    cfg = {
        "enabled": True,
        "backend": "zero_shot",
        "encoder": {"backend": "dummy", "embed_dim": 64},
        "num_hypotheses": 3,
        "top_m": 2,
        "cost": {"mode": "geodesic"},
        "ranker": {"policy": "risk_averse", "beta": 0.5},
        "log_decisions": False,
    }
    pipe = WorldModelPipeline(cfg, mgr, goal="microwave", save_dir=None)

    rng = np.random.default_rng(1)
    rgb = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    W_T_C2 = np.eye(4)

    pipe.attach_context(fts, rgb, depth, W_T_C2, step=0)
    for ft in fts:
        assert ft.features.get("uid")
        assert pipe.memory.get(ft.features["uid"]) is not None

    pipe.step(W_T_C2, step=0)
    assert pipe.memory.stats["wm_calls"] == 2  # top_m = 2
    scored = [ft for ft in fts if "wm_mean" in ft.features]
    assert len(scored) == 2

    # ranker through the manager hook
    mgr.external_utility_fn(mgr.valid_frontiers, W_T_C2[:3, 3])
    for ft in fts:
        assert ft.utility is not None and np.isfinite(ft.utility)
        assert "wm_terms" in ft.features

    # cache: second step with unchanged context must not re-predict
    calls_before = pipe.memory.stats["wm_calls"]
    pipe.step(W_T_C2, step=1)
    assert pipe.memory.stats["wm_calls"] == calls_before
    assert pipe.memory.stats["cache_hits"] >= 2

    # goal selection + closed loop: walk to the best PREDICTED frontier and
    # observe (only frontiers with cached predictions can be verified)
    best = max(scored, key=lambda f: f.utility)
    pipe.notify_goal_selected(best)
    assert pipe.memory.get(best.features["uid"]).status == SELECTED

    near = np.eye(4)
    near[:3, 3] = np.asarray(best.pos3d) + np.array([0.1, 0.0, 0.0])
    near[:3, 2] = np.array([1.0, 0, 0])
    pipe.observe(rgb, near, step=2)
    rec = pipe.memory.get(best.features["uid"])
    assert rec.status == CONSUMED
    assert len(rec.residuals) == 1
    assert pipe.calibrator.snapshot()["global_updates"] == 1

    # merged identity propagates
    pipe._on_frontiers_merged(
        fts[0].features["uid"], [fts[1].features["uid"]]
    )
    assert pipe.memory.get(fts[1].features["uid"]).status == MERGED
    print("test_pipeline_end_to_end OK")


def test_manager_merge_keeps_uid():
    """FrontierManager.merge_frontiers must carry the dominant uid through."""
    try:
        from frontier.manager import FrontierManager
    except Exception as e:  # heavy deps (open3d etc.) may be absent
        print(f"test_manager_merge_keeps_uid SKIPPED ({type(e).__name__}: {e})")
        return

    class _P:  # minimal planner stub
        nav_level = 0.0

        def update_space(self, **kw):
            pass

        def isoccupied(self, p):
            return False

        def isfree(self, p):
            return True

        def set_bounds(self, b):
            pass

    mgr = FrontierManager(params={"filter_min_gain": 0.0}, planner=_P())
    mgr.add_robot_poses([np.eye(4)])
    a = make_frontier([1.0, 0, 1], gain=10.0)
    b = make_frontier([1.3, 0.2, 1], gain=2.0)
    dominant_uid = a.uid
    merged_events = []
    mgr.on_frontiers_merged = lambda kept, absorbed: merged_events.append(
        (kept, absorbed)
    )
    mgr.add_frontiers([a, b], parent_ids=[mgr.current_robot_id])
    mgr.merge_frontiers()
    remaining = mgr.valid_frontiers
    assert len(remaining) == 1
    assert remaining[0].uid == dominant_uid
    assert merged_events and merged_events[0][0] == dominant_uid
    print("test_manager_merge_keeps_uid OK")


def test_recorder_and_dataset(tmp_dir="/tmp/wm_dataset_test"):
    import shutil

    from worldmodel.dataset import BeyondFrontierDataset, WMDataRecorder
    from worldmodel.records import FrontierWMRecord

    shutil.rmtree(tmp_dir, ignore_errors=True)
    enc = DummyEncoder(embed_dim=64)
    recorder = WMDataRecorder(tmp_dir, enc, obs_stride=1)

    rec = FrontierWMRecord(uid="f1")
    rec.pos3d = np.array([2.0, 0.0, 1.0])
    rec.view_direction = np.array([1.0, 0.0, 0.0])
    rec.geom_features = make_geom_features(
        5.0, 4.0, rec.view_direction, 2.0, 2.0, 1
    )
    rec.context_embedding = enc.encode_text("doorway")
    rec.scene_embedding = enc.encode_text("hallway")
    recorder.record_frontier(rec, step=0)

    rng = np.random.default_rng(2)
    for step in range(1, 5):
        pose = np.eye(4)
        pose[:3, 3] = [2.0 + step * 0.5, 0.0, 1.0]  # walking through the frontier
        pose[:3, 2] = [1.0, 0.0, 0.0]
        rgb = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        recorder.record_observation(rgb, pose, step)

    path = recorder.close()
    assert path is not None and os.path.exists(path)
    ds = BeyondFrontierDataset([path])
    assert len(ds) == 1
    sample = ds[0]
    assert sample["future_embedding"].shape == (64,)
    assert sample["room_label"].shape == (len(ROOM_TYPES),)
    assert sample["object_labels"].shape == (len(OBJECT_VOCAB),)
    print("test_recorder_and_dataset OK")


if __name__ == "__main__":
    test_vocab()
    test_memory_merge_and_cache()
    test_zero_shot_predictor_and_evaluator()
    test_calibrator()
    test_pipeline_end_to_end()
    test_manager_merge_keeps_uid()
    test_recorder_and_dataset()
    print("\nAll worldmodel smoke tests passed.")
