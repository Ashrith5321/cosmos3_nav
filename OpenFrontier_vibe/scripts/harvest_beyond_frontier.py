#!/usr/bin/env python
"""
Passive beyond-frontier data harvester on HM3D TRAIN scenes (design section 23).

Generates goal-directed rollouts (navmesh-sampled start->goal shortest paths,
the same frontier-crossing structure as human demonstrations) and records
(frontier context -> realized future) pairs with worldmodel.dataset's
WMDataRecorder. No VLM / SAM3 / planner involved - just habitat-sim,
FrontierNet, and the frozen CLIP encoder, so it is ~10x faster per episode
than eval-run recording and touches only train-split scenes (no benchmark
contamination).

Usage (from OpenFrontier_vibe/):
    OF_CUDA_DEVICE=1 python scripts/harvest_beyond_frontier.py \
        --out output/wm_harvest --scenes 1 4 --trajs-per-scene 5

Shards over scene index with --scenes i n (i in 1..n), resumable per scene
(skips scenes whose output dir already has a manifest).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = "/home/ashed/Documents/cosmos3_nav/OpenFrontier/data/scene_datasets/hm3d_v0.2"
_ROOT = Path(os.environ.get("OF_HM3D_ROOT", _DEFAULT_ROOT))
TRAIN_SCENES = _ROOT / "train"
SCENE_CONFIG = _ROOT / "hm3d_annotated_basis.scene_dataset_config.json"

# camera setup mirroring the OpenFrontier benchmark sensors
WIDTH, HEIGHT, HFOV = 640, 480, 79
SENSOR_HEIGHT = 0.88
MAX_DEPTH = 3.5
FORWARD_M, TURN_DEG = 0.25, 30.0


def make_sim(scene_glb: Path):
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_glb)
    if SCENE_CONFIG.exists():
        backend.scene_dataset_config_file = str(SCENE_CONFIG)
    backend.enable_physics = False

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [HEIGHT, WIDTH]
    rgb.hfov = HFOV
    rgb.position = [0.0, SENSOR_HEIGHT, 0.0]

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [HEIGHT, WIDTH]
    depth.hfov = HFOV
    depth.position = [0.0, SENSOR_HEIGHT, 0.0]

    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [rgb, depth]
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=FORWARD_M)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)
        ),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def cam_pose_world(sim) -> np.ndarray:
    """W_T_C (camera->world) in the OpenFrontier world frame."""
    from utils.transform import from_habitat_position, from_habitat_rotation

    state = sim.get_agent(0).get_state()
    s = state.sensor_states["rgb"]
    T = np.eye(4)
    T[:3, :3] = from_habitat_rotation(s.rotation)
    T[:3, 3] = from_habitat_position(np.asarray(s.position, dtype=float))
    return T


def get_frame(sim):
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3]
    depth = np.asarray(obs["depth"], dtype=np.float32)
    depth = np.clip(depth, 0.0, MAX_DEPTH)
    return rgb, depth


class FrameProcessor:
    """Shared observation/frontier recording for both harvest modes."""

    def __init__(self, sim, scene_name, out_dir, detector, encoder):
        from worldmodel.dataset import WMDataRecorder

        self.sim = sim
        self.scene_name = scene_name
        self.detector = detector
        self.encoder = encoder
        self.recorder = WMDataRecorder(str(out_dir), encoder, obs_stride=2)
        self.frontier_uid_grid = {}
        self.n_frontiers = 0

    def observe(self, step):
        rgb, depth = get_frame(self.sim)
        W_T_C = cam_pose_world(self.sim)
        self.recorder.record_observation(rgb, W_T_C, step)
        return rgb, depth, W_T_C

    def detect(self, step, rgb, depth, W_T_C):
        from worldmodel.predictor import make_geom_features
        from worldmodel.records import FrontierWMRecord

        detector, encoder = self.detector, self.encoder
        try:
            detector.detect(rgb=rgb, depth=depth, df_normalizer=10, df_thr=0.3)
            fts = detector.anchor_fts(depth=depth, extrinsic=np.linalg.inv(W_T_C))
        except Exception:
            return
        if not fts:
            return
        robot_pos = W_T_C[:3, 3]
        crops, owners = [], []
        for ft in fts:
            if ft.pixel_pos is None or ft.pos3d is None:
                continue
            key = tuple(np.round(np.asarray(ft.pos3d, dtype=float) / 0.75).astype(int))
            if key in self.frontier_uid_grid:
                continue
            self.frontier_uid_grid[key] = f"{self.scene_name}_{len(self.frontier_uid_grid)}"
            H, W = rgb.shape[:2]
            s = float(detector.scale_factor)
            mw, mh = detector.img_size_model
            x = int(np.clip((float(ft.pixel_pos[0]) * mw + (s * W - mw) / 2) / s, 0, W - 1))
            y = int(np.clip((float(ft.pixel_pos[1]) * mh + (s * H - mh) / 2) / s, 0, H - 1))
            half = int(0.5 * 0.45 * min(H, W))
            crop = rgb[max(y - half, 0):y + half, max(x - half, 0):x + half]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            crops.append(np.ascontiguousarray(crop))
            owners.append((self.frontier_uid_grid[key], ft, float(depth[y, x])))
        if not crops:
            return
        crop_embs = encoder.encode_images(crops)
        scene_emb = encoder.encode_image(rgb)
        for (uid, ft, d_at), ce in zip(owners, crop_embs):
            rec = FrontierWMRecord(uid=uid)
            rec.pos3d = np.asarray(ft.pos3d, dtype=float)
            rec.view_direction = np.asarray(ft.view_direction, dtype=float)
            rec.context_embedding = ce
            rec.scene_embedding = scene_emb
            rec.geom_features = make_geom_features(
                gain=float(ft.gain or 0.0),
                u_gain=float(ft.u_gain or ft.gain or 0.0),
                view_direction=rec.view_direction,
                rel_distance=float(np.linalg.norm(rec.pos3d - robot_pos)),
                depth_at_frontier=d_at,
                n_parents=1,
            )
            self.recorder.record_frontier(rec, step)
            self.n_frontiers += 1

    def close(self):
        return self.recorder.close()


DEMO_ACTION_MAP = {
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "LOOK_UP": "look_up",
    "LOOK_DOWN": "look_down",
}


def harvest_demo_scene(demo_file: Path, out_dir: Path, args, detector, encoder) -> dict:
    """Replay PIRLNav human demonstrations and record beyond-frontier data."""
    import gzip

    from habitat_sim.utils.common import quat_from_coeffs

    data = json.load(gzip.open(demo_file))
    episodes = data["episodes"]
    scene_rel = episodes[0]["scene_id"]  # e.g. hm3d/train/00744-.../....basis.glb
    scene_glb = TRAIN_SCENES.parent.parent / "hm3d_v0.2" / Path(*Path(scene_rel).parts[1:])
    if not scene_glb.exists():
        # scene ids reference hm3d/, our copy lives in hm3d_v0.2/
        candidates = list(TRAIN_SCENES.glob(f"*{Path(scene_rel).parent.name.split('-')[-1]}*/*.basis.glb"))
        if not candidates:
            return {"scene": demo_file.stem, "skipped": "scene glb not found"}
        scene_glb = candidates[0]

    rng = np.random.default_rng(abs(hash(demo_file.stem)) % 2**31)
    if args.demo_eps_per_scene and len(episodes) > args.demo_eps_per_scene:
        episodes = list(rng.choice(episodes, size=args.demo_eps_per_scene, replace=False))

    sim = make_sim(scene_glb)
    proc = FrameProcessor(sim, demo_file.stem, out_dir, detector, encoder)
    step_counter = 0
    replayed = 0

    for ep in episodes:
        agent = sim.get_agent(0)
        st = agent.get_state()
        st.position = np.asarray(ep["start_position"], dtype=np.float32)
        st.rotation = quat_from_coeffs(ep["start_rotation"])
        st.sensor_states = {}
        agent.set_state(st)

        actions = [
            DEMO_ACTION_MAP[r["action"]]
            for r in ep.get("reference_replay", [])
            if r.get("action") in DEMO_ACTION_MAP
        ]
        if len(actions) < 10:
            continue
        actions = actions[: args.max_steps]
        for act in actions:
            step_counter += 1
            rgb, depth, W_T_C = proc.observe(step_counter)
            if step_counter % args.detect_every == 0:
                proc.detect(step_counter, rgb, depth, W_T_C)
            sim.step(act)
        replayed += 1

    path = proc.close()
    sim.close()
    return {
        "scene": demo_file.stem,
        "demos_replayed": replayed,
        "frontiers_recorded": proc.n_frontiers,
        "dataset": path,
    }


def harvest_scene(scene_dir: Path, out_dir: Path, args, detector, encoder) -> dict:
    import habitat_sim
    from habitat_sim import ShortestPath
    from habitat_sim.nav import GreedyGeodesicFollower

    from worldmodel.dataset import WMDataRecorder
    from worldmodel.predictor import make_geom_features
    from worldmodel.records import FrontierWMRecord

    glbs = list(scene_dir.glob("*.basis.glb"))
    if not glbs:
        return {"scene": scene_dir.name, "skipped": "no glb"}
    sim = make_sim(glbs[0])
    rng = np.random.default_rng(abs(hash(scene_dir.name)) % 2**31)

    recorder = WMDataRecorder(str(out_dir), encoder, obs_stride=2)
    frontier_uid_grid = {}  # quantized position -> uid (cheap re-identification)
    n_frontiers = 0
    step_counter = 0
    trajs_done = 0

    def observe(step):
        rgb, depth, = get_frame(sim)
        W_T_C = cam_pose_world(sim)
        recorder.record_observation(rgb, W_T_C, step)
        return rgb, depth, W_T_C

    def detect(step, rgb, depth, W_T_C):
        nonlocal n_frontiers
        try:
            detector.detect(rgb=rgb, depth=depth, df_normalizer=10, df_thr=0.3)
            fts = detector.anchor_fts(depth=depth, extrinsic=np.linalg.inv(W_T_C))
        except Exception:
            return
        if not fts:
            return
        robot_pos = W_T_C[:3, 3]
        crops, owners = [], []
        for ft in fts:
            if ft.pixel_pos is None or ft.pos3d is None:
                continue
            key = tuple(np.round(np.asarray(ft.pos3d, dtype=float) / 0.75).astype(int))
            if key in frontier_uid_grid:
                continue  # already recorded this physical frontier
            frontier_uid_grid[key] = f"{scene_dir.name}_{len(frontier_uid_grid)}"
            # crop around the frontier pixel (detector coords are normalized
            # to the model's center-cropped frame)
            H, W = rgb.shape[:2]
            s = float(detector.scale_factor)
            mw, mh = detector.img_size_model
            x = int(np.clip((float(ft.pixel_pos[0]) * mw + (s * W - mw) / 2) / s, 0, W - 1))
            y = int(np.clip((float(ft.pixel_pos[1]) * mh + (s * H - mh) / 2) / s, 0, H - 1))
            half = int(0.5 * 0.45 * min(H, W))
            crop = rgb[max(y - half, 0):y + half, max(x - half, 0):x + half]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            crops.append(np.ascontiguousarray(crop))
            owners.append((frontier_uid_grid[key], ft, float(depth[y, x])))
        if not crops:
            return
        crop_embs = encoder.encode_images(crops)
        scene_emb = encoder.encode_image(rgb)
        for (uid, ft, d_at), ce in zip(owners, crop_embs):
            rec = FrontierWMRecord(uid=uid)
            rec.pos3d = np.asarray(ft.pos3d, dtype=float)
            rec.view_direction = np.asarray(ft.view_direction, dtype=float)
            rec.context_embedding = ce
            rec.scene_embedding = scene_emb
            rec.geom_features = make_geom_features(
                gain=float(ft.gain or 0.0),
                u_gain=float(ft.u_gain or ft.gain or 0.0),
                view_direction=rec.view_direction,
                rel_distance=float(np.linalg.norm(rec.pos3d - robot_pos)),
                depth_at_frontier=d_at,
                n_parents=1,
            )
            recorder.record_frontier(rec, step)
            n_frontiers += 1

    pathfinder = sim.pathfinder
    for _ in range(args.trajs_per_scene):
        # sample a start/goal pair with a substantial geodesic separation
        start = goal = None
        for _try in range(40):
            s = pathfinder.get_random_navigable_point()
            g = pathfinder.get_random_navigable_point()
            sp = ShortestPath()
            sp.requested_start, sp.requested_end = s, g
            if pathfinder.find_path(sp) and 5.0 < sp.geodesic_distance < 30.0:
                start, goal = s, g
                break
        if start is None:
            continue

        agent = sim.get_agent(0)
        st = agent.get_state()
        st.position = start
        yaw = rng.uniform(0, 2 * np.pi)
        st.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
        st.sensor_states = {}
        agent.set_state(st)

        follower = GreedyGeodesicFollower(
            pathfinder, agent, goal_radius=0.5,
            forward_key="move_forward", left_key="turn_left", right_key="turn_right",
        )
        for _step in range(args.max_steps):
            step_counter += 1
            rgb, depth, W_T_C = observe(step_counter)
            if step_counter % args.detect_every == 0:
                detect(step_counter, rgb, depth, W_T_C)
            try:
                action = follower.next_action_along(goal)
            except Exception:
                break
            if action is None:
                break
            sim.step(action)
        trajs_done += 1

    path = recorder.close()
    sim.close()
    return {
        "scene": scene_dir.name,
        "trajectories": trajs_done,
        "frontiers_recorded": n_frontiers,
        "dataset": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="output/wm_harvest")
    ap.add_argument("--scenes", type=int, nargs=2, default=(1, 1),
                    help="shard: index (1-based) and total")
    ap.add_argument("--max-scenes", type=int, default=0, help="0 = all")
    ap.add_argument("--trajs-per-scene", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--detect-every", type=int, default=6)
    ap.add_argument("--unet-weight", type=str, default="model_weights/rgbd_11cls.pth")
    ap.add_argument("--demos", type=str, default=None,
                    help="dir of PIRLNav <scene>.json.gz demo files; enables replay mode")
    ap.add_argument("--annotated-only", action="store_true",
                    help="restrict to scenes carrying semantic annotations, "
                         "which ground-truth goal-distance labels require")
    ap.add_argument("--demo-eps-per-scene", type=int, default=25,
                    help="subsample this many human demos per scene (0 = all)")
    args = ap.parse_args()

    import torch

    from frontier.detector import FrontierDetector
    from frontier.model.predict import load_model
    from worldmodel.encoder import build_encoder

    # intrinsics matching the sensor above
    import open3d as o3d
    f = (WIDTH / 2.0) / np.tan(np.deg2rad(HFOV / 2.0))
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(WIDTH, HEIGHT, f, f, (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = load_model(path=args.unet_weight, num_classes=11, use_depth=True)
    detector = FrontierDetector(
        model=unet, camera_intrinsic=intr.intrinsic_matrix.copy(),
        use_depth=True, img_size_model=(320, 320), device=device,
    )
    encoder = build_encoder({"backend": "clip", "device": device})

    if args.demos:
        units = sorted(Path(args.demos).glob("*.json.gz"))
        label = "demo scenes"
    else:
        units = sorted(p for p in TRAIN_SCENES.iterdir() if p.is_dir())
        if args.annotated_only:
            # only 145 of 800 HM3D train scenes ship semantic annotations, and
            # ground-truth goal-distance labels need them; harvesting the rest
            # produces data that cannot be labeled
            units = [u for u in units if list(u.glob("*.semantic.glb"))]
            print(f"annotated-only: {len(units)} scenes", flush=True)
        label = "scenes"
    idx, total = args.scenes
    units = [s for i, s in enumerate(units) if i % total == (idx - 1)]
    if args.max_scenes:
        units = units[: args.max_scenes]
    print(f"harvesting {len(units)} {label} (shard {idx}/{total})", flush=True)

    out_root = Path(args.out)
    manifest_rows = []
    for unit in units:
        name = unit.stem.replace(".json", "")
        scene_out = out_root / name
        manifest = scene_out / "harvest.json"
        if manifest.exists():
            print(f"[skip] {name} already harvested", flush=True)
            continue
        scene_out.mkdir(parents=True, exist_ok=True)
        try:
            if args.demos:
                row = harvest_demo_scene(unit, scene_out, args, detector, encoder)
            else:
                row = harvest_scene(unit, scene_out, args, detector, encoder)
        except Exception as e:  # noqa: BLE001 - one bad scene must not kill the shard
            row = {"scene": name, "error": str(e)[:300]}
        manifest.write_text(json.dumps(row, indent=1))
        manifest_rows.append(row)
        print(f"[done] {json.dumps(row)}", flush=True)

    print(f"harvest complete: {len(manifest_rows)} {label}", flush=True)


if __name__ == "__main__":
    main()
