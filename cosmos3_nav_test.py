#!/usr/bin/env python3
"""One frontier, end to end, timed: Habitat render -> Cosmos3 reasoner -> Cosmos3 generator.

The question this script answers is "what does one Cosmos3-Nano imagination step
cost, and is the image it produces anything like what is actually there".

    1. Habitat renders an ego view in an HM3D scene, picks the deepest opening in
       front of the robot (the frontier), turns to face it, and saves that frame.
    2. The Cosmos3 reasoner looks at that frame and writes one dense caption of
       the room / hallway the robot will be standing in after driving straight
       through the opening.
    3. The Cosmos3 generator turns that caption into ONE image -- text2image from
       the caption alone, and image2image (edit) grounded on the observed frame.
    4. Habitat drives straight to the frontier for real and renders the ground
       truth, and the reasoner judges generated-vs-real side by side.

Every stage is wall-clocked, with model load reported separately from the call.

Three interpreters are involved and none of them can be merged: Habitat 0.3.3
needs its own conda env; the Cosmos3 *generator* runs on cosmos_framework
(torch 2.10 / py3.13); the Cosmos3 *reasoner* needs transformers >= 5.11, because
the released Nano omni checkpoint ships `include_visual=false` and the framework
therefore refuses image-conditioned reasoner prompts -- the same weights loaded
as `Cosmos3OmniForConditionalGeneration` do accept them (this is the backend
OpenFrontier already uses). So this file re-executes itself under each
environment via --role, one model resident at a time, which also keeps the peak
GPU footprint at one model instead of two. Running it plainly does the whole
thing:

    python cosmos3_nav_test.py
    python cosmos3_nav_test.py --scene 00802-wcojb4TFT35 --seed 3 --num-steps 35
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent

DEFAULT_HABITAT_PYTHON = Path.home() / "miniconda3/envs/habitat033/bin/python"
DEFAULT_COSMOS_PYTHON = REPO / "cosmos/packages/cosmos3/.venv/bin/python"
DEFAULT_REASON_PYTHON = REPO / ".venv/bin/python"
DEFAULT_FRONTIERNET_PYTHON = Path.home() / "miniconda3/envs/openfrontier/bin/python"
DEFAULT_SCENES = Path("/home/ashed/Documents/spatial_training/data/scene_datasets/hm3d_v0.2")


def gpu_free_mb() -> list[int]:
    """Free VRAM per GPU, as nvidia-smi sees it (includes other users' jobs)."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise SystemExit(f"nvidia-smi failed: {result.stderr.strip()}")
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def system_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 0


def preflight(args: argparse.Namespace) -> None:
    """Refuse to start unless there is room. This box runs other jobs; a 29 GB
    checkpoint landing on a busy GPU would take them down with it, so the check
    is a hard abort, not a warning."""
    free = gpu_free_mb()
    if not free:
        raise SystemExit("no CUDA devices visible")

    if args.cosmos_gpu is None:
        args.cosmos_gpu = max(range(len(free)), key=lambda i: free[i])
        print(f"[preflight] picking GPU {args.cosmos_gpu} for Cosmos "
              f"({free[args.cosmos_gpu]} MiB free of {len(free)} GPUs: {free})")
    if args.cosmos_gpu >= len(free):
        raise SystemExit(f"--cosmos-gpu {args.cosmos_gpu} does not exist (found {len(free)} GPUs)")
    if free[args.cosmos_gpu] < args.min_free_gpu_mb:
        raise SystemExit(
            f"refusing to start: GPU {args.cosmos_gpu} has {free[args.cosmos_gpu]} MiB free, "
            f"need {args.min_free_gpu_mb} MiB for Cosmos3-Nano (~29 GB of weights plus "
            f"activations). Free the GPU, pick another with --cosmos-gpu, or lower "
            f"--min-free-gpu-mb if you know what else is resident. Per-GPU free: {free}"
        )
    if args.habitat_gpu >= len(free):
        args.habitat_gpu = 0
    if free[args.habitat_gpu] < 2000:
        raise SystemExit(f"GPU {args.habitat_gpu} has only {free[args.habitat_gpu]} MiB free for the Habitat render")

    available = system_available_mb()
    if available < args.min_free_ram_mb:
        raise SystemExit(
            f"refusing to start: {available} MiB RAM available, need {args.min_free_ram_mb} MiB "
            f"(the checkpoint is staged through host memory before it reaches the GPU)"
        )
    print(f"[preflight] GPU {args.cosmos_gpu}: {free[args.cosmos_gpu]} MiB free | "
          f"RAM: {available} MiB available | threads capped at {args.threads}")


def default_checkpoint() -> str:
    """Cosmos3-Nano weights: env override, then the HF snapshot, then the loose copy."""
    env = os.environ.get("COSMOS3_CHECKPOINT")
    if env:
        return env
    snapshots = Path.home() / ".cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots"
    if snapshots.is_dir():
        candidates = sorted(p for p in snapshots.iterdir() if (p / "model_index.json").exists())
        if candidates:
            return str(candidates[-1])
    return "/home/ashed/Documents/Cosmos3-Nano"


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

DEPTH_FRONTIER_NOTE = (
    "Straight ahead, near the centre of the frame, is an unexplored opening -- a doorway, "
    "corridor mouth or room boundary -- with about {free_depth:.1f} m of free space behind it."
)

FRONTIERNET_FRONTIER_NOTE = (
    "A frontier detector has labelled the unexplored boundaries in this view and the robot is "
    "aimed at the most promising one: frontier #{id} of {n}, predicted information gain "
    "{gain:.1f} m^3 (the largest in view), now centred in the frame with about {free_depth:.1f} m "
    "of free space behind it."
)

REASONER_PROMPT = """You are the spatial imagination module of an indoor exploration robot.

The image is the robot's current forward-facing camera view. {frontier_note} The robot is about to travel {advance:.1f} m forward, to that opening and through it, arriving on the far side with the same heading it has now.

Imagine the single camera frame the robot will capture the moment it arrives -- a view of the space BEYOND the opening, not the space it is standing in now -- and describe it as one dense image caption for an image generator.

Requirements:
- Open by naming the kind of space the robot ends up in (hallway, living room, kitchen, bedroom, bathroom, dining room, stairwell, ...).
- Describe the whole space: walls, floor, ceiling, doorways, windows, and the major furniture and objects with their positions in frame.
- The current room is now BEHIND the camera. Do not describe the furniture visible in this image; describe what the opening is hiding.
- Keep the architecture, wall colour, flooring, and lighting consistent with what is already visible in this image -- it is the same home, seen further in.
- Write it as a forward-facing eye-level photograph at about 1 m height, 79 degree field of view.
- One paragraph, 80-150 words, no preamble, no bullet points, no mention of the robot.

Answer with the caption only."""

EDIT_PROMPT = """Move the camera straight forward {advance:.1f} metres, through the opening ahead, and render the view from the new position: {caption}"""

# forward_dynamics is told the motion in metres, through the action tensor, so
# the prompt deliberately says nothing about what lies ahead -- describing the
# unseen room here would hand the model the answer it is being tested on.
FORWARD_DYNAMICS_PROMPT = (
    "Ego-centric video from a camera moving through an indoor residential space. "
    "The camera translates smoothly forward; lighting and surfaces stay consistent "
    "as new parts of the home come into view."
)

JUDGE_PROMPT = """The image shows two photographs side by side of the SAME location in a house, taken from the same spot.

LEFT is an image imagined by a world model. RIGHT is the real photograph.

Compare them and answer with a single JSON object, nothing else. Every list holds AT MOST 5 short entries and must not repeat an entry.

The scores and the verdict come first, and the lists last, so that a long list can never cost you the scores:
{{"predicted_space_type": "<what the LEFT image shows>", "real_space_type": "<what the RIGHT image shows>", "space_type_match": true|false, "layout_similarity_0_to_10": <int>, "appearance_similarity_0_to_10": <int>, "verdict": "<one sentence>", "shared_objects": ["..."], "objects_only_in_prediction": ["..."], "objects_only_in_reality": ["..."]}}"""


# --------------------------------------------------------------------------- #
# Stage 1 -- Habitat (runs under the habitat033 interpreter)
# --------------------------------------------------------------------------- #


def stage_habitat(args: argparse.Namespace) -> None:
    import numpy as np
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    scenes_root = Path(args.scenes_root)
    split_dir = scenes_root / args.split
    scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not scene_dirs:
        raise SystemExit(f"no scenes under {split_dir}")
    if args.scene:
        matches = [p for p in scene_dirs if args.scene in p.name]
        if not matches:
            raise SystemExit(f"scene {args.scene!r} not found in {split_dir}")
        scene_dir = matches[0]
    else:
        scene_dir = scene_dirs[args.scene_index % len(scene_dirs)]
    stem = scene_dir.name.split("-", 1)[1]
    glb = scene_dir / f"{stem}.basis.glb"
    if not glb.is_file():
        raise SystemExit(f"missing scene mesh {glb}")

    width, height, hfov = args.width, args.height, args.hfov
    fx = (width / 2.0) / np.tan(np.radians(hfov) / 2.0)
    cx = width / 2.0
    up = np.array([0.0, 1.0, 0.0])

    started = time.perf_counter()
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(glb)
    dataset_cfg = scenes_root / "hm3d_annotated_basis.scene_dataset_config.json"
    if dataset_cfg.is_file():
        backend.scene_dataset_config_file = str(dataset_cfg)
    backend.gpu_device_id = args.habitat_gpu
    backend.enable_physics = False
    backend.random_seed = args.seed

    specs = []
    for uuid, sensor_type in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [height, width]
        spec.position = [0.0, args.sensor_height, 0.0]
        spec.hfov = hfov
        specs.append(spec)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    agent_cfg.height = args.sensor_height
    agent_cfg.radius = 0.18

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    timings["sim_load_s"] = time.perf_counter() - started

    try:
        pathfinder = sim.pathfinder
        if not pathfinder.is_loaded:
            raise SystemExit(f"no navmesh loaded for {glb}")
        pathfinder.seed(args.seed)
        rng = np.random.default_rng(args.seed)

        agent = sim.get_agent(0)

        def render(position, rotation) -> dict:
            state = habitat_sim.AgentState()
            state.position = np.asarray(position, dtype=np.float32)
            state.rotation = rotation
            agent.set_state(state)
            return sim.get_sensor_observations()

        def profile(depth: np.ndarray, bins: int = 32):
            """Free distance per column bin, over a band around the horizon."""
            rows = depth[int(height * 0.35) : int(height * 0.62)]
            rows = np.where(rows > 1e-3, rows, np.nan)
            edges = np.linspace(0, width, bins + 1).astype(int)
            values = []
            for i in range(bins):
                block = rows[:, edges[i] : edges[i + 1]]
                values.append(
                    float(np.nanpercentile(block, 60)) if np.isfinite(block).any() else 0.0
                )
            centers = (edges[:-1] + edges[1:]) / 2.0
            return np.nan_to_num(np.array(values)), centers

        sensor = {"width": width, "height": height, "hfov": hfov, "height_m": args.sensor_height}

        if args.task == "observe":
            # A start pose with somewhere to go: sample navigable points, scan
            # yaw at each, keep the view with the deepest opening in it. This
            # only picks where the robot *stands*; which frontier it drives at
            # is decided afterwards, by FrontierNet or by the depth fallback.
            started = time.perf_counter()
            best = None
            for _ in range(args.num_candidates):
                position = pathfinder.get_random_navigable_point()
                if not np.isfinite(position).all():
                    continue
                for k in range(args.num_yaws):
                    yaw = 2.0 * np.pi * k / args.num_yaws + float(rng.uniform(0, 0.2))
                    rotation = quat_from_angle_axis(yaw, up)
                    depth = render(position, rotation)["depth"]
                    values, centers = profile(depth)
                    index = int(np.argmax(values))
                    score = float(values[index])
                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
                            "position": np.array(position, dtype=np.float32),
                            "yaw": yaw,
                            "column": float(centers[index]),
                        }
            if best is None:
                raise SystemExit("no navigable start pose found")
            timings["pose_scan_s"] = time.perf_counter() - started

            start = render(best["position"], quat_from_angle_axis(best["yaw"], up))
            Image.fromarray(start["rgb"][:, :, :3]).save(out / "start_view.png")
            np.save(out / "start_depth.npy", start["depth"].astype(np.float32))

            # Depth-only fallback frontier, and the sign check for the yaw
            # convention. Turning the wrong way would silently aim the robot at
            # a wall, so both signs are rendered and the one that actually
            # centres the opening wins; FrontierNet then reuses that convention.
            magnitude = float(np.arctan((best["column"] - cx) / fx))
            candidates = {}
            for name, phi in (("neg", -magnitude), ("pos", magnitude)):
                values, centers = profile(render(best["position"], quat_from_angle_axis(best["yaw"] + phi, up))["depth"])
                index = int(np.argmax(values))
                candidates[name] = {"phi": phi, "offset_px": abs(float(centers[index]) - cx)}
            turn = min(candidates.values(), key=lambda c: c["offset_px"])

            record = {
                "scene": scene_dir.name,
                "scene_glb": str(glb),
                "split": args.split,
                "seed": args.seed,
                "position": [float(v) for v in best["position"]],
                "yaw_rad": float(best["yaw"]),
                "depth_turn_deg": float(np.degrees(turn["phi"])),
                "turn_sign_check_px": {k: round(v["offset_px"], 1) for k, v in candidates.items()},
                "start_view": str((out / "start_view.png").resolve()),
                "start_depth": str((out / "start_depth.npy").resolve()),
                "sensor": sensor,
                "timings": {k: round(v, 3) for k, v in timings.items()},
            }
            (out / "pose.json").write_text(json.dumps(record, indent=2))
            print(json.dumps(record, indent=2))
            return

        # --- advance: turn by the chosen amount and drive at the frontier ----
        pose = json.loads(Path(args.pose_json).read_text())
        position = np.array(pose["position"], dtype=np.float32)
        rotation = quat_from_angle_axis(pose["yaw_rad"] + np.radians(args.turn_deg), up)

        obs = render(position, rotation)
        depth = obs["depth"]
        Image.fromarray(obs["rgb"][:, :, :3]).save(out / "observation.png")
        np.save(out / "observation_depth.npy", depth.astype(np.float32))

        # Free distance dead ahead, from the centre strip of the depth image.
        center_band = depth[int(height * 0.35) : int(height * 0.62), int(cx - 40) : int(cx + 40)]
        center_band = np.where(center_band > 1e-3, center_band, np.nan)
        free_depth = float(np.nanpercentile(center_band, 60)) if np.isfinite(center_band).any() else 0.0

        # Drive straight until the navmesh says stop or we run out of free space.
        forward = np.asarray(quat_rotate_vector(rotation, np.array([0.0, 0.0, -1.0])), dtype=np.float32)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        forward = forward / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # How far to drive. With a target (the detector's distance to the
        # frontier) the robot aims to reach and cross it, and only the navmesh
        # stops it early -- clamping to the centre-band free depth instead would
        # park it short of the opening, which is the whole thing we are trying
        # to see past.
        if args.target_advance > 0:
            limit = min(args.max_advance, args.target_advance)
        else:
            limit = min(args.max_advance, max(0.0, free_depth - args.clearance))

        # Render along the way as well as at the end: the sweep is the ruler
        # that says how far forward a generated image actually corresponds to.
        sweep_dir = out / "sweep"
        sweep_dir.mkdir(exist_ok=True)
        sweep = []
        advance, step, blocked = 0.0, 0.25, False
        started = time.perf_counter()
        while True:
            frame = render(position + forward * advance, rotation)
            path = sweep_dir / f"d_{advance:05.2f}.png"
            Image.fromarray(frame["rgb"][:, :, :3]).save(path)
            sweep.append({"distance_m": round(advance, 2), "image": str(path.resolve())})
            if advance + step > limit:
                break
            probe = position + forward * (advance + step)
            if not pathfinder.is_navigable(probe):
                blocked = True
                break
            advance += step
        timings["sweep_render_s"] = time.perf_counter() - started

        goal = np.asarray(pathfinder.snap_point(position + forward * advance), dtype=np.float32)
        if not np.isfinite(goal).all():
            goal, advance = position, 0.0

        # Driving straight often stops well short of the frontier -- here an
        # island in the middle of the kitchen -- and a view from short of the
        # opening is not a view beyond it. So when a target was given and the
        # straight line was blocked, navigate: take the furthest point along the
        # frontier bearing that the navmesh can actually reach, and render from
        # there. The robot would get there by going around; for a single image
        # only the final pose matters.
        navigation = {"used": False}
        if args.target_advance > 0 and blocked:
            for distance in np.arange(args.target_advance, advance, -0.25):
                candidate = np.asarray(
                    pathfinder.snap_point(position + forward * float(distance)), dtype=np.float32
                )
                if not np.isfinite(candidate).all():
                    continue
                # The snap must not have slid us sideways into a different room.
                if float(np.linalg.norm(candidate - (position + forward * float(distance)))) > args.snap_tolerance:
                    continue
                path = habitat_sim.ShortestPath()
                path.requested_start = position
                path.requested_end = candidate
                if not pathfinder.find_path(path):
                    continue
                goal = candidate
                navigation = {
                    "used": True,
                    "straight_line_blocked_at_m": round(float(advance), 2),
                    "euclidean_m": round(float(np.linalg.norm(goal - position)), 2),
                    "geodesic_m": round(float(path.geodesic_distance), 2),
                    "requested_m": round(float(distance), 2),
                }
                advance = float(np.linalg.norm(goal - position))
                Image.fromarray(render(goal, rotation)["rgb"][:, :, :3]).save(
                    sweep_dir / f"d_{advance:05.2f}.png"
                )
                sweep.append({
                    "distance_m": round(advance, 2),
                    "image": str((sweep_dir / f"d_{advance:05.2f}.png").resolve()),
                    "reached_by": "navmesh path",
                })
                break

        # Flying ruler: the straight line to the goal rendered every 0.25 m,
        # ignoring navigability -- the camera can fly where the robot cannot.
        # Pure measurement infrastructure: generated frames are matched against
        # these to say, in metres, how far forward a rollout actually got.
        fly_dir = out / "sweep_fly"
        fly_dir.mkdir(exist_ok=True)
        fly = []
        distance = 0.0
        while distance <= advance + 1e-6:
            frame = render(position + forward * distance, rotation)
            fly_path = fly_dir / f"d_{distance:05.2f}.png"
            Image.fromarray(frame["rgb"][:, :, :3]).save(fly_path)
            fly.append({"distance_m": round(distance, 2), "image": str(fly_path.resolve())})
            distance += 0.25

        gt = render(goal, rotation)
        Image.fromarray(gt["rgb"][:, :, :3]).save(out / "ground_truth.png")

        # The commanded trajectory, for Cosmos's forward_dynamics mode: the
        # camera translating straight at the frontier in habitat's own 0.25 m
        # steps, no rotation. Nominal, not executed -- the robot cannot know
        # where the navmesh would have stopped it, and feeding the executed path
        # would hand the model part of the answer.
        #
        # Habitat is y-up with the agent down its local -z; Cosmos camera_pose
        # actions are camera-to-world in OpenCV axes (x right, y down,
        # z forward). The conversion is a fixed 180-degree flip about x, applied
        # on the right (see frontierworld/models/nominal_poses.py).
        yaw = float(pose["yaw_rad"] + np.radians(args.turn_deg))
        cos, sin = np.cos(yaw), np.sin(yaw)
        yaw_matrix = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        habitat_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
        # The rollout length must satisfy frames = 4k+1 (the VAE's temporal
        # compression), and frames = actions + 1, so the action count has to be
        # a multiple of 4. Asking for 43 silently yields 41 frames, which slides
        # the actions out of step with the frames and produces a rollout that
        # barely moves.
        n_steps = max(4, (args.fd_steps // 4) * 4)
        poses = []
        for i in range(n_steps + 1):
            translation = position + forward * (advance * i / n_steps)
            agent = np.eye(4)
            agent[:3, :3] = yaw_matrix
            agent[:3, 3] = translation + np.array([0.0, args.sensor_height, 0.0])
            poses.append(agent @ habitat_to_opencv)
        poses = np.stack(poses).astype(np.float32)
        np.save(out / "camera_poses.npy", poses)

        record = {
            "scene": pose["scene"],
            "scene_glb": pose["scene_glb"],
            "split": args.split,
            "seed": args.seed,
            "frontier_source": args.frontier_source,
            "start_position": [float(v) for v in position],
            "goal_position": [float(v) for v in goal],
            "yaw_scan_deg": float(np.degrees(pose["yaw_rad"])),
            "turn_to_frontier_deg": float(args.turn_deg),
            "turn_sign_check_px": pose["turn_sign_check_px"],
            "frontier_free_depth_m": round(free_depth, 3),
            "target_advance_m": round(float(args.target_advance), 3),
            "advance_m": round(float(advance), 3),
            # The target is frontier + overshoot, so crossing means clearing the
            # frontier distance itself, not the overshoot as well.
            "reached_frontier": bool(
                args.target_advance > 0 and advance >= args.target_advance - args.overshoot
            ),
            "stopped_by": ("navmesh path" if navigation["used"] else
                           "navmesh" if blocked else
                           "target" if args.target_advance > 0 else "free depth"),
            "navigation": navigation,
            "camera_poses": str((out / "camera_poses.npy").resolve()),
            "camera_poses_n": int(poses.shape[0]),
            "camera_poses_step_m": round(float(advance / n_steps), 3),
            "sweep": sweep,
            "sweep_fly": fly,
            "observation": str((out / "observation.png").resolve()),
            "ground_truth": str((out / "ground_truth.png").resolve()),
            "start_view": pose["start_view"],
            "sensor": sensor,
            "timings": {**pose["timings"], **{k: round(v, 3) for k, v in timings.items()}},
        }
        (out / "habitat.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2))
    finally:
        sim.close()


# --------------------------------------------------------------------------- #
# Stage 2 -- FrontierNet: label the frontiers in the observed image
# --------------------------------------------------------------------------- #


def stage_frontiernet(args: argparse.Namespace) -> None:
    """RGB+depth -> labelled frontiers (mask, per-frontier information gain).

    Runs FrontierNet's own detector on the habitat frame, using the simulator's
    true metric depth instead of the monocular estimate the single-image demo
    falls back to. The frontier with the highest predicted gain is the one the
    robot drives at, and the one Cosmos3 is asked to imagine beyond.
    """
    root = Path(args.frontiernet_root)
    if not (root / "frontier" / "detector.py").is_file():
        raise SystemExit(f"FrontierNet not found at {root}")
    sys.path.insert(0, str(root))
    os.chdir(root)

    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw

    from frontier.detector import FrontierDetector
    from frontier.model.predict import load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pose = json.loads(Path(args.pose_json).read_text())
    config = yaml.safe_load((root / "config" / args.frontiernet_config).read_text())

    rgb = np.asarray(Image.open(pose["start_view"]).convert("RGB"), dtype=np.uint8)
    depth = np.load(pose["start_depth"]).astype(np.float32)
    height, width = rgb.shape[:2]

    # The habitat camera's real intrinsics, not the config's placeholder focal
    # length -- the pixel->bearing conversion below has to match the sensor the
    # image actually came from.
    hfov = float(pose["sensor"]["hfov"])
    fx = fy = (width / 2.0) / np.tan(np.radians(hfov) / 2.0)
    cx, cy = width / 2.0 - 0.5, height / 2.0 - 0.5
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    timings: dict[str, float] = {}
    started = time.perf_counter()
    unet = load_model(
        path=root / args.frontiernet_weights, num_classes=config["num_classes"], use_depth=True
    )
    timings["load_s"] = time.perf_counter() - started

    detector = FrontierDetector(
        model=unet,
        camera_intrinsic=intrinsics,
        use_depth=True,
        img_size_model=tuple(config["input_img_size"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    started = time.perf_counter()
    ft_region, info_gain = detector.detect(
        rgb=rgb, depth=depth,
        df_normalizer=config["df_normalizer"], df_thr=config["df_thr"],
    )
    timings["detect_s"] = time.perf_counter() - started

    # Anchor the 2D detections into 3D clusters. The extrinsic is the demo's
    # front-facing camera frame, so the 3D positions it reports are relative to
    # the camera, not habitat world coordinates -- they are recorded as labels,
    # while the robot is steered from the image bearing.
    extrinsic = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
    started = time.perf_counter()
    frontiers = detector.anchor_fts(depth=depth, extrinsic=extrinsic)
    timings["anchor_s"] = time.perf_counter() - started

    # Map model-resolution pixels back to the original frame: preprocess()
    # resizes by scale_factor then centre-crops to img_size_model.
    scale = float(detector.scale_factor)
    crop_h, crop_w = config["input_img_size"]
    top = int(round((height * scale - crop_h) / 2.0))
    left = int(round((width * scale - crop_w) / 2.0))

    def to_original(u: float, v: float) -> tuple[float, float]:
        """Frontier pixel_pos is normalised by the model input size (see
        FrontierDetector.get_ft_feature: `xs / img_size_model[1]`), so undo the
        normalisation before undoing the crop and the resize."""
        return (u * crop_w + left) / scale, (v * crop_h + top) / scale

    def depth_at(x: float, y: float, half: int = 6) -> float:
        patch = depth[
            max(0, int(y) - half) : int(y) + half, max(0, int(x) - half) : int(x) + half
        ]
        patch = np.where(patch > 1e-3, patch, np.nan)
        return float(np.nanmedian(patch)) if np.isfinite(patch).any() else 0.0

    labels = []
    for index, frontier in enumerate(frontiers or []):
        u, v = frontier.features["pixel_pos"]
        x, y = to_original(float(u), float(v))
        labels.append({
            "id": index,
            "gain_m3": round(float(frontier.features["gain"]), 3),
            "pixel": [round(x, 1), round(y, 1)],
            "pixel_normalised": [round(float(u), 3), round(float(v), 3)],
            # Same sign convention the depth fallback validated by rendering both.
            "turn_deg": round(float(-np.degrees(np.arctan((x - cx) / fx))), 2),
            "depth_at_pixel_m": round(depth_at(x, y), 3),
            "direction_angle_rad": round(float(frontier.features["direct_angle"]), 3),
            "pos3d_camera": [round(float(c), 3) for c in frontier.features["3d_pos"]],
            "view_direction": [round(float(c), 3) for c in frontier.features["vd"]],
        })
    # Highest predicted gain wins. Gains tie often (the clusters share a
    # gain class), so the deeper frontier breaks it -- same information, more
    # room to actually drive into.
    labels.sort(key=lambda f: (-f["gain_m3"], -f["depth_at_pixel_m"]))
    chosen = labels[0] if labels else None

    # Overlay: the frontier mask tinted onto the frame, every frontier marked
    # with its predicted gain, the chosen one in green.
    overlay = Image.open(pose["start_view"]).convert("RGB")
    if ft_region is not None:
        mask = Image.fromarray((np.asarray(ft_region) > 0).astype(np.uint8) * 255)
        canvas = Image.new("L", (int(width * scale), int(height * scale)), 0)
        canvas.paste(mask, (left, top))
        mask_full = canvas.resize((width, height), Image.NEAREST)
        tint = Image.new("RGB", (width, height), (255, 60, 60))
        overlay = Image.composite(Image.blend(overlay, tint, 0.45), overlay, mask_full)
    draw = ImageDraw.Draw(overlay)
    for label in labels:
        x, y = label["pixel"]
        color = (60, 255, 90) if chosen and label["id"] == chosen["id"] else (255, 220, 60)
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=color, width=3)
        draw.text((x + 12, y - 8), f"#{label['id']}  {label['gain_m3']:.1f} m3", fill=color)
    overlay_path = out / "frontiers.png"
    overlay.save(overlay_path)

    result = {
        "config": args.frontiernet_config,
        "weights": str(root / args.frontiernet_weights),
        "intrinsics_fx": round(float(fx), 2),
        "n_frontiers": len(labels),
        "frontiers": labels,
        "chosen": chosen,
        # Per-pixel map in the network's own units; only the clustered per-frontier
        # gain is rescaled to m^3 (get_3D_ft_clusters multiplies by 10 * 0.001).
        "info_gain_max_raw": round(float(np.nanmax(info_gain)), 3) if info_gain is not None else None,
        "frontier_pixels": int((np.asarray(ft_region) > 0).sum()) if ft_region is not None else 0,
        "overlay": str(overlay_path.resolve()),
        "timings": {k: round(v, 3) for k, v in timings.items()},
    }
    (out / "frontiernet.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "frontiers"}, indent=2))
    if not labels:
        print("[warn] FrontierNet found no frontiers in this view", flush=True)


# --------------------------------------------------------------------------- #
# Stages 3 and 5 -- Cosmos3 reasoner (transformers runtime, image-capable)
# --------------------------------------------------------------------------- #


def stage_reason(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    habitat = json.loads(Path(args.habitat_json).read_text())
    torch.set_num_threads(args.threads)

    device = 0  # CUDA_VISIBLE_DEVICES already pins the physical GPU
    free_mb = torch.cuda.mem_get_info(device)[0] // 2**20
    if free_mb < args.min_free_gpu_mb:
        raise SystemExit(f"aborting before load: only {free_mb} MiB free on the assigned GPU")
    print(f"[reasoner] {free_mb} MiB free; loading {args.checkpoint}", flush=True)

    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    load_s = time.perf_counter() - started
    results: dict[str, object] = {"task": args.task, "load_s": round(load_s, 3)}
    print(f"[reasoner] loaded in {load_s:.1f}s", flush=True)

    def ask(image, prompt: str, repetition_penalty: float = 1.0) -> tuple[str, float]:
        content = [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": prompt},
        ]
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                repetition_penalty=repetition_penalty,
            )
        elapsed = time.perf_counter() - started
        text = processor.tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        return text, elapsed

    if args.task == "caption":
        free_depth = habitat["frontier_free_depth_m"]
        note = DEPTH_FRONTIER_NOTE.format(free_depth=free_depth)
        if args.frontiernet_json and Path(args.frontiernet_json).is_file():
            detected = json.loads(Path(args.frontiernet_json).read_text())
            if detected.get("chosen"):
                note = FRONTIERNET_FRONTIER_NOTE.format(
                    id=detected["chosen"]["id"], n=detected["n_frontiers"],
                    gain=detected["chosen"]["gain_m3"], free_depth=free_depth,
                )
        results["frontier_note"] = note
        prompt = REASONER_PROMPT.format(frontier_note=note, advance=habitat["advance_m"])
        caption, elapsed = ask(Image.open(habitat["observation"]), prompt)
        results |= {"caption": caption, "caption_chars": len(caption), "generate_s": round(elapsed, 3)}
        (out / "caption.txt").write_text(caption + "\n")
        print(f"\n[caption] ({elapsed:.1f}s)\n{caption}\n", flush=True)
    else:
        generated = json.loads(Path(args.generated_json).read_text())
        truth = Image.open(habitat["ground_truth"]).convert("RGB")
        verdicts: dict[str, object] = {}
        for mode, path in generated.items():
            pair = _pair(Image.open(path).convert("RGB"), truth, "IMAGINED", "REAL", Image, ImageDraw)
            pair_path = out / f"compare_{mode}.png"
            pair.save(pair_path)
            # Greedy decoding loops on the object lists here (it re-emits the
            # same nouns until the token cap and the JSON never closes), so the
            # judge gets a repetition penalty the caption does not need. The
            # schema also puts the scores ahead of the lists, so a list that
            # still runs away costs nothing that matters.
            verdict, elapsed = ask(pair, JUDGE_PROMPT, repetition_penalty=1.15)
            verdicts[mode] = {
                "raw": verdict,
                "parsed": _parse_json(verdict),
                "compare_file": str(pair_path),
                "evaluate_s": round(elapsed, 3),
            }
            print(f"\n[judge:{mode}] ({elapsed:.1f}s)\n{verdict}\n", flush=True)
        results["verdicts"] = verdicts

    results["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    (out / f"reasoner_{args.task}.json").write_text(json.dumps(results, indent=2))


# --------------------------------------------------------------------------- #
# Stage 3 -- Cosmos3 generator (cosmos_framework runtime)
# --------------------------------------------------------------------------- #


def stage_cosmos(args: argparse.Namespace) -> None:
    from cosmos_framework.inference.common.init import init_output_dir, init_script

    init_script()

    import torch

    from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    habitat = json.loads(Path(args.habitat_json).read_text())
    caption = Path(args.caption_file).read_text().strip()
    timings: dict[str, float] = {}
    results: dict[str, object] = {}

    device = 0  # CUDA_VISIBLE_DEVICES already pins the physical GPU
    torch.set_num_threads(args.threads)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    free_mb = free_bytes // 2**20
    if free_mb < args.min_free_gpu_mb:
        raise SystemExit(
            f"aborting before load: {free_mb} MiB free on the assigned GPU, need {args.min_free_gpu_mb} MiB. "
            "Something else claimed the memory after the preflight check."
        )
    print(f"[cosmos] {free_mb} MiB free of {total_bytes // 2**20} MiB on the assigned GPU", flush=True)
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    setup = OmniSetupOverrides(
        checkpoint_path=args.checkpoint,
        output_dir=out,
        parallelism_preset="latency",
        # torch.compile is on by default and spends minutes tracing a 14B MoT
        # before the first image appears, which buries the number we came here
        # to measure. --compile turns it back on for steady-state throughput.
        use_torch_compile=args.compile,
        use_cuda_graphs=args.compile,
    ).build_setup()
    init_output_dir(setup.output_dir)
    pipe = OmniInference.create(setup)
    timings["model_load_s"] = time.perf_counter() - started
    results["load_peak_gpu_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)

    def run(name: str, **overrides):
        sample_overrides = OmniSampleOverrides(name=name, **overrides)
        sample_overrides.output_dir = out / name
        sample = sample_overrides.build_sample(model_config=pipe.model_config)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        outputs = pipe.generate([sample])
        elapsed = time.perf_counter() - started
        peak = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        # Hand the transient activation memory back so the other jobs on this
        # box can use it between our calls.
        torch.cuda.empty_cache()
        if not outputs or outputs[0].status != "success":
            message = outputs[0].message if outputs else "no output"
            return None, elapsed, peak, message
        return outputs[0], elapsed, peak, None

    # -- One image of the space beyond the frontier --------------------------
    generated: dict[str, str] = {}
    for mode in args.gen_modes:
        common = dict(
            model_mode=mode,
            seed=args.seed,
            num_steps=args.num_steps,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
        )
        if mode == "forward_dynamics":
            # The only mode that is told the motion in metres rather than in
            # words: the commanded camera trajectory becomes a 9-D action per
            # step ([translation(3), rot6d(6)]), converted by the framework's
            # own utility so handedness and ordering match the weights.
            import numpy as np
            from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel

            poses = np.load(habitat["camera_poses"]).astype(np.float32)
            actions = pose_abs_to_rel(
                poses, rotation_format="rot6d", pose_convention="backward_framewise"
            )
            if not np.isfinite(actions).all():
                raise SystemExit("action tensor contains non-finite values")
            action_path = out / "action.json"
            action_path.write_text(json.dumps(actions.astype(float).tolist()))
            results["forward_dynamics_actions"] = {
                "n_actions": int(actions.shape[0]),
                "action_dim": int(actions.shape[1]),
                "step_translation_m": round(
                    float(np.linalg.norm(actions[:, :3], axis=1).mean()), 4
                ),
                "total_path_m": round(float(np.linalg.norm(actions[:, :3], axis=1).sum()), 3),
            }
            common.update(
                prompt=FORWARD_DYNAMICS_PROMPT,
                action_path=str(action_path),
                action_chunk_size=int(actions.shape[0]),
                domain_name="camera_pose",
                view_point="ego_view",
                image_size=args.image_size,
                aspect_ratio="4,3",
                # Sampler settings from the spec validated against this
                # checkpoint, not the mode defaults.
                fps=args.fd_fps,
                shift=args.fd_shift,
                num_steps=args.fd_num_steps,
            )

            # Autoregressive chaining. The checkpoint only responds to actions
            # at its trained chunk length (16); a longer journey is made by
            # feeding each chunk's final frame back as the next chunk's
            # conditioning image. Every chunk re-commands the same relative
            # trajectory -- relative actions carry no absolute position.
            from cosmos_framework.inference.vision import read_media_frames
            from PIL import Image as PILImage

            def last_frame_still(video_path: Path, still_path: Path) -> Path:
                frames, _ = read_media_frames(video_path, 10_000)
                array = frames[:, -1].numpy().transpose(1, 2, 0)
                still_path.parent.mkdir(parents=True, exist_ok=True)
                PILImage.fromarray(array.astype("uint8")).save(still_path)
                return still_path

            n_chunks = max(1, args.fd_chain)
            vision = habitat["observation"]
            chunks = []
            for k in range(n_chunks):
                name = f"gen_{mode}" if n_chunks == 1 else f"gen_{mode}_chunk{k}"
                output, elapsed, peak, error = run(name, **dict(common, vision_path=vision))
                key = f"generate_{mode}_s" if n_chunks == 1 else f"generate_{mode}_chunk{k}_s"
                timings[key] = elapsed
                results[f"{name}_peak_gpu_mb"] = peak
                if error:
                    results[f"generate_{mode}_error"] = error
                    print(f"[warn] {mode} chunk {k} failed: {error}", flush=True)
                    break
                video = Path(str(output.outputs[0].files[0]))
                still = last_frame_still(video, out / name / "vision_last.jpg")
                chunks.append({"chunk": k, "video": str(video), "last_frame": str(still)})
                vision = str(still)
            if chunks:
                generated[mode] = chunks[-1]["last_frame"]
                results[f"generate_{mode}_file"] = chunks[-1]["last_frame"]
                results[f"generate_{mode}_chunks"] = chunks
            continue
        elif mode == "image2image":
            common["prompt"] = EDIT_PROMPT.format(advance=habitat["advance_m"], caption=caption)
            common["vision_path"] = habitat["observation"]
        else:
            common["prompt"] = caption
        output, elapsed, peak, error = run(f"gen_{mode}", **common)
        timings[f"generate_{mode}_s"] = elapsed
        results[f"generate_{mode}_peak_gpu_mb"] = peak
        if error:
            results[f"generate_{mode}_error"] = error
            print(f"[warn] {mode} failed: {error}", flush=True)
            continue
        files = [str(f) for f in output.outputs[0].files]
        generated[mode] = files[0]
        results[f"generate_{mode}_file"] = files[0]

    results["generated"] = generated
    results["timings"] = {k: round(v, 3) for k, v in timings.items()}
    results["checkpoint"] = args.checkpoint
    results["num_steps"] = args.num_steps
    results["torch_compile"] = args.compile
    results["resolution"] = f"{args.resolution}p {args.aspect_ratio}"
    (out / "generated.json").write_text(json.dumps(generated, indent=2))
    (out / "cosmos.json").write_text(json.dumps(results, indent=2))


def measure_travel(habitat: dict, generated: dict[str, str]) -> dict:
    """How far forward does each generated image actually correspond to?

    The habitat sweep holds the true view at every 0.25 m along the drive, so
    each generated image is scored against all of them and reported at its best
    match. A model that really renders the view from D metres ahead should peak
    near D; one that just cleans up its input peaks at 0.

    Photometric scores only compare like with like: they are meaningful for
    image2image (same house, same renderer) and close to meaningless for
    text2image, which invents a different room -- hence both scores, and the
    caveat carried in the output.
    """
    import numpy as np
    from PIL import Image

    def features(path: str, size=(160, 120)):
        gray = np.asarray(Image.open(path).convert("L").resize(size), dtype=np.float32) / 255.0
        centred = gray - gray.mean()
        norm = float(np.linalg.norm(centred))
        return gray, (centred / norm if norm > 1e-6 else centred)

    # Prefer the flying ruler: it covers the whole straight line to the goal,
    # not just the navmesh-walkable prefix.
    frames_list = habitat.get("sweep_fly") or habitat.get("sweep") or []
    sweep = [(float(s["distance_m"]), *features(s["image"])) for s in frames_list]
    if not sweep:
        return {}

    results: dict[str, object] = {"asked_for_m": habitat["advance_m"], "per_mode": {}}
    for mode, path in generated.items():
        gray, unit = features(path)
        curve = []
        for distance, sweep_gray, sweep_unit in sweep:
            curve.append({
                "distance_m": distance,
                "mae": round(float(np.abs(gray - sweep_gray).mean()), 4),
                "ncc": round(float((unit * sweep_unit).sum()), 4),
            })
        best_mae = min(curve, key=lambda c: c["mae"])
        best_ncc = max(curve, key=lambda c: c["ncc"])
        results["per_mode"][mode] = {
            "best_match_by_mae_m": best_mae["distance_m"],
            "best_match_by_ncc_m": best_ncc["distance_m"],
            "ncc_at_0m": curve[0]["ncc"],
            "ncc_at_target": curve[-1]["ncc"],
            "curve": curve,
        }
    return results


def _pair(left, right, left_label: str, right_label: str, Image, ImageDraw):
    """Two images side by side with labels, for the reasoner to compare."""
    height = max(left.height, right.height)
    left = left.resize((int(left.width * height / left.height), height))
    right = right.resize((int(right.width * height / right.height), height))
    canvas = Image.new("RGB", (left.width + right.width + 8, height + 28), (16, 16, 16))
    canvas.paste(left, (0, 28))
    canvas.paste(right, (left.width + 8, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), left_label, fill=(255, 255, 255))
    draw.text((left.width + 14, 8), right_label, fill=(255, 255, 255))
    return canvas


def _parse_json(text: str):
    """The judge is asked for JSON; the raw text is kept either way."""
    start = text.find("{")
    if start < 0:
        return None
    blob = text[start:]
    end = blob.rfind("}")
    if end > 0:
        try:
            return json.loads(blob[: end + 1])
        except json.JSONDecodeError:
            pass
    # Truncated object (the token cap landed mid-list): cut back to the last
    # complete field and close what is still open, rather than losing the
    # verdict entirely.
    for cut in reversed([i for i, char in enumerate(blob) if char == ","]):
        for closers in ("}", "]}", '"]}'):
            try:
                return json.loads(blob[:cut] + closers)
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def stage_orchestrate(args: argparse.Namespace) -> int:
    interpreters = {
        "habitat": Path(args.habitat_python),
        "cosmos": Path(args.cosmos_python),
        "reason": Path(args.reason_python),
    }
    if args.frontier_source == "frontiernet":
        interpreters["frontiernet"] = Path(args.frontiernet_python)
    for label, path in interpreters.items():
        if not path.is_file():
            raise SystemExit(f"{label} interpreter not found: {path}")
    preflight(args)

    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    habitat_dir, cosmos_dir = run_dir / "habitat", run_dir / "cosmos"
    reason_dir = run_dir / "reason"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Other jobs are sharing this machine: stay off their cores as well as their
    # VRAM. `nice` deprioritises us, the thread caps stop torch/BLAS from
    # grabbing all 32 cores.
    limits = {
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
    }
    be_nice = (lambda: os.nice(args.nice)) if args.nice else None

    wall: dict[str, float] = {}

    this = str(Path(__file__).resolve())
    gpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.cosmos_gpu), **limits)
    gpu_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    shared = [
        "--checkpoint", args.checkpoint, "--habitat-json", str(habitat_dir / "habitat.json"),
        "--seed", str(args.seed), "--max-new-tokens", str(args.max_new_tokens),
        "--threads", str(args.threads), "--min-free-gpu-mb", str(args.min_free_gpu_mb),
    ]

    def step(number: int, label: str, command: list[str], env: dict, cwd: str | None = None) -> None:
        print(f"\n[{number}/{args.total_steps}] {label}", flush=True)
        started = time.perf_counter()
        code = subprocess.run(command, env=env, cwd=cwd, preexec_fn=be_nice).returncode
        wall[f"{label.split(':')[0]}_wall_s"] = time.perf_counter() - started
        if code != 0:
            raise SystemExit(f"stage {label!r} failed with exit code {code}")

    habitat_cmd = [
        str(interpreters["habitat"]), this, "--role", "habitat",
        "--out", str(habitat_dir),
        "--scenes-root", args.scenes_root, "--split", args.split,
        "--scene-index", str(args.scene_index), "--seed", str(args.seed),
        "--width", str(args.width), "--height", str(args.height), "--hfov", str(args.hfov),
        "--sensor-height", str(args.sensor_height), "--max-advance", str(args.max_advance),
        "--clearance", str(args.clearance), "--num-candidates", str(args.num_candidates),
        "--num-yaws", str(args.num_yaws), "--habitat-gpu", str(args.habitat_gpu),
        "--frontier-source", args.frontier_source, "--pose-json", str(habitat_dir / "pose.json"),
        "--overshoot", str(args.overshoot), "--snap-tolerance", str(args.snap_tolerance),
        "--fd-steps", str(args.fd_steps),
    ]
    if args.scene:
        habitat_cmd += ["--scene", args.scene]
    habitat_env = dict(os.environ, MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", **limits)

    step(1, "habitat: render the observation", [*habitat_cmd, "--task", "observe"], habitat_env)
    pose = json.loads((habitat_dir / "pose.json").read_text())

    detected: dict = {}
    turn_deg = pose["depth_turn_deg"]
    target = 0.0
    frontiernet_json = habitat_dir / "frontiernet.json"
    if args.frontier_source == "frontiernet":
        step(2, f"frontiernet: label the frontiers in that image (GPU {args.frontiernet_gpu})",
             [str(interpreters["frontiernet"]), this, "--role", "frontiernet",
              "--out", str(habitat_dir), "--pose-json", str(habitat_dir / "pose.json"),
              "--frontiernet-root", args.frontiernet_root,
              "--frontiernet-config", args.frontiernet_config,
              "--frontiernet-weights", args.frontiernet_weights],
             dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.frontiernet_gpu), **limits))
        detected = json.loads(frontiernet_json.read_text())
        if detected.get("chosen"):
            turn_deg = detected["chosen"]["turn_deg"]
            # Drive to the frontier and through it, not to some fixed distance:
            # stopping short leaves the "beyond" unseen by both the robot and
            # the model, which makes the whole comparison meaningless.
            target = detected["chosen"]["depth_at_pixel_m"] + args.overshoot
        else:
            print("[warn] FrontierNet found nothing; falling back to the depth heuristic", flush=True)

    step(3 if args.frontier_source == "frontiernet" else 2,
         f"habitat: drive at the frontier ({turn_deg:+.1f} deg, target {target or args.max_advance:.1f} m)",
         [*habitat_cmd, "--task", "advance", "--turn-deg", str(turn_deg),
          "--target-advance", str(target)], habitat_env)
    habitat = json.loads((habitat_dir / "habitat.json").read_text())
    print(f"      drove {habitat['advance_m']:.2f} m of {target:.2f} m "
          f"(stopped by {habitat['stopped_by']}); crossed frontier: {habitat['reached_frontier']}", flush=True)

    offset = 1 if args.frontier_source == "frontiernet" else 0
    reason_cmd = [str(interpreters["reason"]), this, "--role", "reason", "--task", "caption",
                  "--out", str(reason_dir), *shared]
    if detected.get("chosen"):
        reason_cmd += ["--frontiernet-json", str(frontiernet_json)]
    step(3 + offset, f"reasoner: imagine beyond the labelled frontier (GPU {args.cosmos_gpu})",
         reason_cmd, gpu_env)
    caption_file = reason_dir / "caption.txt"

    cosmos_cmd = [
        str(interpreters["cosmos"]), this, "--role", "cosmos",
        "--out", str(cosmos_dir), "--caption-file", str(caption_file),
        "--num-steps", str(args.num_steps), "--resolution", args.resolution,
        "--aspect-ratio", args.aspect_ratio, "--gen-modes", ",".join(args.gen_modes),
        "--image-size", str(args.image_size), "--fd-chain", str(args.fd_chain),
        *shared,
    ]
    if args.compile:
        cosmos_cmd.append("--compile")
    step(4 + offset, f"generator: one image beyond the frontier (GPU {args.cosmos_gpu})",
         cosmos_cmd, gpu_env, cwd=str(REPO / "cosmos/packages/cosmos3"))
    cosmos = json.loads((cosmos_dir / "cosmos.json").read_text())

    judge = {}
    if not args.skip_eval and cosmos.get("generated"):
        step(5 + offset, f"reasoner: evaluate against ground truth (GPU {args.cosmos_gpu})",
             [str(interpreters["reason"]), this, "--role", "reason", "--task", "judge",
              "--out", str(reason_dir), "--generated-json", str(cosmos_dir / "generated.json"),
              *shared],
             gpu_env)
        judge = json.loads((reason_dir / "reasoner_judge.json").read_text())

    travel = measure_travel(habitat, cosmos.get("generated") or {})

    report = {
        "run_dir": str(run_dir),
        "habitat": habitat,
        "frontiernet": detected,
        "travel": travel,
        "reasoner": json.loads((reason_dir / "reasoner_caption.json").read_text()),
        "cosmos": cosmos,
        "judge": judge,
        "wall_clock": {k: round(v, 3) for k, v in wall.items()},
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    _summarize(report)
    return 0


def _summarize(report: dict) -> None:
    habitat, cosmos = report["habitat"], report["cosmos"]
    reasoner, judge = report.get("reasoner", {}), report.get("judge", {})
    timings = cosmos.get("timings", {})
    verdicts = judge.get("verdicts", {}) if judge else {}
    line = "-" * 72
    compile_note = "torch.compile on" if cosmos.get("torch_compile") else "no torch.compile"
    print(f"\n{line}\n TIMING  (Cosmos3-Nano, {cosmos.get('resolution')}, "
          f"{cosmos.get('num_steps')} steps, {compile_note})\n{line}")

    detected = report.get("frontiernet") or {}
    rows: list[tuple[str, float | None]] = [
        ("habitat: simulator load", habitat["timings"].get("sim_load_s")),
        ("habitat: pose scan + renders", habitat["timings"].get("pose_scan_s")),
        ("frontiernet load", (detected.get("timings") or {}).get("load_s")),
        ("frontiernet INFER: label frontiers", (detected.get("timings") or {}).get("detect_s")),
        ("frontiernet: anchor frontiers to 3D", (detected.get("timings") or {}).get("anchor_s")),
        ("cosmos load: reasoner runtime", reasoner.get("load_s")),
        ("cosmos INFER: reason beyond the frontier", reasoner.get("generate_s")),
        ("cosmos load: generator runtime", timings.get("model_load_s")),
    ]
    for key, value in timings.items():
        if key.startswith("generate_"):
            rows.append((f"cosmos INFER: generate 1 image ({key[9:-2]})", value))
    if judge:
        rows.append(("cosmos load: reasoner runtime (2nd)", judge.get("load_s")))
    for mode, verdict in verdicts.items():
        rows.append((f"cosmos INFER: evaluate vs ground truth ({mode})", verdict.get("evaluate_s")))
    for label, value in rows:
        if value is not None:
            print(f"  {label:<50} {value:>8.1f} s")

    inference = [reasoner.get("generate_s")]
    inference += [v for k, v in timings.items() if k.startswith("generate_")]
    inference += [v.get("evaluate_s") for v in verdicts.values()]
    loads = [reasoner.get("load_s"), timings.get("model_load_s"), judge.get("load_s") if judge else None]
    print(f"  {'-' * 50} {'-' * 10}")
    print(f"  {'COSMOS3 INFERENCE ONLY (all calls)':<50} {sum(v for v in inference if v):>8.1f} s")
    print(f"  {'checkpoint loading (paid once per process)':<50} {sum(v for v in loads if v):>8.1f} s")
    print(f"  {'end to end wall clock':<50} {sum(report['wall_clock'].values()):>8.1f} s")
    peaks = [v for k, v in cosmos.items() if k.endswith("_peak_gpu_mb")]
    peaks += [reasoner.get("peak_gpu_mb"), judge.get("peak_gpu_mb") if judge else None]
    peak = max((v for v in peaks if v), default=None)
    if peak:
        print(f"  {'peak GPU (one model resident at a time)':<50} {peak / 1024:>8.1f} GiB")

    print(f"{line}\n FRONTIER  (source: {habitat.get('frontier_source', 'depth')})\n{line}")
    if detected.get("frontiers"):
        print(f"  FrontierNet labelled {detected['n_frontiers']} frontier(s) over "
              f"{detected['frontier_pixels']} frontier pixels:")
        for label in detected["frontiers"]:
            mark = "->" if detected.get("chosen") and label["id"] == detected["chosen"]["id"] else "  "
            print(f"   {mark} #{label['id']}  gain {label['gain_m3']:>6.2f} m3   "
                  f"pixel {str(label['pixel']):<16} turn {label['turn_deg']:+6.1f} deg   "
                  f"depth {label['depth_at_pixel_m']:.2f} m")
        print(f"  overlay  : {detected['overlay']}")
    print(f"  scene {habitat['scene']}: turned {habitat['turn_to_frontier_deg']:.1f} deg to the frontier, "
          f"drove {habitat['advance_m']:.2f} m of a {habitat.get('target_advance_m', 0):.2f} m target "
          f"(stopped by {habitat.get('stopped_by', '?')})")
    if habitat.get("target_advance_m"):
        print(f"  crossed the frontier: {habitat.get('reached_frontier')}")

    travel = report.get("travel") or {}
    if travel.get("per_mode"):
        print(f"{line}\n HOW FAR AHEAD DID IT ACTUALLY IMAGINE?\n{line}")
        print(f"  the prompt asked for {travel['asked_for_m']:.2f} m of forward motion; each generated")
        print(f"  image is matched against the true view every 0.25 m along that drive:")
        for mode, scores in travel["per_mode"].items():
            print(f"   {mode:<12} best match at {scores['best_match_by_ncc_m']:>5.2f} m (NCC) / "
                  f"{scores['best_match_by_mae_m']:>5.2f} m (MAE)   "
                  f"NCC {scores['ncc_at_0m']:+.3f} at 0 m -> {scores['ncc_at_target']:+.3f} at the end")
        print("  (photometric, so it is evidence for image2image and only a hint for text2image,")
        print("   which paints a different room and cannot align with any sweep frame)")

    for mode, verdict in verdicts.items():
        parsed = verdict.get("parsed")
        print(f"{line}\n EVALUATION -- imagined ({mode}) vs real\n{line}")
        if not parsed:
            print(f"  (unparsed) {verdict.get('raw', '')[:400]}")
            continue
        print(f"  imagined : {parsed.get('predicted_space_type')}")
        print(f"  real     : {parsed.get('real_space_type')}")
        print(f"  same type: {parsed.get('space_type_match')}   "
              f"layout {parsed.get('layout_similarity_0_to_10')}/10   "
              f"appearance {parsed.get('appearance_similarity_0_to_10')}/10")
        shared = list(dict.fromkeys(parsed.get("shared_objects") or []))[:8]
        print(f"  shared   : {', '.join(shared) or '-'}")
        print(f"  verdict  : {parsed.get('verdict')}")

    print(f"{line}\n  images   : {report['run_dir']}")
    print(f"  report   : {report['run_dir']}/report.json\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("orchestrate", "habitat", "frontiernet", "reason", "cosmos"),
                        default="orchestrate")
    parser.add_argument("--task", choices=("caption", "judge", "observe", "advance"), default="caption",
                        help="caption/judge for --role reason, observe/advance for --role habitat")
    parser.add_argument("--out", default=str(REPO / "eval/cosmos3_nav_test"))

    scene = parser.add_argument_group("scene")
    scene.add_argument("--scenes-root", default=str(DEFAULT_SCENES))
    scene.add_argument("--split", default="val")
    scene.add_argument("--scene", default=None, help="scene name or fragment, e.g. 00802-wcojb4TFT35")
    scene.add_argument("--scene-index", type=int, default=0)
    scene.add_argument("--seed", type=int, default=0)
    scene.add_argument("--width", type=int, default=640)
    scene.add_argument("--height", type=int, default=480)
    scene.add_argument("--hfov", type=float, default=79.0)
    scene.add_argument("--sensor-height", type=float, default=0.88)
    scene.add_argument("--max-advance", type=float, default=12.0,
                       help="hard cap on the drive; the frontier distance is the real target")
    scene.add_argument("--target-advance", type=float, default=0.0,
                       help="drive this far (metres) instead of stopping at the free depth")
    scene.add_argument("--overshoot", type=float, default=1.0,
                       help="metres to continue past the frontier, so the view is genuinely beyond it")
    scene.add_argument("--snap-tolerance", type=float, default=1.0,
                       help="how far the navmesh snap may move the goal before it is rejected")
    scene.add_argument("--clearance", type=float, default=0.6, help="stand-off from the far wall")
    scene.add_argument("--num-candidates", type=int, default=4, help="navigable points to try")
    scene.add_argument("--num-yaws", type=int, default=12, help="headings scanned per point")
    scene.add_argument("--habitat-gpu", type=int, default=0)
    scene.add_argument("--pose-json", default=None)
    scene.add_argument("--turn-deg", type=float, default=0.0, help="for --task advance")

    frontier = parser.add_argument_group("frontier detection")
    frontier.add_argument("--frontier-source", choices=("frontiernet", "depth"), default="frontiernet",
                          help="frontiernet: label the image with FrontierNet and drive at the "
                               "highest-gain frontier; depth: the built-in deepest-opening heuristic")
    frontier.add_argument("--frontiernet-root", default=str(REPO / "FrontierNet"))
    frontier.add_argument("--frontiernet-config", default="hm3d.yaml")
    frontier.add_argument("--frontiernet-weights", default="model_weights/rgbd_11cls.pth")
    frontier.add_argument("--frontiernet-json", default=None)
    frontier.add_argument("--frontiernet-gpu", type=int, default=0)

    cosmos = parser.add_argument_group("cosmos")
    cosmos.add_argument("--checkpoint", default=default_checkpoint())
    cosmos.add_argument("--cosmos-gpu", type=int, default=None,
                        help="default: whichever GPU has the most free memory")
    cosmos.add_argument("--num-steps", type=int, default=35)
    cosmos.add_argument("--resolution", default="480", choices=("256", "480", "720", "768", "1080"))
    cosmos.add_argument("--aspect-ratio", default="4,3", choices=("1,1", "4,3", "3,4", "16,9", "9,16"))
    cosmos.add_argument("--max-new-tokens", type=int, default=512)
    cosmos.add_argument("--gen-modes", default="text2image,image2image,forward_dynamics",
                        help="comma separated: text2image, image2image, forward_dynamics")
    cosmos.add_argument("--image-size", type=int, default=480,
                        help="forward_dynamics rollout resolution")
    # forward_dynamics defaults mirror the spec frontierworld validated against
    # this checkpoint (phase9d); the mode's own defaults (shift 10, fps 24)
    # produce a rollout that barely moves.
    cosmos.add_argument("--fd-steps", type=int, default=16,
                        help="commanded actions in the rollout. Must be a multiple of 4: the "
                             "video is 4k+1 frames and frames = actions + 1")
    cosmos.add_argument("--fd-fps", type=int, default=30)
    cosmos.add_argument("--fd-shift", type=float, default=5.0)
    cosmos.add_argument("--fd-num-steps", type=int, default=30)
    cosmos.add_argument("--fd-chain", type=int, default=1,
                        help="forward_dynamics: autoregressive chunks -- each chunk's last frame "
                             "becomes the next chunk's conditioning image")
    cosmos.add_argument("--compile", action="store_true",
                        help="enable torch.compile (slow first call, faster steady state)")
    cosmos.add_argument("--skip-eval", action="store_true", help="skip the ground-truth comparison")
    cosmos.add_argument("--habitat-json", default=None)
    cosmos.add_argument("--caption-file", default=None)
    cosmos.add_argument("--generated-json", default=None)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--habitat-python", default=str(DEFAULT_HABITAT_PYTHON))
    runtime.add_argument("--cosmos-python", default=str(DEFAULT_COSMOS_PYTHON),
                         help="interpreter with cosmos_framework (the generator)")
    runtime.add_argument("--reason-python", default=str(DEFAULT_REASON_PYTHON),
                         help="interpreter with transformers >= 5.11 (the image-capable reasoner)")
    runtime.add_argument("--frontiernet-python", default=str(DEFAULT_FRONTIERNET_PYTHON),
                         help="interpreter with FrontierNet's deps (hdbscan, smp)")
    runtime.add_argument("--min-free-gpu-mb", type=int, default=36000,
                         help="abort rather than start on a GPU with less free VRAM than this")
    runtime.add_argument("--min-free-ram-mb", type=int, default=45000,
                         help="abort rather than start with less host RAM available than this")
    runtime.add_argument("--threads", type=int, default=8, help="cap on BLAS/torch CPU threads")
    runtime.add_argument("--nice", type=int, default=5, help="niceness for the worker processes (0 disables)")

    args = parser.parse_args()
    args.gen_modes = [m.strip() for m in str(args.gen_modes).split(",") if m.strip()]
    # observe, [frontiernet,] advance, reason, generate, [judge]
    args.total_steps = 4 + (args.frontier_source == "frontiernet") + (not args.skip_eval)

    if args.role == "habitat":
        if args.task not in ("observe", "advance"):
            raise SystemExit("--role habitat needs --task observe or --task advance")
        if not args.pose_json:
            raise SystemExit("--pose-json is required for --role habitat")
        stage_habitat(args)
        return 0
    if args.role == "frontiernet":
        if not args.pose_json:
            raise SystemExit("--pose-json is required for --role frontiernet")
        stage_frontiernet(args)
        return 0
    if args.role in ("reason", "cosmos") and not args.habitat_json:
        raise SystemExit(f"--habitat-json is required for --role {args.role}")
    if args.role == "reason":
        if args.task == "judge" and not args.generated_json:
            raise SystemExit("--generated-json is required for --task judge")
        stage_reason(args)
        return 0
    if args.role == "cosmos":
        if not args.caption_file:
            raise SystemExit("--caption-file is required for --role cosmos")
        stage_cosmos(args)
        return 0
    return stage_orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
