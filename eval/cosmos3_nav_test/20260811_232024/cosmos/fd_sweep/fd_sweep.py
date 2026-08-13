"""Does Cosmos3 travel further if we give the motion more time?

Every rollout commands the SAME 10.71 m to the same frontier. What varies is how
long the model is told that takes: 17 frames at 30 fps is 0.57 s (68 km/h, which
no indoor camera does), while 33 frames at 2 fps is 16.5 s (a slow walk). If the
under-travel is a speed prior rather than a scale failure, the low-fps rollouts
should reach further.

Model is loaded once; each config costs one generate call.
"""

from cosmos_framework.inference.common.init import init_output_dir, init_script

init_script()

import json
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel
from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.vision import read_media_frames

RUN = Path("/home/ashed/Documents/cosmos3_nav/eval/cosmos3_nav_test/20260811_232024")
OUT = Path("/tmp/claude-995200228/-home-ashed-Documents-cosmos3-nav/d43ae068-b0e5-44cc-ba2c-b7068e5c545c/scratchpad/fd_sweep")
CHECKPOINT = "/home/ashed/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/411f42a8fdfb8c5b2583cb8786e0938f49796eaa"

PROMPT = (
    "Ego-centric video from a camera moving through an indoor residential space. "
    "The camera translates smoothly forward; lighting and surfaces stay consistent "
    "as new parts of the home come into view."
)

# (actions, fps). actions must be a multiple of 4 (frames = actions + 1 = 4k+1).
# Round 1 showed fps is inert and longer sequences move LESS: travel tracks the
# per-step translation only. So push that lever the other way -- fewer, bigger
# steps -- and see whether it keeps scaling or hits a ceiling.
CONFIGS = [
    (4, 30),    # 2.68 m per step
    (8, 30),    # 1.34 m per step
    (4, 8),     # 2.68 m per step, slower nominal playback
]


def build_poses(habitat: dict, n_steps: int) -> np.ndarray:
    """Straight-line camera-to-world poses (OpenCV axes) from start to goal."""
    start = np.array(habitat["start_position"], dtype=np.float64)
    goal = np.array(habitat["goal_position"], dtype=np.float64)
    yaw = np.radians(habitat["yaw_scan_deg"] + habitat["turn_to_frontier_deg"])
    cos, sin = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    height = np.array([0.0, habitat["sensor"]["height_m"], 0.0])

    poses = []
    for i in range(n_steps + 1):
        agent = np.eye(4)
        agent[:3, :3] = rotation
        agent[:3, 3] = start + (goal - start) * (i / n_steps) + height
        poses.append(agent @ flip)
    return np.stack(poses).astype(np.float32)


def main() -> None:
    habitat = json.loads((RUN / "habitat" / "habitat.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    free = torch.cuda.mem_get_info(0)[0] // 2**20
    if free < 36000:
        raise SystemExit(f"aborting: only {free} MiB free on the assigned GPU")
    print(f"[sweep] {free} MiB free; commanding {habitat['advance_m']} m in every config", flush=True)

    setup = OmniSetupOverrides(
        checkpoint_path=CHECKPOINT, output_dir=OUT,
        parallelism_preset="latency", use_torch_compile=False, use_cuda_graphs=False,
    ).build_setup()
    init_output_dir(setup.output_dir)
    started = time.perf_counter()
    pipe = OmniInference.create(setup)
    print(f"[sweep] model loaded in {time.perf_counter() - started:.1f}s", flush=True)

    results = []
    for n_actions, fps in CONFIGS:
        name = f"a{n_actions}_fps{fps}"
        poses = build_poses(habitat, n_actions)
        actions = pose_abs_to_rel(poses, rotation_format="rot6d", pose_convention="backward_framewise")
        action_path = OUT / f"{name}_action.json"
        action_path.write_text(json.dumps(actions.astype(float).tolist()))

        duration = (n_actions + 1) / fps
        speed = float(habitat["advance_m"]) / duration
        print(f"\n[sweep] {name}: {n_actions} actions, {fps} fps -> "
              f"{duration:.2f} s, {speed:.2f} m/s", flush=True)

        overrides = OmniSampleOverrides(
            name=name, model_mode="forward_dynamics", prompt=PROMPT,
            vision_path=habitat["observation"], action_path=str(action_path),
            action_chunk_size=int(actions.shape[0]), domain_name="camera_pose",
            view_point="ego_view", image_size=480, aspect_ratio="4,3",
            resolution="480", fps=fps, shift=5.0, num_steps=30, seed=0,
        )
        overrides.output_dir = OUT / name
        sample = overrides.build_sample(model_config=pipe.model_config)

        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        outputs = pipe.generate([sample])
        elapsed = time.perf_counter() - started
        torch.cuda.empty_cache()

        if not outputs or outputs[0].status != "success":
            print(f"[sweep] {name} FAILED: {outputs[0].message if outputs else 'no output'}", flush=True)
            results.append({"name": name, "n_actions": n_actions, "fps": fps, "error": True})
            continue

        video = Path(str(outputs[0].outputs[0].files[0]))
        frames, _ = read_media_frames(video, 10_000)
        array = frames.numpy().astype(np.float32)
        motion = [float(np.abs(array[:, i + 1] - array[:, i]).mean()) for i in range(array.shape[1] - 1)]

        from PIL import Image
        still = OUT / f"{name}_last.jpg"
        Image.fromarray(frames[:, -1].numpy().transpose(1, 2, 0).astype("uint8")).save(still)

        results.append({
            "name": name, "n_actions": n_actions, "fps": fps,
            "duration_s": round(duration, 2), "commanded_speed_mps": round(speed, 2),
            "frames": int(array.shape[1]), "video": str(video), "last_frame": str(still),
            "consecutive_diff_mean": round(float(np.mean(motion)), 2),
            "first_vs_last_diff": round(float(np.abs(array[:, -1] - array[:, 0]).mean()), 2),
            "generate_s": round(elapsed, 1),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated(0) / 2**20, 1),
        })
        print(f"[sweep] {name}: {elapsed:.1f}s, {array.shape[1]} frames, "
              f"motion {results[-1]['consecutive_diff_mean']:.2f}/frame, "
              f"first-vs-last {results[-1]['first_vs_last_diff']:.2f}", flush=True)

    (OUT / "sweep.json").write_text(json.dumps(results, indent=2))
    print("\n[sweep] wrote " + str(OUT / "sweep.json"), flush=True)


if __name__ == "__main__":
    main()
