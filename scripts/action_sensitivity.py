"""Does the model actually obey the action, or just continue the video?

The smoke rollout showed motion, but motion alone proves nothing: an
unconditioned video model also produces motion. The decisive test holds the
conditioning frame, seed and prompt fixed and varies ONLY the action, then asks
whether the outputs differ in the commanded direction.

Four probes, each one primitive action per frame so the per-frame motion stays
in the range the model was trained on:

    turn_left     +30 deg yaw per frame, no translation
    turn_right    -30 deg yaw per frame, no translation
    forward       +0.25 m along the camera axis per frame, no rotation
    still         identity

`still` is the control. If it moves as much as the others, the model is
ignoring the action and the other comparisons mean nothing.

Sign convention is deliberately not asserted -- the claim under test is that
left and right produce OPPOSITE horizontal image motion, and that both differ
from `still`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STEP_METRES = 0.25
STEP_DEGREES = 30.0


def yaw_delta(angle_rad: float) -> np.ndarray:
    """4x4 relative transform: rotation about the camera's y axis."""
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    return delta


def translation_delta(distance: float) -> np.ndarray:
    """4x4 relative transform: translation along the camera's +z (forward)."""
    delta = np.eye(4, dtype=np.float64)
    delta[2, 3] = distance
    return delta


def compose(delta: np.ndarray, n_steps: int) -> np.ndarray:
    """Absolute camera-to-world poses from a repeated relative transform."""
    poses = [np.eye(4, dtype=np.float64)]
    for _ in range(n_steps):
        poses.append(poses[-1] @ delta)
    return np.stack(poses)


def probes(n_steps: int) -> dict[str, np.ndarray]:
    return {
        "turn_left": compose(yaw_delta(np.deg2rad(STEP_DEGREES)), n_steps),
        "turn_right": compose(yaw_delta(np.deg2rad(-STEP_DEGREES)), n_steps),
        "forward": compose(translation_delta(STEP_METRES), n_steps),
        "still": compose(np.eye(4), n_steps),
    }


def write_specs(
    out_dir: Path,
    conditioning_rgb: Path,
    prompt: str,
    seed: int,
    n_frames: int,
    num_steps: int,
    guidance: float,
    shift: float,
    fps: int,
    image_size: int,
) -> list[Path]:
    from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel

    out_dir.mkdir(parents=True, exist_ok=True)
    spec_paths = []
    for name, poses in probes(n_frames - 1).items():
        actions = pose_abs_to_rel(
            poses.astype(np.float32),
            rotation_format="rot6d",
            pose_convention="backward_framewise",
        )
        action_path = out_dir / f"{name}_action.json"
        action_path.write_text(json.dumps(actions.astype(float).tolist()))

        spec = {
            "name": name,
            "model_mode": "forward_dynamics",
            "domain_name": "camera_pose",
            "view_point": "ego_view",
            "action_chunk_size": n_frames - 1,
            "action_path": str(action_path.resolve()),
            "vision_path": str(conditioning_rgb.resolve()),
            "prompt": prompt,
            "seed": seed,
            "num_steps": num_steps,
            "guidance": guidance,
            "shift": shift,
            "fps": fps,
            "image_size": image_size,
        }
        spec_path = out_dir / f"{name}.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        spec_paths.append(spec_path)
    return spec_paths


def horizontal_flow(video_path: Path) -> list[float]:
    import cv2
    import imageio.v3 as iio

    frames = np.asarray(iio.imread(video_path, plugin="pyav"))
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    return [
        float(
            cv2.calcOpticalFlowFarneback(
                grey[i], grey[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0
            )[..., 0].mean()
        )
        for i in range(len(grey) - 1)
    ]


def analyse(generated_root: Path) -> dict:
    import imageio.v3 as iio

    results: dict = {}
    for name in probes(1):
        video = generated_root / name / "vision.mp4"
        if not video.exists():
            results[name] = {"error": "missing"}
            continue
        frames = np.asarray(iio.imread(video, plugin="pyav"))
        differences = np.abs(
            frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)
        ).mean(axis=(1, 2, 3))
        flow = horizontal_flow(video)
        results[name] = {
            "n_frames": int(frames.shape[0]),
            "mean_frame_difference": round(float(differences.mean()), 3),
            "mean_horizontal_flow": round(float(np.mean(flow)), 3),
            "per_step_flow": [round(f, 3) for f in flow],
        }

    verdict: dict = {}
    if all("error" not in results[k] for k in ("turn_left", "turn_right", "still")):
        left = results["turn_left"]["mean_horizontal_flow"]
        right = results["turn_right"]["mean_horizontal_flow"]
        still = results["still"]["mean_frame_difference"]
        verdict["left_and_right_have_opposite_sign"] = bool(left * right < 0)
        verdict["left_right_separation"] = round(abs(left - right), 3)
        verdict["still_is_quietest"] = bool(
            still < min(results[k]["mean_frame_difference"] for k in ("turn_left", "turn_right", "forward"))
        )
        verdict["model_consumes_action"] = bool(
            verdict["left_and_right_have_opposite_sign"] and verdict["still_is_quietest"]
        )
    return {"per_probe": results, "verdict": verdict}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning-rgb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    if not args.analyse_only:
        paths = write_specs(
            args.out, args.conditioning_rgb, args.prompt, args.seed, args.frames,
            args.num_steps, args.guidance, args.shift, args.fps, args.image_size,
        )
        print("\n".join(str(p) for p in paths))
        return

    report = analyse(args.out / "generated")
    (args.out / "sensitivity_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
