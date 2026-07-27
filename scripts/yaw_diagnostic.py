"""Final in-distribution yaw diagnostic. Criteria fixed in
`archive/phase9d/prereg_in_distribution_yaw.md` before any rollout was made.

Two arms, five yaw levels each, plus one repeat for determinism. Everything
except the commanded yaw is held fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

YAW_LEVELS_DEG = [-0.50, -0.25, 0.00, 0.25, 0.50]
REFERENCE_TRANSLATION_M = 0.884  # mean of camera_action_44.json
ARMS = {"a_zerotrans": 0.0, "b_reftrans": REFERENCE_TRANSLATION_M}
N_STEPS = 16


def condition_name(arm: str, yaw_deg: float) -> str:
    return f"{arm}_yaw{yaw_deg:+.2f}".replace(".", "p").replace("+", "p").replace("-", "m")


def poses_for(yaw_deg: float, translation_m: float, n_steps: int) -> np.ndarray:
    """Absolute camera-to-world poses for a constant per-frame twist."""
    angle = np.deg2rad(yaw_deg)
    cos, sin = np.cos(angle), np.sin(angle)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    delta[2, 3] = translation_m

    poses = [np.eye(4, dtype=np.float64)]
    for _ in range(n_steps):
        poses.append(poses[-1] @ delta)
    return np.stack(poses)


def write_specs(out_dir: Path, conditioning_rgb: Path, prompt: str, seed: int) -> list[Path]:
    from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel

    out_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    jobs = [(arm, yaw, condition_name(arm, yaw)) for arm, _ in ARMS.items() for yaw in YAW_LEVELS_DEG]
    # Determinism check: one condition repeated with an identical action and seed.
    jobs.append(("a_zerotrans", 0.50, "repeat_" + condition_name("a_zerotrans", 0.50)))

    for arm, yaw, name in jobs:
        actions = pose_abs_to_rel(
            poses_for(yaw, ARMS[arm], N_STEPS).astype(np.float32),
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
            "action_chunk_size": N_STEPS,
            "action_path": str(action_path.resolve()),
            "vision_path": str(conditioning_rgb.resolve()),
            "prompt": prompt,
            "seed": seed,
            "num_steps": 30,
            "guidance": 1.0,
            "shift": 5.0,
            "fps": 30,
            "image_size": 480,
        }
        spec_path = out_dir / f"{name}.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        specs.append(spec_path)
    return specs


def mean_horizontal_flow(video: Path) -> float:
    import cv2
    import imageio.v3 as iio

    frames = np.asarray(iio.imread(video, plugin="pyav"))
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    values = [
        float(
            cv2.calcOpticalFlowFarneback(
                grey[i], grey[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0
            )[..., 0].mean()
        )
        for i in range(len(grey) - 1)
    ]
    return float(np.mean(values))


def spearman(x: list[float], y: list[float]) -> float:
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    if np.std(rank_x) < 1e-12 or np.std(rank_y) < 1e-12:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def analyse(generated: Path) -> dict:
    import imageio.v3 as iio

    report: dict = {"arms": {}}

    for arm in ARMS:
        flows, levels = [], []
        for yaw in YAW_LEVELS_DEG:
            video = generated / condition_name(arm, yaw) / "vision.mp4"
            if not video.exists():
                continue
            flows.append(mean_horizontal_flow(video))
            levels.append(yaw)

        if len(flows) != len(YAW_LEVELS_DEG):
            report["arms"][arm] = {"error": "missing rollouts", "n": len(flows)}
            continue

        zero_index = levels.index(0.0)
        relative = [f - flows[zero_index] for f in flows]
        # Arm B mixes translation-induced flow into every condition, so the
        # sign test is applied to flow relative to the zero-yaw control.
        signal = relative if arm == "b_reftrans" else flows

        negative = [signal[i] for i, y in enumerate(levels) if y < 0]
        positive = [signal[i] for i, y in enumerate(levels) if y > 0]
        opposite = bool(np.mean(negative) * np.mean(positive) < 0)

        rho = spearman(levels, signal)
        correlation = (
            float(np.corrcoef(levels, signal)[0, 1])
            if np.std(signal) > 1e-12 else 0.0
        )

        report["arms"][arm] = {
            "translation_m_per_frame": ARMS[arm],
            "yaw_levels_deg": levels,
            "mean_horizontal_flow": [round(f, 4) for f in flows],
            "flow_relative_to_zero": [round(f, 4) for f in relative],
            "signal_used": "relative_to_zero" if arm == "b_reftrans" else "absolute",
            "criterion_1_opposite_signs": opposite,
            "criterion_2_monotonic": bool(abs(rho) == 1.0),
            "spearman": round(rho, 4),
            "criterion_3_correlation_gt_0p30": bool(abs(correlation) > 0.30),
            "correlation": round(correlation, 4),
        }

    first = generated / condition_name("a_zerotrans", 0.50) / "vision.mp4"
    repeat = generated / ("repeat_" + condition_name("a_zerotrans", 0.50)) / "vision.mp4"
    if first.exists() and repeat.exists():
        a = np.asarray(iio.imread(first, plugin="pyav")).astype(np.float32)
        b = np.asarray(iio.imread(repeat, plugin="pyav")).astype(np.float32)
        difference = float(np.abs(a - b).mean()) if a.shape == b.shape else float("inf")
        report["criterion_4_determinism"] = {
            "mean_abs_pixel_difference": round(difference, 5),
            "threshold": 1.0,
            "pass": bool(difference < 1.0),
        }

    per_arm_pass = {
        arm: all(
            values.get(k, False)
            for k in (
                "criterion_1_opposite_signs",
                "criterion_2_monotonic",
                "criterion_3_correlation_gt_0p30",
            )
        )
        for arm, values in report["arms"].items()
    }
    report["per_arm_pass"] = per_arm_pass
    report["arms_agree"] = len(set(per_arm_pass.values())) == 1
    report["determinism_pass"] = report.get("criterion_4_determinism", {}).get("pass", False)
    report["overall_pass"] = bool(
        all(per_arm_pass.values()) and report["determinism_pass"]
    )
    report["verdict"] = (
        "PASS" if report["overall_pass"]
        else ("INCONCLUSIVE — arms disagree" if not report["arms_agree"] else "FAIL")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning-rgb", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    if args.analyse_only:
        report = analyse(args.out / "generated")
        (args.out / "yaw_diagnostic_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    for path in write_specs(args.out, args.conditioning_rgb, args.prompt, args.seed):
        print(path)


if __name__ == "__main__":
    main()
