"""Verify a generated Cosmos rollout before anything downstream trusts it.

Checks that a video exists is not the same as checking it is usable. The
failures that matter here are quiet ones: the right number of frames but all
identical (the model ignored the action), or a frame count that silently
disagrees with the action sequence, or values that are finite but frozen after
frame 3.

Every check is reported with its measured number so a PASS is auditable rather
than asserted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_video(path: Path) -> np.ndarray:
    """Decode to `(T, H, W, 3)` uint8."""
    import imageio.v3 as iio

    return np.asarray(iio.imread(path, plugin="pyav"))


def frame_differences(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive frames, per step."""
    a = frames[:-1].astype(np.float32)
    b = frames[1:].astype(np.float32)
    return np.abs(b - a).mean(axis=(1, 2, 3))


def optical_flow_x(frames: np.ndarray) -> list[float]:
    """Mean horizontal flow per step; sign indicates apparent turn direction.

    Uses Farneback if OpenCV is available. Returns an empty list otherwise
    rather than substituting a weaker proxy and calling it flow.
    """
    try:
        import cv2
    except ImportError:
        return []

    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    out = []
    for i in range(len(grey) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grey[i], grey[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        out.append(float(flow[..., 0].mean()))
    return out


def verify(video_path: Path, request: dict, conditioning: Path, action_path: Path) -> dict:
    frames = read_video(video_path)
    n_expected = request["n_frames"]
    actions = np.asarray(json.loads(action_path.read_text()))

    differences = frame_differences(frames)
    checks: dict = {}

    checks["decodes"] = {"pass": frames.ndim == 4 and frames.shape[-1] == 3,
                         "shape": list(frames.shape), "dtype": str(frames.dtype)}

    checks["frame_count"] = {
        "pass": int(frames.shape[0]) == n_expected,
        "expected": n_expected,
        "actual": int(frames.shape[0]),
        "note": "num_frames must equal action_chunk_size + 1",
    }

    checks["finite"] = {
        "pass": bool(np.isfinite(frames.astype(np.float32)).all()),
        "min": int(frames.min()),
        "max": int(frames.max()),
    }

    # A model that ignored the action would emit a still image. Requiring every
    # consecutive pair to differ is a stronger check than requiring the mean to
    # differ, which a single moving frame could satisfy.
    checks["no_frame_collapse"] = {
        "pass": bool((differences > 1.0).all()),
        "min_consecutive_difference": float(differences.min()),
        "mean_consecutive_difference": float(differences.mean()),
        "per_step": [round(float(d), 3) for d in differences],
        "threshold": 1.0,
    }

    checks["no_duplicate_frames"] = {
        "pass": bool(len({frames[i].tobytes() for i in range(len(frames))}) == len(frames)),
        "n_unique": len({frames[i].tobytes() for i in range(len(frames))}),
    }

    # Temporal order: content should drift monotonically away from frame 0, not
    # oscillate back to it, which would indicate the rollout is looping.
    from_first = np.abs(
        frames.astype(np.float32) - frames[0].astype(np.float32)
    ).mean(axis=(1, 2, 3))
    checks["drifts_from_conditioning_frame"] = {
        "pass": bool(from_first[-1] > from_first[1]),
        "distance_from_frame0": [round(float(d), 3) for d in from_first],
    }

    # Conditioning consistency: the generated first frame should resemble the
    # conditioning image far more than a later frame does.
    import imageio.v3 as iio

    conditioning_rgb = np.asarray(
        iio.imread(conditioning / "conditioning_rgb.png")
    )[..., :3]
    if conditioning_rgb.shape[:2] != frames.shape[1:3]:
        import cv2

        conditioning_rgb = cv2.resize(
            conditioning_rgb, (frames.shape[2], frames.shape[1])
        )
    first_error = float(
        np.abs(frames[0].astype(np.float32) - conditioning_rgb.astype(np.float32)).mean()
    )
    last_error = float(
        np.abs(frames[-1].astype(np.float32) - conditioning_rgb.astype(np.float32)).mean()
    )
    checks["conditioning_consistency"] = {
        "pass": first_error < last_error,
        "frame0_vs_conditioning": round(first_error, 3),
        "frameN_vs_conditioning": round(last_error, 3),
    }

    # Turn direction: yaw sign from the commanded rot6d block vs apparent flow.
    # A camera turning left makes the scene sweep right, so the two should be
    # anti-correlated. Reported, not asserted, when flow is unavailable.
    yaw_command = []
    for row in actions:
        rotation = np.array([row[3:6], row[6:9]])
        yaw_command.append(float(np.arctan2(rotation[0, 2], rotation[0, 0])))
    flow = optical_flow_x(frames)
    correlation = None
    if flow and len(flow) == len(yaw_command):
        if np.std(yaw_command) > 1e-6 and np.std(flow) > 1e-6:
            correlation = float(np.corrcoef(yaw_command, flow)[0, 1])
    checks["turn_direction"] = {
        "pass": None if correlation is None else bool(abs(correlation) > 0.3),
        "commanded_yaw_per_step": [round(y, 4) for y in yaw_command],
        "mean_horizontal_flow_per_step": [round(f, 3) for f in flow] or None,
        "correlation": None if correlation is None else round(correlation, 3),
        "note": "camera yaw and apparent horizontal flow should be anti-correlated; "
                "sign convention is not asserted, magnitude of correlation is",
    }

    # Forward motion: commanded translation magnitude should relate to how much
    # the image changes. Correlation, not a threshold, because image change also
    # depends on scene content.
    translation = np.linalg.norm(actions[:, :3], axis=1)
    motion_correlation = None
    if np.std(translation) > 1e-6 and np.std(differences) > 1e-6:
        motion_correlation = float(np.corrcoef(translation, differences)[0, 1])
    checks["forward_motion_tracks_command"] = {
        "pass": None if motion_correlation is None else bool(motion_correlation > 0.0),
        "commanded_translation_per_step_m": [round(float(t), 4) for t in translation],
        "correlation_with_frame_difference": (
            None if motion_correlation is None else round(motion_correlation, 3)
        ),
    }

    asserted = [k for k, v in checks.items() if v.get("pass") is not None]
    return {
        "video": str(video_path),
        "checks": checks,
        "n_checks_asserted": len(asserted),
        "n_passed": sum(1 for k in asserted if checks[k]["pass"]),
        "all_passed": all(checks[k]["pass"] for k in asserted),
    }


def contact_sheet(video_path: Path, out_path: Path, columns: int = 6) -> None:
    import imageio.v3 as iio

    frames = read_video(video_path)
    rows = int(np.ceil(len(frames) / columns))
    height, width = frames.shape[1:3]
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        r, c = divmod(index, columns)
        sheet[r * height : (r + 1) * height, c * width : (c + 1) * width] = frame
    iio.imwrite(out_path, sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()

    request = json.loads(args.request.read_text())
    report = verify(args.video, request, args.conditioning, args.request.parent / "action.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    if args.contact_sheet:
        contact_sheet(args.video, args.contact_sheet)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
