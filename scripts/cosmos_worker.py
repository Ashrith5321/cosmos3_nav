"""Cosmos-side worker: conditioning manifest -> generated rollout.

Runs in the **Cosmos** environment (Python 3.13 / torch 2.10). It never imports
Habitat, and it reads nothing except the conditioning directory produced by
`build_cosmos_manifest.py`.

Two stages, deliberately separable so the first can be verified without a GPU:

    prepare   manifest + nominal poses -> 9-D action JSON + inference input spec
    generate  invoke the Cosmos inference CLI on that spec, guardrails enabled

The prompt describes only appearance and camera motion. The ObjectNav goal is
never included; `prepare` fails loudly if the goal string appears in it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Neutral, scene-agnostic. It carries no information about what lies beyond the
# frontier, which is the quantity the experiment is trying to measure.
DEFAULT_PROMPT = (
    "Ego-centric video from a camera moving through an indoor residential space. "
    "The camera translates and rotates smoothly; lighting and surfaces stay "
    "consistent as new parts of the room come into view."
)

DOMAIN_NAME = "camera_pose"
RAW_ACTION_DIM = 9


def to_action_vectors(poses_camera: np.ndarray) -> np.ndarray:
    """`(T, 4, 4)` camera-to-world -> `(T-1, 9)` `[translation(3), rot6d(6)]`.

    Uses the framework's own converter so the handedness and ordering match the
    weights exactly.
    """
    from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel

    return pose_abs_to_rel(
        np.asarray(poses_camera, dtype=np.float32),
        rotation_format="rot6d",
        pose_convention="backward_framewise",
    )


def request_hash(payload: dict) -> str:
    """Cache key over everything that can change a pixel."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def prepare(
    conditioning: Path,
    out_dir: Path,
    checkpoint: Path,
    prompt: str,
    seed: int,
    num_steps: int,
    guidance: float,
    shift: float,
    fps: int,
    image_size: int,
    checkpoint_hash: str,
) -> dict:
    manifest = json.loads((conditioning / "manifest.json").read_text())
    poses = np.load(conditioning / "nominal_camera_poses.npy")

    goal = str(manifest["prompt_policy"]["goal_for_audit_only"] or "").strip()
    if goal and goal.lower() in prompt.lower():
        raise SystemExit(
            f"refusing to generate: the ObjectNav goal {goal!r} appears in the prompt. "
            "The goal must never condition generation."
        )

    actions = to_action_vectors(poses)
    n_frames = int(poses.shape[0])
    action_chunk_size = n_frames - 1

    if actions.shape != (action_chunk_size, RAW_ACTION_DIM):
        raise SystemExit(
            f"action tensor is {actions.shape}, expected {(action_chunk_size, RAW_ACTION_DIM)}"
        )
    if not np.isfinite(actions).all():
        raise SystemExit("action tensor contains non-finite values")

    out_dir.mkdir(parents=True, exist_ok=True)
    action_path = out_dir / "action.json"
    action_path.write_text(json.dumps(actions.astype(float).tolist()))

    spec = {
        "name": f"{manifest['group_id']}_f{manifest['frontier_id']}",
        "model_mode": "forward_dynamics",
        "domain_name": DOMAIN_NAME,
        "view_point": "ego_view",
        "action_chunk_size": action_chunk_size,
        "action_path": str(action_path.resolve()),
        "vision_path": str((conditioning / "conditioning_rgb.png").resolve()),
        "prompt": prompt,
        "seed": seed,
        "num_steps": num_steps,
        "guidance": guidance,
        "shift": shift,
        "fps": fps,
        "image_size": image_size,
    }
    spec_path = out_dir / "inference_input.json"
    spec_path.write_text(json.dumps(spec, indent=2))

    key_payload = {
        "checkpoint_hash": checkpoint_hash,
        "preprocessing_version": manifest["schema_version"],
        "scheduler": "UniPCMultistepScheduler",
        "num_steps": num_steps,
        "guidance": guidance,
        "shift": shift,
        "image_size": image_size,
        "n_frames": n_frames,
        "fps": fps,
        "seed": seed,
        "dtype": "bfloat16",
        "action_mode": "forward_dynamics",
        "domain_name": DOMAIN_NAME,
        "prompt": prompt,
        "conditioning_rgb_hash": manifest["source_rgb_hash"],
        "pose_hash": manifest["poses"]["hash"],
        "action_vectors": actions.astype(float).round(6).tolist(),
    }
    key = request_hash(key_payload)

    record = {
        "cache_key": key,
        "key_payload": key_payload,
        "spec": spec,
        "group_id": manifest["group_id"],
        "frontier_id": manifest["frontier_id"],
        "branch_index": manifest["branch_index"],
        "n_frames": n_frames,
        "action_chunk_size": action_chunk_size,
        "action_dim": RAW_ACTION_DIM,
        "action_stats": {
            "translation_norm_per_step_m": [
                float(v) for v in np.linalg.norm(actions[:, :3], axis=1)
            ],
            "total_path_length_m": float(np.linalg.norm(actions[:, :3], axis=1).sum()),
        },
        "checkpoint": str(checkpoint),
        "goal_leak_check": "passed: ObjectNav goal absent from prompt",
    }
    (out_dir / "request.json").write_text(json.dumps(record, indent=2))
    return record


def generate(out_dir: Path, checkpoint: Path, extra: list[str]) -> int:
    """Invoke the framework CLI. Guardrails stay enabled."""
    command = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.inference",
        "-i",
        str((out_dir / "inference_input.json").resolve()),
        "--checkpoint-path",
        str(checkpoint),
        "--output-dir",
        str((out_dir / "generated").resolve()),
        *extra,
    ]
    if any(argument.startswith("--no-guardrail") for argument in extra):
        raise SystemExit("refusing to run with guardrails disabled")
    print(" ".join(command))
    return subprocess.run(command, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-hash", default="unknown")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("extra", nargs="*", default=[])
    args = parser.parse_args()

    record = prepare(
        conditioning=args.conditioning,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance=args.guidance,
        shift=args.shift,
        fps=args.fps,
        image_size=args.image_size,
        checkpoint_hash=args.checkpoint_hash,
    )
    print(json.dumps({k: v for k, v in record.items() if k != "key_payload"}, indent=2))

    if args.generate:
        code = generate(args.out, args.checkpoint, list(args.extra))
        raise SystemExit(code)


if __name__ == "__main__":
    main()
