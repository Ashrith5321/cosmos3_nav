"""Cache-key and leak-guard tests for the Cosmos request builder.

These run in the Habitat environment and need neither a GPU nor the checkpoint:
the pose->action conversion is stubbed with an independent reference
implementation, which doubles as a cross-check on the layout the worker
expects (translation first, then rot6d).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_worker():
    spec = importlib.util.spec_from_file_location("cosmos_worker", SCRIPTS / "cosmos_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_worker()


def reference_action_vectors(poses: np.ndarray) -> np.ndarray:
    """Independent `backward_framewise` [translation(3), rot6d(6)] converter.

    Deliberately not the framework's function: if the worker's expectations
    about layout drift, these tests should notice.
    """
    out = []
    for i in range(len(poses) - 1):
        delta = np.linalg.inv(poses[i].astype(np.float64)) @ poses[i + 1].astype(np.float64)
        rotation = delta[:3, :3]
        out.append(np.concatenate([delta[:3, 3], rotation[:, 0], rotation[:, 1]]))
    return np.stack(out).astype(np.float32)


@pytest.fixture(autouse=True)
def stub_converter(monkeypatch):
    monkeypatch.setattr(worker, "to_action_vectors", reference_action_vectors)


def make_poses(n_frames: int = 17, step: float = 0.25) -> np.ndarray:
    poses = np.zeros((n_frames, 4, 4), dtype=np.float32)
    for i in range(n_frames):
        poses[i] = np.eye(4)
        poses[i][2, 3] = i * step
    return poses


@pytest.fixture
def conditioning(tmp_path) -> Path:
    directory = tmp_path / "conditioning"
    directory.mkdir()
    np.save(directory / "nominal_camera_poses.npy", make_poses())
    (directory / "conditioning_rgb.png").write_bytes(b"not-a-real-png")
    manifest = {
        "schema_version": "frontierworld-cosmos-conditioning-1",
        "group_id": "scene_0_t12_nearest",
        "frontier_id": 3,
        "branch_index": 1,
        "source_rgb_hash": "aaaaaaaaaaaaaaaa",
        "poses": {"hash": "bbbbbbbbbbbbbbbb"},
        "prompt_policy": {"goal_for_audit_only": "chair"},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def run(conditioning: Path, tmp_path: Path, name: str = "out", **overrides) -> dict:
    parameters = dict(
        conditioning=conditioning,
        out_dir=tmp_path / name,
        checkpoint=Path("/checkpoint"),
        prompt=worker.DEFAULT_PROMPT,
        seed=0,
        num_steps=30,
        guidance=1.0,
        shift=5.0,
        fps=30,
        image_size=480,
        checkpoint_hash="cccccccccccccccc",
    )
    parameters.update(overrides)
    return worker.prepare(**parameters)


def test_action_tensor_shape_and_dim(conditioning, tmp_path):
    record = run(conditioning, tmp_path)
    assert record["n_frames"] == 17
    assert record["action_chunk_size"] == 16
    assert record["action_dim"] == 9
    actions = json.loads((tmp_path / "out" / "action.json").read_text())
    assert np.asarray(actions).shape == (16, 9)


def test_frame_count_relation_holds():
    """`num_frames = action_chunk_size + 1` is the framework's contract; a 17
    frame rollout must request exactly 16 actions."""
    poses = make_poses(17)
    assert len(reference_action_vectors(poses)) == 16


def test_identical_inputs_give_identical_key(conditioning, tmp_path):
    first = run(conditioning, tmp_path, name="a")
    second = run(conditioning, tmp_path, name="b")
    assert first["cache_key"] == second["cache_key"]


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 1},
        {"num_steps": 31},
        {"guidance": 1.5},
        {"shift": 4.0},
        {"image_size": 720},
        {"fps": 24},
        {"checkpoint_hash": "dddddddddddddddd"},
        {"prompt": "A completely different description of the scene."},
    ],
)
def test_changing_a_generation_parameter_changes_the_key(conditioning, tmp_path, override):
    baseline = run(conditioning, tmp_path, name="base")
    changed = run(conditioning, tmp_path, name="changed", **override)
    assert baseline["cache_key"] != changed["cache_key"], override


def test_changing_the_action_changes_the_key(conditioning, tmp_path):
    """The whole point of action conditioning: a different option must be a
    different request, or counterfactuals would collide in the cache."""
    baseline = run(conditioning, tmp_path, name="base")
    np.save(conditioning / "nominal_camera_poses.npy", make_poses(step=0.5))
    changed = run(conditioning, tmp_path, name="changed")
    assert baseline["cache_key"] != changed["cache_key"]


def test_changing_the_conditioning_frame_changes_the_key(conditioning, tmp_path):
    baseline = run(conditioning, tmp_path, name="base")
    manifest = json.loads((conditioning / "manifest.json").read_text())
    manifest["source_rgb_hash"] = "ffffffffffffffff"
    (conditioning / "manifest.json").write_text(json.dumps(manifest))
    changed = run(conditioning, tmp_path, name="changed")
    assert baseline["cache_key"] != changed["cache_key"]


def test_changing_the_preprocessing_version_changes_the_key(conditioning, tmp_path):
    baseline = run(conditioning, tmp_path, name="base")
    manifest = json.loads((conditioning / "manifest.json").read_text())
    manifest["schema_version"] = "frontierworld-cosmos-conditioning-2"
    (conditioning / "manifest.json").write_text(json.dumps(manifest))
    changed = run(conditioning, tmp_path, name="changed")
    assert baseline["cache_key"] != changed["cache_key"]


def test_goal_in_prompt_is_refused(conditioning, tmp_path):
    """The ObjectNav goal must never reach the generator."""
    with pytest.raises(SystemExit, match="goal"):
        run(conditioning, tmp_path, prompt="Find the chair in the next room.")


def test_goal_check_is_case_insensitive(conditioning, tmp_path):
    with pytest.raises(SystemExit):
        run(conditioning, tmp_path, prompt="A CHAIR sits by the window.")


def test_default_prompt_contains_no_goal_vocabulary(conditioning, tmp_path):
    """A prompt naming any ObjectNav category would leak the answer even when
    it happens not to be this episode's goal."""
    categories = {
        "chair", "bed", "plant", "toilet", "tv_monitor", "sofa",
        "television", "couch", "potted plant",
    }
    lowered = worker.DEFAULT_PROMPT.lower()
    assert not [c for c in categories if c in lowered]


def test_non_finite_actions_are_refused(conditioning, tmp_path, monkeypatch):
    def broken(poses):
        actions = reference_action_vectors(poses)
        actions[3, 0] = np.nan
        return actions

    monkeypatch.setattr(worker, "to_action_vectors", broken)
    with pytest.raises(SystemExit, match="non-finite"):
        run(conditioning, tmp_path)


def test_wrong_action_dimension_is_refused(conditioning, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "to_action_vectors", lambda poses: np.zeros((16, 7), np.float32))
    with pytest.raises(SystemExit, match="expected"):
        run(conditioning, tmp_path)


def test_generate_refuses_to_disable_guardrails(tmp_path):
    """Guardrails are a required safety component; the worker must not be
    talked into turning them off through pass-through arguments."""
    (tmp_path / "inference_input.json").write_text("{}")
    with pytest.raises(SystemExit, match="guardrails disabled"):
        worker.generate(tmp_path, Path("/checkpoint"), ["--no-guardrails"])


def test_spec_declares_forward_dynamics_and_camera_pose(conditioning, tmp_path):
    record = run(conditioning, tmp_path)
    assert record["spec"]["model_mode"] == "forward_dynamics"
    assert record["spec"]["domain_name"] == "camera_pose"
    assert record["spec"]["action_chunk_size"] == record["n_frames"] - 1
