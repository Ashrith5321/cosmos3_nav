"""Frontier-centred tensorisation.

Turns a stored decision group into model inputs and targets:

    X_{t,i} = (M^local_{t,i}, F_{t,i}, O_{t,i}, omega_i, g)
    Y_{t,i} = (dM^occ_i, dM^sem_i, y^g_i, chi_i, A^new_i)

Everything is expressed in a **frontier-centred frame**: origin at the frontier
centroid, +y pointing along the crossing normal into unknown space, fixed
metric extent and resolution. Two consequences that matter.

First, a model working in the global map frame would have to learn "what lies
beyond a boundary" separately for every orientation the boundary can take. In
the frontier frame, "beyond" is always the same direction, so the geometry is
shared across every example.

Second, the transform must be exactly invertible. A prediction is only useful
if it can be put back on the global map, and an alignment error there is
invisible in the loss but wrong on the map. `to_global` is the inverse of
`to_local` and the round-trip is tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN

# Input channel order. Fixed here so the model, the visualiser and any
# checkpoint agree without a separate convention to keep in sync.
CH_FREE = 0
CH_OCCUPIED = 1
CH_UNKNOWN = 2
CH_BOUNDARY = 3
CH_AGENT = 4
CH_TRAJECTORY = 5
N_INPUT_CHANNELS = 6

# Target channel order.
TGT_REVEALED_FREE = 0
TGT_REVEALED_OCCUPIED = 1
TGT_REVEALED_SEMANTIC = 2
N_TARGET_CHANNELS = 3

INPUT_CHANNEL_NAMES = [
    "observed_free",
    "observed_occupied",
    "unknown",
    "frontier_boundary",
    "agent_position",
    "crossing_trajectory",
]
TARGET_CHANNEL_NAMES = ["revealed_free", "revealed_occupied", "revealed_semantic"]


@dataclass(frozen=True)
class FrameSpec:
    """The frontier-centred window: fixed metric extent and resolution."""

    extent_m: float = 8.0  # side length of the square window
    resolution: float = 0.10  # metres per cell in the model frame

    @property
    def size(self) -> int:
        return int(round(self.extent_m / self.resolution))

    @property
    def half(self) -> float:
        return self.extent_m / 2.0


@dataclass
class FrontierFrame:
    """Rigid transform between the global map and one frontier's frame.

    The frame is centred on the frontier centroid with +v along the crossing
    normal, so "into the unknown" is always +v regardless of which way the
    boundary faces in the world.
    """

    centroid_world: np.ndarray  # (2,) world (x, z)
    normal: np.ndarray  # (2,) unit world (x, z), into unknown
    spec: FrameSpec

    @property
    def rotation(self) -> np.ndarray:
        """World -> local rotation. Rows are the local axes in world terms."""
        v = self.normal / max(float(np.linalg.norm(self.normal)), 1e-9)
        u = np.array([v[1], -v[0]])  # right-handed perpendicular
        return np.stack([u, v])

    def to_local(self, points_world: np.ndarray) -> np.ndarray:
        """World (x, z) -> local (u, v) metres. Accepts (..., 2)."""
        points = np.asarray(points_world, dtype=np.float64)
        return (points - self.centroid_world) @ self.rotation.T

    def to_global(self, points_local: np.ndarray) -> np.ndarray:
        """Local (u, v) metres -> world (x, z). Exact inverse of to_local."""
        points = np.asarray(points_local, dtype=np.float64)
        return points @ self.rotation + self.centroid_world

    def local_to_cell(self, points_local: np.ndarray) -> np.ndarray:
        """Local metres -> integer cell indices (row, col) in the window."""
        points = np.asarray(points_local, dtype=np.float64)
        cols = (points[..., 0] + self.spec.half) / self.spec.resolution
        rows = (points[..., 1] + self.spec.half) / self.spec.resolution
        return np.stack([rows, cols], axis=-1).astype(np.int32)

    def cell_to_local(self, cells: np.ndarray) -> np.ndarray:
        """Cell centres -> local metres. Inverse of local_to_cell up to
        half-cell quantisation, which is why cell centres are used."""
        cells = np.asarray(cells, dtype=np.float64)
        u = (cells[..., 1] + 0.5) * self.spec.resolution - self.spec.half
        v = (cells[..., 0] + 0.5) * self.spec.resolution - self.spec.half
        return np.stack([u, v], axis=-1)

    def cell_to_global(self, cells: np.ndarray) -> np.ndarray:
        return self.to_global(self.cell_to_local(cells))

    def in_window(self, cells: np.ndarray) -> np.ndarray:
        size = self.spec.size
        rows, cols = cells[..., 0], cells[..., 1]
        return (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)


def frame_from_example(example: dict, spec: FrameSpec | None = None) -> FrontierFrame:
    geometry = example["frontier_geometry"]
    centroid = np.asarray(geometry["centroid_world"], dtype=np.float64)[[0, 2]]
    normal = np.asarray(geometry["orientation"], dtype=np.float64)
    return FrontierFrame(centroid, normal, spec or FrameSpec())


def resample_grid(
    grid: np.ndarray,
    map_origin: np.ndarray,
    map_resolution: float,
    frame: FrontierFrame,
    fill: int = UNKNOWN,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a global grid into the frontier window.

    Returns (values, valid). `valid` is false where the window falls outside
    the global map -- those cells carry no evidence and must be excluded from
    the loss rather than treated as unknown-but-observed.
    """
    size = frame.spec.size
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    cells = np.stack([rows, cols], axis=-1)

    world = frame.cell_to_global(cells)  # (size, size, 2) in world x/z
    global_cols = np.floor((world[..., 0] - map_origin[0]) / map_resolution).astype(int)
    global_rows = np.floor((world[..., 1] - map_origin[1]) / map_resolution).astype(int)

    valid = (
        (global_rows >= 0)
        & (global_rows < grid.shape[0])
        & (global_cols >= 0)
        & (global_cols < grid.shape[1])
    )
    values = np.full((size, size), fill, dtype=grid.dtype)
    values[valid] = grid[global_rows[valid], global_cols[valid]]
    return values, valid


def rasterise_cells(
    cells_global: np.ndarray,
    map_origin: np.ndarray,
    map_resolution: float,
    frame: FrontierFrame,
) -> np.ndarray:
    """Draw global grid cells (row, col) into the frontier window."""
    size = frame.spec.size
    canvas = np.zeros((size, size), dtype=np.float32)
    if cells_global.size == 0:
        return canvas

    world_x = map_origin[0] + (cells_global[:, 1] + 0.5) * map_resolution
    world_z = map_origin[1] + (cells_global[:, 0] + 0.5) * map_resolution
    local = frame.to_local(np.stack([world_x, world_z], axis=-1))
    cells = frame.local_to_cell(local)
    keep = frame.in_window(cells)
    if keep.any():
        canvas[cells[keep][:, 0], cells[keep][:, 1]] = 1.0
    return canvas


def rasterise_points(
    points_world: np.ndarray, frame: FrontierFrame, value: float = 1.0
) -> np.ndarray:
    """Draw world (x, z) points into the frontier window."""
    size = frame.spec.size
    canvas = np.zeros((size, size), dtype=np.float32)
    if points_world.size == 0:
        return canvas
    cells = frame.local_to_cell(frame.to_local(points_world))
    keep = frame.in_window(cells)
    if keep.any():
        canvas[cells[keep][:, 0], cells[keep][:, 1]] = value
    return canvas


@dataclass
class BranchTensors:
    """One branch, ready for a model."""

    inputs: np.ndarray  # (C, H, W)
    targets: np.ndarray  # (T, H, W)
    target_valid: np.ndarray  # (H, W) bool
    option: np.ndarray  # encoded omega_i
    goal: str | None
    scalars: dict
    frame: FrontierFrame
    frontier_id: int


def encode_option(example: dict, frame: FrontierFrame) -> np.ndarray:
    """Encode omega_i = (tau_approach, tau_cross, H) as a fixed vector.

    Deliberately includes the *counts* of approach and crossing actions and the
    probe geometry rather than a raw action list: the horizon is fixed and the
    model must be sensitive to how far the probe reaches, which is what changes
    between candidate options at one state.
    """
    option = example["candidate_option"]
    geometry = example["frontier_geometry"]
    return np.array(
        [
            option.get("n_approach_actions", 0) / 50.0,
            option.get("n_cross_actions", 0) / 20.0,
            option.get("probe_distance_m", 0.0) / 5.0,
            option.get("horizon", 0) / 20.0,
            float(geometry.get("geodesic_distance_m") or 0.0) / 20.0,
            geometry.get("information_gain_m2", 0.0) / 20.0,
            geometry.get("boundary_length_m", 0.0) / 5.0,
            # The normal is (0, 1) in the frontier frame by construction, so
            # the world heading is kept for models that want absolute context.
            float(geometry.get("yaw", 0.0)) / np.pi,
        ],
        dtype=np.float32,
    )


def build_branch_tensors(
    example: dict,
    arrays: dict,
    spec: FrameSpec | None = None,
) -> BranchTensors:
    """Tensorise one stored branch."""
    spec = spec or FrameSpec()
    frame = frame_from_example(example, spec)
    map_origin = np.asarray(arrays["map_origin"], dtype=np.float64)
    map_resolution = float(arrays["resolution"])

    grid_before = arrays["grid_before"]
    local_grid, in_map = resample_grid(
        grid_before, map_origin, map_resolution, frame, fill=UNKNOWN
    )

    size = spec.size
    inputs = np.zeros((N_INPUT_CHANNELS, size, size), dtype=np.float32)
    inputs[CH_FREE] = (local_grid == FREE) & in_map
    inputs[CH_OCCUPIED] = (local_grid == OCCUPIED) & in_map
    inputs[CH_UNKNOWN] = ((local_grid == UNKNOWN) | ~in_map).astype(np.float32)

    boundary = np.asarray(
        example["frontier_geometry"].get("boundary_cells", []), dtype=np.int32
    ).reshape(-1, 2)
    inputs[CH_BOUNDARY] = rasterise_cells(boundary, map_origin, map_resolution, frame)

    history = example.get("observation_history", {})
    start = history.get("branch_start_position")
    if start is not None:
        agent = np.asarray(start, dtype=np.float64)[[0, 2]]
        inputs[CH_AGENT] = rasterise_points(agent[None, :], frame)

    trajectory = history.get("trajectory") or []
    crossing = np.asarray(
        [p["position"] for p in trajectory if p.get("phase") == "cross"],
        dtype=np.float64,
    )
    if crossing.size:
        inputs[CH_TRAJECTORY] = rasterise_points(crossing[:, [0, 2]], frame)

    # -- targets ---------------------------------------------------------
    targets = np.zeros((N_TARGET_CHANNELS, size, size), dtype=np.float32)
    target_keys = example.get("target_arrays", {})
    packed_shape = target_keys.get("packed_shape")

    if packed_shape and target_keys.get("grid_after") in arrays:
        from frontierworld.data.records import unpack_mask

        grid_after = arrays[target_keys["grid_after"]]
        revealed = unpack_mask(arrays[target_keys["revealed_mask"]], packed_shape)
        semantic = unpack_mask(
            arrays[target_keys["semantic_revealed_mask"]], packed_shape
        )

        local_after, _ = resample_grid(
            grid_after, map_origin, map_resolution, frame, fill=UNKNOWN
        )
        local_revealed, _ = resample_grid(
            revealed.astype(np.uint8), map_origin, map_resolution, frame, fill=0
        )
        local_semantic, _ = resample_grid(
            semantic.astype(np.uint8), map_origin, map_resolution, frame, fill=0
        )

        revealed_mask = local_revealed.astype(bool) & in_map
        targets[TGT_REVEALED_FREE] = revealed_mask & (local_after == FREE)
        targets[TGT_REVEALED_OCCUPIED] = revealed_mask & (local_after == OCCUPIED)
        targets[TGT_REVEALED_SEMANTIC] = local_semantic.astype(bool) & in_map

    # Cells outside the global map have no ground truth; excluding them keeps
    # the loss from rewarding confident predictions where nothing was observed.
    target_valid = in_map.copy()

    revelation = example.get("future_revelation", {})
    scalars = {
        "target_present": float(bool(revelation.get("target_became_visible"))),
        "crossing_success": float(bool(revelation.get("crossed"))),
        "revealed_area_m2": float(revelation.get("newly_observed_area_m2", 0.0)),
        "collisions": float(revelation.get("collisions", 0)),
        "n_new_frontiers": float(revelation.get("n_new_frontiers", 0)),
    }

    return BranchTensors(
        inputs=inputs,
        targets=targets,
        target_valid=target_valid,
        option=encode_option(example, frame),
        goal=example.get("navigation_goal"),
        scalars=scalars,
        frame=frame,
        frontier_id=example["frontier_geometry"]["frontier_id"],
    )


def build_group_tensors(group, spec: FrameSpec | None = None) -> dict:
    """Tensorise a whole decision group, padded with a candidate mask.

    Returns arrays with a leading candidate axis. Padded slots are zero and
    `candidate_mask` is false there; every loss and every ranking must respect
    it, or the model is scored on frontiers that do not exist.
    """
    spec = spec or FrameSpec()
    branches = [build_branch_tensors(e, group.arrays, spec) for e in group.examples]
    n = len(branches)
    size = spec.size

    inputs = np.zeros((n, N_INPUT_CHANNELS, size, size), dtype=np.float32)
    targets = np.zeros((n, N_TARGET_CHANNELS, size, size), dtype=np.float32)
    valid = np.zeros((n, size, size), dtype=bool)
    options = np.zeros((n, branches[0].option.shape[0]), dtype=np.float32) if n else np.zeros((0, 8))
    scalars = {key: np.zeros(n, dtype=np.float32) for key in
               ("target_present", "crossing_success", "revealed_area_m2",
                "collisions", "n_new_frontiers")}

    for index, branch in enumerate(branches):
        inputs[index] = branch.inputs
        targets[index] = branch.targets
        valid[index] = branch.target_valid
        options[index] = branch.option
        for key in scalars:
            scalars[key][index] = branch.scalars[key]

    return {
        "group_id": group.group_id,
        "scene_id": group.scene_id,
        "episode_id": group.episode_id,
        "decision_timestep": group.decision_timestep,
        "goal": group.navigation_goal,
        "inputs": inputs,
        "targets": targets,
        "target_valid": valid,
        "options": options,
        "scalars": scalars,
        "candidate_mask": np.ones(n, dtype=bool),
        "frontier_ids": [b.frontier_id for b in branches],
        "frames": [b.frame for b in branches],
    }


def collate_tensor_groups(groups: list[dict]) -> dict:
    """Pad a batch of tensorised groups to a common candidate count."""
    if not groups:
        raise ValueError("cannot collate an empty batch")

    max_candidates = max(g["inputs"].shape[0] for g in groups)
    batch = len(groups)
    _, channels, size, _ = groups[0]["inputs"].shape
    target_channels = groups[0]["targets"].shape[1]
    option_dim = groups[0]["options"].shape[1]

    out = {
        "inputs": np.zeros((batch, max_candidates, channels, size, size), dtype=np.float32),
        "targets": np.zeros((batch, max_candidates, target_channels, size, size), dtype=np.float32),
        "target_valid": np.zeros((batch, max_candidates, size, size), dtype=bool),
        "options": np.zeros((batch, max_candidates, option_dim), dtype=np.float32),
        "candidate_mask": np.zeros((batch, max_candidates), dtype=bool),
        "group_ids": [g["group_id"] for g in groups],
        "goals": [g["goal"] for g in groups],
        "scalars": {},
    }
    keys = list(groups[0]["scalars"].keys())
    for key in keys:
        out["scalars"][key] = np.zeros((batch, max_candidates), dtype=np.float32)

    for i, group in enumerate(groups):
        n = group["inputs"].shape[0]
        out["inputs"][i, :n] = group["inputs"]
        out["targets"][i, :n] = group["targets"]
        out["target_valid"][i, :n] = group["target_valid"]
        out["options"][i, :n] = group["options"]
        out["candidate_mask"][i, :n] = True
        for key in keys:
            out["scalars"][key][i, :n] = group["scalars"][key]
    return out
