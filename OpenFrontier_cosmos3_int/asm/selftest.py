"""Offline self-test: builds an ASM from a synthetic room, no habitat, no SAM3.

    python -m asm.selftest [out_dir]

Renders a 6 m x 6 m room seen from four viewpoints, injects two fake object
masks, and checks that geometry, semantics, annotation and the prompt sentence
all come out consistent.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from asm.config import ASMConfig                      # noqa: E402
from asm.semantic_map import AnnotatedSemanticMap     # noqa: E402

W, H, HFOV = 320, 240, 79.0


def intrinsics() -> np.ndarray:
    f = (W / 2.0) / np.tan(np.deg2rad(HFOV) / 2.0)
    return np.array([[f, 0, (W - 1) / 2.0],
                     [0, f, (H - 1) / 2.0],
                     [0, 0, 1.0]])


def pose(x: float, y: float, yaw_deg: float, z: float = 0.88) -> np.ndarray:
    """Camera-to-world, world Z-up, camera +z forward and +y down."""
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    forward = np.array([c, s, 0.0])
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(down, forward)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, down, forward
    T[:3, 3] = [x, y, z]
    return T


def synth_depth(wall_distance: float = 3.0) -> np.ndarray:
    """A flat wall with a strip of floor along the bottom of the frame."""
    K = intrinsics()
    depth = np.full((H, W), wall_distance, np.float32)
    vs = np.arange(H).reshape(-1, 1).astype(np.float32)
    below = vs > K[1, 2] + 12
    # Floor: depth grows with the ray's downward angle.
    ray_y = (vs - K[1, 2]) / K[1, 1]
    floor_depth = np.divide(0.88, np.maximum(ray_y, 1e-3))
    depth = np.where(below & (floor_depth < wall_distance),
                     floor_depth.astype(np.float32), depth)
    return np.clip(depth, 0.0, 3.5).repeat(W, axis=1)[:, :W] if depth.shape[1] == 1 \
        else np.clip(depth, 0.0, 3.5)


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ASMConfig(categories=["chair", "bed"], resolution_m=0.05, extent_m=20.0,
                    min_votes_per_cell=1, min_blob_area_cells=4)
    amap = AnnotatedSemanticMap(cfg, intrinsics())

    depth = synth_depth()
    assert depth.shape == (H, W), depth.shape
    floor_z = 0.0

    failures = []

    # Four viewpoints around the origin.
    for i, yaw in enumerate((0, 90, 180, 270)):
        T = pose(0.0, 0.0, yaw)
        amap.integrate_geometry(depth, T, floor_z)

        masks = {}
        if yaw == 0:
            m = np.zeros((H, W), bool); m[100:150, 120:200] = True
            masks["chair"] = m
        elif yaw == 180:
            m = np.zeros((H, W), bool); m[100:150, 100:220] = True
            masks["bed"] = m
        if masks:
            amap.integrate_semantics(depth, T, floor_z, masks)

    # --- checks -----------------------------------------------------------
    if amap.n_geometry_updates != 4:
        failures.append(f"geometry updates {amap.n_geometry_updates} != 4")
    if amap.free_votes.sum() == 0:
        failures.append("no free-space votes accumulated")
    if amap.occ_votes.sum() == 0:
        failures.append("no occupied votes accumulated")

    grid = amap.label_grid()
    n_sem = int((grid >= 3).sum())
    if n_sem == 0:
        failures.append("no semantic cells in the label grid")

    image, objects = amap.annotate()
    labels = sorted({o.label for o in objects})
    if labels != ["bed", "chair"]:
        failures.append(f"expected ['bed', 'chair'], got {labels}")
    if image.ndim != 3 or image.shape[2] != 3:
        failures.append(f"bad annotated image shape {image.shape}")

    # The chair was seen facing +x, the bed facing -x: their centroids must
    # land on opposite sides of the origin.
    by_label = {o.label: o for o in objects}
    if "chair" in by_label and "bed" in by_label:
        if not (by_label["chair"].centroid_world[0] > by_label["bed"].centroid_world[0]):
            failures.append(
                "object centroids are not on the expected sides: "
                f"chair={by_label['chair'].centroid_world}, "
                f"bed={by_label['bed'].centroid_world}"
            )

    sentence = amap.prompt_text(objects)
    if "chair" not in sentence or "bed" not in sentence:
        failures.append(f"prompt text missing objects: {sentence!r}")

    import cv2
    cv2.imwrite(str(out_dir / "selftest_asm.png"), image[:, :, ::-1])

    print(f"free cells      : {int((amap.free_votes > 0).sum())}")
    print(f"occupied cells  : {int((amap.occ_votes > 0).sum())}")
    print(f"semantic cells  : {n_sem}")
    print(f"objects         : {[o.to_dict() for o in objects]}")
    print(f"prompt text     : {sentence}")
    print(f"image           : {image.shape} -> {out_dir / 'selftest_asm.png'}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nOK: all self-test checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
