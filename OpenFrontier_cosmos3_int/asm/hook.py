"""Attach the ASM builder to a live NavigationAgent without editing it.

`NavigationAgent.__init__` stores its simulator callables as plain instance
attributes (nav/agent.py:77-80), so the observation stream can be tapped by
replacing `agent.get_rgbd` with a wrapper. Nothing in the navigation pipeline is
modified: the wrapper forwards the original return value untouched and only
performs a non-blocking queue put on the side.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from .builder import ASMBuilder
from .config import ASMConfig


def _intrinsic_matrix(agent) -> np.ndarray:
    """Pull a 3x3 K out of whatever `cam_intrinsic` happens to be.

    habitat_agent.py:93 passes an open3d PinholeCameraIntrinsic; the demos pass
    a bare numpy array.
    """
    K = getattr(agent, "cam_intrinsic", None)
    if K is None:
        raise ValueError("agent has no cam_intrinsic")
    if hasattr(K, "intrinsic_matrix"):
        return np.asarray(K.intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    return np.asarray(K, dtype=np.float64).reshape(3, 3)


def _camera_height(agent) -> float:
    """Camera height above the agent's floor.

    habitat_agent.py:52-53 builds cam_to_agent as identity with the height in
    [2, 3]; agent.py:314 uses the same element to derive `nav_level`.
    """
    C_T_R = getattr(agent, "C_T_R", None)
    if C_T_R is None:
        return 0.0
    return float(np.asarray(C_T_R)[2, 3])


def attach(agent, cfg: Optional[ASMConfig] = None) -> Optional[ASMBuilder]:
    """Start an ASM builder fed by `agent`'s observations. Returns the builder.

    Safe to call twice; the second call is a no-op. Any failure here leaves the
    agent exactly as it was.
    """
    if getattr(agent, "_asm_builder", None) is not None:
        return agent._asm_builder

    cfg = cfg or ASMConfig()
    if not cfg.enabled:
        return None

    def _log(msg: str) -> None:
        try:
            agent.log("info", agent.logging_file, msg)
        except Exception:
            print(msg)

    try:
        K = _intrinsic_matrix(agent)
        cam_h = _camera_height(agent)
        out_dir = Path(getattr(agent, "save_dir", ".")) / cfg.out_subdir
        builder = ASMBuilder(cfg, K, out_dir, logger=_log)
    except Exception as exc:
        print(f"ASM: attach failed, continuing without it: {exc!r}")
        return None

    original_get_rgbd = agent.get_rgbd
    original_get_extrinsic = agent.get_cam_extrinsic
    original_close = agent.close

    state = {"last_pose": None}

    def get_rgbd_tee():
        rgb, depth = original_get_rgbd()
        try:
            C2_T_W = original_get_extrinsic()
            W_T_C2 = np.linalg.inv(C2_T_W)

            # get_rgbd is called several times per navigation step (agent.py:327,
            # :972, :1211); skip the duplicates at an unchanged pose.
            last = state["last_pose"]
            if last is None or not np.allclose(last, W_T_C2, atol=1e-6):
                state["last_pose"] = W_T_C2
                builder.submit(
                    rgb=rgb,
                    depth=depth,
                    W_T_C2=W_T_C2,
                    floor_z=float(W_T_C2[2, 3]) - cam_h,
                    step=int(getattr(agent, "navigation_steps", 0)),
                )
        except Exception as exc:
            builder.log(f"ASM: submit failed: {exc!r}")
        return rgb, depth

    def close_with_asm():
        try:
            builder.close(final_step=int(getattr(agent, "navigation_steps", 0)))
            _log(f"ASM: {builder.stats()}")
        except Exception as exc:
            print(f"ASM: close failed: {exc!r}")
        finally:
            agent._asm_builder = None
        return original_close()

    agent.get_rgbd = get_rgbd_tee
    agent.close = close_with_asm
    agent._asm_builder = builder
    _log(f"ASM: attached, writing to {out_dir}")
    return builder


def detach(agent) -> None:
    """Stop the builder if one is attached (the agent keeps the wrappers)."""
    builder = getattr(agent, "_asm_builder", None)
    if builder is not None:
        builder.close(final_step=int(getattr(agent, "navigation_steps", 0)))
        agent._asm_builder = None
