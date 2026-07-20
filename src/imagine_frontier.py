"""World-model frontier imagination with the Cosmos3-Nano generator tower.

The generator tower of the *same* omni model the reasoner runs on does
image->video with 9D camera-motion conditioning -- exactly the "imagine walking
through that doorway" primitive from md_files/imagine_frontier.md (section 2).

Two conditioning paths are exposed:

  * ``imagine_i2v(image, prompt, ...)`` -- plain image-to-video. Condition on a
    stored keyframe and steer with text ("the camera moves forward through the
    doorway into the next room"). Simplest; enough to eyeball usability (build
    order step 2).
  * ``imagine_camera(image, trajectory, ...)`` -- action-conditioned
    forward_dynamics with ``domain_name="camera_pose"``. The 9D action is a
    ``3D translation + 6D rotation`` pose (Zhou et al. 2019 over-parameterized
    rotation), per the diffusers pipeline. ``forward_trajectory()`` builds a
    straight-ahead walk with no rotation.

VRAM: the diffusers path loads the FULL checkpoint (both experts + VAE +
tokenizers ~33 GB), which does NOT fit a 32 GB card. We therefore load onto CPU
and use ``enable_model_cpu_offload()`` so only the active component sits on the
GPU during its forward. This is the "generator as a second on-demand server"
plan -- run it while the reasoner server is stopped (they cannot co-reside).

Usable standalone as a probe (see ``__main__`` / src/imagine_probe.py):
    .venv/bin/python src/imagine_frontier.py --frames 17 --steps 10 --size 256x320
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

COSMOS3_PATH = "/home/ashed/Documents/Cosmos3-Nano"
CAMERA_DOMAIN = "camera_pose"      # 9D: 3D translation + 6D rotation
CAMERA_ACTION_DIM = 9

# Identity rotation in the Zhou-2019 6D representation = first two columns of I.
_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


@dataclass
class Imagination:
    """One imagined rollout beyond a frontier."""
    frames: list          # list[PIL.Image] decoded video frames
    seconds: float        # wall-clock generation time
    peak_vram_gb: float   # torch peak allocated during generation
    meta: dict            # generation settings used


def forward_trajectory(n_steps: int, step_m: float = 0.25,
                       axis: int = 2, sign: float = -1.0) -> np.ndarray:
    """Straight-ahead camera walk as a [n_steps, 9] raw_actions array.

    Each row is a 9D pose = (tx, ty, tz, r6d...). Translation advances by
    ``step_m`` per step along ``axis`` (default z, camera-forward = -z in the
    habitat/OpenGL convention used across this repo), rotation held at identity.

    NOTE: the exact translation sign/scale/frame the model was trained on is not
    documented in the checkpoint; treat this as the starting guess to calibrate
    against decoded output in the probe, not ground truth.
    """
    traj = np.tile(np.concatenate([[0.0, 0.0, 0.0], _IDENTITY_ROT6D]),
                   (n_steps, 1)).astype(np.float32)
    traj[:, axis] = sign * step_m * np.arange(1, n_steps + 1)
    return traj


class FrontierImaginer:
    """Loads the Cosmos3 generator (diffusers) and imagines frontier rollouts."""

    def __init__(self, model_path: str = COSMOS3_PATH,
                 offload: str = "model", flow_shift: float = 10.0,
                 vae_tiling: bool = True, group_blocks: int = 4) -> None:
        """
        offload:
          "model"      -- enable_model_cpu_offload; transformer resident on GPU
                          during its forward. Fastest, but needs ~29 GB free.
          "group"      -- block-level group offload with a CUDA stream that
                          overlaps CPU->GPU transfer with compute. Low VRAM
                          (~group_blocks * 0.8 GB) and much faster than
                          sequential. Best on a shared GPU.
          "sequential" -- lowest VRAM (~1 GB), slowest (streams every submodule).
          "none"       -- all on GPU (only if the card can hold ~33 GB).
        group_blocks: transformer blocks onloaded together in "group" mode
                      (higher = faster + more VRAM).
        """
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        t0 = time.time()
        # enable_safety_checker=False skips constructing CosmosSafetyChecker at
        # load (it hard-requires the cosmos_guardrail package); we don't need the
        # guardrail for local sim rollouts. This is a from_pretrained/__init__
        # kwarg -- distinct from the per-call `enable_safety_check` below.
        self.pipe = Cosmos3OmniPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, enable_safety_checker=False)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=flow_shift)

        if offload == "model":
            self.pipe.enable_model_cpu_offload()
        elif offload == "group":
            self.pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=group_blocks,
                use_stream=True,       # overlap transfer with compute
                record_stream=True,
            )
        elif offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        elif offload == "none":
            self.pipe.to("cuda")
        else:
            raise ValueError(f"unknown offload mode {offload!r}")

        if vae_tiling and hasattr(self.pipe, "vae"):
            try:
                self.pipe.vae.enable_tiling()
            except Exception:
                pass
        self.load_seconds = round(time.time() - t0, 1)
        self.offload = offload

    # ---------- conditioning paths ----------
    def imagine_i2v(self, image, prompt: str, num_frames: int = 17,
                    height: int = 256, width: int = 320, steps: int = 15,
                    guidance_scale: float = 6.0, negative_prompt: str = "",
                    seed: int = 1234) -> Imagination:
        """Image-to-video: extend a keyframe under a text steer."""
        image = _to_pil(image)
        return self._run(dict(
            prompt=prompt, negative_prompt=negative_prompt, image=image,
            num_frames=num_frames, height=height, width=width, steps=steps,
            guidance_scale=guidance_scale, seed=seed, mode="i2v"))

    def imagine_i2v_batch(self, images, prompts, num_frames: int = 17,
                          height: int = 256, width: int = 320, steps: int = 15,
                          guidance_scale: float = 6.0, seed: int = 1234) -> list:
        """Imagine several frontiers in ONE diffusion call (batch dimension).

        The model weights stream once for the whole batch, so N frontiers cost
        roughly the wall-time of one -- the right way to "parallelize" on a
        single GPU (running N separate pipelines would need N copies of the
        29 GB model in CPU RAM). Returns a list of Imagination, one per input.
        """
        pil = [_to_pil(im) for im in images]
        prompts = list(prompts)
        assert len(pil) == len(prompts), "images and prompts must be same length"
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator(device="cuda").manual_seed(seed)
        t0 = time.time()
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompts, image=pil, num_frames=num_frames,
                height=height, width=width, num_inference_steps=steps,
                guidance_scale=guidance_scale, enable_sound=False, generator=gen,
                enable_safety_check=False, add_resolution_template=False,
                add_duration_template=False)
        secs = round(time.time() - t0, 1)
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        vids = result.video if _is_batched(result.video) else [result.video]
        return [Imagination(frames=list(v), seconds=secs, peak_vram_gb=peak,
                            meta={"batch": len(pil), "mode": "i2v_batch"}) for v in vids]

    def imagine_camera(self, image, trajectory: np.ndarray, prompt: str = "",
                       num_frames: int | None = None, steps: int = 15,
                       guidance_scale: float = 6.0, resolution_tier: int = 256,
                       seed: int = 1234) -> Imagination:
        """Action-conditioned forward_dynamics along a 9D camera trajectory."""
        from diffusers import CosmosActionCondition

        image = _to_pil(image)
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] != CAMERA_ACTION_DIM:
            raise ValueError(f"trajectory must be [T, {CAMERA_ACTION_DIM}], got {traj.shape}")
        chunk_size = traj.shape[0]
        action = CosmosActionCondition(
            mode="forward_dynamics", chunk_size=chunk_size,
            domain_name=CAMERA_DOMAIN, resolution_tier=resolution_tier,
            raw_actions=torch.from_numpy(traj), image=image, view_point="ego_view")
        return self._run(dict(
            prompt=prompt, action=action, num_frames=num_frames, steps=steps,
            guidance_scale=guidance_scale, seed=seed, mode="camera",
            trajectory_len=chunk_size))

    # ---------- core ----------
    def _run(self, cfg: dict) -> Imagination:
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
        kwargs = dict(
            prompt=cfg["prompt"], num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance_scale"], enable_sound=False,
            generator=gen, enable_safety_check=False,
            add_resolution_template=False, add_duration_template=False)
        if cfg.get("num_frames") is not None:
            kwargs["num_frames"] = cfg["num_frames"]
        if cfg["mode"] == "i2v":
            kwargs.update(image=cfg["image"], height=cfg["height"],
                          width=cfg["width"], negative_prompt=cfg["negative_prompt"])
        else:  # camera / action
            kwargs["action"] = cfg["action"]

        t0 = time.time()
        with torch.inference_mode():
            result = self.pipe(**kwargs)
        secs = round(time.time() - t0, 1)
        peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        frames = list(result.video[0]) if _is_batched(result.video) else list(result.video)
        return Imagination(frames=frames, seconds=secs, peak_vram_gb=peak, meta=cfg)


# ---------- helpers ----------
def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")


def _is_batched(video) -> bool:
    # diffusers returns list[frames] or list[list[frames]] for batch>1
    return len(video) > 0 and isinstance(video[0], list)


def save_video(frames, path: str, fps: int = 8) -> None:
    from diffusers.utils import export_to_video
    export_to_video(frames, path, fps=fps, macro_block_size=1)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cosmos3 frontier-imagination probe")
    p.add_argument("--image", default=f"{COSMOS3_PATH}/assets/example_i2v_input.jpg")
    p.add_argument("--mode", choices=["i2v", "camera"], default="i2v")
    p.add_argument("--prompt", default="The camera moves forward through the "
                   "doorway into the next room, revealing more of the home.")
    p.add_argument("--frames", type=int, default=17, help="num_frames (~1 mod 4)")
    p.add_argument("--steps", type=int, default=10, help="denoising steps")
    p.add_argument("--size", default="256x320", help="HxW for i2v")
    p.add_argument("--offload", default="model",
                   choices=["model", "group", "sequential", "none"])
    p.add_argument("--group-blocks", type=int, default=4,
                   help="transformer blocks per onload group (offload=group)")
    p.add_argument("--out", default="/tmp/claude-114414985/imagine_probe.mp4")
    args = p.parse_args()

    h, w = (int(x) for x in args.size.lower().split("x"))
    print(f"Loading Cosmos3 generator (offload={args.offload}) ...", flush=True)
    imag = FrontierImaginer(offload=args.offload, group_blocks=args.group_blocks)
    print(f"Pipeline ready in {imag.load_seconds}s. Imagining ...", flush=True)

    if args.mode == "i2v":
        out = imag.imagine_i2v(args.image, args.prompt, num_frames=args.frames,
                               height=h, width=w, steps=args.steps)
    else:
        traj = forward_trajectory(args.frames, step_m=0.25)
        out = imag.imagine_camera(args.image, traj, prompt=args.prompt,
                                  num_frames=args.frames, steps=args.steps)

    save_video(out.frames, args.out)
    print("\n=== PROBE RESULT ===")
    print(f"  frames generated : {len(out.frames)}")
    print(f"  generation time  : {out.seconds}s "
          f"({out.seconds / max(len(out.frames),1):.2f}s/frame)")
    print(f"  peak VRAM        : {out.peak_vram_gb} GB  (budget 32 GB)")
    print(f"  saved            : {args.out}")
    print("  -> eyeball the mp4: are the imaginations geometrically usable?")
