"""Load Cosmos3-Nano once and record what it costs.

Runs in the isolated Cosmos environment (Python 3.13 / torch 2.10), NEVER in
the Habitat 0.3.3 environment. Nothing here imports Habitat.

The point is to establish, before any generation, whether a single A6000 can
hold the model in BF16, and to record the checkpoint identity so that every
later rollout can be attributed to exactly these weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def system_ram_mb() -> dict:
    import psutil

    virtual = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    return {
        "process_rss_mb": round(process.memory_info().rss / 2**20, 1),
        "system_used_mb": round((virtual.total - virtual.available) / 2**20, 1),
        "system_total_mb": round(virtual.total / 2**20, 1),
    }


def gpu_mb(device: int) -> dict:
    import torch

    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 2**20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 2**20, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
    }


def checkpoint_hash(root: Path, sample_bytes: int = 1 << 20) -> str:
    """Stable identity for a multi-shard checkpoint.

    Hashing 33 GB in full would take minutes on every run, so each file
    contributes its name, size, and first `sample_bytes`. That is enough to
    distinguish checkpoints while staying cheap; it is an identity check, not a
    tamper-proof seal.
    """
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() and not path.exists():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(size).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    import torch

    profile: dict = {
        "checkpoint_path": str(args.checkpoint),
        "requested_device": args.device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype_requested": "bfloat16",
        "gpus": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_mb": round(torch.cuda.get_device_properties(i).total_memory / 2**20, 1),
            }
            for i in range(torch.cuda.device_count())
        ],
    }

    if not args.skip_hash:
        started = time.time()
        profile["checkpoint_hash"] = checkpoint_hash(args.checkpoint)
        profile["checkpoint_hash_seconds"] = round(time.time() - started, 1)

    try:
        profile["nvidia_smi_topo"] = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - topology is informational
        profile["nvidia_smi_topo"] = f"unavailable: {error}"

    profile["ram_before"] = system_ram_mb()
    torch.cuda.reset_peak_memory_stats(args.device)
    profile["gpu_before"] = gpu_mb(args.device)

    # Weights are loaded component-by-component rather than through
    # `Cosmos3OmniPipeline.from_pretrained`, because the pipeline constructor
    # instantiates the guardrail safety checker eagerly and that requires a
    # gated checkpoint. This profile measures weight residency only; it is NOT
    # a generation path and nothing here disables a safety component. Actual
    # generation goes through `cosmos_framework.scripts.inference` with
    # guardrails enabled.
    profile["load_path"] = "per-component (pipeline constructor needs gated guardrail)"

    from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer

    loaders = {
        "transformer": (Cosmos3OmniTransformer, "transformer"),
        "vae": (AutoencoderKLWan, "vae"),
    }

    modules: dict = {}
    load_seconds: dict = {}
    for name, (cls, subfolder) in loaders.items():
        started = time.time()
        modules[name] = cls.from_pretrained(
            str(args.checkpoint), subfolder=subfolder, torch_dtype=torch.bfloat16
        )
        load_seconds[name] = round(time.time() - started, 1)

    profile["load_seconds_cpu"] = load_seconds
    profile["ram_after_from_pretrained"] = system_ram_mb()

    started = time.time()
    for module in modules.values():
        module.to(f"cuda:{args.device}")
    profile["to_device_seconds"] = round(time.time() - started, 1)

    profile["ram_after_to_device"] = system_ram_mb()
    profile["gpu_after"] = gpu_mb(args.device)

    components: dict = {}
    for name, module in modules.items():
        parameters = list(module.parameters())
        if not parameters:
            continue
        components[name] = {
            "n_parameters": sum(p.numel() for p in parameters),
            "dtypes": sorted({str(p.dtype) for p in parameters}),
            "devices": sorted({str(p.device) for p in parameters}),
            "size_mb": round(sum(p.numel() * p.element_size() for p in parameters) / 2**20, 1),
        }
    profile["components"] = components
    profile["total_parameters"] = sum(c["n_parameters"] for c in components.values())
    profile["total_weight_mb"] = round(sum(c["size_mb"] for c in components.values()), 1)
    total_mb = profile["total_weight_mb"]
    device_total = profile["gpus"][args.device]["total_mb"]
    profile["fits_single_gpu"] = bool(profile["gpu_after"]["reserved_mb"] < device_total)
    profile["headroom_mb"] = round(device_total - profile["gpu_after"]["reserved_mb"], 1)
    profile["generation_can_begin"] = False
    profile["generation_blocked_by"] = (
        "nvidia/Cosmos-Guardrail1 is a gated HuggingFace repo and no HF token is "
        "configured. Guardrails are a required safety component and were not disabled."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2))
    print(json.dumps({k: v for k, v in profile.items() if k != "nvidia_smi_topo"}, indent=2))


if __name__ == "__main__":
    main()
