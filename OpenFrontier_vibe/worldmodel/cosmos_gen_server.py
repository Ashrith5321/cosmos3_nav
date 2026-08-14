"""
Cosmos3 generative rollout server (design section 42, Phase 3).

Standalone stdlib-HTTP server that must run under the cosmos_framework venv
(NOT the benchmark env):

    CUDA_VISIBLE_DEVICES=1 \
    /home/ashed/Documents/cosmos3_nav/cosmos/packages/cosmos3/.venv/bin/python \
        worldmodel/cosmos_gen_server.py --port 12186

Protocol (JSON over POST /generate):
    request:  {"image": <b64 png>, "prompt": str, "seeds": [int, ...],
               "num_steps": int, "resolution": str}
    response: {"result": "success", "images": [<b64 png>, ...],
               "elapsed_s": float}
GET /health -> {"status": "ok"}

Only stdlib + PIL + numpy + torch are used so the cosmos venv needs no
extra installs (no flask/cv2).
"""

import argparse
import base64
import io
import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def default_checkpoint() -> str:
    env = os.environ.get("COSMOS3_CHECKPOINT")
    if env:
        return env
    snapshots = Path.home() / ".cache/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots"
    if snapshots.is_dir():
        candidates = sorted(p for p in snapshots.iterdir() if (p / "model_index.json").exists())
        if candidates:
            return str(candidates[-1])
    return str(Path.home() / "Documents/Cosmos3-Nano")


class CosmosGenerator:
    """One resident OmniInference pipeline, image2image (edit) mode."""

    def __init__(self, checkpoint: str, out_dir: str):
        from cosmos_framework.inference.common.init import init_output_dir, init_script

        init_script()
        from cosmos_framework.inference.args import (
            OmniSampleOverrides,
            OmniSetupOverrides,
        )
        from cosmos_framework.inference.inference import OmniInference

        self.OmniSampleOverrides = OmniSampleOverrides
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        setup = OmniSetupOverrides(
            checkpoint_path=checkpoint,
            output_dir=self.out_dir,
            parallelism_preset="latency",
            use_torch_compile=False,
            use_cuda_graphs=False,
        ).build_setup()
        init_output_dir(setup.output_dir)
        self.pipe = OmniInference.create(setup)
        self._counter = 0

    def generate(
        self,
        image_png: bytes,
        prompt: str,
        seeds,
        num_steps: int = 12,
        resolution: str = "480",
    ):
        import torch
        from PIL import Image

        self._counter += 1
        cond_path = self.out_dir / f"cond_{self._counter}.png"
        cond_path.write_bytes(image_png)

        out_images = []
        for seed in seeds:
            name = f"gen_{self._counter}_{seed}"
            overrides = self.OmniSampleOverrides(
                name=name,
                model_mode="image2image",
                prompt=prompt,
                vision_path=str(cond_path),
                seed=int(seed),
                num_steps=int(num_steps),
                resolution=str(resolution),
            )
            overrides.output_dir = self.out_dir / name
            sample = overrides.build_sample(model_config=self.pipe.model_config)
            outputs = self.pipe.generate([sample])
            torch.cuda.empty_cache()
            if not outputs or outputs[0].status != "success":
                continue
            # the framework reports the produced files on the output object
            files = [str(f) for f in outputs[0].outputs[0].files]
            images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if not images:
                # fall back to scanning the sample's output dir
                gen_dir = self.out_dir / name
                images = [
                    str(p)
                    for p in sorted(gen_dir.rglob("*.png"))
                    + sorted(gen_dir.rglob("*.jpg"))
                ]
            if not images:
                continue
            img = Image.open(images[-1]).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out_images.append(base64.b64encode(buf.getvalue()).decode())
        cond_path.unlink(missing_ok=True)
        return out_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("OF_COSMOS_GEN_PORT", 12186)))
    parser.add_argument("--checkpoint", type=str, default=default_checkpoint())
    parser.add_argument("--out-dir", type=str, default=os.path.join(tempfile.gettempdir(), "cosmos_gen_server"))
    args = parser.parse_args()

    print(f"loading Cosmos3 generator from {args.checkpoint} ...", flush=True)
    generator = CosmosGenerator(args.checkpoint, args.out_dir)
    print("generator ready", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"result": "error", "message": "unknown path"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"result": "error", "message": "unknown path"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                images = generator.generate(
                    image_png=base64.b64decode(req["image"]),
                    prompt=req["prompt"],
                    seeds=req.get("seeds", [0]),
                    num_steps=int(req.get("num_steps", 12)),
                    resolution=str(req.get("resolution", "480")),
                )
                self._send(
                    200,
                    {
                        "result": "success",
                        "images": images,
                        "elapsed_s": round(time.perf_counter() - started, 2),
                    },
                )
            except Exception as e:  # noqa: BLE001
                self._send(500, {"result": "error", "message": str(e)})

        def log_message(self, fmt, *a):  # quiet
            pass

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"cosmos generative server listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
