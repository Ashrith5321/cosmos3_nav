"""Async frontier-imagination server backed by the Cosmos3 generator tower.

Loads the world model once and serves imagined rollouts on demand. Teleop / eval
enqueue frontiers (each = a conditioning keyframe + a text steer) as the agent
moves; a single worker thread imagines them one at a time and caches the result,
so the caller never blocks on the ~15-30 s diffusion. Same HTTP pattern as
eval/cosmos3_server.py and src/frontiernet_server.py.

POST /enqueue
    {"frontiers": [{"key": str, "prompt": str, "image_jpg": b64}, ...],
     "steps": int?, "frames": int?, "height": int?, "width": int?}
  -> {"qlen": int, "running": key|None, "done": [key, ...]}
     New keys are queued; keys already done/running/queued are skipped (so the
     same frontier is imagined once even if re-sent every step).
GET  /results -> {key: {"status": "done"|"running"|"error",
                        "last_jpg": b64, "strip_jpg": b64,
                        "frames": int, "seconds": float}}
GET  /health  -> {"status", "vram_gb", "qlen", "running", "done"}

Run (cosmos3_nav venv):
    .venv/bin/python src/generator_server.py --offload model   # fast, needs ~30 GB
    .venv/bin/python src/generator_server.py --offload sequential  # ~1 GB, ~50 s/frontier
"""
import argparse
import base64
import io
import json
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagine_frontier import FrontierImaginer

PORT = 8402
MAX_RESULTS = 16          # LRU cap on cached imaginations
STRIP_N = 5               # frames sampled into the preview strip
STRIP_HW = (102, 128)     # per-frame thumbnail size in the strip (H, W)

IMAGINER = None
QUEUE = deque()
QUEUED = set()            # keys currently queued or running
RESULTS = OrderedDict()   # key -> result dict
RUNNING = {"key": None}
LOCK = threading.Lock()
CFG = {"steps": 16, "frames": 17, "height": 256, "width": 320}


import os  # noqa: E402
SAVE_DIR = os.environ.get("GENERATOR_SAVE_DIR")  # if set, persist mp4 + frames per frontier
SAVE_FPS = int(os.environ.get("GENERATOR_SAVE_FPS", "8"))


def _jpg_b64(pil_img, quality=80):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _safe(key):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(key))


def _save_rollout(key, frames):
    """Persist an imagined rollout as an mp4 + last-frame PNG under SAVE_DIR."""
    if not SAVE_DIR:
        return None
    import cv2
    d = Path(SAVE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    stem = _safe(key)
    arrs = [np.asarray(f.convert("RGB")) for f in frames]
    h, w = arrs[0].shape[:2]
    mp4 = str(d / f"{stem}.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), SAVE_FPS, (w, h))
    for a in arrs:
        vw.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    vw.release()
    frames[-1].convert("RGB").save(str(d / f"{stem}_last.png"))
    return mp4


def _worker():
    """Single consumer: imagine queued frontiers one at a time."""
    while True:
        with LOCK:
            job = QUEUE.popleft() if QUEUE else None
            RUNNING["key"] = job["key"] if job else None
            if job:
                RESULTS[job["key"]] = {"status": "running"}
                RESULTS.move_to_end(job["key"])
        if job is None:
            time.sleep(0.15)
            continue
        try:
            img = Image.open(io.BytesIO(base64.b64decode(job["image_jpg"]))).convert("RGB")
            out = IMAGINER.imagine_i2v(
                img, job["prompt"], num_frames=CFG["frames"],
                height=CFG["height"], width=CFG["width"], steps=CFG["steps"])
            last = out.frames[-1]
            idx = np.linspace(0, len(out.frames) - 1, STRIP_N).astype(int)
            strip = np.concatenate(
                [np.asarray(out.frames[i].resize((STRIP_HW[1], STRIP_HW[0]))) for i in idx],
                axis=1)
            mp4 = _save_rollout(job["key"], out.frames)
            res = {"status": "done", "last_jpg": _jpg_b64(last),
                   "strip_jpg": _jpg_b64(Image.fromarray(strip)),
                   "frames": len(out.frames), "seconds": out.seconds,
                   "mp4": mp4}
        except Exception as e:  # noqa: BLE001
            res = {"status": "error", "error": str(e)[:200]}
            torch.cuda.empty_cache()
        with LOCK:
            RESULTS[job["key"]] = res
            RESULTS.move_to_end(job["key"])
            QUEUED.discard(job["key"])
            RUNNING["key"] = None
            while len(RESULTS) > MAX_RESULTS:
                old, _ = RESULTS.popitem(last=False)
                QUEUED.discard(old)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            with LOCK:
                self._send(200, {"status": "ok",
                                 "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                                 "qlen": len(QUEUE), "running": RUNNING["key"],
                                 "done": [k for k, v in RESULTS.items()
                                          if v.get("status") == "done"]})
        elif self.path == "/results":
            with LOCK:
                self._send(200, dict(RESULTS))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/enqueue":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            for k in ("steps", "frames", "height", "width"):
                if body.get(k):
                    CFG[k] = int(body[k])
            with LOCK:
                for fr in body.get("frontiers", []):
                    key = fr["key"]
                    if key in QUEUED:
                        continue
                    if RESULTS.get(key, {}).get("status") in ("done", "running"):
                        continue
                    QUEUE.append({"key": key, "prompt": fr.get("prompt", ""),
                                  "image_jpg": fr["image_jpg"]})
                    QUEUED.add(key)
                self._send(200, {"qlen": len(QUEUE), "running": RUNNING["key"],
                                 "done": [k for k, v in RESULTS.items()
                                          if v.get("status") == "done"]})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})


def main():
    global IMAGINER
    p = argparse.ArgumentParser()
    p.add_argument("--offload", default="model",
                   choices=["model", "group", "sequential", "none"],
                   help="model=fast/~30GB (needs card mostly free); "
                        "sequential=~1GB/~50s (co-resides with habitat)")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--size", default="256x320", help="HxW")
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()
    CFG["steps"], CFG["frames"] = args.steps, args.frames
    CFG["height"], CFG["width"] = (int(x) for x in args.size.lower().split("x"))

    print(f"Loading generator (offload={args.offload}) ...", flush=True)
    IMAGINER = FrontierImaginer(offload=args.offload)
    print(f"Generator ready in {IMAGINER.load_seconds}s. Config={CFG}", flush=True)

    threading.Thread(target=_worker, daemon=True).start()
    print(f"Serving on 127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
