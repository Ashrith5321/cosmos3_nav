"""Open-vocabulary object detector server (Grounding DINO) for the planning stack.

Replaces "Cosmos as the detector" with a real detector. Given the goal class and
the current RGB frame, returns the highest-scoring detection box for that class.
The driver lifts the box to a 3D world position via depth+pose and uses it for
goal approach and a verified STOP.

POST /detect {"goal": str, "rgb_jpg": b64}
  -> {"found": bool, "score": float, "box": [x0,y0,x1,y1]}   (box in original px)
GET  /health -> {"status": "ok", "vram_gb": float}

Run (cosmos3_nav venv):  CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/detector_server.py 12188
"""
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("DETECTOR_PORT", 12188))
MODEL_ID = "IDEA-Research/grounding-dino-base"
BOX_THRESH = float(os.environ.get("DET_BOX_THRESH", "0.30"))
TEXT_THRESH = float(os.environ.get("DET_TEXT_THRESH", "0.20"))

# HM3D-v2 goal classes -> open-vocab query phrases (with synonyms)
SYNONYMS = {
    "chair": "a chair",
    "sofa": "a couch. a sofa",
    "couch": "a couch. a sofa",
    "bed": "a bed",
    "plant": "a potted plant. a house plant",
    "potted plant": "a potted plant. a house plant",
    "toilet": "a toilet",
    "tv_monitor": "a tv. a television. a monitor",
    "tv": "a tv. a television. a monitor",
}


def goal_query(goal):
    return (SYNONYMS.get(str(goal).lower(), f"a {goal}") + " .").lower()


print(f"Loading detector {MODEL_ID} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).to("cuda").eval()  # fp32 (deformable attn)
print("Detector ready.", flush=True)


GPU_LOCK = threading.Lock()   # serialize GPU inference; threads only queue on connections


def detect(goal, rgb_jpg):
    img = Image.open(io.BytesIO(base64.b64decode(rgb_jpg))).convert("RGB")
    inputs = processor(images=img, text=goal_query(goal), return_tensors="pt").to("cuda")
    with GPU_LOCK, torch.inference_mode():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=BOX_THRESH, text_threshold=TEXT_THRESH,
        target_sizes=[img.size[::-1]])[0]
    if len(res["boxes"]) == 0:
        return {"found": False}
    i = int(torch.argmax(res["scores"]))
    return {"found": True, "score": float(res["scores"][i]),
            "box": [float(x) for x in res["boxes"][i]]}


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
            self._send(200, {"status": "ok",
                             "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/detect":
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, detect(req["goal"], req["rgb_jpg"]))
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"Serving on 127.0.0.1:{PORT}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.request_queue_size = 128   # accept a burst of concurrent worker connections
    srv.serve_forever()
