"""Minimal HTTP action server holding the Cosmos3-Nano reasoner.

POST /act  {"goal": str, "images": [b64-jpeg, ...], "past_actions": [str, ...]}
        -> {"action": "stop|forward|left|right", "text": str}
GET  /health -> {"status": "ok", "vram_gb": float}

Run in the cosmos3_nav venv:  .venv/bin/python eval/cosmos3_server.py
"""
import base64
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "/home/ashed/Documents/Cosmos3-Nano"
PORT = 8399
ACTIONS = ["stop", "forward", "left", "right"]
FRAME_SIZE = (int(os.environ.get("COSMOS3_FRAME_W", 384)),
              int(os.environ.get("COSMOS3_FRAME_H", 288)))  # keep visual tokens low for speed

PROMPT = (
    'You are a robot navigating an indoor home to find a: {goal}.\n'
    "You see your {n} most recent first-person views, oldest to newest "
    "(the last image is your CURRENT view). Your recent actions: {past}.\n"
    "Actions: forward (move 0.25m), left / right (rotate 30 degrees), "
    "stop (ONLY if the {goal} is clearly visible and within 1 meter, close enough to touch).\n"
    "If the view did not change after a forward action, you collided - turn instead.\n"
    "Briefly reason in 1-2 sentences, then end with exactly: ACTION: <forward|left|right|stop>"
)

print(f"Loading {MODEL_ID} ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Model ready.", flush=True)


def pick_action(goal, images, past_actions):
    frames = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB").resize(FRAME_SIZE)
              for b in images]
    past = ", ".join(past_actions[-10:]) if past_actions else "none yet"
    content = [{"type": "image", "image": f} for f in frames]
    content.append({"type": "text",
                    "text": PROMPT.format(goal=goal, n=len(frames), past=past)})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=80)
    text = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    m = re.findall(r"ACTION:\s*(forward|left|right|stop)", text, re.IGNORECASE)
    if m:
        return m[-1].lower(), text
    for a in ACTIONS:  # fallback: last action word mentioned
        if re.search(rf"\b{a}\b", text, re.IGNORECASE):
            return a, text
    return "forward", text  # unparsable: keep exploring


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence per-request access logs

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
        if self.path != "/act":
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            action, text = pick_action(req["goal"], req["images"], req.get("past_actions", []))
            self._send(200, {"action": action, "text": text})
        except Exception as e:
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"Serving on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
