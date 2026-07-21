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
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "/home/ashed/Documents/Cosmos3-Nano"
# Port: CLI arg wins (also gives each instance a distinct cmdline for the watchdog),
# else COSMOS3_PORT env, else default. Device is chosen via CUDA_VISIBLE_DEVICES.
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("COSMOS3_PORT", 8399))
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

MAP_SECTION = (
    "\nThe FINAL image is your top-down exploration map of the current floor: "
    "white = explored floor, black = walls/obstacles, gray = UNEXPLORED, "
    "blue dots = your path so far, green arrow = your position and heading, "
    "red regions labeled A/B/C... = unexplored openings (frontiers), "
    "magenta = predicted frontiers.\n"
    "Frontier directions relative to you:\n{frontier_text}\n"
    "Use the map: avoid re-walking your blue path, and head toward frontiers "
    "likely to contain a {goal}."
)

# planning branch: Cosmos only SELECTS a frontier; an external planner drives to it.
FRONTIER_PROMPT = (
    'You are a robot exploring an indoor home to find a: {goal}.\n'
    "You see your {n} most recent first-person views, oldest to newest (last = CURRENT view), "
    "and as the FINAL image a top-down map: white = explored floor, black = walls, "
    "gray = UNEXPLORED, blue = your path, green arrow = you, red regions labeled "
    "A/B/C... = candidate frontiers (openings into unexplored space).\n"
    "Frontier directions relative to you:\n{frontier_text}\n"
    "Choose the SINGLE best frontier to move toward next to find the {goal}: prefer large, "
    "promising openings and unexplored directions; avoid areas you already searched. "
    "Choose STOP only if the {goal} is clearly visible and within ~1 meter.\n"
    "Available choices: {labels}, or STOP.\n"
    "Briefly reason in 1-2 sentences, then end with exactly: CHOICE: <one of {labels} or STOP>"
)

# stop-gate verification: strict single-image yes/no that the goal is right here
VERIFY_PROMPT = (
    "Look carefully at this first-person camera image. Is a {goal} clearly and "
    "unmistakably visible in the image AND very close to the camera (within about 1 "
    "meter, close enough to reach out and touch)?\n"
    "Only answer YES if you are confident a {goal} is right in front of you and nearby. "
    "If it is far away, only partially visible, or you are unsure, answer NO.\n"
    "Answer with exactly one word: YES or NO."
)

GPU_LOCK = threading.Lock()   # serialize GPU generate(); threads only queue on connections

print(f"Loading {MODEL_ID} ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
print("Model ready.", flush=True)


def pick_action(goal, images, past_actions, map_image=None, frontier_text=""):
    frames = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB").resize(FRAME_SIZE)
              for b in images]
    past = ", ".join(past_actions[-10:]) if past_actions else "none yet"
    content = [{"type": "image", "image": f} for f in frames]
    prompt = PROMPT.format(goal=goal, n=len(frames), past=past)
    if map_image is not None:
        m = Image.open(io.BytesIO(base64.b64decode(map_image))).convert("RGB")
        m.thumbnail((384, 384))
        content.append({"type": "image", "image": m})
        prompt += MAP_SECTION.format(frontier_text=frontier_text or "(none detected yet)",
                                     goal=goal)
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with GPU_LOCK, torch.inference_mode():
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


def pick_frontier(goal, images, map_image, frontier_text, labels):
    """Select which frontier to head toward (or STOP). Returns (choice, text)."""
    frames = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB").resize(FRAME_SIZE)
              for b in images]
    content = [{"type": "image", "image": f} for f in frames]
    lbl = ", ".join(labels) if labels else "(none)"
    prompt = FRONTIER_PROMPT.format(goal=goal, n=len(frames),
                                    frontier_text=frontier_text or "(none detected yet)",
                                    labels=lbl)
    if map_image is not None:
        m = Image.open(io.BytesIO(base64.b64decode(map_image))).convert("RGB")
        m.thumbnail((384, 384))
        content.append({"type": "image", "image": m})
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with GPU_LOCK, torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=80)
    text = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    valid = set(labels) | {"STOP"}
    m = re.findall(r"CHOICE:\s*([A-Za-z]+)", text)
    if m and m[-1].upper() in valid:
        return m[-1].upper(), text
    if re.search(r"\bstop\b", text, re.IGNORECASE):
        return "STOP", text
    for lb in labels:  # fallback: last label mentioned in the text
        if re.search(rf"\b{lb}\b", text):
            return lb, text
    return (labels[0] if labels else "STOP"), text  # unparsable: default to first frontier


def verify_goal(goal, image):
    """Stop-gate: strict single-image check that the goal is visible and within reach."""
    img = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB").resize(FRAME_SIZE)
    content = [{"type": "image", "image": img},
               {"type": "text", "text": VERIFY_PROMPT.format(goal=goal)}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with GPU_LOCK, torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=8)
    text = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    verified = bool(re.search(r"\byes\b", text, re.IGNORECASE)) and \
        not bool(re.search(r"\bno\b", text, re.IGNORECASE))
    return verified, text


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
        if self.path not in ("/act", "/select_frontier", "/verify_goal"):
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/select_frontier":
                choice, text = pick_frontier(req["goal"], req["images"], req.get("map_image"),
                                             req.get("frontier_text", ""), req.get("labels", []))
                self._send(200, {"choice": choice, "text": text})
                return
            if self.path == "/verify_goal":
                verified, text = verify_goal(req["goal"], req["image"])
                self._send(200, {"verified": verified, "text": text})
                return
            action, text = pick_action(req["goal"], req["images"], req.get("past_actions", []),
                                       map_image=req.get("map_image"),
                                       frontier_text=req.get("frontier_text", ""))
            self._send(200, {"action": action, "text": text})
        except Exception as e:
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"Serving on 127.0.0.1:{PORT}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.request_queue_size = 128   # accept a burst of concurrent worker connections
    srv.serve_forever()
