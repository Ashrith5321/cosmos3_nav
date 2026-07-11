"""Offline smoke eval: Cosmos3-Nano reasoner on HM3D-v2 oracle rollout frames.

Feeds sampled RGB observations from the two oracle smoke episodes
(scene 4ok3usBNeis, goals bed / toilet) to the reasoner and asks for a
navigation action. Checks it explores on early frames and recognizes /
stops at the goal on the final frame. The goal/distance text overlay is
masked out so the model cannot read the answer off the frame.

Run:  .venv/bin/python eval/eval_smoke_cosmos3.py
"""
import json
import os
import re
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import av
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "/home/ashed/Documents/Cosmos3-Nano"
HERE = Path(__file__).resolve().parent
VIDEOS = [
    ("bed", HERE / "smoke_videos" / "0000_4ok3usBNeis_0_bed.mp4"),
    ("toilet", HERE / "smoke_videos" / "0001_4ok3usBNeis_1_toilet.mp4"),
]
OBS_BOX = (0, 0, 640, 480)      # left half of the eval video = RGB observation
OVERLAY_BOX = (0, 0, 215, 62)   # "Goal: X / Distance: Y" text overlay to mask
SAMPLE_EVERY = 6                # eval every Nth step (plus the final frame)
ACTIONS = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]

PROMPT = (
    "You are a robot navigating an indoor home to find a: {goal}.\n"
    "This is your current first-person camera view.\n"
    "In 1-2 sentences, describe what you see and whether a {goal} is visible.\n"
    "Then choose exactly ONE action:\n"
    "- MOVE_FORWARD: move ahead 0.25m\n"
    "- TURN_LEFT / TURN_RIGHT: rotate 30 degrees\n"
    "- STOP: ONLY if you are within 1 meter of the {goal} and it is clearly visible\n"
    "End your answer with a final line: ACTION: <action>"
)


def load_obs_frames(path):
    frames = []
    container = av.open(str(path))
    for frame in container.decode(video=0):
        img = frame.to_image().crop(OBS_BOX)
        ImageDraw.Draw(img).rectangle(OVERLAY_BOX, fill=(90, 90, 90))
        frames.append(img)
    return frames


def parse_action(text):
    m = re.findall(r"ACTION:\s*([A-Z_]+)", text)
    if m and m[-1] in ACTIONS:
        return m[-1]
    for a in ACTIONS:  # fallback: last mentioned action
        if a in text:
            return a
    return "UNPARSED"


def main():
    print(f"Loading {MODEL_ID} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    results = []
    for goal, video in VIDEOS:
        frames = load_obs_frames(video)
        steps = list(range(0, len(frames) - 1, SAMPLE_EVERY)) + [len(frames) - 1]
        print(f"\n=== episode goal={goal} | {len(frames)} frames, evaluating steps {steps} ===", flush=True)

        for t in steps:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": frames[t]},
                {"type": "text", "text": PROMPT.format(goal=goal)}]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=160)
            text = processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            action = parse_action(text)
            is_final = (t == len(frames) - 1)
            results.append({"goal": goal, "step": t, "final": is_final,
                            "action": action, "response": text})
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"[{goal} step {t:2d}{' FINAL' if is_final else ''}] "
                  f"action={action} | vram={vram:.1f}GB\n  {text}", flush=True)

    # Mini-score: no STOP before the final frame; STOP (or goal seen) at the final frame
    early = [r for r in results if not r["final"]]
    final = [r for r in results if r["final"]]
    false_stops = sum(r["action"] == "STOP" for r in early)
    final_stops = sum(r["action"] == "STOP" for r in final)
    print("\n" + "=" * 40)
    print(f"early frames: {len(early)} | false STOPs: {false_stops}")
    print(f"final frames: {len(final)} | STOP chosen: {final_stops}/{len(final)}")
    print(f"peak VRAM (torch): {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    out_path = HERE / "smoke_results_cosmos3.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
