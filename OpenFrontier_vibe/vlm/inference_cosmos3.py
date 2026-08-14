"""OpenFrontier VLM backend backed by the Cosmos3-Nano reasoner.

Same interface as the other vlm/inference_*.py backends: predict(image, prompt,
system_prompt) -> str. Loads the Cosmos3-Nano image-text-to-text model locally
and is served by vlm_server.py on port 12185 (the "vlm" endpoint OpenFrontier's
local branch talks to). Runs in the cosmos3_nav .venv (transformers>=5.11).
"""
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlm.inference_base import InferenceBase

MODEL_ID = os.environ.get("COSMOS3_MODEL_ID", "/home/ashed/Documents/Cosmos3-Nano")


class InferenceCosmos3(InferenceBase):
    def __init__(self, model_name="cosmos3-nano") -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"[cosmos3] loading {MODEL_ID} ...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": 0})
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        # Needs to be large enough for a JSON block scoring many frontiers at once,
        # otherwise the response truncates and OpenFrontier can't parse it.
        self.max_new_tokens = int(os.environ.get("COSMOS3_MAX_NEW_TOKENS", "640"))
        print("[cosmos3] ready", flush=True)

    def predict(self, image, prompt: str, system_prompt: str = "none") -> str:
        content = []
        if not (isinstance(image, str) and image == "none"):
            img = image if isinstance(image, Image.Image) else \
                Image.fromarray(np.array(image, dtype=np.uint8))
            content.append({"type": "image", "image": img.convert("RGB")})
        # Cosmos3's chat template is single-turn user; fold any system steer into the text.
        text = str(prompt)
        if system_prompt and system_prompt != "none":
            text = f"{system_prompt}\n\n{text}"
        content.append({"type": "text", "text": text})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=self.max_new_tokens)
        return self.processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
