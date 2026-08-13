"""OpenFrontier VLM backend for a locally fine-tuned Qwen3-VL checkpoint.

Same interface as the other vlm/inference_*.py backends:
`predict(image, prompt, system_prompt) -> str`. Served by vlm_server.py on the
"vlm" endpoint (port 12185 by default). Runs in the cosmos3_nav .venv.

IMPORTANT -- repetition penalty is not optional here. The checkpoint this was
written for (`qwen3_sft_..._single_action`) is SFT'd to emit a single navigation
action, and OpenFrontier's prompts ask it for JSON instead. Measured on a real
SoM frame:

    greedy, no penalty            -> '```json\\n{"probability": 0.9, "reason":
                                      "The camera is facing forward. The camera
                                      is facing forward. ...'   never terminates
    greedy, repetition_penalty    -> parses cleanly
    1.15

Without the penalty every response degenerates and `json.loads` fails, so every
VLM call in the episode fails. Set QWEN3_REPETITION_PENALTY=1.0 to disable it
only if you have swapped in a checkpoint that does not need it.
"""
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlm.inference_base import InferenceBase

DEFAULT_MODEL_ID = (
    "/home/ashed/Documents/cosmos3_nav/OpenFrontier_cosmos3_int"
    "/model_weights/qwen3_sft_single_action"
)
MODEL_ID = os.environ.get("QWEN3_MODEL_ID", DEFAULT_MODEL_ID)


class InferenceQwen3VL(InferenceBase):
    def __init__(self, model_name: str = "qwen3-vl-sft-local") -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        print(f"[qwen3vl] loading {MODEL_ID} ...", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": 0})
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Large enough for a JSON block scoring many frontiers at once.
        self.max_new_tokens = int(os.environ.get("QWEN3_MAX_NEW_TOKENS", "512"))
        self.repetition_penalty = float(
            os.environ.get("QWEN3_REPETITION_PENALTY", "1.15"))
        # Greedy by default so episodes are reproducible; the checkpoint's own
        # generation_config asks for sampling, which we deliberately override.
        self.do_sample = os.environ.get("QWEN3_DO_SAMPLE", "0") == "1"
        print(f"[qwen3vl] ready (rep_penalty={self.repetition_penalty}, "
              f"max_new_tokens={self.max_new_tokens}, do_sample={self.do_sample})",
              flush=True)

    def predict(self, image, prompt: str, system_prompt: str = "none") -> str:
        content = []
        if not (isinstance(image, str) and image == "none"):
            img = image if isinstance(image, Image.Image) else \
                Image.fromarray(np.array(image, dtype=np.uint8))
            content.append({"type": "image", "image": img.convert("RGB")})

        text = str(prompt)
        if system_prompt and system_prompt != "none":
            text = f"{system_prompt}\n\n{text}"
        content.append({"type": "text", "text": text})

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True,
            return_tensors="pt").to(self.model.device)

        gen = dict(max_new_tokens=self.max_new_tokens,
                   repetition_penalty=self.repetition_penalty)
        if self.do_sample:
            gen.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        else:
            gen.update(do_sample=False)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        return self.processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
