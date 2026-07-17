"""Cosmos3-Nano as an OpenFrontier-compatible VLM inference backend.

Kept in cosmos3_nav/src (not inside OpenFrontier) so the vendored repo stays
clean. See cosmos3_vlm_server.py for the drop-in server.
"""
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

COSMOS3_PATH = "/home/ashed/Documents/Cosmos3-Nano"


class InferenceCosmos3:
    """Matches OpenFrontier's InferenceBase.predict(image, prompt, system_prompt)."""

    def __init__(self, model_path: str = COSMOS3_PATH) -> None:
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": 0},
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def predict(self, image, prompt: str, system_prompt: str = "none") -> str:
        messages = []
        if isinstance(system_prompt, str) and system_prompt not in ("", "none"):
            messages.append({"role": "system",
                             "content": [{"type": "text", "text": system_prompt}]})
        content = []
        if isinstance(image, Image.Image):
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)
        return self.processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
