"""Drop-in replacement for OpenFrontier's vlm_server.py, serving Cosmos3-Nano.

Binds the same route (/vlm) and port (12185) that OpenFrontier's pipeline
calls, so no changes inside the OpenFrontier repo are needed — just run this
instead of `python vlm_server.py --model ...`:

    cd ~/Documents/cosmos3_nav
    .venv/bin/python src/cosmos3_vlm_server.py
"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
OPENFRONTIER = Path(__file__).resolve().parents[1] / "OpenFrontier"
sys.path.insert(0, str(OPENFRONTIER))

from utils.server_wrapper import ServerMixin, host_agent  # OpenFrontier's Flask wrapper
from cosmos3_vlm import InferenceCosmos3


class Cosmos3VLMServer(ServerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.model = InferenceCosmos3()
        self.cleanup_count = 0

    def process_payload(self, payload: dict) -> dict:
        self.cleanup_count += 1
        if self.cleanup_count % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        prompt = payload["prompt"]
        try:
            image = Image.fromarray(np.array(payload["image"], dtype=np.uint8))
        except Exception:
            image = "none"
        system_prompt = payload.get("system_prompt", "none")

        response = self.model.predict(image, prompt, system_prompt)
        return {"result": "success", "response": response}


if __name__ == "__main__":
    server = Cosmos3VLMServer()
    print("cosmos3-nano loaded!")
    host_agent(server, name="vlm", port=12185)
