import gc
import torch
import argparse
import numpy as np
from PIL import Image

from vlm.models import VLMModel
# NB: backend modules are imported lazily inside VLMServer.__init__ so that
# selecting one model (e.g. cosmos3-nano) does not require the others' heavy
# deps (llava / internvl / gemma) to be installed.
from utils.server_wrapper import ServerMixin, host_agent

if __name__ == "__main__":

    class VLMServer(ServerMixin):

        def __init__(self, model: str) -> None:
            super().__init__()

            try:
                model = VLMModel(model)
            except ValueError:
                raise ValueError(f"Unsupported VLM model: {model}")

            if model.value.lower().startswith("gemma"):
                from vlm.inference_gemma3 import InferenceGemma3
                self.model = InferenceGemma3(model.value)

            elif model.value.lower().startswith("intern"):
                from vlm.inference_internvl import InferenceInternVL
                self.model = InferenceInternVL(model)

            elif "llava" in model.value.lower():
                from vlm.inference_llava import InferenceLlava
                self.model = InferenceLlava(model.value)

            elif "cosmos3" in model.value.lower():
                from vlm.inference_cosmos3 import InferenceCosmos3
                self.model = InferenceCosmos3(model.value)
            else:
                raise ValueError(f"Unsupported local VLM model: {model.value}")

            self.cleanup_count = 0

        def process_payload(self, payload: dict) -> dict:
            self.cleanup_count += 1
            if self.cleanup_count % 10 == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()

            # try:
            
            prompt = payload["prompt"]
            
            try:
                image = payload["image"]
                image = np.array(image, dtype=np.uint8)
                image = Image.fromarray(image)
            except:
                image = "none"
            
            try:
                system_prompt = payload["system_prompt"]
            except:
                system_prompt = "none"
                
            response = self.model.predict(image, prompt, system_prompt)
            return {"result": "success", "response": response}

            # except Exception as e:
            # return {"result": "error", "message": str(e)}

    parser = argparse.ArgumentParser(description="VLM Server for OpenFrontier")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="VLM model to use",
    )

    model = parser.parse_args().model

    server = VLMServer(model)
    print(f"{model} loaded!")
    import os
    host_agent(server, name="vlm", port=int(os.environ.get("OF_VLM_PORT", "12185")))
