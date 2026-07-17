"""HTTP server exposing OpenFrontier's learned frontier detector (FrontierNet).

Runs in the cosmos3_nav .venv (torch + segmentation-models-pytorch); the
habitat-side teleop/eval talks to it over HTTP, same pattern as the VLM server.

POST /frontiernet  {"rgb": HxWx3 list, "depth": HxW list (meters)}
  -> {"ft_region": 320x320 list, "info_gain": 320x320 list,
      "scale": float, "offset": [offx, offy]}
The mask/gain live in the model's scaled+center-cropped space; map a model
pixel (x, y) back to the original image with ((x + offx) / scale, (y + offy) / scale).

Run:  .venv/bin/python src/frontiernet_server.py
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

OPENFRONTIER = Path(__file__).resolve().parents[1] / "OpenFrontier"
sys.path.insert(0, str(OPENFRONTIER))

from utils.server_wrapper import ServerMixin, host_agent
from frontier.model.predict import load_model
from frontier.detector import FrontierDetector

HFOV_DEG = 79.0
W, H = 640, 480
PORT = 12186


def intrinsics():
    f = (W / 2.0) / np.tan(np.deg2rad(HFOV_DEG) / 2.0)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


class FrontierNetServer(ServerMixin):
    def __init__(self) -> None:
        super().__init__()
        model = load_model(str(OPENFRONTIER / "model_weights" / "rgbd_11cls.pth"))
        self.detector = FrontierDetector(model, intrinsics(), use_depth=True)

    def process_payload(self, payload: dict) -> dict:
        if "rgb_jpg" in payload:  # compact mode: b64 JPEG + b64 16-bit PNG (mm)
            rgb = cv2.cvtColor(cv2.imdecode(
                np.frombuffer(base64.b64decode(payload["rgb_jpg"]), np.uint8),
                cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            depth = cv2.imdecode(
                np.frombuffer(base64.b64decode(payload["depth_png"]), np.uint8),
                cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        else:
            rgb = np.array(payload["rgb"], dtype=np.uint8)
            depth = np.array(payload["depth"], dtype=np.float32)
        ft_region, info_gain = self.detector.detect(rgb, depth)

        s = self.detector.scale_factor
        mh, mw = ft_region.shape[:2]
        offx = (rgb.shape[1] * s - mw) / 2.0
        offy = (rgb.shape[0] * s - mh) / 2.0
        return {
            "result": "success",
            "ft_region": np.asarray(ft_region).astype(np.uint8).tolist(),
            "info_gain": np.asarray(info_gain).astype(float).round(4).tolist(),
            "scale": float(s),
            "offset": [float(offx), float(offy)],
        }


if __name__ == "__main__":
    server = FrontierNetServer()
    print("frontiernet loaded!")
    host_agent(server, name="frontiernet", port=PORT)
