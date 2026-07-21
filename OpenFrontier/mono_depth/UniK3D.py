from unik3d.models import UniK3D
from unik3d.utils.camera import OPENCV
import numpy as np
import torch

_MODEL = None  # load the ViT-L backbone once and reuse across steps


def _get_model():
    global _MODEL
    if _MODEL is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device).eval()
    return _MODEL


def metric_depth_from_rgb(rgb_input, intrinsic_mat):
    model = _get_model()
    device = next(model.parameters()).device
    rgb = (
        torch.from_numpy(np.array(rgb_input).astype(np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        if intrinsic_mat is None:
            predictions = model.infer(rgb)
        else:
            cam_params = [
                float(intrinsic_mat[0, 0]),  # fx
                float(intrinsic_mat[1, 1]),  # fy
                float(intrinsic_mat[0, 2]),  # cx
                float(intrinsic_mat[1, 2]),  # cy
            ] + [0.0] * 12
            camera = OPENCV(params=torch.tensor(cam_params).to(device))
            predictions = model.infer(rgb, camera)

    depth = predictions["depth"]
    return depth.squeeze().cpu().numpy()
