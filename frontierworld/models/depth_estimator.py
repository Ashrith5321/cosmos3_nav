"""Monocular metric-depth estimation for Phase 9.

The converter needs metric depth. Generated video carries no scale of its own,
so this is where scale enters the pipeline, and every downstream area number
inherits whatever error it makes.

Three calibration modes, deliberately separated because they are not equally
deployable:

    none         raw metric output from the estimator
    first_frame  one scale factor per rollout, fitted on the conditioning frame
                 against the real depth sensor. DEPLOYABLE: the conditioning
                 frame is already observed, so nothing about the future is used.
    oracle       per-video scale fitted against ground-truth depth for the
                 whole rollout. DIAGNOSTIC ONLY -- an upper bound, never a
                 system component.

Reporting all three separates "the estimator cannot see shape" from "the
estimator cannot fix scale", which need different remedies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


@dataclass
class DepthMetrics:
    """Standard monocular-depth metrics, plus the scale behaviour that matters
    for mapping."""

    abs_rel: float = float("nan")
    rmse: float = float("nan")
    delta1: float = float("nan")  # fraction with max(d/d*, d*/d) < 1.25
    median_scale_ratio: float = float("nan")  # gt / predicted
    scale_drift: float = float("nan")  # sd of per-frame scale across a rollout
    n_frames: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class MetricDepthEstimator:
    """Wraps a HuggingFace metric-depth model.

    Kept behind the same injectable interface the converter expects, so the
    converter never learns which estimator produced its input.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_depth: float = 10.0,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.max_depth = float(max_depth)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(
            self.device
        ).eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.predict(image)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) uint8 -> metric depth (H, W) in metres."""
        return self.predict_batch(image[None])[0]

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        import torch

        images = np.asarray(images)
        height, width = images.shape[1:3]
        inputs = self.processor(
            images=[img[..., :3].astype(np.uint8) for img in images],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            predicted = self.model(**inputs).predicted_depth

        if predicted.ndim == 3:
            predicted = predicted.unsqueeze(1)
        resized = torch.nn.functional.interpolate(
            predicted, size=(height, width), mode="bicubic", align_corners=False
        )
        return resized.squeeze(1).float().cpu().numpy()


def depth_metrics(
    predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None
) -> DepthMetrics:
    """Standard depth metrics over pixels with usable ground truth."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = (truth > 1e-3) & np.isfinite(truth) & np.isfinite(predicted) & (predicted > 1e-6)
    if valid is not None:
        mask &= valid
    if not mask.any():
        return DepthMetrics(n_frames=int(predicted.shape[0]) if predicted.ndim == 3 else 1)

    p, t = predicted[mask], truth[mask]
    ratio = np.maximum(p / t, t / p)

    # Per-frame scale, so drift across a rollout is visible. A drifting scale
    # is worse for mapping than a constant offset: it warps geometry over time
    # instead of translating it.
    per_frame = []
    if predicted.ndim == 3:
        for index in range(predicted.shape[0]):
            frame_mask = mask[index]
            if frame_mask.sum() > 100:
                per_frame.append(
                    float(np.median(truth[index][frame_mask] / predicted[index][frame_mask]))
                )

    return DepthMetrics(
        abs_rel=float(np.mean(np.abs(p - t) / t)),
        rmse=float(np.sqrt(np.mean((p - t) ** 2))),
        delta1=float(np.mean(ratio < 1.25)),
        median_scale_ratio=float(np.median(t / p)),
        scale_drift=float(np.std(per_frame)) if len(per_frame) > 1 else 0.0,
        n_frames=len(per_frame) if per_frame else 1,
        extra={"per_frame_scale": per_frame},
    )


def first_frame_scale(
    predicted_conditioning: np.ndarray,
    sensor_conditioning: np.ndarray,
    min_valid: int = 500,
) -> float:
    """One scale factor per rollout, from the conditioning frame only.

        c_i = median( D_sensor(p) / D_hat(p) )  over valid pixels p

    Deployable: the conditioning frame is already observed by the real depth
    sensor before the rollout begins, so this uses nothing about the future.
    Median rather than mean because monocular depth error has heavy tails and a
    few bad pixels would drag a mean badly.
    """
    predicted = np.asarray(predicted_conditioning, dtype=np.float64)
    sensor = np.asarray(sensor_conditioning, dtype=np.float64)
    mask = (
        (sensor > 1e-3)
        & (predicted > 1e-6)
        & np.isfinite(sensor)
        & np.isfinite(predicted)
    )
    if mask.sum() < min_valid:
        return 1.0
    return float(np.median(sensor[mask] / predicted[mask]))


def oracle_scale(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Best single scale for a whole rollout. DIAGNOSTIC ONLY -- uses the
    future, so it is an upper bound and never a deployable component."""
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mask = (truth > 1e-3) & (predicted > 1e-6) & np.isfinite(truth) & np.isfinite(predicted)
    if not mask.any():
        return 1.0
    return float(np.median(truth[mask] / predicted[mask]))


def apply_scale(depths: np.ndarray, scale: float, max_depth: float) -> np.ndarray:
    return np.clip(np.asarray(depths, dtype=np.float32) * float(scale), 0.0, max_depth)
