"""
Closed-loop prediction verification and episodic calibration
(design sections 17, 18, 30).

After the robot crosses (or sees past) a frontier, the predicted future
embedding is compared with what was actually observed. The residual
stream drives a lightweight episodic calibrator that shrinks
world-model scores toward the uninformative prior when predictions of a
given room type keep missing - no online model training required.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# residual = 1 - cos(z_pred, z_obs); for unit vectors this lives in [0, 2].
# A residual around this value carries no evidence either way.
NEUTRAL_RESIDUAL = 0.7


class EpisodicCalibrator:
    def __init__(self, params: Optional[dict] = None):
        p = params or {}
        self.enabled = bool(p.get("enabled", True))
        self.error_gain = float(p.get("error_gain", 2.0))
        self.ema_alpha = float(p.get("ema_alpha", 0.3))
        self.min_updates = int(p.get("min_updates", 2))
        self.uncertainty_bonus = float(p.get("uncertainty_bonus", 0.05))

        self._global_err: Optional[float] = None
        self._global_n = 0
        self._room_err: Dict[str, float] = {}
        self._room_n: Dict[str, int] = {}

    # ---------------- residual intake ----------------

    @staticmethod
    def residual(z_pred: np.ndarray, z_obs: np.ndarray) -> float:
        z_pred = z_pred / max(np.linalg.norm(z_pred), 1e-8)
        z_obs = z_obs / max(np.linalg.norm(z_obs), 1e-8)
        return float(1.0 - z_pred @ z_obs)

    def record(self, residual: float, room_top: Optional[str] = None) -> None:
        if not self.enabled:
            return
        err = max(residual - NEUTRAL_RESIDUAL, 0.0)  # only worse-than-neutral counts
        if self._global_err is None:
            self._global_err = err
        else:
            self._global_err = (
                1 - self.ema_alpha
            ) * self._global_err + self.ema_alpha * err
        self._global_n += 1
        if room_top is not None:
            prev = self._room_err.get(room_top)
            self._room_err[room_top] = (
                err if prev is None else (1 - self.ema_alpha) * prev + self.ema_alpha * err
            )
            self._room_n[room_top] = self._room_n.get(room_top, 0) + 1

    # ---------------- calibration ----------------

    def reliability(self, room_top: Optional[str] = None) -> float:
        """1.0 = fully trust the world model; approaches 0 as errors mount."""
        if not self.enabled:
            return 1.0
        err = None
        if (
            room_top is not None
            and self._room_n.get(room_top, 0) >= self.min_updates
        ):
            err = self._room_err[room_top]
        elif self._global_n >= self.min_updates:
            err = self._global_err
        if err is None:
            return 1.0
        return float(np.exp(-self.error_gain * err))

    def calibrate(
        self, mu: float, sigma: float, room_top: Optional[str] = None
    ) -> Tuple[float, float]:
        """Shrink mu toward the uninformative 0.5 and inflate sigma when the
        model has been unreliable for this frontier type (design section 18)."""
        if not self.enabled:
            return mu, sigma
        r = self.reliability(room_top)
        mu_c = 0.5 + (mu - 0.5) * r
        sigma_c = sigma + (1.0 - r) * self.uncertainty_bonus
        return float(mu_c), float(sigma_c)

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "global_error_ema": self._global_err,
            "global_updates": self._global_n,
            "room_error_ema": dict(self._room_err),
            "room_updates": dict(self._room_n),
        }
