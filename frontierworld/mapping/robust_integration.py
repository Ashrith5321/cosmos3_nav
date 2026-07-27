"""A sensor model for PREDICTED depth.

`OccupancyMap` assumes simulator-perfect depth: every ray carves free space up
to its endpoint and marks exactly that endpoint occupied. Under estimated depth
that is catastrophic -- a ray reading long carves a corridor through a wall,
and the errors accumulate rather than cancel because every ray carves
independently.

The goal here is explicitly NOT to recover the ground-truth-depth ceiling. A
robust integrator cannot place a wall the estimator never saw. What it can do
is degrade gracefully: stop claiming free space it cannot justify, and spread
occupied evidence over the interval where the surface plausibly lies.

    free space    carved only to   d_free = max(0, d_hat - k_f * sigma(d_hat))
    occupied      distributed over [d_hat - k_o*sigma, d_hat + k_o*sigma],
                  peaked at d_hat

sigma is not a guessed percentage: it is fitted from real residuals on a
calibration subset (see `fit_error_envelope`), depth-binned, so the uncertainty
band reflects how the estimator actually behaves at each range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from frontierworld.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, MapGeometry, _bresenham


@dataclass
class ErrorEnvelope:
    """Depth-dependent error magnitude, fitted from real residuals.

    Stored as a quantile per depth bin rather than a single percentage: a
    monocular estimator is not uniformly wrong, and an envelope that ignores
    range either over-trusts far readings or throws away near ones.
    """

    bin_edges: np.ndarray
    sigma: np.ndarray  # one value per bin
    quantile: float = 0.9
    n_samples: int = 0

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        index = np.clip(
            np.digitize(depth, self.bin_edges) - 1, 0, len(self.sigma) - 1
        )
        return self.sigma[index]

    def to_dict(self) -> dict:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "sigma": self.sigma.tolist(),
            "quantile": self.quantile,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ErrorEnvelope":
        return cls(
            bin_edges=np.asarray(payload["bin_edges"]),
            sigma=np.asarray(payload["sigma"]),
            quantile=payload.get("quantile", 0.9),
            n_samples=payload.get("n_samples", 0),
        )

    @classmethod
    def constant(cls, sigma: float, max_depth: float = 10.0) -> "ErrorEnvelope":
        return cls(np.array([0.0, max_depth]), np.array([sigma]), 1.0, 0)


def fit_error_envelope(
    predicted: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 8,
    max_depth: float = 5.0,
    quantile: float = 0.9,
) -> ErrorEnvelope:
    """Fit sigma(d_hat) = Q_q(|c*d_hat - d_gt|) per predicted-depth bin.

    Inputs must already be scale-corrected; the envelope describes residual
    error after calibration, which is what the integrator has to tolerate.
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    mask = (
        (truth > 1e-3) & (predicted > 1e-3)
        & np.isfinite(truth) & np.isfinite(predicted)
        & (predicted < max_depth)
    )
    if mask.sum() < 100:
        return ErrorEnvelope.constant(0.5, max_depth)

    predicted, truth = predicted[mask], truth[mask]
    error = np.abs(predicted - truth)
    edges = np.linspace(0.0, max_depth, n_bins + 1)
    sigma = np.zeros(n_bins)
    for index in range(n_bins):
        in_bin = (predicted >= edges[index]) & (predicted < edges[index + 1])
        sigma[index] = (
            float(np.quantile(error[in_bin], quantile)) if in_bin.sum() > 20
            else (sigma[index - 1] if index else 0.5)
        )
    # Monotone non-decreasing: uncertainty should not fall with range, and a
    # sparsely populated far bin otherwise produces a spuriously tight band.
    sigma = np.maximum.accumulate(sigma)
    return ErrorEnvelope(edges, sigma, quantile, int(mask.sum()))


@dataclass
class RobustIntegratorConfig:
    """Frozen after selection on the converter-validation subset."""

    k_free: float = 1.0  # carve only to d_hat - k_free * sigma
    k_occupied: float = 1.0  # spread occupied over +/- k_occupied * sigma
    conservative_free: bool = True
    soft_occupied: bool = True
    confidence_weighting: bool = True
    edge_filter: bool = True
    # Predicted depth deserves less trust per frame than sensor depth, so a
    # single confident-looking frame cannot dominate the map.
    evidence_weight: float = 0.5
    max_log_odds: float = 4.0  # bounded, so no cell becomes unrevisable
    min_depth: float = 0.3
    outlier_gradient: float = 1.5  # metres per pixel; above this is an artefact
    column_stride: int = 2


class RobustOccupancyMap:
    """Occupancy mapping from predicted depth, with an uncertainty band.

    Keeps log-odds rather than counts: evidence has to be weighted by
    confidence, and bounded so accumulating many uncertain frames cannot
    manufacture certainty.
    """

    def __init__(
        self,
        geometry: MapGeometry,
        envelope: ErrorEnvelope,
        config: RobustIntegratorConfig | None = None,
        floor_y: float = 0.0,
        obstacle_band: tuple[float, float] = (0.20, 1.50),
    ) -> None:
        self.geometry = geometry
        self.envelope = envelope
        self.config = config or RobustIntegratorConfig()
        self.floor_y = floor_y
        self.obstacle_band = obstacle_band
        size = geometry.size_cells
        self.log_odds_free = np.zeros((size, size), dtype=np.float32)
        self.log_odds_occupied = np.zeros((size, size), dtype=np.float32)

    # -- preprocessing ---------------------------------------------------

    def _clean(self, depth: np.ndarray, max_depth: float) -> np.ndarray:
        """Reject invalid and extreme depths; optionally edge-preserving filter."""
        depth = np.asarray(depth, dtype=np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < self.config.min_depth] = 0.0
        depth[depth > max_depth] = 0.0

        if self.config.edge_filter:
            try:
                import cv2

                # Bilateral: smooths within surfaces without dragging depth
                # across boundaries, which is exactly where a naive blur would
                # invent geometry between a wall and the space beyond it.
                depth = cv2.bilateralFilter(depth, d=5, sigmaColor=0.15, sigmaSpace=5)
            except Exception:  # noqa: BLE001 - filtering is optional
                pass

        # Drop pixels sitting on a large depth discontinuity: predicted depth
        # is least reliable exactly at boundaries, and those pixels are the
        # ones that carve furthest into real geometry.
        gradient_y, gradient_x = np.gradient(depth)
        steep = np.hypot(gradient_x, gradient_y) > self.config.outlier_gradient
        depth[steep] = 0.0
        return depth

    # -- integration -----------------------------------------------------

    def integrate(
        self,
        depth: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        agent_position: np.ndarray,
        hfov_deg: float,
        max_depth: float,
    ) -> None:
        from frontierworld.mapping.occupancy import OccupancyMap

        config = self.config
        depth = self._clean(depth, max_depth)
        if not np.any(depth > 0):
            return

        helper = OccupancyMap(
            resolution=self.geometry.resolution,
            size_m=self.geometry.size_cells * self.geometry.resolution,
            column_stride=config.column_stride,
        )
        points, valid = helper.unproject(depth, rotation, translation, hfov_deg)
        valid &= depth > 0
        valid &= depth < (max_depth - 1e-3)

        stride = config.column_stride
        points, valid, depth = points[::stride, ::stride], valid[::stride, ::stride], depth[::stride, ::stride]

        heights = points[..., 1] - self.floor_y
        is_obstacle = valid & (heights >= self.obstacle_band[0]) & (heights <= self.obstacle_band[1])
        is_ground = valid & (heights < self.obstacle_band[0])

        origin = np.asarray([agent_position[0], agent_position[2]], dtype=np.float64)
        agent_row, agent_col = self.geometry.world_to_cell(
            np.asarray([origin[0]]), np.asarray([origin[1]])
        )
        agent_cell = (int(agent_row[0]), int(agent_col[0]))
        if not self.geometry.in_bounds(np.asarray([agent_cell[0]]), np.asarray([agent_cell[1]]))[0]:
            return

        ranges = np.linalg.norm(points[..., [0, 2]] - origin, axis=-1)
        weight = config.evidence_weight if config.confidence_weighting else 1.0

        for column in range(points.shape[1]):
            obstacle_rows = np.flatnonzero(is_obstacle[:, column])
            if obstacle_rows.size:
                row = obstacle_rows[np.argmin(ranges[obstacle_rows, column])]
                endpoint_is_surface = True
            else:
                ground_rows = np.flatnonzero(is_ground[:, column])
                if not ground_rows.size:
                    continue
                row = ground_rows[np.argmax(ranges[ground_rows, column])]
                endpoint_is_surface = False

            measured_range = float(ranges[row, column])
            sigma = float(self.envelope(np.asarray([depth[row, column]]))[0])
            direction = (points[row, column][[0, 2]] - origin)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            direction /= norm

            # Free space only to the conservative lower bound.
            free_range = (
                max(0.0, measured_range - config.k_free * sigma)
                if config.conservative_free else measured_range
            )
            self._carve(agent_cell, origin, direction, free_range, weight)

            if endpoint_is_surface:
                self._mark_occupied(
                    origin, direction, measured_range, sigma, weight
                )

        np.clip(self.log_odds_free, 0.0, config.max_log_odds, out=self.log_odds_free)
        np.clip(self.log_odds_occupied, 0.0, config.max_log_odds, out=self.log_odds_occupied)

    def _carve(self, agent_cell, origin, direction, free_range, weight) -> None:
        if free_range <= self.geometry.resolution:
            return
        end = origin + direction * free_range
        end_row, end_col = self.geometry.world_to_cell(
            np.asarray([end[0]]), np.asarray([end[1]])
        )
        cells = _bresenham(agent_cell[0], agent_cell[1], int(end_row[0]), int(end_col[0]))
        if len(cells) <= 1:
            return
        rows = np.asarray([c[0] for c in cells[:-1]])
        cols = np.asarray([c[1] for c in cells[:-1]])
        keep = self.geometry.in_bounds(rows, cols)
        np.add.at(self.log_odds_free, (rows[keep], cols[keep]), weight)

    def _mark_occupied(self, origin, direction, measured_range, sigma, weight) -> None:
        """Spread occupied evidence over the uncertainty band, peaked at d_hat.

        A uniform band would make a thick wall equally likely everywhere it
        could be, which inflates occupied area and destroys precision. The
        triangular kernel keeps the mode where the estimator actually pointed.
        """
        config = self.config
        if not config.soft_occupied or sigma <= 1e-6:
            offsets, weights = np.array([0.0]), np.array([1.0])
        else:
            span = config.k_occupied * sigma
            steps = max(1, int(np.ceil(span / self.geometry.resolution)))
            offsets = np.linspace(-span, span, 2 * steps + 1)
            weights = 1.0 - np.abs(offsets) / (span + 1e-9)  # triangular peak
            weights /= weights.sum()

        for offset, kernel_weight in zip(offsets, weights):
            point = origin + direction * (measured_range + offset)
            row, col = self.geometry.world_to_cell(
                np.asarray([point[0]]), np.asarray([point[1]])
            )
            if self.geometry.in_bounds(row, col)[0]:
                self.log_odds_occupied[int(row[0]), int(col[0])] += weight * kernel_weight

    # -- readout ---------------------------------------------------------

    def to_grid(self, free_threshold: float = 0.5, occupied_threshold: float = 0.5) -> np.ndarray:
        grid = np.full(self.log_odds_free.shape, UNKNOWN, dtype=np.uint8)
        grid[self.log_odds_free >= free_threshold] = FREE
        grid[self.log_odds_occupied >= occupied_threshold] = OCCUPIED
        return grid
