"""Non-learning baselines for revelation prediction.

These are the references Phase 8's gate is stated against. Each is deliberately
blind to something the full model can see, so a win over one of them says
something specific:

    dataset prior       ignores the input entirely -> is the model using
                        anything at all, or reproducing the training mean?
    geometry only       sees frontier scalars, no map -> does the spatial input
                        carry signal beyond boundary length and info gain?
    current observation sees the map, not the option -> does conditioning on
                        the action matter, or is the answer determined by the
                        state alone?

The third is the one that matters most for this paper. If a model conditioned
only on the decision state matches one conditioned on the option too, then
candidates are not distinguishable and there is no ranking problem to solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BaselinePrediction:
    revealed: np.ndarray  # (T, H, W) probabilities
    target_present: float
    crossing_success: float
    revealed_area_m2: float


class DatasetPrior:
    """Predicts the training mean, ignoring the input completely."""

    name = "dataset_prior"

    def __init__(self) -> None:
        self.mean_revealed: np.ndarray | None = None
        self.mean_target = 0.0
        self.mean_crossing = 0.0
        self.mean_area = 0.0

    def fit(self, batches) -> "DatasetPrior":
        revealed_sum = None
        count = 0
        targets, crossings, areas = [], [], []
        for batch in batches:
            mask = batch["candidate_mask"]
            selected = batch["targets"][mask]  # (N, T, H, W)
            revealed_sum = (
                selected.sum(axis=0)
                if revealed_sum is None
                else revealed_sum + selected.sum(axis=0)
            )
            count += selected.shape[0]
            targets.append(batch["scalars"]["target_present"][mask])
            crossings.append(batch["scalars"]["crossing_success"][mask])
            areas.append(batch["scalars"]["revealed_area_m2"][mask])

        self.mean_revealed = revealed_sum / max(count, 1)
        self.mean_target = float(np.concatenate(targets).mean()) if targets else 0.0
        self.mean_crossing = float(np.concatenate(crossings).mean()) if crossings else 0.0
        self.mean_area = float(np.concatenate(areas).mean()) if areas else 0.0
        return self

    def predict(self, inputs, option, goal) -> BaselinePrediction:
        return BaselinePrediction(
            revealed=self.mean_revealed,
            target_present=self.mean_target,
            crossing_success=self.mean_crossing,
            revealed_area_m2=self.mean_area,
        )


class GeometryOnly:
    """Ridge regression on frontier scalars; no map, no spatial output.

    Spatially it falls back to the prior, so any spatial win over this baseline
    is attributable to the map input rather than to the scalars.
    """

    name = "geometry_only"

    def __init__(self) -> None:
        self.prior = DatasetPrior()
        self.coefficients: dict[str, np.ndarray] = {}

    def fit(self, batches) -> "GeometryOnly":
        batches = list(batches)
        self.prior.fit(batches)

        features, labels = [], {"target_present": [], "crossing_success": [], "revealed_area_m2": []}
        for batch in batches:
            mask = batch["candidate_mask"]
            features.append(batch["options"][mask])
            for key in labels:
                labels[key].append(batch["scalars"][key][mask])

        if not features:
            return self
        X = np.concatenate(features)
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=1)
        for key, values in labels.items():
            y = np.concatenate(values)
            # Ridge rather than least squares: option features are correlated
            # (action count tracks travel distance) and the normal equations
            # are ill-conditioned without it.
            gram = X.T @ X + 1e-3 * np.eye(X.shape[1])
            self.coefficients[key] = np.linalg.solve(gram, X.T @ y)
        return self

    def predict(self, inputs, option, goal) -> BaselinePrediction:
        x = np.concatenate([np.asarray(option, dtype=np.float64), [1.0]])
        values = {
            key: float(np.dot(coefficients, x))
            for key, coefficients in self.coefficients.items()
        }
        return BaselinePrediction(
            revealed=self.prior.mean_revealed,
            target_present=float(np.clip(values.get("target_present", 0.0), 0.0, 1.0)),
            crossing_success=float(np.clip(values.get("crossing_success", 0.0), 0.0, 1.0)),
            revealed_area_m2=max(0.0, values.get("revealed_area_m2", 0.0)),
        )


class CurrentObservationOnly:
    """Predicts revelation from the unknown mask alone: 'everything unknown
    within the window gets revealed'.

    A surprisingly strong geometric heuristic, and the honest thing to beat --
    it captures how much of the answer is fixed by the decision state before
    any option is chosen.
    """

    name = "current_observation_only"

    def __init__(self) -> None:
        self.prior = DatasetPrior()
        self.scale = 1.0

    def fit(self, batches) -> "CurrentObservationOnly":
        from frontierworld.data.tensors import CH_UNKNOWN

        batches = list(batches)
        self.prior.fit(batches)

        unknown_areas, actual_areas = [], []
        for batch in batches:
            mask = batch["candidate_mask"]
            unknown = batch["inputs"][mask][:, CH_UNKNOWN]
            unknown_areas.append(unknown.sum(axis=(1, 2)))
            actual_areas.append(batch["scalars"]["revealed_area_m2"][mask])
        if unknown_areas:
            u = np.concatenate(unknown_areas)
            a = np.concatenate(actual_areas)
            self.scale = float((u @ a) / max(float(u @ u), 1e-9))
        return self

    def predict(self, inputs, option, goal) -> BaselinePrediction:
        from frontierworld.data.tensors import CH_UNKNOWN

        unknown = np.asarray(inputs)[CH_UNKNOWN]
        revealed = np.stack(
            [unknown * 0.5, unknown * 0.1, unknown * 0.4]
        ).astype(np.float32)
        return BaselinePrediction(
            revealed=revealed,
            target_present=self.prior.mean_target,
            crossing_success=self.prior.mean_crossing,
            revealed_area_m2=float(unknown.sum() * self.scale),
        )


BASELINES = {
    "dataset_prior": DatasetPrior,
    "geometry_only": GeometryOnly,
    "current_observation_only": CurrentObservationOnly,
}


@dataclass
class PredictionMetrics:
    """Scores shared by the model and every baseline, so they are comparable."""

    occupancy_iou: float = 0.0
    occupancy_auc: float = float("nan")
    semantic_iou: float = 0.0
    target_auc: float = float("nan")
    target_brier: float = float("nan")
    crossing_accuracy: float = 0.0
    area_mae: float = 0.0
    n: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def masked_iou(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float = 0.5
) -> float:
    predicted = (prediction >= threshold) & valid
    actual = (target >= 0.5) & valid
    union = int((predicted | actual).sum())
    return float((predicted & actual).sum() / union) if union else float("nan")


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via rank statistics; nan when only one class is present."""
    labels = np.asarray(labels).astype(int)
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))
