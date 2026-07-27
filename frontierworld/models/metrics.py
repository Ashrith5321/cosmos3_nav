"""Per-head metrics for revelation prediction.

Reported per head rather than as one number, because the aggregate loss can
fall while a dense head stays weak: the scalar heads converge fast and dominate
the total, so a low loss says nothing on its own about occupancy quality.

Two masking rules, both load-bearing:

  * every spatial metric is restricted to `valid` -- cells outside the global
    map have no ground truth, and scoring them rewards confident predictions
    about regions nothing was ever observed in;
  * semantics are additionally restricted to *newly revealed* cells. Semantic
    labels are only defined where the branch revealed something, so scoring
    them over the whole window measures mostly true negatives and inflates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class HeadMetrics:
    """Everything the Phase 8 report table needs, per head."""

    # revelation occupancy
    free_iou: float = float("nan")
    occupied_iou: float = float("nan")
    macro_iou: float = float("nan")
    # semantics, over newly revealed cells only
    semantic_iou: float = float("nan")
    semantic_cosine: float = float("nan")
    # target presence
    target_auroc: float = float("nan")
    target_auprc: float = float("nan")
    target_brier: float = float("nan")
    target_positive_rate: float = float("nan")
    # crossing
    crossing_accuracy: float = float("nan")
    crossing_auroc: float = float("nan")
    # revealed area
    area_mae: float = float("nan")
    area_correlation: float = float("nan")
    n: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def table_rows(self) -> list[tuple[str, str]]:
        return [
            ("Revelation occupancy",
             f"free IoU {self.free_iou:.3f}  occupied IoU {self.occupied_iou:.3f}  "
             f"macro IoU {self.macro_iou:.3f}"),
            ("Semantics (revealed cells)",
             f"IoU {self.semantic_iou:.3f}  cosine {self.semantic_cosine:.3f}"),
            ("Target presence",
             f"AUROC {self.target_auroc:.3f}  AUPRC {self.target_auprc:.3f}  "
             f"Brier {self.target_brier:.3f}"),
            ("Crossing",
             f"accuracy {self.crossing_accuracy:.3f}  AUROC {self.crossing_auroc:.3f}"),
            ("Revealed area",
             f"MAE {self.area_mae:.2f} m2  corr {self.area_correlation:.3f}"),
        ]


def masked_iou(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, threshold: float = 0.5
) -> float:
    predicted = (prediction >= threshold) & valid
    actual = (target >= 0.5) & valid
    union = int((predicted | actual).sum())
    return float((predicted & actual).sum() / union) if union else float("nan")


def masked_cosine(
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> float:
    a = prediction[valid].ravel().astype(np.float64)
    b = target[valid].ravel().astype(np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator > 1e-12 else float("nan")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC; nan when only one class is present."""
    labels = np.asarray(labels).astype(int)
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(np.asarray(scores), kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float(
        (ranks[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
    )


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision. Reported alongside AUROC because target presence is
    imbalanced, and AUROC flatters a classifier on a rare positive class."""
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    precision = cumulative_tp / np.arange(1, len(sorted_labels) + 1)
    return float((precision * sorted_labels).sum() / labels.sum())


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_head_metrics(
    revealed_probabilities: np.ndarray,  # (N, T, H, W)
    targets: np.ndarray,  # (N, T, H, W)
    valid: np.ndarray,  # (N, H, W) bool
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    crossing_scores: np.ndarray,
    crossing_labels: np.ndarray,
    area_predictions: np.ndarray,
    area_labels: np.ndarray,
) -> HeadMetrics:
    """Score every head separately."""
    free_ious, occupied_ious, semantic_ious, semantic_cosines = [], [], [], []

    for index in range(revealed_probabilities.shape[0]):
        window = valid[index]
        free_ious.append(masked_iou(revealed_probabilities[index, 0], targets[index, 0], window))
        occupied_ious.append(
            masked_iou(revealed_probabilities[index, 1], targets[index, 1], window)
        )

        # Semantics only where the branch actually revealed something.
        revealed = window & (
            (targets[index, 0] >= 0.5) | (targets[index, 1] >= 0.5)
        )
        if revealed.any():
            semantic_ious.append(
                masked_iou(revealed_probabilities[index, 2], targets[index, 2], revealed)
            )
            semantic_cosines.append(
                masked_cosine(revealed_probabilities[index, 2], targets[index, 2], revealed)
            )

    free = float(np.nanmean(free_ious)) if free_ious else float("nan")
    occupied = float(np.nanmean(occupied_ious)) if occupied_ious else float("nan")

    return HeadMetrics(
        free_iou=free,
        occupied_iou=occupied,
        macro_iou=float(np.nanmean([free, occupied])),
        semantic_iou=float(np.nanmean(semantic_ious)) if semantic_ious else float("nan"),
        semantic_cosine=(
            float(np.nanmean(semantic_cosines)) if semantic_cosines else float("nan")
        ),
        target_auroc=auroc(target_scores, target_labels),
        target_auprc=auprc(target_scores, target_labels),
        target_brier=float(np.mean((target_scores - target_labels) ** 2)),
        target_positive_rate=float(np.mean(target_labels)),
        crossing_accuracy=float(np.mean((crossing_scores >= 0.5) == (crossing_labels >= 0.5))),
        crossing_auroc=auroc(crossing_scores, crossing_labels),
        area_mae=float(np.mean(np.abs(area_predictions - area_labels))),
        area_correlation=correlation(area_predictions, area_labels),
        n=int(revealed_probabilities.shape[0]),
    )


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    group_ids: np.ndarray,
    iterations: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap CI on the mean difference a - b, resampling GROUPS.

    Paired because every method is scored on identical samples; an unpaired
    interval would be far wider than the comparison warrants.

    Resampling whole decision groups, not individual branches, is the part that
    matters. Branches within a group share a map, a goal and a starting state,
    so their errors are strongly correlated. Treating them as independent
    samples inflates the effective n by roughly the branch count -- about 5x
    here -- and produces intervals far too narrow, which is exactly how a
    difference that is really within noise gets reported as significant.

    `group_ids` must align with `a` and `b` element-wise.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    group_ids = np.asarray(group_ids)

    finite = np.isfinite(a) & np.isfinite(b)
    a, b, group_ids = a[finite], b[finite], group_ids[finite]
    if a.size == 0:
        return {
            "mean_difference": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "n_branches": 0, "n_groups": 0,
            "significant": False, "resampling_unit": "decision_group",
        }

    difference = a - b
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    by_group = [np.flatnonzero(inverse == index) for index in range(len(unique_groups))]

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        picked = rng.integers(0, len(by_group), size=len(by_group))
        sample = np.concatenate([by_group[index] for index in picked])
        means[iteration] = difference[sample].mean()

    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_branches": int(len(difference)),
        "n_groups": int(len(unique_groups)),
        "significant": bool(low > 0 or high < 0),
        "resampling_unit": "decision_group",
    }
