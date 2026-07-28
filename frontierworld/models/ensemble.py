"""Phase 10: deep-ensemble uncertainty and calibration.

The three Phase 8 seeds are treated as a frozen deep ensemble. Nothing here
trains or fine-tunes them; the only fitted quantities are calibration
parameters, and those are fitted on calibration scenes and applied unchanged to
validation scenes.

Two distinctions the code keeps explicit, because collapsing them is the usual
way ensemble uncertainty gets over-claimed:

*Total* uncertainty (predictive entropy of the mean) mixes irreducible outcome
noise with model disagreement. *Mutual information* isolates the disagreement
part -- the component an ensemble can actually speak to:

    U_MI = H(p_bar) - (1/S) sum_s H(p_s)

And with **S = 3**, the seed-wise min/max is not a prediction interval. Three
draws cannot pin an empirical quantile, so intervals here come from a conformal
procedure calibrated on held-out scenes, never from the spread of three numbers.
`seed_range` is provided for description only and is named so it cannot be
mistaken for a bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-7


# -- uncertainty -----------------------------------------------------------


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Bernoulli entropy in nats, safe at 0 and 1."""
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def ensemble_mean(seed_probabilities: np.ndarray) -> np.ndarray:
    """`(S, ...)` -> `(...)`. Mean over the seed axis."""
    return np.mean(np.asarray(seed_probabilities, dtype=np.float64), axis=0)


def predictive_entropy(seed_probabilities: np.ndarray) -> np.ndarray:
    """H(p_bar): total uncertainty of the ensemble mean."""
    return binary_entropy(ensemble_mean(seed_probabilities))


def expected_entropy(seed_probabilities: np.ndarray) -> np.ndarray:
    """(1/S) sum_s H(p_s): the aleatoric part each member already expects."""
    return np.mean(binary_entropy(np.asarray(seed_probabilities, dtype=np.float64)), axis=0)


def mutual_information(seed_probabilities: np.ndarray) -> np.ndarray:
    """H(p_bar) - mean_s H(p_s): disagreement between members.

    Non-negative by Jensen; tiny negative values from floating point are
    clipped rather than left to propagate into a correlation.
    """
    return np.maximum(
        predictive_entropy(seed_probabilities) - expected_entropy(seed_probabilities), 0.0
    )


def seed_variance(seed_probabilities: np.ndarray) -> np.ndarray:
    """Variance of the per-seed probabilities."""
    return np.var(np.asarray(seed_probabilities, dtype=np.float64), axis=0)


def seed_range(seed_values: np.ndarray) -> np.ndarray:
    """max - min across seeds. DESCRIPTIVE ONLY.

    With three seeds this is not a prediction interval and must never be
    reported as coverage. Named `seed_range` rather than anything
    interval-shaped for exactly that reason.
    """
    values = np.asarray(seed_values, dtype=np.float64)
    return values.max(axis=0) - values.min(axis=0)


def pairwise_cosine_disagreement(seed_maps: np.ndarray) -> np.ndarray:
    """Mean pairwise (1 - cosine similarity) across seeds.

    `seed_maps` is `(S, N, ...)`; each seed's per-example map is flattened and
    compared. Used for the semantic head, where the quantity of interest is
    whether seeds agree on *which* cells carry semantic mass, not on a scalar.
    """
    maps = np.asarray(seed_maps, dtype=np.float64)
    n_seeds = maps.shape[0]
    flat = maps.reshape(n_seeds, maps.shape[1], -1)
    norms = np.linalg.norm(flat, axis=-1)

    total = np.zeros(maps.shape[1], dtype=np.float64)
    count = 0
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            denominator = norms[i] * norms[j]
            similarity = np.where(
                denominator > EPSILON,
                np.sum(flat[i] * flat[j], axis=-1) / np.maximum(denominator, EPSILON),
                # Two all-zero maps agree perfectly; one zero and one not do not.
                np.where((norms[i] <= EPSILON) & (norms[j] <= EPSILON), 1.0, 0.0),
            )
            total += 1.0 - similarity
            count += 1
    return total / max(count, 1)


# -- calibration -----------------------------------------------------------


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


@dataclass
class TemperatureScaler:
    """One parameter, fitted by minimising NLL on calibration data.

    Temperature cannot reorder predictions, so it changes calibration metrics
    (Brier, NLL, ECE) without touching ranking metrics (AUROC, AUPRC). That
    separation is why it is registered before the evaluation table is seen.
    """

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if labels.size == 0 or len(np.unique(labels)) < 2:
            self.temperature = 1.0
            return self

        grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
        best, best_nll = 1.0, np.inf
        for temperature in grid:
            probability = _sigmoid(logits / temperature)
            nll = -np.mean(
                labels * np.log(np.clip(probability, EPSILON, 1.0))
                + (1.0 - labels) * np.log(np.clip(1.0 - probability, EPSILON, 1.0))
            )
            if nll < best_nll:
                best, best_nll = float(temperature), float(nll)
        self.temperature = best
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)


@dataclass
class PlattScaler:
    """Affine recalibration in logit space: sigmoid(a * z + b).

    Strictly more flexible than temperature: `b` can shift the base rate, which
    temperature alone cannot. `a > 0` is enforced so ranking is preserved.
    """

    slope: float = 1.0
    intercept: float = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, iterations: int = 200) -> "PlattScaler":
        z = np.asarray(logits, dtype=np.float64).ravel()
        y = np.asarray(labels, dtype=np.float64).ravel()
        if y.size == 0 or len(np.unique(y)) < 2:
            self.slope, self.intercept = 1.0, 0.0
            return self

        slope, intercept = 1.0, 0.0
        learning_rate = 0.1
        for _ in range(iterations):
            probability = _sigmoid(slope * z + intercept)
            residual = probability - y
            gradient_slope = float(np.mean(residual * z))
            gradient_intercept = float(np.mean(residual))
            slope -= learning_rate * gradient_slope
            intercept -= learning_rate * gradient_intercept
            slope = max(slope, 1e-3)  # keep the map monotone
        self.slope, self.intercept = float(slope), float(intercept)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(self.slope * np.asarray(logits, dtype=np.float64) + self.intercept)


@dataclass
class ConformalAreaInterval:
    """Split-conformal intervals for area, width scaled by ensemble spread.

        r_i = |A_gt - A_bar| / (U_i + eps)
        interval = A_bar +/- q_{1-alpha}(r) * (U + eps)

    Normalising the residual by the ensemble standard deviation is what makes
    the interval adaptive: it is wide exactly where the seeds disagree. If the
    disagreement carries no information about error, the quantile simply absorbs
    it and the interval degenerates to a constant width -- which is a real
    result about the ensemble, not a failure of the procedure.
    """

    quantile: float = 1.0
    alpha: float = 0.10
    epsilon: float = 1.0
    n_calibration: int = 0

    def fit(
        self, truth: np.ndarray, predicted: np.ndarray, spread: np.ndarray
    ) -> "ConformalAreaInterval":
        truth = np.asarray(truth, dtype=np.float64).ravel()
        predicted = np.asarray(predicted, dtype=np.float64).ravel()
        spread = np.asarray(spread, dtype=np.float64).ravel()
        residual = np.abs(truth - predicted) / (spread + self.epsilon)
        n = residual.size
        if n == 0:
            self.quantile = 0.0
            return self
        # Finite-sample conformal correction: the ceil((n+1)(1-alpha))/n
        # quantile, not the plain empirical one, is what carries the coverage
        # guarantee.
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile = float(np.quantile(residual, level))
        self.n_calibration = int(n)
        return self

    def interval(self, predicted: np.ndarray, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predicted = np.asarray(predicted, dtype=np.float64)
        width = self.quantile * (np.asarray(spread, dtype=np.float64) + self.epsilon)
        return predicted - width, predicted + width

    def coverage(
        self, truth: np.ndarray, predicted: np.ndarray, spread: np.ndarray
    ) -> dict:
        lower, upper = self.interval(predicted, spread)
        truth = np.asarray(truth, dtype=np.float64)
        inside = (truth >= lower) & (truth <= upper)
        return {
            "coverage": float(np.mean(inside)),
            "target_coverage": 1.0 - self.alpha,
            "mean_width": float(np.mean(upper - lower)),
            "median_width": float(np.median(upper - lower)),
            "n": int(truth.size),
        }


# -- scalar metrics --------------------------------------------------------


def brier(probability: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))


def negative_log_likelihood(probability: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def expected_calibration_error(
    probability: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Equal-width binned |confidence - accuracy|, weighted by bin occupancy."""
    p = np.asarray(probability, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (p > lower) & (p <= upper) if index else (p >= lower) & (p <= upper)
        if not in_bin.any():
            continue
        total += in_bin.mean() * abs(p[in_bin].mean() - y[in_bin].mean())
    return float(total)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation; the uncertainty-error criterion is monotone, not linear."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2:
        return float("nan")
    rank_x = np.argsort(np.argsort(x)).astype(np.float64)
    rank_y = np.argsort(np.argsort(y)).astype(np.float64)
    if np.std(rank_x) < EPSILON or np.std(rank_y) < EPSILON:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def risk_coverage_curve(
    uncertainty: np.ndarray, error: np.ndarray, n_points: int = 20
) -> dict:
    """Error among the most-confident fraction, as that fraction grows.

    A useful uncertainty gives a curve rising with coverage: keeping only
    confident predictions should reduce mean error. AURC summarises it; lower
    is better, and it is only interpretable against the same quantity for a
    random ordering.
    """
    uncertainty = np.asarray(uncertainty, dtype=np.float64).ravel()
    error = np.asarray(error, dtype=np.float64).ravel()
    n = uncertainty.size
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan")}

    order = np.argsort(uncertainty)  # most confident first
    sorted_error = error[order]
    coverages, risks = [], []
    for fraction in np.linspace(1.0 / n_points, 1.0, n_points):
        k = max(1, int(round(fraction * n)))
        coverages.append(k / n)
        risks.append(float(np.mean(sorted_error[:k])))

    return {
        "coverage": [round(c, 4) for c in coverages],
        "risk": [round(r, 6) for r in risks],
        "aurc": float(np.trapz(risks, coverages)),
        "random_aurc": float(np.mean(error)),
    }


def ranking_stability(seed_scores: np.ndarray, group_ids: np.ndarray) -> dict:
    """Do the seeds agree on which frontier is best within a decision group?

    This is the quantity that actually matters for a planner: an ensemble whose
    members disagree about ranking cannot be trusted to choose an option, even
    if their averaged maps look fine.
    """
    seed_scores = np.asarray(seed_scores, dtype=np.float64)
    group_ids = np.asarray(group_ids)
    n_seeds = seed_scores.shape[0]

    agreements, kendalls = [], []
    for group in np.unique(group_ids):
        mask = group_ids == group
        if mask.sum() < 2:
            continue
        block = seed_scores[:, mask]
        best = [int(np.argmax(block[s])) for s in range(n_seeds)]
        agreements.append(float(len(set(best)) == 1))

        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                kendalls.append(_kendall_tau(block[i], block[j]))

    return {
        "top1_agreement_rate": float(np.mean(agreements)) if agreements else float("nan"),
        "mean_pairwise_kendall_tau": float(np.mean(kendalls)) if kendalls else float("nan"),
        "n_groups": len(agreements),
    }


def scene_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    scene_ids: np.ndarray,
    iterations: int = 5000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap resampling whole SCENES, not decision groups.

    Decision groups within one scene are not independent: they share geometry,
    layout and semantics, and often overlap in observed map. Resampling groups
    therefore treats correlated units as independent and understates the
    interval -- the more so here, where 494 groups come from only 22 scenes.

    Resampling scenes with replacement, and taking *every* group from each
    sampled scene, respects that clustering. It is strictly a sensitivity
    analysis: the registered group-level bootstrap stands as the primary
    result, and this is reported beside it, never in place of it.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    scene_ids = np.asarray(scene_ids).ravel()
    if a.shape != b.shape or a.shape != scene_ids.shape:
        raise ValueError("a, b and scene_ids must align element-wise")

    unique = np.unique(scene_ids)
    index_by_scene = {scene: np.flatnonzero(scene_ids == scene) for scene in unique}
    observed = float(np.mean(a) - np.mean(b))

    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([index_by_scene[scene] for scene in chosen])
        differences[iteration] = np.mean(a[index]) - np.mean(b[index])

    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "mean_difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "n_units": int(len(a)),
        "n_scenes": int(len(unique)),
        "significant": bool(low > 0 or high < 0),
        "resampling_unit": "scene",
    }


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0
