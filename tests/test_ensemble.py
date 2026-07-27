"""Tests for Phase 10 ensemble uncertainty and calibration.

Pure numpy; no GPU, no checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontierworld.models.ensemble import (
    ConformalAreaInterval,
    PlattScaler,
    TemperatureScaler,
    binary_entropy,
    brier,
    ensemble_mean,
    expected_calibration_error,
    expected_entropy,
    mutual_information,
    negative_log_likelihood,
    pairwise_cosine_disagreement,
    predictive_entropy,
    ranking_stability,
    risk_coverage_curve,
    seed_range,
    seed_variance,
    spearman,
)


# -- uncertainty -----------------------------------------------------------


def test_binary_entropy_is_maximal_at_one_half():
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2), abs=1e-9)


def test_binary_entropy_is_zero_at_certainty():
    assert binary_entropy(np.array([0.0, 1.0])) == pytest.approx([0.0, 0.0], abs=1e-5)


def test_agreeing_seeds_have_zero_mutual_information():
    """Identical members disagree about nothing, however uncertain each one is."""
    probabilities = np.tile(np.array([[0.3, 0.7, 0.5]]), (3, 1))
    assert mutual_information(probabilities) == pytest.approx([0, 0, 0], abs=1e-9)
    # ...but total uncertainty is emphatically not zero.
    assert predictive_entropy(probabilities)[2] == pytest.approx(np.log(2), abs=1e-9)


def test_disagreeing_seeds_have_positive_mutual_information():
    probabilities = np.array([[0.01], [0.5], [0.99]])
    assert mutual_information(probabilities)[0] > 0.1


def test_mutual_information_is_entropy_minus_expected_entropy():
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0.01, 0.99, size=(3, 50))
    expected = predictive_entropy(probabilities) - expected_entropy(probabilities)
    assert mutual_information(probabilities) == pytest.approx(expected, abs=1e-12)


def test_mutual_information_is_never_negative():
    """Jensen guarantees it; floating point must not be allowed to violate it."""
    probabilities = np.full((3, 20), 0.42)
    assert (mutual_information(probabilities) >= 0).all()


def test_confident_disagreement_exceeds_uncertain_agreement_in_mi():
    """The point of MI: three confident-but-opposed members are epistemically
    more uncertain than three members that agree they do not know."""
    opposed = np.array([[0.02], [0.98], [0.02]])
    agreed_unsure = np.array([[0.5], [0.5], [0.5]])
    assert mutual_information(opposed)[0] > mutual_information(agreed_unsure)[0]
    # Total entropy ranks them the other way, which is exactly why both are reported.
    assert predictive_entropy(agreed_unsure)[0] > predictive_entropy(opposed)[0]


def test_ensemble_mean_and_variance():
    probabilities = np.array([[0.2], [0.4], [0.6]])
    assert ensemble_mean(probabilities)[0] == pytest.approx(0.4)
    assert seed_variance(probabilities)[0] == pytest.approx(np.var([0.2, 0.4, 0.6]))


def test_seed_range_is_descriptive_only():
    """Documented as not an interval; the test pins the arithmetic, and the
    coverage claim lives with the conformal procedure instead."""
    assert seed_range(np.array([[1.0], [5.0], [3.0]]))[0] == pytest.approx(4.0)


def test_cosine_disagreement_zero_for_identical_maps():
    # (3 seeds, 2 examples, 2 values) -> one disagreement score per example.
    maps = np.tile(np.array([[[1.0, 2.0], [3.0, 4.0]]]), (3, 1, 1))
    assert maps.shape == (3, 2, 2)
    assert pairwise_cosine_disagreement(maps) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_cosine_disagreement_positive_for_orthogonal_maps():
    maps = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]])
    assert pairwise_cosine_disagreement(maps)[0] > 0.5


def test_cosine_disagreement_treats_two_empty_maps_as_agreeing():
    """Two seeds that both predict no semantics agree; scoring that as maximal
    disagreement would make empty regions dominate the metric."""
    maps = np.zeros((3, 1, 4))
    assert pairwise_cosine_disagreement(maps) == pytest.approx([0.0], abs=1e-9)


def test_cosine_disagreement_empty_versus_nonempty_disagrees():
    maps = np.array([[[0.0, 0.0]], [[1.0, 1.0]], [[0.0, 0.0]]])
    assert pairwise_cosine_disagreement(maps)[0] > 0.0


# -- calibration -----------------------------------------------------------


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(1)
    labels = rng.binomial(1, 0.5, size=4000).astype(float)
    # Directionally right but far too sharp.
    logits = np.where(labels > 0, 4.0, -4.0) + rng.normal(0, 3.0, size=labels.size)
    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.temperature > 1.0

    from frontierworld.models.ensemble import _sigmoid

    before = negative_log_likelihood(_sigmoid(logits), labels)
    after = negative_log_likelihood(scaler.transform(logits), labels)
    assert after < before


def test_temperature_scaling_preserves_ranking():
    """Calibration must not change AUROC; if it does, it is not calibration."""
    logits = np.array([-2.0, -0.5, 0.3, 1.7, 4.0])
    scaled = TemperatureScaler(temperature=2.5).transform(logits)
    assert list(np.argsort(scaled)) == list(np.argsort(logits))


def test_temperature_handles_single_class_calibration_data():
    scaler = TemperatureScaler().fit(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert scaler.temperature == 1.0


def test_platt_can_shift_the_base_rate():
    """Temperature cannot fix a wrong prior; Platt's intercept can."""
    rng = np.random.default_rng(2)
    labels = rng.binomial(1, 0.1, size=4000).astype(float)
    logits = rng.normal(2.0, 1.0, size=labels.size)  # uniformly far too high
    platt = PlattScaler().fit(logits, labels)
    assert platt.transform(logits).mean() < 0.5
    assert abs(platt.transform(logits).mean() - labels.mean()) < 0.1


def test_platt_keeps_slope_positive():
    rng = np.random.default_rng(3)
    labels = rng.binomial(1, 0.5, size=500).astype(float)
    logits = -rng.normal(0, 1, size=500)  # deliberately anti-correlated
    platt = PlattScaler().fit(logits, labels)
    assert platt.slope > 0


# -- conformal area intervals ---------------------------------------------


def test_conformal_interval_achieves_nominal_coverage():
    rng = np.random.default_rng(4)
    n = 4000
    spread = rng.uniform(0.5, 3.0, size=n)
    predicted = rng.uniform(5.0, 40.0, size=n)
    truth = predicted + rng.normal(0, 1.0, size=n) * spread

    calibration = slice(0, n // 2)
    validation = slice(n // 2, n)

    conformal = ConformalAreaInterval(alpha=0.10).fit(
        truth[calibration], predicted[calibration], spread[calibration]
    )
    result = conformal.coverage(truth[validation], predicted[validation], spread[validation])
    assert 0.85 < result["coverage"] < 0.95


def test_conformal_interval_widens_where_seeds_disagree():
    """Adaptivity is the reason for normalising by spread."""
    conformal = ConformalAreaInterval(alpha=0.10, quantile=2.0, epsilon=1.0)
    narrow = conformal.interval(np.array([10.0]), np.array([0.0]))
    wide = conformal.interval(np.array([10.0]), np.array([5.0]))
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_conformal_uses_finite_sample_corrected_quantile():
    """The plain empirical quantile under-covers at small n; the (n+1)
    correction is what carries the guarantee."""
    truth = np.arange(10, dtype=float)
    predicted = np.zeros(10)
    spread = np.zeros(10)
    conformal = ConformalAreaInterval(alpha=0.10, epsilon=1.0).fit(truth, predicted, spread)
    plain = np.quantile(np.abs(truth - predicted) / 1.0, 0.90)
    assert conformal.quantile >= plain


def test_conformal_with_empty_calibration_is_degenerate_not_crashing():
    conformal = ConformalAreaInterval().fit(np.array([]), np.array([]), np.array([]))
    assert conformal.quantile == 0.0


# -- scalar metrics --------------------------------------------------------


def test_brier_and_nll_reward_correctness():
    good = np.array([0.9, 0.1])
    bad = np.array([0.1, 0.9])
    labels = np.array([1.0, 0.0])
    assert brier(good, labels) < brier(bad, labels)
    assert negative_log_likelihood(good, labels) < negative_log_likelihood(bad, labels)


def test_perfectly_calibrated_predictions_have_low_ece():
    rng = np.random.default_rng(5)
    probability = rng.uniform(0, 1, size=20000)
    labels = rng.binomial(1, probability).astype(float)
    assert expected_calibration_error(probability, labels) < 0.02


def test_ece_detects_systematic_overconfidence():
    probability = np.full(1000, 0.95)
    labels = np.zeros(1000)
    labels[:500] = 1.0  # actual accuracy 0.5 against claimed 0.95
    assert expected_calibration_error(probability, labels) == pytest.approx(0.45, abs=0.02)


def test_spearman_is_monotone_not_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x**3) == pytest.approx(-1.0)


def test_risk_coverage_rewards_useful_uncertainty():
    """Uncertainty that tracks error must beat uncertainty that does not."""
    rng = np.random.default_rng(6)
    error = rng.uniform(0, 1, size=500)
    informative = error + rng.normal(0, 0.02, size=500)
    useless = rng.uniform(0, 1, size=500)
    assert risk_coverage_curve(informative, error)["aurc"] < risk_coverage_curve(useless, error)["aurc"]


def test_risk_coverage_random_baseline_is_the_mean_error():
    error = np.array([0.0, 1.0, 0.5, 0.5])
    assert risk_coverage_curve(error, error)["random_aurc"] == pytest.approx(0.5)


def test_ranking_stability_detects_agreement():
    scores = np.array([[3.0, 1.0, 2.0], [3.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
    groups = np.array(["g", "g", "g"])
    result = ranking_stability(scores, groups)
    assert result["top1_agreement_rate"] == 1.0
    assert result["mean_pairwise_kendall_tau"] == pytest.approx(1.0)


def test_ranking_stability_detects_disagreement():
    scores = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])
    groups = np.array(["g", "g", "g"])
    assert ranking_stability(scores, groups)["top1_agreement_rate"] == 0.0


def test_ranking_stability_ignores_singleton_groups():
    """A group with one frontier has nothing to rank and must not be counted as
    perfect agreement."""
    scores = np.array([[1.0], [2.0], [3.0]])
    result = ranking_stability(scores, np.array(["only"]))
    assert result["n_groups"] == 0
