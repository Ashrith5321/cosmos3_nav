"""Revelation predictor and baseline tests.

The property that matters most here is action conditioning. Candidates at one
decision state share almost all of their input -- same map, same goal, only the
frontier and option differ -- so a model that ignores the option would give
every candidate the same score and the ranking problem this project is about
would be vacuous. Several tests below exist only to catch that.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from frontierworld.data.tensors import N_INPUT_CHANNELS, N_TARGET_CHANNELS
from frontierworld.models.baselines import (
    BASELINES,
    CurrentObservationOnly,
    DatasetPrior,
    GeometryOnly,
    binary_auc,
    masked_iou,
)
from frontierworld.models.predictor import (
    GOAL_CATEGORIES,
    PredictorConfig,
    RevelationPredictor,
    goal_index,
    revelation_loss,
)

SIZE = 32


def make_model(option_dim: int = 8) -> RevelationPredictor:
    torch.manual_seed(0)
    return RevelationPredictor(PredictorConfig(option_dim=option_dim, width=8, depth=1))


def make_inputs(batch: int = 2):
    torch.manual_seed(1)
    return (
        torch.rand(batch, N_INPUT_CHANNELS, SIZE, SIZE),
        torch.rand(batch, 8),
        torch.zeros(batch, dtype=torch.long),
    )


# -- forward pass ----------------------------------------------------------


def test_forward_shapes():
    model = make_model()
    inputs, option, goal = make_inputs(3)
    out = model(inputs, option, goal)

    assert out["revealed_logits"].shape == (3, N_TARGET_CHANNELS, SIZE, SIZE)
    assert out["target_logit"].shape == (3,)
    assert out["crossing_logit"].shape == (3,)
    assert out["area"].shape == (3,)


def test_predicted_area_is_non_negative():
    """Area is a physical quantity; a softplus head cannot emit negatives."""
    model = make_model()
    inputs, option, goal = make_inputs(4)
    assert torch.all(model(inputs, option, goal)["area"] >= 0)


def test_output_is_finite():
    model = make_model()
    inputs, option, goal = make_inputs(2)
    for value in model(inputs, option, goal).values():
        assert torch.all(torch.isfinite(value))


def test_changing_the_option_changes_the_prediction():
    """The core property: identical map, different action -> different answer."""
    model = make_model()
    inputs, option, goal = make_inputs(1)
    first = model(inputs, option, goal)
    second = model(inputs, option + 1.0, goal)

    spatial_delta = (
        (torch.sigmoid(first["revealed_logits"]) - torch.sigmoid(second["revealed_logits"]))
        .abs()
        .mean()
    )
    assert float(spatial_delta.detach()) > 1e-6
    assert float((first["area"] - second["area"]).abs()) > 1e-6


def test_changing_the_goal_changes_the_prediction():
    model = make_model()
    inputs, option, _ = make_inputs(1)
    first = model(inputs, option, torch.zeros(1, dtype=torch.long))
    second = model(inputs, option, torch.ones(1, dtype=torch.long))
    assert float((first["target_logit"] - second["target_logit"]).abs()) > 1e-9


def test_gradients_reach_the_option_encoder():
    """If the option encoder gets no gradient, conditioning is decorative."""
    model = make_model()
    inputs, option, goal = make_inputs(2)
    out = model(inputs, option, goal)
    (out["revealed_logits"].mean() + out["area"].mean()).backward()

    grads = [
        p.grad for p in model.option_encoder.parameters() if p.grad is not None
    ]
    assert grads, "option encoder received no gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_accepts_variable_spatial_sizes():
    model = make_model()
    for size in (32, 48, 80):
        inputs = torch.rand(1, N_INPUT_CHANNELS, size, size)
        out = model(inputs, torch.rand(1, 8), torch.zeros(1, dtype=torch.long))
        assert out["revealed_logits"].shape[-2:] == (size, size)


# -- goal indexing ---------------------------------------------------------


@pytest.mark.parametrize("goal", GOAL_CATEGORIES)
def test_known_goals_get_distinct_indices(goal):
    assert 0 <= goal_index(goal) < len(GOAL_CATEGORIES)


def test_unknown_and_missing_goals_share_the_spare_slot():
    assert goal_index(None) == len(GOAL_CATEGORIES)
    assert goal_index("teapot") == len(GOAL_CATEGORIES)


def test_goal_index_stays_inside_the_embedding():
    vocab = PredictorConfig().goal_vocab
    for goal in GOAL_CATEGORIES + [None, "unknown"]:
        assert goal_index(goal) < vocab


# -- loss ------------------------------------------------------------------


def make_loss_inputs(batch: int = 2):
    torch.manual_seed(2)
    prediction = {
        "revealed_logits": torch.zeros(batch, N_TARGET_CHANNELS, SIZE, SIZE, requires_grad=True),
        "target_logit": torch.zeros(batch, requires_grad=True),
        "crossing_logit": torch.zeros(batch, requires_grad=True),
        "area": torch.ones(batch, requires_grad=True),
    }
    targets = torch.zeros(batch, N_TARGET_CHANNELS, SIZE, SIZE)
    targets[:, 0, :8, :8] = 1.0
    valid = torch.ones(batch, SIZE, SIZE, dtype=torch.bool)
    scalars = {
        "target_present": torch.ones(batch),
        "crossing_success": torch.ones(batch),
        "revealed_area_m2": torch.full((batch,), 5.0),
    }
    return prediction, targets, valid, scalars


def test_loss_is_finite_and_positive():
    prediction, targets, valid, scalars = make_loss_inputs()
    loss, parts = revelation_loss(prediction, targets, valid, scalars)
    assert torch.isfinite(loss) and float(loss) > 0
    assert set(parts) == {"loss", "occ", "sem", "goal", "cross", "area"}


def test_loss_backpropagates_to_every_head():
    prediction, targets, valid, scalars = make_loss_inputs()
    loss, _ = revelation_loss(prediction, targets, valid, scalars)
    loss.backward()
    for key in ("revealed_logits", "target_logit", "crossing_logit", "area"):
        assert prediction[key].grad is not None, key
        assert float(prediction[key].grad.abs().sum()) > 0, key


def test_invalid_cells_are_excluded_from_the_spatial_loss():
    """Cells outside the map have no ground truth; scoring them would reward
    confident predictions about regions nothing was observed in.

    The error has to be spatially non-uniform for this to be observable: with a
    constant per-cell loss, masking scales numerator and denominator equally
    and the mean does not move.
    """
    _, targets, valid, scalars = make_loss_inputs()

    # Put positive target mass in the bottom half, then predict "empty"
    # everywhere. The top half is then correct and the bottom half is
    # confidently wrong, so masking the bottom half must lower the loss.
    targets = targets.clone()
    targets[:, 0, SIZE // 2 :, :] = 1.0
    logits = torch.full_like(targets, -10.0)
    logits[:, :, : SIZE // 2] = targets[:, :, : SIZE // 2] * 20.0 - 10.0
    prediction = {
        "revealed_logits": logits,
        "target_logit": torch.zeros(targets.shape[0]),
        "crossing_logit": torch.zeros(targets.shape[0]),
        "area": torch.ones(targets.shape[0]),
    }

    full, full_parts = revelation_loss(prediction, targets, valid, scalars)

    top_only = valid.clone()
    top_only[:, SIZE // 2 :, :] = False  # drop the wrong half
    masked, masked_parts = revelation_loss(prediction, targets, top_only, scalars)

    assert masked_parts["occ"] < full_parts["occ"], (
        "masking the region the model gets wrong must lower the occupancy loss"
    )
    assert float(masked) < float(full)


def test_a_perfect_prediction_scores_better_than_a_wrong_one():
    _, targets, valid, scalars = make_loss_inputs()
    good = {
        "revealed_logits": (targets * 20.0 - 10.0),
        "target_logit": torch.full((2,), 10.0),
        "crossing_logit": torch.full((2,), 10.0),
        "area": scalars["revealed_area_m2"].clone(),
    }
    bad = {
        "revealed_logits": ((1 - targets) * 20.0 - 10.0),
        "target_logit": torch.full((2,), -10.0),
        "crossing_logit": torch.full((2,), -10.0),
        "area": torch.full((2,), 40.0),
    }
    good_loss, _ = revelation_loss(good, targets, valid, scalars)
    bad_loss, _ = revelation_loss(bad, targets, valid, scalars)
    assert float(good_loss) < float(bad_loss)


# -- baselines -------------------------------------------------------------


def fake_batch(n: int = 4, area: float = 5.0) -> dict:
    rng = np.random.default_rng(0)
    return {
        "inputs": rng.random((1, n, N_INPUT_CHANNELS, SIZE, SIZE)).astype(np.float32),
        "targets": (rng.random((1, n, N_TARGET_CHANNELS, SIZE, SIZE)) > 0.7).astype(np.float32),
        "target_valid": np.ones((1, n, SIZE, SIZE), dtype=bool),
        "options": rng.random((1, n, 8)).astype(np.float32),
        "candidate_mask": np.ones((1, n), dtype=bool),
        "scalars": {
            "target_present": np.zeros((1, n), dtype=np.float32),
            "crossing_success": np.ones((1, n), dtype=np.float32),
            "revealed_area_m2": np.full((1, n), area, dtype=np.float32),
        },
    }


def test_dataset_prior_learns_the_mean():
    prior = DatasetPrior().fit([fake_batch(area=5.0), fake_batch(area=7.0)])
    assert prior.mean_area == pytest.approx(6.0, abs=1e-5)
    assert prior.mean_crossing == pytest.approx(1.0)
    assert prior.mean_target == pytest.approx(0.0)


def test_dataset_prior_ignores_its_input():
    """It must be blind by construction; otherwise it is not a prior."""
    prior = DatasetPrior().fit([fake_batch()])
    first = prior.predict(np.zeros((N_INPUT_CHANNELS, SIZE, SIZE)), np.zeros(8), None)
    second = prior.predict(np.ones((N_INPUT_CHANNELS, SIZE, SIZE)), np.ones(8), None)
    assert first.revealed_area_m2 == second.revealed_area_m2


def test_geometry_only_fits_and_predicts():
    baseline = GeometryOnly().fit([fake_batch(), fake_batch(area=9.0)])
    prediction = baseline.predict(
        np.zeros((N_INPUT_CHANNELS, SIZE, SIZE)), np.ones(8), None
    )
    assert np.isfinite(prediction.revealed_area_m2)
    assert 0.0 <= prediction.target_present <= 1.0


def test_current_observation_scales_with_unknown_area():
    baseline = CurrentObservationOnly().fit([fake_batch()])
    from frontierworld.data.tensors import CH_UNKNOWN

    small = np.zeros((N_INPUT_CHANNELS, SIZE, SIZE), dtype=np.float32)
    small[CH_UNKNOWN, :4, :4] = 1.0
    large = np.zeros((N_INPUT_CHANNELS, SIZE, SIZE), dtype=np.float32)
    large[CH_UNKNOWN] = 1.0

    assert (
        baseline.predict(large, np.zeros(8), None).revealed_area_m2
        > baseline.predict(small, np.zeros(8), None).revealed_area_m2
    )


def test_every_registered_baseline_fits_and_predicts():
    batches = [fake_batch()]
    for name, factory in BASELINES.items():
        baseline = factory().fit(batches)
        prediction = baseline.predict(
            np.zeros((N_INPUT_CHANNELS, SIZE, SIZE), dtype=np.float32), np.zeros(8), None
        )
        assert np.isfinite(prediction.revealed_area_m2), name


# -- metric helpers --------------------------------------------------------


def test_masked_iou_perfect_and_disjoint():
    prediction = np.zeros((10, 10))
    prediction[:5, :5] = 1.0
    valid = np.ones((10, 10), dtype=bool)

    assert masked_iou(prediction, prediction, valid) == pytest.approx(1.0)
    assert masked_iou(prediction, 1.0 - prediction, valid) == pytest.approx(0.0)


def test_masked_iou_respects_the_validity_mask():
    prediction = np.ones((10, 10))
    target = np.zeros((10, 10))
    target[:5] = 1.0

    valid = np.zeros((10, 10), dtype=bool)
    valid[:5] = True  # only the region where they agree
    assert masked_iou(prediction, target, valid) == pytest.approx(1.0)


def test_auc_is_one_for_a_perfect_ranking_and_nan_for_one_class():
    assert binary_auc(np.array([0.1, 0.9]), np.array([0, 1])) == pytest.approx(1.0)
    assert binary_auc(np.array([0.9, 0.1]), np.array([0, 1])) == pytest.approx(0.0)
    assert np.isnan(binary_auc(np.array([0.5, 0.7]), np.array([1, 1])))
