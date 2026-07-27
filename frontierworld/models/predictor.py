"""Deterministic structured revelation predictor.

    Y_hat = F_theta(M_local, E(omega_i), E(g))

A compact encoder-decoder over the frontier-centred window, deliberately built
before Cosmos 3. If a video foundation model later underperforms, the failure
has to be attributable to the model rather than to the dataset or the
representation, and that is only possible with a working structured baseline to
compare against.

The action option is injected via FiLM at the bottleneck rather than
concatenated as extra channels. Candidates at one decision state share almost
all of their spatial input -- the map is the same, only the frontier and the
option differ -- so the conditioning has to modulate the whole feature map, not
sit in a corner of it. A model that ignores the option would rank every
candidate identically, which is exactly the failure the Phase 8 debug
progression checks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from frontierworld.data.tensors import N_INPUT_CHANNELS, N_TARGET_CHANNELS


@dataclass(frozen=True)
class PredictorConfig:
    """FROZEN as of Phase 8.

    The implementation and optimisation gates pass; empirical generalisation is
    still pending the held-out dataset. Tuning width/depth against validation
    before that table exists would be selecting an architecture on the same
    scenes used to claim generalisation. Retune only after the held-out
    evaluation has been run once.
    """

    input_channels: int = N_INPUT_CHANNELS
    target_channels: int = N_TARGET_CHANNELS
    option_dim: int = 8
    goal_vocab: int = 8  # HM3D ObjectNav has 6 goal categories; spare room
    width: int = 48
    depth: int = 3
    dropout: float = 0.0


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(),
    )


class FiLM(nn.Module):
    """Feature-wise modulation from the option and goal embedding."""

    def __init__(self, conditioning_dim: int, channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(conditioning_dim, channels * 2)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(conditioning).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + scale) + shift


class RevelationPredictor(nn.Module):
    """Predicts what crossing a frontier would reveal."""

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        width = self.config.width

        self.goal_embedding = nn.Embedding(self.config.goal_vocab, 16)
        self.option_encoder = nn.Sequential(
            nn.Linear(self.config.option_dim, 32), nn.SiLU(), nn.Linear(32, 32)
        )
        conditioning_dim = 32 + 16

        self.stem = conv_block(self.config.input_channels, width)
        self.down1 = conv_block(width, width * 2, stride=2)
        self.down2 = conv_block(width * 2, width * 4, stride=2)

        self.film1 = FiLM(conditioning_dim, width * 2)
        self.film2 = FiLM(conditioning_dim, width * 4)

        self.bottleneck = nn.Sequential(
            *[conv_block(width * 4, width * 4) for _ in range(self.config.depth)]
        )

        self.up2 = conv_block(width * 4 + width * 2, width * 2)
        self.up1 = conv_block(width * 2 + width, width)
        self.head_spatial = nn.Conv2d(width, self.config.target_channels, 1)

        # Scalar heads read the pooled bottleneck: target presence and crossing
        # success are properties of the whole revelation, not of a pixel.
        self.head_scalar = nn.Sequential(
            nn.Linear(width * 4 + conditioning_dim, 64),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(64, 3),  # target presence, crossing success, revealed area
        )

    def forward(
        self,
        inputs: torch.Tensor,  # (B, C, H, W)
        option: torch.Tensor,  # (B, option_dim)
        goal: torch.Tensor,  # (B,) long
    ) -> dict[str, torch.Tensor]:
        conditioning = torch.cat(
            [self.option_encoder(option), self.goal_embedding(goal)], dim=-1
        )

        x0 = self.stem(inputs)
        x1 = self.film1(self.down1(x0), conditioning)
        x2 = self.film2(self.down2(x1), conditioning)
        x2 = self.bottleneck(x2)

        pooled = torch.cat([x2.mean(dim=(2, 3)), conditioning], dim=-1)
        scalars = self.head_scalar(pooled)

        y = F.interpolate(x2, size=x1.shape[-2:], mode="nearest")
        y = self.up2(torch.cat([y, x1], dim=1))
        y = F.interpolate(y, size=x0.shape[-2:], mode="nearest")
        y = self.up1(torch.cat([y, x0], dim=1))
        spatial = self.head_spatial(y)

        return {
            "revealed_logits": spatial,  # (B, T, H, W)
            "target_logit": scalars[:, 0],
            "crossing_logit": scalars[:, 1],
            "area": F.softplus(scalars[:, 2]),
        }


GOAL_CATEGORIES = ["chair", "bed", "plant", "toilet", "tv_monitor", "sofa"]


def goal_index(goal: str | None) -> int:
    """Map a goal category to an embedding index; unknown goals share a slot."""
    if goal is None:
        return len(GOAL_CATEGORIES)
    try:
        return GOAL_CATEGORIES.index(goal)
    except ValueError:
        return len(GOAL_CATEGORIES)


def revelation_loss(
    prediction: dict[str, torch.Tensor],
    targets: torch.Tensor,  # (B, T, H, W)
    target_valid: torch.Tensor,  # (B, H, W) bool
    scalars: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L = w_occ L_occ + w_sem L_sem + w_goal L_goal + w_cross L_cross + w_area L_area.

    Spatial losses are masked by `target_valid`: cells outside the global map
    carry no ground truth, and averaging over them would let the model reduce
    the loss by being confident about regions nothing was ever observed in.

    KNOWN DEFECT (v0, frozen; fix belongs in v1). Both occupancy channels are
    averaged against one full-window denominator. Occupied cells are NOT rare
    conditional on revelation -- p(occupied | revealed) ~ 0.34 -- but they are
    rare under this denominator, p(revealed and occupied) ~ 0.038. The
    imbalance is therefore created by the loss formulation, not by the dataset,
    and near-zero prediction minimises it: measured recall 0.049 at precision
    0.458 on validation.

    v1 should train a genuinely factorised objective instead:
        p(free)     = p(reveal) * p(free | reveal)
        p(occupied) = p(reveal) * p(occupied | reveal)
    with the reveal head over the whole window and the free/occupied softmax
    only over ground-truth revealed cells, optionally plus a boundary or
    distance-transform term for thin-wall alignment.
    """
    weights = weights or {
        "occ": 1.0, "sem": 0.5, "goal": 1.0, "cross": 0.5, "area": 0.1
    }

    mask = target_valid.unsqueeze(1).float()
    logits = prediction["revealed_logits"]

    per_cell = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denominator = mask.sum().clamp(min=1.0)
    occupancy = (per_cell[:, :2] * mask).sum() / (denominator * 2)
    semantic = (per_cell[:, 2:] * mask).sum() / denominator

    goal_loss = F.binary_cross_entropy_with_logits(
        prediction["target_logit"], scalars["target_present"]
    )
    crossing_loss = F.binary_cross_entropy_with_logits(
        prediction["crossing_logit"], scalars["crossing_success"]
    )
    # Areas span 0-40 m2; a log target keeps the gradient from being dominated
    # by the few very large revelations.
    area_loss = F.smooth_l1_loss(
        torch.log1p(prediction["area"]), torch.log1p(scalars["revealed_area_m2"])
    )

    total = (
        weights["occ"] * occupancy
        + weights["sem"] * semantic
        + weights["goal"] * goal_loss
        + weights["cross"] * crossing_loss
        + weights["area"] * area_loss
    )
    return total, {
        "loss": float(total.detach()),
        "occ": float(occupancy.detach()),
        "sem": float(semantic.detach()),
        "goal": float(goal_loss.detach()),
        "cross": float(crossing_loss.detach()),
        "area": float(area_loss.detach()),
    }
