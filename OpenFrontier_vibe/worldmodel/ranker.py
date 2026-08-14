"""
Predictive frontier utility Q_i (design sections 14-16, 49).

Replaces OpenFrontier's utility = (p^sharp * gain)^f / distance with the
additive, normalized combination:

    Q_i = l_obs*P_obs + l_wm*mu_wm + l_ig*IG + l_nov*N
          - l_cost*C - l_unc*sigma - l_visit*V - l_risk*R

All terms are normalized over the current candidate set before mixing.
Decision policies: argmax, UCB (mu + beta*sigma), risk-averse
(mu - beta*sigma); risk-averse is expressed through l_unc/beta.
Per-frontier term breakdowns are stashed in ft.features["wm_terms"] for
the interpretability logging of design section 46.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LAMBDAS = {
    "obs": 0.25,
    "wm": 0.40,
    "ig": 0.20,
    "cost": 0.10,
    "unc": 0.05,
    "nov": 0.0,
    "visit": 0.10,
    "risk": 0.0,
}


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


class FrontierRanker:
    def __init__(self, memory, cost_estimator, params: Optional[dict] = None):
        p = params or {}
        self.memory = memory
        self.cost_estimator = cost_estimator
        lambdas = dict(DEFAULT_LAMBDAS)
        lambdas.update(p.get("lambdas", {}))
        self.lambdas = {k: float(v) for k, v in lambdas.items()}
        self.policy = p.get("policy", "risk_averse")  # argmax | ucb | risk_averse
        self.beta = float(p.get("beta", 0.5))
        self.visit_radius = float(p.get("visit_radius", 1.5))
        # budget-aware urgency schedule: as the step budget depletes, shift
        # from exploration (info gain, UCB bonus) to greedy goal-seeking
        # (world-model relevance, path cost). SPL/timeout analysis showed the
        # agent searches broadly but slowly; late-episode wandering is wasted.
        self.urgency_enabled = bool(p.get("urgency_schedule", False))
        self.urgency_start = float(p.get("urgency_start", 0.4))  # progress where shift begins
        self._budget_progress = 0.0
        # frontiers with geodesic cost estimated this step (top-M candidates)
        self._geodesic_uids = set()

    def set_budget_progress(self, progress: float) -> None:
        """Fraction of the episode step budget consumed (0..1)."""
        self._budget_progress = float(np.clip(progress, 0.0, 1.0))

    def _urgency(self) -> float:
        """0 = explore freely, 1 = fully greedy (budget nearly gone)."""
        if not self.urgency_enabled:
            return 0.0
        span = max(1.0 - self.urgency_start, 1e-6)
        return float(np.clip((self._budget_progress - self.urgency_start) / span, 0.0, 1.0))

    def allow_geodesic_for(self, uids) -> None:
        self._geodesic_uids = set(uids)

    # ---------------- term extraction ----------------

    def _revisit_fraction(self, ft, robot_positions: np.ndarray) -> float:
        """Fraction-of-history proxy for P(predicted region already explored)."""
        if robot_positions.shape[0] == 0:
            return 0.0
        d = np.linalg.norm(robot_positions - np.asarray(ft.pos3d)[None, :], axis=1)
        n_close = int((d < self.visit_radius).sum())
        return float(min(n_close / 25.0, 1.0))

    def _risk(self, ft, manager) -> float:
        """P(frontier unreachable/unsafe) proxy from unreachable history."""
        risk = 0.0
        for pos in getattr(manager, "_unreachable_positions", []):
            if np.linalg.norm(np.asarray(ft.pos3d) - np.asarray(pos)) < 1.0:
                risk = 1.0
                break
        return risk

    # ---------------- main entry ----------------

    def update_utilities(
        self,
        frontiers: List,
        current_pose: np.ndarray,
        manager,
        goal: str,
    ) -> None:
        """Compute Q_i for every valid frontier and write ft.utility.

        Called by FrontierManager.update_utility via the external hook, so
        object frontiers keep their dominate-everything behavior.
        """
        current_pos = np.asarray(current_pose[:3, 3], dtype=float).reshape(3)

        regular = []
        for ft in frontiers:
            if ft.is_object:
                d = max(float(np.linalg.norm(np.asarray(ft.pos3d) - current_pos)), 1e-6)
                ft.utility = 1e20 / d
            else:
                regular.append(ft)

        if not regular:
            return

        robot_positions = (
            np.stack([p[:3, 3] for p in manager.robot_poses.values()], axis=0)
            if manager.robot_poses
            else np.zeros((0, 3))
        )

        n = len(regular)
        p_obs = np.zeros(n)
        wm_mu = np.zeros(n)
        wm_sigma = np.zeros(n)
        has_wm = np.zeros(n, dtype=bool)
        ig = np.zeros(n)
        cost = np.zeros(n)
        nov = np.zeros(n)
        visit = np.zeros(n)
        risk = np.zeros(n)

        for i, ft in enumerate(regular):
            p_obs[i] = float(ft.probability if ft.probability is not None else 0.5)
            gain = ft.u_gain if ft.u_gain is not None else (ft.gain or 0.0)
            ig[i] = np.log1p(max(float(gain), 0.0))

            uid = ft.features.get("uid")
            gs = self.memory.get_goal_score(uid, goal) if uid else None
            if gs is not None:
                wm_mu[i] = gs.value
                wm_sigma[i] = gs.spread
                has_wm[i] = True
            else:
                wm_mu[i] = 0.5  # uninformative prior
                wm_sigma[i] = 0.0

            cost[i] = self.cost_estimator.cost(
                ft, current_pose, allow_geodesic=(uid in self._geodesic_uids)
            )
            visit[i] = self._revisit_fraction(ft, robot_positions)
            nov[i] = 1.0 - visit[i]
            risk[i] = self._risk(ft, manager)

        ig_n = _minmax(ig)
        cost_n = _minmax(cost)

        l = dict(self.lambdas)
        beta = self.beta
        u = self._urgency()
        if u > 0:
            # deplete exploration terms, boost goal-seeking and proximity
            l["ig"] *= 1.0 - 0.75 * u
            l["nov"] *= 1.0 - u
            l["wm"] *= 1.0 + 0.5 * u
            l["obs"] *= 1.0 + 0.5 * u
            l["cost"] *= 1.0 + 1.0 * u
            beta *= 1.0 - u  # UCB optimism fades with the clock

        mu_eff = wm_mu.copy()
        if self.policy == "ucb":
            mu_eff = wm_mu + beta * wm_sigma
        elif self.policy == "risk_averse":
            mu_eff = wm_mu - beta * wm_sigma

        q = (
            l["obs"] * p_obs
            + l["wm"] * mu_eff
            + l["ig"] * ig_n
            + l["nov"] * nov
            - l["cost"] * cost_n
            - l["unc"] * wm_sigma
            - l["visit"] * visit
            - l["risk"] * risk
        )

        for i, ft in enumerate(regular):
            ft.utility = float(q[i])
            ft.features["wm_terms"] = {
                "p_obs": float(p_obs[i]),
                "wm_mu": float(wm_mu[i]) if has_wm[i] else None,
                "wm_sigma": float(wm_sigma[i]) if has_wm[i] else None,
                "info_gain_norm": float(ig_n[i]),
                "cost_m": float(cost[i]),
                "cost_norm": float(cost_n[i]),
                "novelty": float(nov[i]),
                "revisit": float(visit[i]),
                "risk": float(risk[i]),
                "Q": float(q[i]),
            }

    def ambiguity(self, frontiers: List) -> float:
        """Gap between the two best utilities (section 21, condition 3)."""
        utils = sorted(
            (
                float(ft.utility)
                for ft in frontiers
                if ft.utility is not None and np.isfinite(ft.utility) and not ft.is_object
            ),
            reverse=True,
        )
        if len(utils) < 2:
            return float("inf")
        return utils[0] - utils[1]
