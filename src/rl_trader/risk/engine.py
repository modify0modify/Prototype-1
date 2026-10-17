"""Deterministic risk engine — independent from the RL agent.

Approves, rejects, or resizes target portfolios before any execution adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from rl_trader.portfolio.targets import TargetPortfolio


class RiskDecision(StrEnum):
    """Risk engine outcome."""

    APPROVE = "approve"
    REJECT = "reject"
    RESIZE = "resize"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Hard limits applied to target portfolios."""

    max_gross_exposure: float = 1.0
    max_single_weight: float = 0.25
    min_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class RiskResult:
    """Approved (possibly resized) portfolio plus decision metadata."""

    decision: RiskDecision
    portfolio: TargetPortfolio | None
    reason: str
    details: dict[str, Any] | None = None


class RiskEngine:
    """Deterministic target-portfolio gate (stub rules)."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()

    def review(self, target: TargetPortfolio) -> RiskResult:
        """Approve / reject / resize a proposed target portfolio."""
        weights = np.asarray(target.weights, dtype=np.float64)
        if weights.size == 0 or not np.isfinite(weights).all():
            return RiskResult(
                decision=RiskDecision.REJECT,
                portfolio=None,
                reason="non-finite or empty weights",
            )

        clipped = np.clip(weights, self.limits.min_weight, self.limits.max_single_weight)
        gross = float(np.sum(np.abs(clipped)))
        if gross <= 0.0:
            return RiskResult(
                decision=RiskDecision.REJECT,
                portfolio=None,
                reason="zero gross exposure after clipping",
            )

        if gross > self.limits.max_gross_exposure:
            clipped = clipped * (self.limits.max_gross_exposure / gross)
            return RiskResult(
                decision=RiskDecision.RESIZE,
                portfolio=TargetPortfolio(weights=clipped, as_of=target.as_of),
                reason="gross exposure exceeded; resized",
            )

        if not np.allclose(clipped, weights):
            total = float(np.sum(clipped))
            if total > 0:
                clipped = clipped / total
            return RiskResult(
                decision=RiskDecision.RESIZE,
                portfolio=TargetPortfolio(weights=clipped, as_of=target.as_of),
                reason="single-name cap applied; resized",
            )

        return RiskResult(
            decision=RiskDecision.APPROVE,
            portfolio=target,
            reason="within limits",
        )
