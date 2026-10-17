"""Paper-trading adapter stubs.

Simulated fills only. Default runtime path. Never contacts a live broker.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from rl_trading.risk.engine import RiskResult
from rl_trading.rl.agents import TargetPortfolio


@dataclass(frozen=True, slots=True)
class PaperFill:
    """Simulated fill record."""

    portfolio: TargetPortfolio
    filled_at: datetime
    metadata: dict[str, Any]


class PaperBroker:
    """Records risk-approved target portfolios as simulated fills."""

    def __init__(self) -> None:
        self.fills: list[PaperFill] = []

    def submit_target(self, risk_result: RiskResult) -> PaperFill:
        """Accept an approved/resized portfolio; reject otherwise."""
        if risk_result.portfolio is None:
            raise ValueError(f"cannot paper-trade rejected target: {risk_result.reason}")
        weights = np.asarray(risk_result.portfolio.weights)
        if not np.isfinite(weights).all():
            raise ValueError("paper broker refuses non-finite weights")
        fill = PaperFill(
            portfolio=risk_result.portfolio,
            filled_at=datetime.now(UTC),
            metadata={"mode": "paper", "decision": risk_result.decision.value},
        )
        self.fills.append(fill)
        return fill
