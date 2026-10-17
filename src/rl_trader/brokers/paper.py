"""Paper broker adapter (simulated fills; uses paper credentials only)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from rl_trader.brokers.base import BrokerFill
from rl_trader.risk.engine import RiskResult


class PaperBroker:
    """Records risk-approved target portfolios as simulated fills."""

    mode = "paper"

    def __init__(self) -> None:
        self.fills: list[BrokerFill] = []

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Accept an approved/resized portfolio; reject otherwise."""
        if risk_result.portfolio is None:
            raise ValueError(f"cannot paper-trade rejected target: {risk_result.reason}")
        weights = np.asarray(risk_result.portfolio.weights)
        if not np.isfinite(weights).all():
            raise ValueError("paper broker refuses non-finite weights")
        fill = BrokerFill(
            portfolio=risk_result.portfolio,
            filled_at=datetime.now(UTC),
            mode=self.mode,
            metadata={"decision": risk_result.decision.value, "credential_profile": "paper"},
        )
        self.fills.append(fill)
        return fill
