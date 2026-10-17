"""Paper broker adapter — gated until research cost model is applied.

The historical stub recorded risk-approved weights as fills with **zero**
commission/spread/slippage. That is unsafe for evaluating live readiness.
Paper trading remains disabled unless ``allow_incomplete=True`` is passed
explicitly for demos.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from rl_trader.brokers.base import BrokerFill
from rl_trader.risk.engine import RiskResult


class PaperTradingDisabledError(RuntimeError):
    """Raised when paper trading is attempted without cost-model readiness."""


class PaperBroker:
    """Records risk-approved target portfolios as simulated fills.

    By default construction fails closed — paper path must not silently ignore
    transaction costs used by the research ``PortfolioEnv``.
    """

    mode = "paper"

    def __init__(self, *, allow_incomplete: bool = False) -> None:
        if not allow_incomplete:
            raise PaperTradingDisabledError(
                "Paper broker is gated: fills do not apply the research cost / "
                "execution model (commission, spread, slippage, impact). "
                "Do not enable paper trading until costs are wired. "
                "Pass allow_incomplete=True only for explicit stub demos."
            )
        self.fills: list[BrokerFill] = []
        self.incomplete = True

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
            metadata={
                "decision": risk_result.decision.value,
                "credential_profile": "paper",
                "costs_applied": False,
                "incomplete_paper_broker": True,
            },
        )
        self.fills.append(fill)
        return fill
