"""Paper execution orchestration (risk → paper broker).

Paper path stays gated via :class:`PaperBroker` until costs are implemented.
"""

from __future__ import annotations

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.paper import PaperBroker, PaperTradingDisabledError
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.engine import RiskEngine

__all__ = ["PaperExecutor", "PaperTradingDisabledError"]


class PaperExecutor:
    """Route a target portfolio through risk, then the paper broker."""

    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        broker: PaperBroker | None = None,
        *,
        allow_incomplete: bool = False,
    ) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self.broker = broker or PaperBroker(allow_incomplete=allow_incomplete)

    def execute(self, target: TargetPortfolio) -> BrokerFill:
        """Review with risk engine and submit to paper broker."""
        result = self.risk_engine.review(target)
        return self.broker.submit_target(result)
