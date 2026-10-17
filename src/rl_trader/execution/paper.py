"""Paper execution orchestration (risk → paper broker)."""

from __future__ import annotations

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.paper import PaperBroker
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.engine import RiskEngine


class PaperExecutor:
    """Route a target portfolio through risk, then the paper broker."""

    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        broker: PaperBroker | None = None,
    ) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        self.broker = broker or PaperBroker()

    def execute(self, target: TargetPortfolio) -> BrokerFill:
        """Review with risk engine and submit to paper broker."""
        result = self.risk_engine.review(target)
        return self.broker.submit_target(result)
