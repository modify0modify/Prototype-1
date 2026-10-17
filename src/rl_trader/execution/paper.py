"""Paper execution orchestration (risk → paper broker)."""

from __future__ import annotations

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig, PaperTradingDisabledError
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.decisions import RiskDecision
from rl_trader.risk.engine import RiskEngine

__all__ = ["PaperExecutor", "PaperTradingDisabledError"]


class PaperExecutor:
    """Route a target portfolio through risk, then the paper broker."""

    def __init__(
        self,
        risk_engine: RiskEngine | None = None,
        broker: PaperBroker | None = None,
        *,
        config: PaperBrokerConfig | None = None,
        allow_incomplete: bool | None = None,
        auto_connect: bool = True,
    ) -> None:
        self.risk_engine = risk_engine or RiskEngine()
        if broker is not None:
            self.broker = broker
        else:
            self.broker = PaperBroker(config=config, allow_incomplete=allow_incomplete)
        if auto_connect and not self.broker.connected:
            self.broker.connect()

    def execute(self, target: TargetPortfolio) -> BrokerFill:
        """Review with risk engine and submit to paper broker.

        Rejected risk decisions never reach the broker.
        """
        result = self.risk_engine.review(target)
        if result.decision is RiskDecision.REJECTED or result.portfolio is None:
            raise ValueError(f"risk rejected target — broker not called: {result.reason}")
        return self.broker.submit_target(result)
