"""Live execution orchestration stub.

Research modules must never import this module.
"""

from __future__ import annotations

import inspect

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.guards import assert_live_trading_allowed
from rl_trader.brokers.live import LiveBroker
from rl_trader.config.settings import Settings
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.engine import RiskEngine
from rl_trader.utils.boundaries import guard_live_import


def _enforce_caller_boundary() -> None:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        while caller is not None:
            module_name = caller.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rl_trader."):
                guard_live_import(module_name)
            caller = caller.f_back
    finally:
        del frame


class LiveExecutor:
    """Risk → live broker path (gated; stubbed)."""

    def __init__(self, settings: Settings, risk_engine: RiskEngine | None = None) -> None:
        _enforce_caller_boundary()
        assert_live_trading_allowed(settings)
        self.settings = settings
        self.risk_engine = risk_engine or RiskEngine()
        self.broker = LiveBroker(settings)

    def execute(self, target: TargetPortfolio) -> BrokerFill:
        """Review with risk engine and submit to live broker (stub)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self.settings)
        result = self.risk_engine.review(target)
        return self.broker.submit_target(result)
