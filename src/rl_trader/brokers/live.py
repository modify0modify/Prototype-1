"""Live broker adapter stub — blocked by default via activation guards.

Research modules must never import this module.
"""

from __future__ import annotations

import inspect

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.guards import assert_live_trading_allowed
from rl_trader.config.settings import Settings
from rl_trader.risk.engine import RiskResult
from rl_trader.utils.boundaries import guard_live_import


def _enforce_caller_boundary() -> None:
    """Reject imports/calls originating from research packages."""
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


class LiveBroker:
    """Live routing stub. Construction requires full activation."""

    mode = "live"

    def __init__(self, settings: Settings) -> None:
        _enforce_caller_boundary()
        assert_live_trading_allowed(settings)
        self._settings = settings

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Refuse until a real live adapter is implemented (still gated)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        _ = risk_result
        raise NotImplementedError(
            "LiveBroker.submit_target is a stub; live routing remains gated. "
            "Use RL_TRADER_LIVE_BROKER_* credentials only (never paper keys)."
        )
