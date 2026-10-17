"""Live broker adapter stub — blocked by default via activation guards.

No live routing is implemented. Research modules must never import this module.
"""

from __future__ import annotations

import inspect

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.guards import assert_live_trading_allowed
from rl_trader.brokers.types import (
    Account,
    MarketClock,
    Order,
    OrderRequest,
    Position,
    Quote,
)
from rl_trader.config.settings import Settings
from rl_trader.risk.engine import RiskResult
from rl_trader.utils.boundaries import guard_live_import

_STUB_MSG = (
    "LiveBroker is unimplemented; live routing remains gated. "
    "Use PaperBroker for simulated trading. "
    "Use RL_TRADER_LIVE_BROKER_* credentials only when a real adapter exists "
    "(never paper keys)."
)


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
    """Live routing stub. Construction requires full activation; methods raise."""

    mode = "live"

    def __init__(self, settings: Settings) -> None:
        _enforce_caller_boundary()
        assert_live_trading_allowed(settings)
        self._settings = settings

    def connect(self) -> None:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        raise NotImplementedError(_STUB_MSG)

    def disconnect(self) -> None:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        raise NotImplementedError(_STUB_MSG)

    def get_account(self) -> Account:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        raise NotImplementedError(_STUB_MSG)

    def get_positions(self) -> list[Position]:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        raise NotImplementedError(_STUB_MSG)

    def get_open_orders(self) -> list[Order]:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        raise NotImplementedError(_STUB_MSG)

    def get_latest_quote(self, symbol: str) -> Quote:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        _ = symbol
        raise NotImplementedError(_STUB_MSG)

    def submit_order(self, request: OrderRequest) -> Order:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        _ = request
        raise NotImplementedError(_STUB_MSG)

    def cancel_order(self, order_id: str) -> Order:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        _ = order_id
        raise NotImplementedError(_STUB_MSG)

    def cancel_all_orders(self) -> list[Order]:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        raise NotImplementedError(_STUB_MSG)

    def get_order_status(self, order_id: str) -> Order:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        _ = order_id
        raise NotImplementedError(_STUB_MSG)

    def get_market_clock(self) -> MarketClock:
        """Refuse — live broker not implemented."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        raise NotImplementedError(_STUB_MSG)

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Refuse until a real live adapter is implemented (still gated)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        _ = risk_result
        raise NotImplementedError(_STUB_MSG)
