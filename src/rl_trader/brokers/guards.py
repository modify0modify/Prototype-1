"""Live-trading guards — re-export config live gates for broker adapters.

Fail-fast validation runs **before** any live broker connection.
"""

from __future__ import annotations

from rl_trader.config.live_gates import (
    ActivationReport,
    GuardCheck,
    LiveTradingBlockedError,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)

__all__ = [
    "ActivationReport",
    "GuardCheck",
    "LiveTradingBlockedError",
    "assert_live_trading_allowed",
    "evaluate_live_activation",
    "is_live_trading_blocked",
]
