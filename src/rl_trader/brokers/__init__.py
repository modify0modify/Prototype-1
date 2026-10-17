"""Broker adapters behind a shared interface.

Import ``paper`` freely from orchestration. Do **not** import ``live`` from
research packages — see ``rl_trader.utils.boundaries``.
"""

from rl_trader.brokers.base import BrokerAdapter, BrokerFill
from rl_trader.brokers.guards import (
    ActivationReport,
    LiveTradingBlockedError,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)
from rl_trader.brokers.paper import PaperBroker, PaperTradingDisabledError

__all__ = [
    "ActivationReport",
    "BrokerAdapter",
    "BrokerFill",
    "LiveTradingBlockedError",
    "PaperBroker",
    "PaperTradingDisabledError",
    "assert_live_trading_allowed",
    "evaluate_live_activation",
    "is_live_trading_blocked",
]
