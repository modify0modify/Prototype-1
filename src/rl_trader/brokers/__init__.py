"""Broker adapters behind a shared interface.

Import ``paper`` freely from orchestration. Do **not** import ``live`` from
research packages — see ``rl_trader.utils.boundaries``.

LiveBroker is intentionally not re-exported here so research code cannot
accidentally pull it through the package root.
"""

from rl_trader.brokers.base import BrokerAdapter, BrokerFill
from rl_trader.brokers.guards import (
    ActivationReport,
    LiveTradingBlockedError,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)
from rl_trader.brokers.paper import (
    PaperBroker,
    PaperBrokerConfig,
    PaperTradingDisabledError,
)
from rl_trader.brokers.types import (
    Account,
    MarketClock,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)

__all__ = [
    "Account",
    "ActivationReport",
    "BrokerAdapter",
    "BrokerFill",
    "LiveTradingBlockedError",
    "MarketClock",
    "Order",
    "OrderRequest",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperBroker",
    "PaperBrokerConfig",
    "PaperTradingDisabledError",
    "Position",
    "Quote",
    "assert_live_trading_allowed",
    "evaluate_live_activation",
    "is_live_trading_blocked",
]
