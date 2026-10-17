"""Shared broker adapter interface.

Research logic must not import live adapters. Orchestration code selects
paper or live implementations behind this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from rl_trader.brokers.types import (
    Account,
    MarketClock,
    Order,
    OrderRequest,
    Position,
    Quote,
)
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.engine import RiskResult


@dataclass(frozen=True, slots=True)
class BrokerFill:
    """Normalized fill / acceptance record from a broker adapter."""

    portfolio: TargetPortfolio
    filled_at: datetime
    mode: str
    metadata: dict[str, Any]


@runtime_checkable
class BrokerAdapter(Protocol):
    """Shared interface for paper and live broker adapters.

    Order-level methods are the primary trading API. ``submit_target`` remains
    for the risk-engine path (approved weights → broker).
    """

    mode: str

    def connect(self) -> None:
        """Establish a session / load persistent state."""
        ...

    def disconnect(self) -> None:
        """Tear down the session and flush persistence."""
        ...

    def get_account(self) -> Account:
        """Return cash, equity, and buying power."""
        ...

    def get_positions(self) -> list[Position]:
        """Return open positions."""
        ...

    def get_open_orders(self) -> list[Order]:
        """Return working (non-terminal) orders."""
        ...

    def get_latest_quote(self, symbol: str) -> Quote:
        """Return the latest quote for ``symbol``."""
        ...

    def submit_order(self, request: OrderRequest) -> Order:
        """Submit a new order (honours idempotency keys)."""
        ...

    def cancel_order(self, order_id: str) -> Order:
        """Cancel one open order."""
        ...

    def cancel_all_orders(self) -> list[Order]:
        """Cancel all open orders."""
        ...

    def get_order_status(self, order_id: str) -> Order:
        """Return the current status of one order."""
        ...

    def get_market_clock(self) -> MarketClock:
        """Return exchange open/closed state."""
        ...

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Submit a risk-approved target portfolio for execution/simulation."""
        ...
