"""Shared broker domain types (paper and live)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class OrderSide(StrEnum):
    """Buy or sell."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(StrEnum):
    """Order lifecycle statuses."""

    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Account:
    """Broker account snapshot."""

    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"
    account_id: str = "paper-sim"


@dataclass(frozen=True, slots=True)
class Position:
    """Open position in one symbol."""

    symbol: str
    qty: float
    avg_price: float
    market_price: float
    market_value: float
    unrealized_pnl: float


@dataclass(frozen=True, slots=True)
class Quote:
    """Latest quote for a symbol."""

    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: datetime
    bid_size: float = 0.0
    ask_size: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketClock:
    """Exchange clock snapshot."""

    is_open: bool
    timestamp: datetime
    next_open: datetime | None = None
    next_close: datetime | None = None
    timezone: str = "UTC"


@dataclass(slots=True)
class OrderRequest:
    """Client order submission."""

    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    idempotency_key: str | None = None
    client_order_id: str | None = None
    time_in_force: str = "day"


@dataclass(slots=True)
class Order:
    """Persisted order with lifecycle fields."""

    order_id: str
    symbol: str
    side: OrderSide
    qty: float
    order_type: OrderType
    status: OrderStatus
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    limit_price: float | None = None
    idempotency_key: str | None = None
    client_order_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    reject_reason: str | None = None
    commission: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        payload = asdict(self)
        payload["side"] = str(self.side)
        payload["order_type"] = str(self.order_type)
        payload["status"] = str(self.status)
        if self.created_at is not None:
            payload["created_at"] = self.created_at.isoformat()
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Order:
        """Restore from JSON object."""
        created = data.get("created_at")
        updated = data.get("updated_at")
        return cls(
            order_id=str(data["order_id"]),
            symbol=str(data["symbol"]).upper(),
            side=OrderSide(str(data["side"])),
            qty=float(data["qty"]),
            order_type=OrderType(str(data["order_type"])),
            status=OrderStatus(str(data["status"])),
            filled_qty=float(data.get("filled_qty", 0.0)),
            avg_fill_price=(
                None
                if data.get("avg_fill_price") is None
                else float(data["avg_fill_price"])
            ),
            limit_price=(
                None if data.get("limit_price") is None else float(data["limit_price"])
            ),
            idempotency_key=data.get("idempotency_key"),
            client_order_id=data.get("client_order_id"),
            created_at=datetime.fromisoformat(created) if created else None,
            updated_at=datetime.fromisoformat(updated) if updated else None,
            reject_reason=data.get("reject_reason"),
            commission=float(data.get("commission", 0.0)),
            metadata=dict(data.get("metadata") or {}),
        )
