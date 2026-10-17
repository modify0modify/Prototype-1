"""Unit tests for PaperBroker edge cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
)


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _broker(tmp_path: Path, **kwargs: object) -> tuple[PaperBroker, _Clock]:
    clock = _Clock(datetime(2024, 6, 3, 15, 0, tzinfo=UTC))  # weekday, open UTC hour
    cfg_kwargs: dict[str, object] = {
        "data_dir": tmp_path / "ledger",
        "initial_cash": 10_000.0,
        "commission_bps": 10.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "execution_delay_seconds": 0.0,
        "max_quote_age_seconds": 30.0,
        "rng_seed": 1,
    }
    cfg_kwargs.update(kwargs)
    cfg = PaperBrokerConfig(**cfg_kwargs)  # type: ignore[arg-type]
    broker = PaperBroker(config=cfg, now_fn=clock)
    broker.connect()
    return broker, clock


def _fresh_quote(symbol: str, clock: _Clock, *, mid: float = 100.0) -> Quote:
    return Quote(
        symbol=symbol,
        bid=mid - 0.05,
        ask=mid + 0.05,
        last=mid,
        timestamp=clock.now,
    )


def test_insufficient_buying_power_rejects(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path)
    broker.set_quote(_fresh_quote("AAPL", clock, mid=100.0))
    order = broker.submit_order(
        OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=10_000.0)
    )
    assert order.status == OrderStatus.REJECTED
    assert order.reject_reason == "insufficient_buying_power"
    assert broker.get_account().cash == pytest.approx(10_000.0)


def test_stale_quote_rejects(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path, max_quote_age_seconds=5.0)
    broker.set_quote(
        Quote(
            symbol="AAPL",
            bid=99.0,
            ask=101.0,
            last=100.0,
            timestamp=clock.now - timedelta(seconds=60),
        )
    )
    order = broker.submit_order(
        OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=1.0)
    )
    assert order.status == OrderStatus.REJECTED
    assert order.reject_reason == "stale_quote"


def test_cancel_open_order(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path, execution_delay_seconds=10.0)
    broker.set_quote(_fresh_quote("MSFT", clock))
    order = broker.submit_order(
        OrderRequest(symbol="MSFT", side=OrderSide.BUY, qty=1.0)
    )
    assert order.status == OrderStatus.ACCEPTED
    cancelled = broker.cancel_order(order.order_id)
    assert cancelled.status == OrderStatus.CANCELLED
    assert broker.get_open_orders() == []


def test_idempotency_key_returns_same_order(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path)
    broker.set_quote(_fresh_quote("QQQ", clock))
    first = broker.submit_order(
        OrderRequest(
            symbol="QQQ",
            side=OrderSide.BUY,
            qty=2.0,
            idempotency_key="client-abc",
        )
    )
    second = broker.submit_order(
        OrderRequest(
            symbol="QQQ",
            side=OrderSide.BUY,
            qty=2.0,
            idempotency_key="client-abc",
        )
    )
    assert first.order_id == second.order_id
    assert first.status == OrderStatus.FILLED
    # Only one fill deducted cash once.
    cash = broker.get_account().cash
    assert cash < 10_000.0
    # Re-submit does not double-spend.
    third = broker.submit_order(
        OrderRequest(
            symbol="QQQ",
            side=OrderSide.BUY,
            qty=2.0,
            idempotency_key="client-abc",
        )
    )
    assert third.order_id == first.order_id
    assert broker.get_account().cash == pytest.approx(cash)


def test_limit_order_waits_for_cross(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path)
    broker.set_quote(_fresh_quote("SPY", clock, mid=100.0))
    order = broker.submit_order(
        OrderRequest(
            symbol="SPY",
            side=OrderSide.BUY,
            qty=1.0,
            order_type=OrderType.LIMIT,
            limit_price=90.0,
        )
    )
    assert order.status == OrderStatus.ACCEPTED
    # Cross the limit.
    broker.set_quote(_fresh_quote("SPY", clock, mid=89.0))
    filled = broker.advance_time()
    assert any(o.order_id == order.order_id for o in filled)
    assert broker.get_order_status(order.order_id).status == OrderStatus.FILLED


def test_fractional_shares_gate(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path, allow_fractional_shares=False)
    broker.set_quote(_fresh_quote("AAPL", clock))
    order = broker.submit_order(
        OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=1.5)
    )
    assert order.status == OrderStatus.REJECTED
    assert order.reject_reason == "fractional_shares_disabled"


def test_market_hours_enforcement(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path, enforce_market_hours=True)
    clock.now = datetime(2024, 6, 1, 3, 0, tzinfo=UTC)  # Saturday / closed hour
    broker.set_quote(_fresh_quote("AAPL", clock))
    order = broker.submit_order(
        OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=1.0)
    )
    assert order.status == OrderStatus.REJECTED
    assert order.reject_reason == "market_closed"


def test_commission_reduces_cash(tmp_path: Path) -> None:
    broker, clock = _broker(tmp_path, commission_bps=100.0, spread_bps=0.0, slippage_bps=0.0)
    broker.set_quote(
        Quote(symbol="AAPL", bid=100.0, ask=100.0, last=100.0, timestamp=clock.now)
    )
    order = broker.submit_order(
        OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=10.0)
    )
    assert order.status == OrderStatus.FILLED
    # 10 * 100 = 1000 notional + 1% commission = 10 → cash 8990
    assert broker.get_account().cash == pytest.approx(8_990.0)
    assert order.commission == pytest.approx(10.0)


def test_persistence_reload(tmp_path: Path) -> None:
    data_dir = tmp_path / "ledger"
    broker, clock = _broker(tmp_path)
    broker.set_quote(_fresh_quote("AAPL", clock))
    broker.submit_order(OrderRequest(symbol="AAPL", side=OrderSide.BUY, qty=3.0))
    broker.disconnect()

    cfg = PaperBrokerConfig(data_dir=data_dir, initial_cash=10_000.0, rng_seed=1)
    restored = PaperBroker(config=cfg, now_fn=clock)
    restored.connect()
    positions = restored.get_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == pytest.approx(3.0)
