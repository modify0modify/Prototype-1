"""Integration: full paper-trading cycle with persistent ledger."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from rl_trader.brokers.base import BrokerAdapter
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Quote,
)
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.decisions import DecisionRecord, RiskDecision, RiskResult


def test_full_paper_trading_cycle(tmp_path: Path) -> None:
    """Connect → quote → buy → partial sell → cancel resting limit → reconnect."""
    data_dir = tmp_path / "paper_broker"
    clock_now = datetime(2024, 6, 3, 15, 30, tzinfo=UTC)

    def now_fn() -> datetime:
        return clock_now

    cfg = PaperBrokerConfig(
        data_dir=data_dir,
        initial_cash=50_000.0,
        commission_bps=5.0,
        spread_bps=2.0,
        slippage_bps=1.0,
        partial_fill_fraction=1.0,
        execution_delay_seconds=0.0,
        allow_fractional_shares=True,
        max_quote_age_seconds=120.0,
        enforce_market_hours=False,
        rng_seed=7,
    )
    broker: BrokerAdapter = PaperBroker(config=cfg, now_fn=now_fn)
    assert isinstance(broker, PaperBroker)

    broker.connect()
    assert broker.get_market_clock().is_open is True

    q = Quote(
        symbol="AAPL",
        bid=149.90,
        ask=150.10,
        last=150.0,
        timestamp=clock_now,
    )
    broker.set_quote(q)

    buy = broker.submit_order(
        OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=20.5,
            order_type=OrderType.MARKET,
            idempotency_key="cycle-buy",
        )
    )
    assert buy.status == OrderStatus.FILLED
    assert buy.filled_qty == pytest.approx(20.5)
    assert buy.commission > 0.0

    account_after_buy = broker.get_account()
    assert account_after_buy.cash < 50_000.0
    assert account_after_buy.equity == pytest.approx(
        account_after_buy.cash + 20.5 * 150.0, rel=1e-4
    )

    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].qty == pytest.approx(20.5)

    # Refresh quote and sell half.
    clock_now = clock_now + timedelta(seconds=1)
    broker.set_quote(
        Quote(symbol="AAPL", bid=151.0, ask=151.2, last=151.1, timestamp=clock_now)
    )
    sell = broker.submit_order(
        OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=10.0,
            order_type=OrderType.MARKET,
            idempotency_key="cycle-sell",
        )
    )
    assert sell.status == OrderStatus.FILLED

    # Resting limit that will not fill — then cancel.
    limit = broker.submit_order(
        OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=5.0,
            order_type=OrderType.LIMIT,
            limit_price=200.0,
            idempotency_key="cycle-limit",
        )
    )
    assert limit.status == OrderStatus.ACCEPTED
    open_orders = broker.get_open_orders()
    assert any(o.order_id == limit.order_id for o in open_orders)

    cancelled = broker.cancel_order(limit.order_id)
    assert cancelled.status == OrderStatus.CANCELLED
    assert broker.get_open_orders() == []

    # cancel_all on empty is fine
    assert broker.cancel_all_orders() == []

    status = broker.get_order_status(buy.order_id)
    assert status.status == OrderStatus.FILLED

    report = broker.reconcile()
    assert report["ok"] is True

    broker.disconnect()

    # Persistence: reconnect and see remaining long position + ledger.
    broker2 = PaperBroker(config=cfg, now_fn=now_fn)
    broker2.connect()
    pos2 = broker2.get_positions()
    assert len(pos2) == 1
    assert pos2[0].qty == pytest.approx(10.5)
    assert (data_dir / "account.json").is_file()
    assert (data_dir / "orders.json").is_file()
    assert (data_dir / "fills.jsonl").is_file()

    # Idempotent replay of the original buy key returns same order id.
    replay = broker2.submit_order(
        OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=20.5,
            idempotency_key="cycle-buy",
        )
    )
    assert replay.order_id == buy.order_id
    broker2.disconnect()


def test_submit_target_applies_costs(tmp_path: Path) -> None:
    clock = datetime(2024, 6, 3, 16, 0, tzinfo=UTC)
    broker = PaperBroker(
        config=PaperBrokerConfig(
            data_dir=tmp_path / "ledger",
            initial_cash=100_000.0,
            commission_bps=10.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            rng_seed=0,
        ),
        now_fn=lambda: clock,
    )
    broker.connect()
    broker.set_quotes(
        [
            Quote(symbol="AAA", bid=10.0, ask=10.0, last=10.0, timestamp=clock),
            Quote(symbol="BBB", bid=20.0, ask=20.0, last=20.0, timestamp=clock),
        ]
    )
    weights = np.array([0.5, 0.0], dtype=np.float64)
    portfolio = TargetPortfolio(weights=weights, as_of="2024-06-03")
    record = DecisionRecord(
        decision_status=RiskDecision.APPROVED,
        original_order=(0.5, 0.0),
        approved_order=(0.5, 0.0),
        violated_rules=(),
        reason="ok",
        timestamp=clock,
        configuration_version="test",
        model_identifier="unit",
        details={"symbols": ["AAA", "BBB"]},
    )
    result = RiskResult(
        decision=RiskDecision.APPROVED,
        portfolio=portfolio,
        reason="ok",
        record=record,
        details={"symbols": ["AAA", "BBB"]},
    )
    fill = broker.submit_target(result)
    assert fill.metadata["costs_applied"] is True
    assert fill.metadata["incomplete_paper_broker"] is False
    assert broker.get_account().cash < 100_000.0
    assert any(p.symbol == "AAA" for p in broker.get_positions())
    broker.disconnect()
