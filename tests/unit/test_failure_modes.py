"""Failure-mode unit tests (synthetic / fixture labelled)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    load_checkpoint_for_features,
    save_checkpoint_metadata,
)
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import OrderRequest, OrderSide, OrderStatus, OrderType, Quote
from rl_trader.config.models import RiskSettings
from rl_trader.features.engineering import FeatureSpec
from rl_trader.paper_trading.market_data import (
    StaleMarketDataError,
    assert_snapshot_fresh,
)
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import RiskDecision, RiskEngine, RiskReviewContext
from rl_trader.validation.features import FeatureMismatchError


def _now() -> datetime:
    return datetime(2024, 6, 3, 15, 0, tzinfo=UTC)


def _broker(tmp_path: Path, **kwargs: object) -> PaperBroker:
    cfg_kwargs: dict[str, object] = {
        "data_dir": tmp_path / "ledger",
        "initial_cash": 10_000.0,
        "commission_bps": 1.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "max_quote_age_seconds": 30.0,
        "rng_seed": 2,
        "partial_fill_fraction": 1.0,
        "reject_probability": 0.0,
    }
    cfg_kwargs.update(kwargs)
    broker = PaperBroker(
        PaperBrokerConfig(**cfg_kwargs),  # type: ignore[arg-type]
        now_fn=_now,
    )
    broker.connect()
    return broker


def _risk_ctx(**overrides: object) -> RiskReviewContext:
    now = _now()
    base: dict[str, object] = {
        "symbols": ("AAPL",),
        "prices": np.array([100.0]),
        "prior_prices": np.array([100.0]),
        "price_timestamps": (now - timedelta(seconds=5),),
        "now": now,
        "portfolio_value": 10_000.0,
        "cash": 10_000.0,
        "current_weights": np.array([0.0]),
        "day_pnl_frac": 0.0,
        "week_pnl_frac": 0.0,
        "peak_nav": 10_000.0,
        "current_nav": 10_000.0,
        "realized_volatility": 0.1,
        "bid_ask_spread_bps": np.array([5.0]),
        "sector_by_symbol": {"AAPL": "tech"},
        "market_is_open": True,
        "orders_submitted_today": 0,
        "last_order_fingerprint": None,
        "pending_order_ids": (),
        "consecutive_losses": 0,
        "database_available": True,
        "account_confirmed": True,
        "trading_mode": "live",
        "model_id": "fail-model",
        "config_version": "fail-v1",
    }
    base.update(overrides)
    return RiskReviewContext(**base)  # type: ignore[arg-type]


def _risk_settings(**overrides: object) -> RiskSettings:
    base: dict[str, object] = {
        "max_position_size": 0.5,
        "max_portfolio_exposure": 1.0,
        "max_sector_exposure": 1.0,
        "min_cash_reserve": 0.0,
        "max_order_notional": 50_000.0,
        "max_daily_turnover": 2.0,
        "max_orders_per_day": 50,
        "daily_loss_limit": 0.05,
        "weekly_loss_limit": 0.10,
        "max_drawdown_limit": 0.20,
        "max_volatility_target": 1.0,
        "max_consecutive_losses": 10,
        "max_data_age_seconds": 60.0,
        "abnormal_price_jump": 0.10,
        "max_spread_bps": 50.0,
        "configuration_version": "fail-v1",
        "kill_switch_reset_phrase": "MANUAL_KILL_SWITCH_RESET",
    }
    base.update(overrides)
    return RiskSettings(**base)  # type: ignore[arg-type]


def test_missing_market_data() -> None:
    """Empty / incomplete snapshot is rejected as stale/missing."""
    from rl_trader.paper_trading.market_data import MarketSnapshot

    now = _now()
    broken = MarketSnapshot(
        symbols=(),
        prices=np.array([]),
        bids=np.array([]),
        asks=np.array([]),
        price_timestamps=(),
        observation=np.array([]),
        as_of=now,
        feature_as_of=now,
    )
    with pytest.raises(StaleMarketDataError, match="incomplete"):
        assert_snapshot_fresh(broken, max_age_seconds=60.0, now=now)


def test_malformed_price(tmp_path: Path) -> None:
    """Invalid bid/ask quote is refused by paper broker validation."""
    broker = _broker(tmp_path)
    broker.set_quote(
        Quote(symbol="AAA", bid=0.0, ask=-1.0, last=100.0, timestamp=_now())
    )
    order = broker.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=1.0,
            order_type=OrderType.MARKET,
            idempotency_key="bad-px",
        )
    )
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason == "invalid_quote"


def test_stale_quote(tmp_path: Path) -> None:
    broker = _broker(tmp_path, max_quote_age_seconds=5.0)
    old = _now() - timedelta(seconds=60)
    broker.set_quote(Quote(symbol="AAA", bid=99.0, ask=101.0, last=100.0, timestamp=old))
    order = broker.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=1.0,
            order_type=OrderType.MARKET,
            idempotency_key="stale",
        )
    )
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason is not None
    assert "stale" in order.reject_reason.lower() or "age" in order.reject_reason.lower()


def test_disconnected_broker(tmp_path: Path) -> None:
    broker = PaperBroker(
        PaperBrokerConfig(data_dir=tmp_path / "d", initial_cash=1_000.0),
        now_fn=_now,
    )
    # Not connected — submit must fail.
    with pytest.raises(RuntimeError, match="not connected|connect"):
        broker.submit_order(
            OrderRequest(
                symbol="AAA",
                side=OrderSide.BUY,
                qty=1.0,
                order_type=OrderType.MARKET,
            )
        )
    broker.connect()
    broker.disconnect()
    with pytest.raises(RuntimeError, match="not connected|connect"):
        broker.get_account()


def test_database_failure_fail_closed() -> None:
    eng = RiskEngine(settings=_risk_settings())
    result = eng.review(
        TargetPortfolio(weights=np.array([0.1])),
        context=_risk_ctx(database_available=False, trading_mode="live"),
    )
    assert result.decision is RiskDecision.REJECTED


def test_partial_fill(tmp_path: Path) -> None:
    broker = _broker(tmp_path, partial_fill_fraction=0.4, execution_delay_seconds=0.0)
    broker.set_quote(
        Quote(symbol="AAA", bid=99.9, ask=100.1, last=100.0, timestamp=_now())
    )
    order = broker.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=10.0,
            order_type=OrderType.MARKET,
            idempotency_key="partial",
        )
    )
    assert order.filled_qty == pytest.approx(4.0)
    assert order.status in {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED}
    if order.filled_qty < order.qty - 1e-9:
        assert order.status is OrderStatus.PARTIALLY_FILLED


def test_rejected_order(tmp_path: Path) -> None:
    broker = _broker(tmp_path, reject_probability=1.0)
    broker.set_quote(
        Quote(symbol="AAA", bid=99.9, ask=100.1, last=100.0, timestamp=_now())
    )
    order = broker.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=1.0,
            order_type=OrderType.MARKET,
            idempotency_key="rej",
        )
    )
    assert order.status is OrderStatus.REJECTED
    assert order.filled_qty == 0.0


def test_process_restart_idempotency(tmp_path: Path) -> None:
    """Same idempotency key after reconnect returns the original order."""
    data_dir = tmp_path / "ledger"
    now = _now()

    def make() -> PaperBroker:
        b = PaperBroker(
            PaperBrokerConfig(
                data_dir=data_dir,
                initial_cash=10_000.0,
                commission_bps=0.0,
                spread_bps=0.0,
                slippage_bps=0.0,
                rng_seed=3,
            ),
            now_fn=lambda: now,
        )
        b.connect()
        b.set_quote(
            Quote(symbol="AAA", bid=99.9, ask=100.1, last=100.0, timestamp=now)
        )
        return b

    b1 = make()
    o1 = b1.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=2.0,
            order_type=OrderType.MARKET,
            idempotency_key="restart-key",
        )
    )
    b1.disconnect()

    b2 = make()
    o2 = b2.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=2.0,
            order_type=OrderType.MARKET,
            idempotency_key="restart-key",
        )
    )
    assert o2.order_id == o1.order_id
    positions = b2.get_positions()
    total_qty = sum(p.qty for p in positions)
    assert total_qty == pytest.approx(2.0)


def test_corrupted_model(tmp_path: Path) -> None:
    """Corrupt checkpoint metadata JSON cannot load."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "metadata.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises((json.JSONDecodeError, ValueError, OSError, KeyError, TypeError)):
        load_checkpoint_for_features(
            ckpt,
            (FeatureSpec(name="returns_1d", lookback=1),),
        )


def test_mismatched_features(tmp_path: Path) -> None:
    specs = (FeatureSpec(name="returns_1d", lookback=1),)
    save_checkpoint_metadata(
        tmp_path,
        CheckpointMetadata(
            model_id="m",
            algorithm="stub",
            feature_specs=specs,
            training_period=TrainingPeriod(
                start=datetime(2023, 1, 1).date(),
                end=datetime(2023, 6, 1).date(),
            ),
            seed=1,
            package_version="0",
        ),
    )
    with pytest.raises(FeatureMismatchError):
        load_checkpoint_for_features(
            tmp_path,
            (FeatureSpec(name="vol_20d", lookback=20),),
        )


def test_invalid_risk_configuration() -> None:
    with pytest.raises(ValidationError):
        RiskSettings(max_position_size=0.9, max_portfolio_exposure=0.5)
    with pytest.raises(ValidationError):
        RiskSettings(daily_loss_limit=1.5)  # type: ignore[arg-type]
