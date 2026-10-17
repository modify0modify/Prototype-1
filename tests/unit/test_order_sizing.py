"""Unit: order sizing via risk-engine notional / cash-reserve rules."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from rl_trader.config.models import RiskSettings
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import RiskDecision, RiskEngine, RiskReviewContext


def _ctx(**overrides: object) -> RiskReviewContext:
    now = datetime(2024, 6, 3, 15, 30, tzinfo=UTC)
    base: dict[str, object] = {
        "symbols": ("AAPL", "MSFT"),
        "prices": np.array([100.0, 200.0]),
        "prior_prices": np.array([100.0, 200.0]),
        "price_timestamps": (now - timedelta(seconds=5), now - timedelta(seconds=5)),
        "now": now,
        "portfolio_value": 100_000.0,
        "cash": 50_000.0,
        "current_weights": np.array([0.0, 0.0]),
        "day_pnl_frac": 0.0,
        "week_pnl_frac": 0.0,
        "peak_nav": 100_000.0,
        "current_nav": 100_000.0,
        "realized_volatility": 0.10,
        "bid_ask_spread_bps": np.array([5.0, 5.0]),
        "sector_by_symbol": {"AAPL": "tech", "MSFT": "tech"},
        "market_is_open": True,
        "orders_submitted_today": 0,
        "last_order_fingerprint": None,
        "pending_order_ids": (),
        "consecutive_losses": 0,
        "database_available": True,
        "account_confirmed": True,
        "trading_mode": "research",
        "model_id": "sizing-model",
        "config_version": "size-v1",
    }
    base.update(overrides)
    return RiskReviewContext(**base)  # type: ignore[arg-type]


def _settings(**overrides: object) -> RiskSettings:
    base: dict[str, object] = {
        "max_position_size": 0.50,
        "max_portfolio_exposure": 1.0,
        "max_sector_exposure": 1.0,
        "min_cash_reserve": 0.0,
        "max_order_notional": 5_000.0,
        "max_daily_turnover": 2.0,
        "max_orders_per_day": 50,
        "daily_loss_limit": 0.50,
        "weekly_loss_limit": 0.50,
        "max_drawdown_limit": 0.90,
        "max_volatility_target": 1.0,
        "max_consecutive_losses": 50,
        "max_data_age_seconds": 300.0,
        "abnormal_price_jump": 0.50,
        "max_spread_bps": 500.0,
        "configuration_version": "size-v1",
        "kill_switch_reset_phrase": "MANUAL_KILL_SWITCH_RESET",
    }
    base.update(overrides)
    return RiskSettings(**base)  # type: ignore[arg-type]


def test_order_notional_resizes_to_cap() -> None:
    """Oversized target weight is resized so order notional ≤ max_order_notional."""
    eng = RiskEngine(settings=_settings(max_order_notional=5_000.0))
    # 20% of 100k = 20k notional > 5k cap → must resize
    result = eng.review(TargetPortfolio(weights=np.array([0.20, 0.0])), context=_ctx())
    assert result.decision in {RiskDecision.RESIZED, RiskDecision.APPROVED}
    assert result.portfolio is not None
    approved = np.asarray(result.portfolio.weights)
    # Delta notional for first asset ≈ |w| * portfolio_value ≤ 5000 (+eps)
    notional = abs(float(approved[0])) * 100_000.0
    assert notional <= 5_000.0 + 1e-6


def test_cash_reserve_scales_order() -> None:
    """min_cash_reserve forces exposure below 1 - reserve."""
    eng = RiskEngine(
        settings=_settings(
            max_order_notional=1_000_000.0,
            min_cash_reserve=0.25,
            max_position_size=0.80,
            max_portfolio_exposure=1.0,
            max_sector_exposure=1.0,
        )
    )
    result = eng.review(
        TargetPortfolio(weights=np.array([0.60, 0.30])),
        context=_ctx(cash=100_000.0),
    )
    assert result.decision in {RiskDecision.RESIZED, RiskDecision.APPROVED}
    assert result.portfolio is not None
    exposure = float(np.sum(result.portfolio.weights))
    assert exposure <= 0.75 + 1e-6
