"""Unit tests for every deterministic risk-engine rule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from rl_trader.config.models import RiskSettings
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import (
    KillSwitch,
    RiskDecision,
    RiskEngine,
    RiskReviewContext,
)
from rl_trader.utils.boundaries import (
    assert_risk_has_no_agent_imports,
    scan_file_for_risk_agent_imports,
)


def _now() -> datetime:
    return datetime(2024, 6, 3, 15, 30, tzinfo=UTC)


def _ctx(**overrides: object) -> RiskReviewContext:
    now = _now()
    base: dict[str, object] = {
        "symbols": ("AAPL", "MSFT"),
        "prices": np.array([100.0, 200.0]),
        "prior_prices": np.array([100.0, 200.0]),
        "price_timestamps": (now - timedelta(seconds=10), now - timedelta(seconds=10)),
        "now": now,
        "portfolio_value": 100_000.0,
        "cash": 10_000.0,
        "current_weights": np.array([0.2, 0.2]),
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
        "pending_order_ids": ("p1",),
        "consecutive_losses": 0,
        "database_available": True,
        "account_confirmed": True,
        "trading_mode": "research",
        "model_id": "unit-model",
        "config_version": "risk-v1-test",
    }
    base.update(overrides)
    return RiskReviewContext(**base)  # type: ignore[arg-type]


def _settings(**overrides: object) -> RiskSettings:
    base: dict[str, object] = {
        "max_position_size": 0.25,
        "max_portfolio_exposure": 0.80,
        "max_sector_exposure": 0.40,
        "min_cash_reserve": 0.10,
        "max_order_notional": 20_000.0,
        "max_daily_turnover": 0.50,
        "max_orders_per_day": 5,
        "daily_loss_limit": 0.05,
        "weekly_loss_limit": 0.10,
        "max_drawdown_limit": 0.20,
        "max_volatility_target": 0.30,
        "max_consecutive_losses": 3,
        "max_data_age_seconds": 60.0,
        "abnormal_price_jump": 0.10,
        "max_spread_bps": 20.0,
        "configuration_version": "risk-v1-test",
        "kill_switch_reset_phrase": "MANUAL_KILL_SWITCH_RESET",
    }
    base.update(overrides)
    return RiskSettings(**base)  # type: ignore[arg-type]


def _engine(**overrides: object) -> RiskEngine:
    return RiskEngine(settings=_settings(**overrides))


def _target(w0: float = 0.2, w1: float = 0.2) -> TargetPortfolio:
    return TargetPortfolio(weights=np.array([w0, w1], dtype=np.float64))


def test_approved_within_limits() -> None:
    eng = _engine()
    result = eng.review(_target(0.2, 0.2), _ctx())
    assert result.decision is RiskDecision.APPROVED
    assert result.record.decision_status is RiskDecision.APPROVED
    assert result.portfolio is not None
    assert result.record.model_identifier == "unit-model"
    assert result.record.configuration_version == "risk-v1-test"
    assert result.record.original_order == (0.2, 0.2)
    assert result.record.approved_order is not None


def test_maximum_position_weight_resizes() -> None:
    eng = _engine(max_position_size=0.25)
    result = eng.review(_target(0.9, 0.05), _ctx())
    assert result.decision is RiskDecision.RESIZED
    assert "maximum_position_weight" in result.record.violated_rules
    assert result.portfolio is not None
    assert float(np.max(result.portfolio.weights)) <= 0.25 + 1e-9


def test_maximum_portfolio_exposure_resizes() -> None:
    eng = _engine(max_portfolio_exposure=0.5, max_position_size=0.4, min_cash_reserve=0.0)
    result = eng.review(_target(0.4, 0.4), _ctx())
    assert result.decision is RiskDecision.RESIZED
    assert "maximum_portfolio_exposure" in result.record.violated_rules
    assert float(np.sum(result.portfolio.weights)) <= 0.5 + 1e-9  # type: ignore[union-attr]


def test_maximum_sector_exposure_resizes() -> None:
    eng = _engine(
        max_sector_exposure=0.30,
        max_portfolio_exposure=1.0,
        max_position_size=0.5,
        min_cash_reserve=0.0,
        max_daily_turnover=2.0,
        max_order_notional=1e9,
    )
    result = eng.review(_target(0.25, 0.25), _ctx())
    assert result.decision is RiskDecision.RESIZED
    assert "maximum_sector_exposure" in result.record.violated_rules


def test_maximum_order_notional_resizes() -> None:
    eng = _engine(
        max_order_notional=1_000.0,
        max_daily_turnover=2.0,
        max_portfolio_exposure=1.0,
        min_cash_reserve=0.0,
        max_sector_exposure=1.0,
        max_position_size=0.5,
    )
    # Large rebalance vs current 0.2/0.2
    result = eng.review(_target(0.5, 0.0), _ctx(portfolio_value=100_000.0))
    assert result.decision is RiskDecision.RESIZED
    assert "maximum_order_notional" in result.record.violated_rules


def test_maximum_daily_turnover_resizes() -> None:
    eng = _engine(
        max_daily_turnover=0.05,
        max_order_notional=1e9,
        max_sector_exposure=1.0,
        max_portfolio_exposure=1.0,
        min_cash_reserve=0.0,
        max_position_size=0.5,
    )
    result = eng.review(_target(0.5, 0.5), _ctx(current_weights=np.array([0.0, 0.0])))
    assert result.decision is RiskDecision.RESIZED
    assert "maximum_daily_turnover" in result.record.violated_rules


def test_minimum_cash_reserve_resizes() -> None:
    eng = _engine(
        min_cash_reserve=0.20,
        max_portfolio_exposure=1.0,
        max_sector_exposure=1.0,
        max_daily_turnover=2.0,
        max_order_notional=1e9,
        max_position_size=0.5,
    )
    result = eng.review(_target(0.5, 0.5), _ctx())
    assert result.decision is RiskDecision.RESIZED
    # Exposure cap uses min(max_portfolio_exposure, 1-min_cash); either rule may fire.
    assert (
        "minimum_cash_reserve" in result.record.violated_rules
        or "maximum_portfolio_exposure" in result.record.violated_rules
    )
    assert 1.0 - float(np.sum(result.portfolio.weights)) >= 0.20 - 1e-9  # type: ignore[union-attr]


def test_maximum_daily_loss_trips_kill_switch() -> None:
    eng = _engine(daily_loss_limit=0.05)
    result = eng.review(_target(), _ctx(day_pnl_frac=-0.06))
    assert result.decision is RiskDecision.REJECTED
    assert eng.kill_switch.is_active
    assert result.record.cancel_pending_orders is True
    assert result.record.pending_order_ids_cancelled == ("p1",)
    assert "maximum_daily_loss" in result.record.violated_rules
    # Subsequent order still blocked until manual reset
    blocked = eng.review(_target(), _ctx(day_pnl_frac=0.0))
    assert blocked.decision is RiskDecision.REJECTED
    assert "kill_switch" in blocked.record.violated_rules


def test_maximum_weekly_loss_trips_kill_switch() -> None:
    eng = _engine(weekly_loss_limit=0.08)
    result = eng.review(_target(), _ctx(week_pnl_frac=-0.09))
    assert result.decision is RiskDecision.REJECTED
    assert eng.kill_switch.is_active
    assert "maximum_weekly_loss" in result.record.violated_rules


def test_maximum_portfolio_drawdown_trips_kill_switch() -> None:
    eng = _engine(max_drawdown_limit=0.10)
    result = eng.review(
        _target(),
        _ctx(peak_nav=100_000.0, current_nav=85_000.0),
    )
    assert result.decision is RiskDecision.REJECTED
    assert eng.kill_switch.is_active
    assert "maximum_portfolio_drawdown" in result.record.violated_rules


def test_maximum_volatility_target_rejects() -> None:
    eng = _engine(max_volatility_target=0.15)
    result = eng.review(_target(), _ctx(realized_volatility=0.25))
    assert result.decision is RiskDecision.REJECTED
    assert "maximum_volatility_target" in result.record.violated_rules


def test_stale_data_rejection() -> None:
    eng = _engine(max_data_age_seconds=30.0)
    old = _now() - timedelta(seconds=120)
    result = eng.review(
        _target(),
        _ctx(price_timestamps=(old, old)),
    )
    assert result.decision is RiskDecision.REJECTED
    assert "stale_data_rejection" in result.record.violated_rules


def test_invalid_market_data_rejects() -> None:
    eng = _engine()
    result = eng.review(_target(), _ctx(prices=np.array([np.nan, 200.0])))
    assert result.decision is RiskDecision.REJECTED
    assert "stale_or_invalid_market_data" in result.record.violated_rules


def test_missing_context_fail_closed() -> None:
    eng = _engine()
    result = eng.review(_target(), context=None)
    assert result.decision is RiskDecision.REJECTED


def test_abnormal_price_rejection() -> None:
    eng = _engine(abnormal_price_jump=0.05)
    result = eng.review(
        _target(),
        _ctx(prices=np.array([130.0, 200.0]), prior_prices=np.array([100.0, 200.0])),
    )
    assert result.decision is RiskDecision.REJECTED
    assert "abnormal_price_rejection" in result.record.violated_rules


def test_spread_threshold_rejects() -> None:
    eng = _engine(max_spread_bps=10.0)
    result = eng.review(
        _target(),
        _ctx(bid_ask_spread_bps=np.array([5.0, 40.0])),
    )
    assert result.decision is RiskDecision.REJECTED
    assert "spread_threshold" in result.record.violated_rules


def test_market_hours_check() -> None:
    eng = _engine()
    result = eng.review(_target(), _ctx(market_is_open=False))
    assert result.decision is RiskDecision.REJECTED
    assert "market_hours_check" in result.record.violated_rules


def test_duplicate_order_prevention() -> None:
    eng = _engine()
    from rl_trader.risk.rules import order_fingerprint

    fp = order_fingerprint(np.array([0.2, 0.2]))
    result = eng.review(_target(0.2, 0.2), _ctx(last_order_fingerprint=fp))
    assert result.decision is RiskDecision.REJECTED
    assert "duplicate_order_prevention" in result.record.violated_rules


def test_maximum_orders_per_day() -> None:
    eng = _engine(max_orders_per_day=2)
    result = eng.review(_target(), _ctx(orders_submitted_today=2))
    assert result.decision is RiskDecision.REJECTED
    assert "maximum_orders_per_day" in result.record.violated_rules


def test_maximum_consecutive_loss_safeguard() -> None:
    eng = _engine(max_consecutive_losses=3)
    result = eng.review(_target(), _ctx(consecutive_losses=3))
    assert result.decision is RiskDecision.REJECTED
    assert "maximum_consecutive_loss_safeguard" in result.record.violated_rules


def test_kill_switch_requires_manual_reset() -> None:
    ks = KillSwitch(reset_phrase="MANUAL_KILL_SWITCH_RESET")
    ks.trip("daily loss")
    assert ks.is_active
    with pytest.raises(PermissionError, match="manual reset"):
        ks.manual_reset("wrong")
    with pytest.raises(PermissionError):
        ks.manual_reset("")
    ks.manual_reset("MANUAL_KILL_SWITCH_RESET")
    assert not ks.is_active
    eng = RiskEngine(settings=_settings(), kill_switch=ks)
    result = eng.review(_target(), _ctx())
    assert result.decision is RiskDecision.APPROVED


def test_live_database_unavailable_fail_closed() -> None:
    eng = _engine()
    result = eng.review(
        _target(),
        _ctx(trading_mode="live", database_available=False, account_confirmed=True),
    )
    assert result.decision is RiskDecision.REJECTED
    assert "database_available" in result.record.violated_rules


def test_live_account_unconfirmed_fail_closed() -> None:
    eng = _engine()
    result = eng.review(
        _target(),
        _ctx(trading_mode="live", database_available=True, account_confirmed=False),
    )
    assert result.decision is RiskDecision.REJECTED
    assert "account_confirmed" in result.record.violated_rules


def test_risk_package_does_not_import_agents() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "rl_trader"
    assert_risk_has_no_agent_imports(package_root)
    for path in (package_root / "risk").rglob("*.py"):
        assert scan_file_for_risk_agent_imports(path) == []


def test_decision_record_fields_complete() -> None:
    eng = _engine()
    result = eng.review(_target(0.9, 0.05), _ctx())
    rec = result.record
    assert rec.timestamp.tzinfo is not None
    assert isinstance(rec.reason, str) and rec.reason
    assert rec.configuration_version
    assert rec.model_identifier
    assert rec.original_order
    payload = rec.to_dict()
    assert payload["decision_status"] in {"approved", "resized", "rejected"}
