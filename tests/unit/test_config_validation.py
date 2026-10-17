"""Unit tests for config validation rules (non-live gates)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rl_trader.config import (
    AppSettings,
    DataSettings,
    FundamentalValuationSettings,
    RiskSettings,
    RuntimeMode,
    mask_account_id,
)
from rl_trader.fundamentals import validate_valuation_assumptions


def test_reject_empty_universe() -> None:
    with pytest.raises(ValidationError, match="universe must not be empty"):
        DataSettings(universe=())


def test_reject_unsupported_ticker() -> None:
    with pytest.raises(ValidationError, match="unsupported ticker"):
        DataSettings(universe=("SPY", "NOTAREALTICKER"))


def test_accept_supported_universe() -> None:
    data = DataSettings(universe=("spy", "qqq"))
    assert data.universe == ("SPY", "QQQ")


def test_reject_max_position_above_exposure() -> None:
    with pytest.raises(
        ValidationError, match="max_position_size must be <= max_portfolio_exposure"
    ):
        RiskSettings(max_position_size=0.8, max_portfolio_exposure=0.5, daily_loss_limit=0.05)


def test_reject_percentage_out_of_range() -> None:
    with pytest.raises(ValidationError):
        RiskSettings(max_position_size=1.5, max_portfolio_exposure=1.0, daily_loss_limit=0.05)


def test_paper_requires_daily_loss_limit() -> None:
    with pytest.raises(ValidationError, match="daily_loss_limit"):
        AppSettings(
            _env_file=None,  # type: ignore[call-arg]
            app_mode=RuntimeMode.PAPER,
            risk=RiskSettings(
                max_position_size=0.2,
                max_portfolio_exposure=1.0,
                daily_loss_limit=None,
            ),
        )


def test_backtest_requires_daily_loss_limit() -> None:
    with pytest.raises(ValidationError, match="daily_loss_limit"):
        AppSettings(
            _env_file=None,  # type: ignore[call-arg]
            app_mode=RuntimeMode.BACKTEST,
            risk=RiskSettings(daily_loss_limit=None),
        )


def test_research_allows_missing_daily_loss_limit() -> None:
    settings = AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        app_mode=RuntimeMode.RESEARCH,
        risk=RiskSettings(daily_loss_limit=None),
    )
    assert settings.app_mode is RuntimeMode.RESEARCH
    assert settings.risk.daily_loss_limit is None


def test_reject_terminal_growth_ge_wacc() -> None:
    with pytest.raises(ValidationError, match="terminal_growth_rate must be strictly less"):
        FundamentalValuationSettings(wacc=0.05, terminal_growth_rate=0.05)
    with pytest.raises(ValidationError, match="terminal_growth_rate must be strictly less"):
        validate_valuation_assumptions(wacc=0.06, terminal_growth_rate=0.07)


def test_accept_terminal_growth_below_wacc() -> None:
    cfg = validate_valuation_assumptions(wacc=0.08, terminal_growth_rate=0.02)
    assert cfg.terminal_growth_rate < cfg.wacc


def test_mask_account_id_never_leaks_full_value() -> None:
    assert mask_account_id("LIVEACCT123456") == "***3456"
    assert mask_account_id("ab") == "****"
    assert mask_account_id("") == "<empty>"
    live_repr = AppSettings(
        _env_file=None,  # type: ignore[call-arg]
        app_mode=RuntimeMode.PAPER,
    ).live.__repr__()
    # Default empty account is masked; secrets must not appear as raw keys in repr.
    assert "account_id=" in live_repr
    assert "api_key" not in live_repr.lower() or "SecretStr" in live_repr


def test_live_enabled_outside_live_mode_rejected() -> None:
    with pytest.raises(ValidationError, match="LIVE_TRADING_ENABLED"):
        AppSettings(
            _env_file=None,  # type: ignore[call-arg]
            app_mode=RuntimeMode.PAPER,
            live_trading_enabled=True,
        )


def test_features_require_causal_only() -> None:
    from rl_trader.config import FeatureSettings

    with pytest.raises(ValidationError, match="causal_only"):
        FeatureSettings(causal_only=False)
