"""Smoke tests: package import, paper defaults, live blocked."""

from __future__ import annotations

import rl_trader
from rl_trader.brokers import (
    LiveTradingBlockedError,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)
from rl_trader.config import RuntimeMode, Settings, get_settings


def _clear_live_env(monkeypatch: object) -> None:
    for key in (
        "APP_MODE",
        "LIVE_TRADING_ENABLED",
        "BROKER_ENVIRONMENT",
        "LIVE_TRADING_ACKNOWLEDGEMENT",
        "LIVE_ACCOUNT_ID",
        "LIVE_MAX_CAPITAL",
        "LIVE_MAX_ORDER_VALUE",
        "APPROVED_MODEL_MANIFEST",
        "RL_TRADER_MODE",
        "RL_TRADER_LIVE_TRADING_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]


def test_import_package() -> None:
    assert isinstance(rl_trader.__version__, str)
    assert rl_trader.__version__


def test_config_defaults_are_paper_mode() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert settings.app_mode is RuntimeMode.PAPER
    assert settings.mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.broker.environment == "paper"
    assert settings.is_paper_default is True
    assert settings.risk.daily_loss_limit is not None
    settings.assert_live_blocked_by_default()


def test_get_settings_loads_without_env_file(monkeypatch: object) -> None:
    _clear_live_env(monkeypatch)
    settings = get_settings()
    assert settings.app_mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False


def test_live_trading_blocked_by_default() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_mode=RuntimeMode.PAPER,
        live_trading_enabled=False,
    )
    assert is_live_trading_blocked(settings) is True
    report = evaluate_live_activation(settings)
    assert report.allowed is False


def test_live_trading_assert_raises_by_default() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    try:
        assert_live_trading_allowed(settings)
        raised = False
    except LiveTradingBlockedError:
        raised = True
    assert raised is True


def test_live_still_blocked_without_full_activation() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_mode=RuntimeMode.LIVE,
        live_trading_enabled=True,
        broker_environment="live",
    )
    assert is_live_trading_blocked(settings) is True
