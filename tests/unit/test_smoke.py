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


def test_import_package() -> None:
    assert isinstance(rl_trader.__version__, str)
    assert rl_trader.__version__


def test_config_defaults_are_paper_mode() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert settings.mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.is_paper_default is True
    settings.assert_live_blocked_by_default()


def test_get_settings_loads_without_env_file(monkeypatch: object) -> None:
    for key in (
        "RL_TRADER_MODE",
        "RL_TRADER_LIVE_TRADING_ENABLED",
        "RL_TRADER_LIVE_ACTIVATION_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    settings = get_settings()
    assert settings.mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False


def test_live_trading_blocked_by_default() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        mode=RuntimeMode.PAPER,
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
        mode=RuntimeMode.LIVE,
        live_trading_enabled=True,
        live_activation_token=None,
        live_broker_api_key=None,
        live_broker_api_secret=None,
    )
    assert is_live_trading_blocked(settings) is True
