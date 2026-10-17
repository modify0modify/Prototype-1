"""Initial smoke tests for the package skeleton."""

from __future__ import annotations

import rl_trading
from rl_trading.config import RuntimeMode, Settings, get_settings
from rl_trading.live import (
    LiveTradingBlockedError,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)


def test_import_package() -> None:
    assert isinstance(rl_trading.__version__, str)
    assert rl_trading.__version__


def test_config_defaults_are_paper_mode() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert settings.mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.is_paper_default is True
    settings.assert_live_blocked_by_default()


def test_get_settings_loads_without_env_file(monkeypatch: object) -> None:
    # Ensure no accidental env pollution from the test runner.
    for key in (
        "RL_TRADING_MODE",
        "RL_TRADING_LIVE_TRADING_ENABLED",
        "RL_TRADING_LIVE_ACTIVATION_TOKEN",
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
    assert any(c.name == "mode_is_live" and not c.passed for c in report.checks)


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
    """Mode=live alone is insufficient — multi-check must all pass."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        mode=RuntimeMode.LIVE,
        live_trading_enabled=True,
        live_activation_token=None,
        broker_api_key=None,
        broker_api_secret=None,
    )
    assert is_live_trading_blocked(settings) is True
