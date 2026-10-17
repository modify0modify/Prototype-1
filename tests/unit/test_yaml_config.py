"""YAML config loading smoke."""

from __future__ import annotations

from pathlib import Path

from rl_trader.config import RuntimeMode, get_settings, load_yaml_config


def test_load_paper_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "paper.yaml"
    data = load_yaml_config(path)
    assert data["app_mode"] == "paper"
    assert data["live_trading_enabled"] is False
    assert data["risk"]["daily_loss_limit"] == 0.05


def test_get_settings_from_paper_yaml(monkeypatch: object) -> None:
    for key in (
        "APP_MODE",
        "LIVE_TRADING_ENABLED",
        "BROKER_ENVIRONMENT",
    ):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    settings = get_settings(config_file=root / "configs" / "paper.yaml")
    assert settings.app_mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.broker.environment == "paper"
    assert settings.data.universe
    assert settings.paper_trading.evaluation_period_id == "paper-default"
    assert settings.paper_trading.use_fixture_market_data is False


def test_get_settings_from_backtest_yaml(monkeypatch: object) -> None:
    for key in ("APP_MODE", "LIVE_TRADING_ENABLED", "BROKER_ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    settings = get_settings(config_file=root / "configs" / "backtest.yaml")
    assert settings.app_mode is RuntimeMode.BACKTEST
    assert settings.risk.daily_loss_limit is not None


def test_get_settings_from_research_yaml(monkeypatch: object) -> None:
    for key in ("APP_MODE", "LIVE_TRADING_ENABLED", "BROKER_ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    settings = get_settings(config_file=root / "configs" / "research.yaml")
    assert settings.app_mode is RuntimeMode.RESEARCH
