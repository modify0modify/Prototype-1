"""YAML config loading smoke."""

from __future__ import annotations

from pathlib import Path

from rl_trader.config import RuntimeMode, get_settings, load_yaml_config


def test_load_paper_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "paper.yaml"
    data = load_yaml_config(path)
    assert data["mode"] == "paper"
    assert data["live_trading_enabled"] is False


def test_get_settings_from_paper_yaml(monkeypatch: object) -> None:
    for key in (
        "RL_TRADER_MODE",
        "RL_TRADER_LIVE_TRADING_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    settings = get_settings(config_file=root / "configs" / "paper.yaml")
    assert settings.mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False
