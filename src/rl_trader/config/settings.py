"""Application settings via pydantic-settings + optional YAML overlays.

Secrets come from environment variables (or a local ``.env`` that is never
committed). Paper and live broker credentials are separate fields.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(StrEnum):
    """Supported runtime modes."""

    RESEARCH = "research"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Validated runtime configuration.

    Defaults intentionally keep the system in paper mode with live trading
    blocked. Never hard-code API keys in source; load them from env vars.
    """

    model_config = SettingsConfigDict(
        env_prefix="RL_TRADER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: RuntimeMode = RuntimeMode.PAPER
    seed: int = 42
    database_url: str = "sqlite:///./reports/rl_trader.db"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    data_api_key: SecretStr | None = None
    data_api_secret: SecretStr | None = None

    paper_broker_api_key: SecretStr | None = None
    paper_broker_api_secret: SecretStr | None = None
    paper_broker_base_url: str | None = None

    live_broker_api_key: SecretStr | None = None
    live_broker_api_secret: SecretStr | None = None
    live_broker_base_url: str | None = None

    live_trading_enabled: bool = False
    live_activation_token: SecretStr | None = None

    project_root: Path = Field(default_factory=lambda: Path.cwd())

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value

    @model_validator(mode="after")
    def _reject_live_flag_outside_live_mode(self) -> Settings:
        if self.live_trading_enabled and self.mode is not RuntimeMode.LIVE:
            raise ValueError(
                "live_trading_enabled=True is only valid when mode=live"
            )
        return self

    @property
    def is_paper_default(self) -> bool:
        """True when running under the safe default paper configuration."""
        return self.mode is RuntimeMode.PAPER and not self.live_trading_enabled

    def assert_live_blocked_by_default(self) -> None:
        """No-op check used by smoke tests; raises if misconfigured."""
        if self.live_trading_enabled and self.mode is not RuntimeMode.LIVE:
            raise ValueError(
                "live_trading_enabled=True is only valid when mode=live"
            )


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def get_settings(*, config_file: Path | None = None) -> Settings:
    """Load settings from environment / optional ``.env`` / YAML overlay.

    YAML values supply non-secret defaults; environment variables always win
    for fields defined on :class:`Settings`.
    """
    if config_file is None:
        return Settings()
    overlay = load_yaml_config(config_file)
    allowed = set(Settings.model_fields)
    filtered = {k: v for k, v in overlay.items() if k in allowed}
    return Settings(**filtered)
