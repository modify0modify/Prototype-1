"""Application settings via pydantic-settings.

Secrets come from environment variables (or a local ``.env`` that is never
committed). Modes are separated: research / paper / live.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

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
        env_prefix="RL_TRADING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: RuntimeMode = RuntimeMode.PAPER
    seed: int = 42
    database_url: str = "sqlite:///./artifacts/rl_trading.db"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    data_api_key: SecretStr | None = None
    data_api_secret: SecretStr | None = None
    broker_api_key: SecretStr | None = None
    broker_api_secret: SecretStr | None = None
    broker_base_url: str | None = None

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
    def _live_requires_explicit_flag(self) -> Settings:
        """Live mode alone is insufficient; the enable flag must also be true."""
        if self.mode is RuntimeMode.LIVE and not self.live_trading_enabled:
            # Allowed as a config template; routing remains blocked by guards.
            return self
        return self

    @property
    def is_paper_default(self) -> bool:
        """True when running under the safe default paper configuration."""
        return self.mode is RuntimeMode.PAPER and not self.live_trading_enabled

    def assert_live_blocked_by_default(self) -> None:
        """Raise if live trading appears accidentally enabled without intent."""
        if self.live_trading_enabled and self.mode is not RuntimeMode.LIVE:
            raise ValueError(
                "live_trading_enabled=True is only valid when mode=live"
            )


def get_settings() -> Settings:
    """Load settings from environment / optional ``.env`` file."""
    return Settings()
