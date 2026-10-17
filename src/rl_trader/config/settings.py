"""Root application settings (pydantic-settings) with nested typed models.

Environment variables (see ``.env.example``) override YAML. Default
``APP_MODE`` is **paper** — never live.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rl_trader.config.models import (
    BacktestSettings,
    BenchmarkSettings,
    BrokerSettings,
    DatabaseSettings,
    DataSettings,
    EnvironmentSettings,
    FeatureSettings,
    FundamentalValuationSettings,
    LiveTradingSettings,
    MonitoringSettings,
    PaperTradingSettings,
    RiskSettings,
    TrainingSettings,
)


class RuntimeMode(StrEnum):
    """Supported application modes."""

    RESEARCH = "research"
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class AppSettings(BaseSettings):
    """Validated root configuration.

    Nested sections cover data, features, training, environment, backtest,
    benchmarks, risk, broker, monitoring, database, fundamentals, and live
    activation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    app_mode: RuntimeMode = Field(default=RuntimeMode.PAPER, validation_alias="APP_MODE")
    live_trading_enabled: bool = Field(
        default=False, validation_alias="LIVE_TRADING_ENABLED"
    )
    seed: int = 42

    data: DataSettings = Field(default_factory=DataSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    benchmarks: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    paper_trading: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    fundamentals: FundamentalValuationSettings = Field(
        default_factory=FundamentalValuationSettings
    )
    live: LiveTradingSettings = Field(default_factory=LiveTradingSettings)

    # Flat env bridges (merged into nested models after load)
    broker_environment: Literal["paper", "live"] | None = Field(
        default=None, validation_alias="BROKER_ENVIRONMENT"
    )
    live_trading_acknowledgement: str | None = Field(
        default=None, validation_alias="LIVE_TRADING_ACKNOWLEDGEMENT"
    )
    live_account_id: str | None = Field(default=None, validation_alias="LIVE_ACCOUNT_ID")
    live_max_capital: float | None = Field(default=None, validation_alias="LIVE_MAX_CAPITAL")
    live_max_order_value: float | None = Field(
        default=None, validation_alias="LIVE_MAX_ORDER_VALUE"
    )
    approved_model_manifest: Path | None = Field(
        default=None, validation_alias="APPROVED_MODEL_MANIFEST"
    )
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")

    project_root: Path = Field(default_factory=lambda: Path.cwd())

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "mode" in normalized and "app_mode" not in normalized and "APP_MODE" not in normalized:
            normalized["app_mode"] = normalized.pop("mode")
        if "broker_profile" in normalized:
            broker = dict(normalized.get("broker") or {})
            broker.setdefault("environment", normalized.pop("broker_profile"))
            normalized["broker"] = broker
        return normalized

    @model_validator(mode="after")
    def _apply_env_bridges_and_rules(self) -> Self:
        live_updates: dict[str, Any] = {}
        if self.live_trading_acknowledgement is not None:
            live_updates["acknowledgement"] = self.live_trading_acknowledgement
        if self.live_account_id is not None:
            live_updates["account_id"] = self.live_account_id
        if self.live_max_capital is not None:
            live_updates["max_live_capital"] = self.live_max_capital
        if self.live_max_order_value is not None:
            live_updates["max_order_value"] = self.live_max_order_value
        if self.approved_model_manifest is not None:
            live_updates["approved_model_manifest"] = self.approved_model_manifest

        if self.broker_environment is not None:
            object.__setattr__(
                self,
                "broker",
                self.broker.model_copy(update={"environment": self.broker_environment}),
            )
        if self.database_url is not None:
            object.__setattr__(
                self,
                "database",
                self.database.model_copy(update={"url": self.database_url}),
            )
        if live_updates:
            object.__setattr__(self, "live", self.live.model_copy(update=live_updates))

        if self.live_trading_enabled and self.app_mode is not RuntimeMode.LIVE:
            raise ValueError("LIVE_TRADING_ENABLED=true is only valid when APP_MODE=live")

        risk_active = self.app_mode in {
            RuntimeMode.PAPER,
            RuntimeMode.LIVE,
            RuntimeMode.BACKTEST,
        }
        if risk_active and self.risk.daily_loss_limit is None:
            raise ValueError(
                f"risk.daily_loss_limit is required when APP_MODE={self.app_mode.value}"
            )
        return self

    @property
    def mode(self) -> RuntimeMode:
        """Alias for ``app_mode`` (legacy call sites)."""
        return self.app_mode

    @property
    def log_level(self) -> str:
        """Monitoring log level."""
        return self.monitoring.log_level

    @property
    def log_json(self) -> bool:
        """Whether JSON logging is enabled."""
        return self.monitoring.log_json

    @property
    def is_paper_default(self) -> bool:
        """True when running under the safe default paper configuration."""
        return (
            self.app_mode is RuntimeMode.PAPER
            and not self.live_trading_enabled
            and self.broker.environment == "paper"
        )

    def assert_live_blocked_by_default(self) -> None:
        """Raise if live trading appears accidentally enabled outside live mode."""
        if self.live_trading_enabled and self.app_mode is not RuntimeMode.LIVE:
            raise ValueError("LIVE_TRADING_ENABLED=true is only valid when APP_MODE=live")


Settings = AppSettings


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def get_settings(*, config_file: Path | None = None) -> AppSettings:
    """Load settings from environment / optional ``.env`` / YAML overlay.

    YAML supplies structured defaults; environment variables override via
    pydantic-settings. Root default mode is paper.
    """
    if config_file is None:
        return AppSettings()

    overlay = load_yaml_config(config_file)
    known = set(AppSettings.model_fields) | {
        "mode",
        "broker_profile",
        "APP_MODE",
        "LIVE_TRADING_ENABLED",
        "BROKER_ENVIRONMENT",
        "LIVE_TRADING_ACKNOWLEDGEMENT",
        "LIVE_ACCOUNT_ID",
        "LIVE_MAX_CAPITAL",
        "LIVE_MAX_ORDER_VALUE",
        "APPROVED_MODEL_MANIFEST",
        "DATABASE_URL",
    }
    filtered = {k: v for k, v in overlay.items() if k in known}
    return AppSettings(**filtered)
