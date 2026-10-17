"""Typed nested configuration models for the trading system."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from rl_trader.config.constants import SUPPORTED_TICKERS
from rl_trader.config.masking import mask_account_id


class DataSettings(BaseModel):
    """Market-data download / validation settings."""

    model_config = ConfigDict(extra="forbid")

    universe: tuple[str, ...] = ("SPY", "QQQ")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None
    allow_missing_bars: bool = False  # silent fills are never allowed when False

    @field_validator("universe", mode="before")
    @classmethod
    def _normalize_universe(cls, value: object) -> object:
        if isinstance(value, str):
            parts = [p.strip().upper() for p in value.split(",") if p.strip()]
            return tuple(parts)
        if isinstance(value, (list, tuple)):
            return tuple(str(v).strip().upper() for v in value)
        return value

    @field_validator("universe")
    @classmethod
    def _validate_universe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("universe must not be empty")
        unsupported = sorted({s for s in value if s not in SUPPORTED_TICKERS})
        if unsupported:
            raise ValueError(
                f"unsupported ticker symbols: {unsupported}; "
                f"supported={sorted(SUPPORTED_TICKERS)}"
            )
        return value


class FeatureSettings(BaseModel):
    """Feature-engineering settings."""

    model_config = ConfigDict(extra="forbid")

    lookback_bars: Annotated[int, Field(ge=1, le=10_000)] = 20
    normalize: bool = True
    causal_only: bool = True
    feature_version: str = "1"

    @field_validator("causal_only")
    @classmethod
    def _require_causal(cls, value: bool) -> bool:
        if not value:
            raise ValueError("causal_only must remain True (look-ahead prevention)")
        return value


class TrainingSettings(BaseModel):
    """RL training hyperparameters (research path)."""

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "ppo"
    total_timesteps: Annotated[int, Field(ge=1)] = 10_000
    learning_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 3e-4
    batch_size: Annotated[int, Field(ge=1)] = 64
    gamma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.99
    seed: Annotated[int, Field(ge=0)] = 42
    checkpoint_dir: Path = Path("models/checkpoints")


class BacktestSettings(BaseModel):
    """Out-of-sample / walk-forward backtest settings."""

    model_config = ConfigDict(extra="forbid")

    initial_cash: Annotated[float, Field(gt=0.0)] = 100_000.0
    commission_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    slippage_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    reports_dir: Path = Path("reports/backtests")
    walk_forward_folds: Annotated[int, Field(ge=1, le=100)] = 3


class RiskSettings(BaseModel):
    """Deterministic risk-engine limits.

    ``daily_loss_limit`` is required whenever risk is active (paper / live /
    backtest). Research configs should still set it explicitly when risk is used.
    """

    model_config = ConfigDict(extra="forbid")

    max_position_size: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    max_portfolio_exposure: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    min_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    # Default set for paper-safe root defaults; may be None only in research configs.
    daily_loss_limit: Annotated[float, Field(gt=0.0, le=1.0)] | None = 0.05
    max_drawdown_limit: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20

    @model_validator(mode="after")
    def _position_within_exposure(self) -> RiskSettings:
        if self.max_position_size > self.max_portfolio_exposure:
            raise ValueError(
                "max_position_size must be <= max_portfolio_exposure "
                f"({self.max_position_size} > {self.max_portfolio_exposure})"
            )
        return self


class BrokerSettings(BaseModel):
    """Broker adapter settings — paper and live credentials stay separate."""

    model_config = ConfigDict(extra="forbid")

    environment: Literal["paper", "live"] = "paper"
    paper_api_key: SecretStr | None = None
    paper_api_secret: SecretStr | None = None
    paper_base_url: str | None = None
    live_api_key: SecretStr | None = None
    live_api_secret: SecretStr | None = None
    live_base_url: str | None = None


class MonitoringSettings(BaseModel):
    """Logging and audit settings — never log secret values."""

    model_config = ConfigDict(extra="forbid")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    audit_enabled: bool = True


class DatabaseSettings(BaseModel):
    """Persistence settings."""

    model_config = ConfigDict(extra="forbid")

    url: str = "sqlite:///./reports/rl_trader.db"
    echo: bool = False


class FundamentalValuationSettings(BaseModel):
    """Config for fundamental valuation — terminal growth must be < WACC."""

    model_config = ConfigDict(extra="forbid")

    wacc: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.08
    terminal_growth_rate: Annotated[float, Field(ge=-0.5, lt=1.0)] = 0.02

    @model_validator(mode="after")
    def _growth_strictly_below_wacc(self) -> FundamentalValuationSettings:
        if self.terminal_growth_rate >= self.wacc:
            raise ValueError(
                "terminal_growth_rate must be strictly less than wacc "
                f"(got growth={self.terminal_growth_rate}, wacc={self.wacc})"
            )
        return self


class LiveTradingSettings(BaseModel):
    """Live-only activation fields (ignored unless app_mode=live).

    Account identifiers are masked in ``repr`` / safe logging helpers.
    """

    model_config = ConfigDict(extra="forbid")

    acknowledgement: str = ""
    account_id: str = ""
    max_live_capital: Annotated[float, Field(gt=0.0)] | None = None
    max_order_value: Annotated[float, Field(gt=0.0)] | None = None
    approved_model_manifest: Path = Path("models/approved/manifest.json")

    def __repr__(self) -> str:
        return (
            "LiveTradingSettings("
            f"acknowledgement={'<set>' if self.acknowledgement else '<unset>'}, "
            f"account_id={mask_account_id(self.account_id)!r}, "
            f"max_live_capital={self.max_live_capital!r}, "
            f"max_order_value={self.max_order_value!r}, "
            f"approved_model_manifest={self.approved_model_manifest!r})"
        )
