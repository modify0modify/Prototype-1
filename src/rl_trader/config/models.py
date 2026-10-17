"""Typed nested configuration models for the trading system."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from rl_trader.config.constants import MARKET_DATA_TIMEZONE, SUPPORTED_TICKERS
from rl_trader.config.masking import mask_account_id


class DataSettings(BaseModel):
    """Market-data download / validation settings."""

    model_config = ConfigDict(extra="forbid")

    universe: tuple[str, ...] = ("SPY", "QQQ", "PLTR", "AAPL", "MSFT")
    provider: Literal["yfinance"] = "yfinance"
    timezone: Literal["UTC"] = "UTC"
    start_date: date | None = None
    end_date: date | None = None
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")
    cache_dir: Path = Path("data/interim/cache")
    quality_report_dir: Path = Path("reports/data_quality")
    # Absolute day-over-day close return above this fraction is flagged (not filled).
    jump_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20
    api_key: SecretStr | None = None
    api_secret: SecretStr | None = None
    allow_missing_bars: bool = False  # silent fills are never allowed when False

    @field_validator("timezone")
    @classmethod
    def _timezone_must_match_canonical(cls, value: str) -> str:
        if value != MARKET_DATA_TIMEZONE:
            raise ValueError(f"timezone must be {MARKET_DATA_TIMEZONE!r} (got {value!r})")
        return value

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
    """Feature-engineering settings (all windows are causal / look-ahead-safe)."""

    model_config = ConfigDict(extra="forbid")

    lookback_bars: Annotated[int, Field(ge=1, le=10_000)] = 20
    normalize: bool = False  # optional z-score; never drops rows
    causal_only: bool = True
    feature_version: str = "1"
    benchmark_symbol: str = "SPY"
    output_dir: Path = Path("data/processed/features")
    manifest_dir: Path = Path("data/processed/features")
    quality_report_dir: Path = Path("reports/feature_quality")

    # Technical windows
    return_windows: tuple[int, ...] = (1, 5, 20, 60)
    ma_windows: tuple[int, ...] = (5, 20, 60)
    rsi_period: Annotated[int, Field(ge=2, le=500)] = 14
    macd_fast: Annotated[int, Field(ge=2, le=200)] = 12
    macd_slow: Annotated[int, Field(ge=3, le=400)] = 26
    macd_signal: Annotated[int, Field(ge=2, le=200)] = 9
    bollinger_period: Annotated[int, Field(ge=2, le=500)] = 20
    bollinger_std: Annotated[float, Field(gt=0.0, le=10.0)] = 2.0
    atr_period: Annotated[int, Field(ge=2, le=500)] = 14
    volatility_window: Annotated[int, Field(ge=2, le=500)] = 20
    downside_vol_window: Annotated[int, Field(ge=2, le=500)] = 20
    volume_zscore_window: Annotated[int, Field(ge=2, le=500)] = 20
    drawdown_window: Annotated[int, Field(ge=2, le=1000)] = 60
    rolling_high_window: Annotated[int, Field(ge=2, le=1000)] = 60

    # Regime
    regime_trend_window: Annotated[int, Field(ge=2, le=500)] = 50
    regime_vol_window: Annotated[int, Field(ge=2, le=500)] = 20
    vol_percentile_window: Annotated[int, Field(ge=5, le=1000)] = 252
    bull_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.02
    bear_threshold: Annotated[float, Field(ge=-1.0, lt=0.0)] = -0.02
    high_vol_percentile: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.80
    low_vol_percentile: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.20

    # Portfolio-state (optional series supplied by caller; defaults produce NaNs)
    turnover_window: Annotated[int, Field(ge=1, le=500)] = 5
    include_portfolio_state: bool = True
    include_fundamentals: bool = True
    include_regime: bool = True

    # Row policy: never drop rows silently — keep warm-up / missing as explicit NaN.
    drop_na_rows: bool = False

    @field_validator("causal_only")
    @classmethod
    def _require_causal(cls, value: bool) -> bool:
        if not value:
            raise ValueError("causal_only must remain True (look-ahead prevention)")
        return value

    @field_validator("drop_na_rows")
    @classmethod
    def _forbid_silent_drops(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "drop_na_rows must be False; warm-up/missing values stay as explicit NaNs"
            )
        return value

    @field_validator("return_windows", "ma_windows", mode="before")
    @classmethod
    def _tuple_ints(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(int(v) for v in value)
        return value

    @model_validator(mode="after")
    def _macd_ordering(self) -> FeatureSettings:
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be < macd_slow")
        if self.low_vol_percentile >= self.high_vol_percentile:
            raise ValueError("low_vol_percentile must be < high_vol_percentile")
        return self


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
    """Config for fundamental DCF valuation — terminal growth must be < WACC."""

    model_config = ConfigDict(extra="forbid")

    wacc: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.08
    terminal_growth_rate: Annotated[float, Field(ge=-0.5, lt=1.0)] = 0.02
    forecast_horizon_years: Annotated[int, Field(ge=1, le=30)] = 5
    # Explicit forecast growth path (declining): year-1 growth, then subtract decline each year
    initial_fcff_growth: Annotated[float, Field(ge=-0.5, lt=1.0)] = 0.08
    growth_decline_per_year: Annotated[float, Field(ge=0.0, le=0.5)] = 0.01
    fcff_method: Literal["ebit_bridge", "ocf_capex", "auto"] = "auto"
    # Optional annual share-count inflation for SBC / dilution (0 = none)
    annual_share_dilution: Annotated[float, Field(ge=0.0, le=0.5)] = 0.0
    include_short_term_investments_in_cash: bool = True
    # Sensitivity grid half-widths around base WACC / g
    sensitivity_wacc_deltas: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02)
    sensitivity_g_deltas: tuple[float, ...] = (-0.01, -0.005, 0.0, 0.005, 0.01)
    currency: str = "USD"
    output_dir: Path = Path("reports/valuations")

    @field_validator("sensitivity_wacc_deltas", "sensitivity_g_deltas", mode="before")
    @classmethod
    def _tuple_floats(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(float(v) for v in value)
        return value

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
