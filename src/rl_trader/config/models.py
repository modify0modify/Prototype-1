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
    # Static ticker lists are survivor-biased unless a true point-in-time universe
    # is supplied. Research may acknowledge this; paper/live must not ignore it.
    point_in_time_universe: bool = False
    survivorship_bias_acknowledged: bool = False

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
    """RL training hyperparameters (research path).

    Date splits are inclusive calendar dates. Feature scalers are fit only on
    the training window; validation is used for early stopping / model
    selection; the test window is never used for fitting or selection.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["ppo", "sac", "td3"] = "ppo"
    model_id: str = "ppo-portfolio"
    total_timesteps: Annotated[int, Field(ge=1)] = 10_000
    learning_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 3e-4
    batch_size: Annotated[int, Field(ge=1)] = 64
    # PPO-specific
    n_steps: Annotated[int, Field(ge=1)] = 256
    n_epochs: Annotated[int, Field(ge=1, le=100)] = 5
    gamma: Annotated[float, Field(gt=0.0, le=1.0)] = 0.99
    gae_lambda: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95
    clip_range: Annotated[float, Field(gt=0.0, le=1.0)] = 0.2
    ent_coef: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    vf_coef: Annotated[float, Field(ge=0.0, le=10.0)] = 0.5
    max_grad_norm: Annotated[float, Field(gt=0.0)] = 0.5
    # SAC / TD3 (off-policy) — continuous Box actions match PortfolioEnv
    buffer_size: Annotated[int, Field(ge=100)] = 50_000
    learning_starts: Annotated[int, Field(ge=0)] = 100
    tau: Annotated[float, Field(gt=0.0, le=1.0)] = 0.005
    train_freq: Annotated[int, Field(ge=1)] = 1
    gradient_steps: Annotated[int, Field(ge=1)] = 1
    # TD3-specific
    policy_delay: Annotated[int, Field(ge=1)] = 2
    target_policy_noise: Annotated[float, Field(ge=0.0)] = 0.2
    target_noise_clip: Annotated[float, Field(ge=0.0)] = 0.5
    seed: Annotated[int, Field(ge=0)] = 42
    device: str = "cpu"
    checkpoint_dir: Path = Path("models/checkpoints")
    comparison_dir: Path = Path("reports/agent_comparison")
    # Inclusive train / validation / test windows (calendar dates).
    train_start: date = date(2020, 1, 1)
    train_end: date = date(2021, 12, 31)
    val_start: date = date(2022, 1, 1)
    val_end: date = date(2022, 6, 30)
    test_start: date = date(2022, 7, 1)
    test_end: date = date(2022, 12, 31)
    # Evaluation / early stopping (validation only — never test).
    eval_freq: Annotated[int, Field(ge=1)] = 500
    n_eval_episodes: Annotated[int, Field(ge=1)] = 1
    early_stopping_patience: Annotated[int, Field(ge=1)] = 5
    min_evals_before_stop: Annotated[int, Field(ge=1)] = 2
    normalize_features: bool = True
    # When True, run a one-shot test evaluation after training for reporting only.
    evaluate_test_after_training: bool = True

    @model_validator(mode="after")
    def _ordered_splits(self) -> TrainingSettings:
        if not (self.train_start <= self.train_end):
            raise ValueError("train_start must be <= train_end")
        if not (self.val_start <= self.val_end):
            raise ValueError("val_start must be <= val_end")
        if not (self.test_start <= self.test_end):
            raise ValueError("test_start must be <= test_end")
        if self.train_end >= self.val_start:
            raise ValueError("train_end must be strictly before val_start")
        if self.val_end >= self.test_start:
            raise ValueError("val_end must be strictly before test_start")
        return self


class EnvironmentSettings(BaseModel):
    """Gymnasium portfolio-env costs, constraints, and reward weights."""

    model_config = ConfigDict(extra="forbid")

    initial_cash: Annotated[float, Field(gt=0.0)] = 100_000.0
    # Long-only target weight cap per asset; residual is cash.
    max_weight: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    # Transaction-cost model (research simulation — not a broker).
    commission_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    bid_ask_spread_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 2.0
    slippage_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    market_impact_coeff: Annotated[float, Field(ge=0.0, le=10.0)] = 0.1
    # Default 1: decide using bar-t information, fill at bar-(t+delay) prices.
    # delay=0 is same-bar close fill (look-ahead vs close features) and requires
    # ``allow_same_bar_execution=True``.
    execution_delay_bars: Annotated[int, Field(ge=0, le=5)] = 1
    allow_same_bar_execution: bool = False
    # Observation / reward helpers
    recent_returns_lookback: Annotated[int, Field(ge=1, le=252)] = 5
    # Reward = return - penalties (configurable risk-adjusted).
    drawdown_penalty: Annotated[float, Field(ge=0.0)] = 0.1
    turnover_penalty: Annotated[float, Field(ge=0.0)] = 0.01
    # When True (default), period return already embeds simulated costs — do not
    # subtract ``transaction_cost_penalty * cost_frac`` again (reward hacking /
    # double-count risk).
    costs_already_in_return: bool = True
    transaction_cost_penalty: Annotated[float, Field(ge=0.0)] = 1.0
    concentration_penalty: Annotated[float, Field(ge=0.0)] = 0.0
    volatility_penalty: Annotated[float, Field(ge=0.0)] = 0.0
    # Safe termination thresholds
    max_drawdown_terminate: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    min_portfolio_value: Annotated[float, Field(gt=0.0)] = 1.0

    @model_validator(mode="after")
    def _execution_delay_gate(self) -> EnvironmentSettings:
        if self.execution_delay_bars == 0 and not self.allow_same_bar_execution:
            raise ValueError(
                "execution_delay_bars=0 enables same-bar fill at close while "
                "observations may include that close — set "
                "allow_same_bar_execution=True only if this look-ahead is intentional"
            )
        return self


class BacktestSettings(BaseModel):
    """Out-of-sample / walk-forward backtest settings.

    Walk-forward windows are measured in bars. Test periods of successive folds
    do not overlap when ``step_bars >= test_bars`` (enforced for combined
    return aggregation). Scalers are fit independently on each fold's train
    window.
    """

    model_config = ConfigDict(extra="forbid")

    initial_cash: Annotated[float, Field(gt=0.0)] = 100_000.0
    commission_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    slippage_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    reports_dir: Path = Path("reports/backtests")
    walk_forward_dir: Path = Path("reports/walk_forward")
    # Legacy / soft cap on number of folds (geometry may produce fewer).
    walk_forward_folds: Annotated[int, Field(ge=1, le=100)] = 3
    train_bars: Annotated[int, Field(ge=4)] = 40
    val_bars: Annotated[int, Field(ge=2)] = 10
    test_bars: Annotated[int, Field(ge=2)] = 10
    step_bars: Annotated[int, Field(ge=1)] = 10
    window_mode: Literal["rolling", "expanding"] = "rolling"
    retrain_mode: Literal["scratch", "fine_tune"] = "scratch"
    max_dd_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20
    # Fold marked unstable when |sharpe - median| > mult * MAD (or MAD floor).
    instability_sharpe_mad_mult: Annotated[float, Field(gt=0.0)] = 3.0
    instability_return_mad_mult: Annotated[float, Field(gt=0.0)] = 3.0
    underperform_eps: Annotated[float, Field(ge=0.0)] = 0.0
    algorithm: Literal["ppo", "sac", "td3"] = "ppo"
    train_timesteps: Annotated[int, Field(ge=1)] = 256
    # When False, evaluate a deterministic equal-weight agent (fast CI / unit tests).
    train_agent: bool = True
    regime_feature_index: Annotated[int, Field(ge=0)] = 0
    n_regime_buckets: Annotated[int, Field(ge=2, le=10)] = 3
    trading_days_per_year: Annotated[int, Field(ge=1, le=366)] = 252

    @model_validator(mode="after")
    def _walk_forward_geometry(self) -> BacktestSettings:
        if self.step_bars < self.test_bars:
            raise ValueError(
                "step_bars must be >= test_bars so successive test windows do not "
                f"overlap (got step_bars={self.step_bars}, test_bars={self.test_bars})"
            )
        return self


class BenchmarkSettings(BaseModel):
    """Deterministic baseline strategy knobs (shared harness with env costs)."""

    model_config = ConfigDict(extra="forbid")

    spy_symbol: str = "SPY"
    ma_window: Annotated[int, Field(ge=2, le=500)] = 50
    momentum_lookback: Annotated[int, Field(ge=2, le=500)] = 20
    momentum_top_k: Annotated[int, Field(ge=1, le=50)] = 3
    mean_reversion_lookback: Annotated[int, Field(ge=2, le=120)] = 5
    mean_reversion_z: Annotated[float, Field(gt=0.0, le=10.0)] = 1.0
    vol_lookback: Annotated[int, Field(ge=5, le=500)] = 20
    risk_parity_vol_floor: Annotated[float, Field(gt=0.0, le=1.0)] = 1e-6
    trading_days_per_year: Annotated[int, Field(ge=1, le=366)] = 252
    output_dir: Path = Path("reports/benchmarks")
    # When True, reuse environment.* cost fields so baselines match the RL env.
    use_environment_costs: bool = True


class RiskSettings(BaseModel):
    """Deterministic risk-engine limits (independent of the RL agent).

    ``daily_loss_limit`` is required whenever risk is active (paper / live /
    backtest). Research configs should still set it explicitly when risk is used.
    """

    model_config = ConfigDict(extra="forbid")

    # Position / exposure
    max_position_size: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    max_portfolio_exposure: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    max_sector_exposure: Annotated[float, Field(gt=0.0, le=1.0)] = 0.50
    min_weight: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    min_cash_reserve: Annotated[float, Field(ge=0.0, le=1.0)] = 0.02
    # Order / turnover
    max_order_notional: Annotated[float, Field(gt=0.0)] = 50_000.0
    max_daily_turnover: Annotated[float, Field(gt=0.0, le=10.0)] = 1.0
    max_orders_per_day: Annotated[int, Field(ge=1, le=10_000)] = 50
    # Loss / drawdown / vol
    daily_loss_limit: Annotated[float, Field(gt=0.0, le=1.0)] | None = 0.05
    weekly_loss_limit: Annotated[float, Field(gt=0.0, le=1.0)] = 0.10
    max_drawdown_limit: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20
    max_volatility_target: Annotated[float, Field(gt=0.0, le=5.0)] = 0.35
    max_consecutive_losses: Annotated[int, Field(ge=1, le=1000)] = 5
    # Market data quality
    max_data_age_seconds: Annotated[float, Field(gt=0.0)] = 300.0
    abnormal_price_jump: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20
    max_spread_bps: Annotated[float, Field(gt=0.0, le=10_000.0)] = 50.0
    # Identity / versioning for decision records
    configuration_version: str = "risk-v1"
    # Kill-switch acknowledgement phrase required for manual reset
    kill_switch_reset_phrase: str = "MANUAL_KILL_SWITCH_RESET"

    @model_validator(mode="after")
    def _position_within_exposure(self) -> RiskSettings:
        if self.max_position_size > self.max_portfolio_exposure:
            raise ValueError(
                "max_position_size must be <= max_portfolio_exposure "
                f"({self.max_position_size} > {self.max_portfolio_exposure})"
            )
        if self.min_cash_reserve + 1e-12 >= 1.0:
            raise ValueError("min_cash_reserve must be < 1")
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


class PaperTradingSettings(BaseModel):
    """Prospective paper-trading service runtime settings.

    The model is frozen for an evaluation period — scheduled retrain is a
    separate hook and never runs inside the decision loop.
    """

    model_config = ConfigDict(extra="forbid")

    state_dir: Path = Path("reports/paper_trading")
    broker_data_dir: Path = Path("data/interim/paper_broker")
    lock_filename: str = "paper_trader.lock"
    heartbeat_filename: str = "heartbeat.json"
    decisions_filename: str = "decisions.jsonl"
    metrics_filename: str = "metrics.json"
    alerts_filename: str = "alerts.jsonl"
    # Freshness gate — shut down safely when market data is older than this.
    max_data_age_seconds: Annotated[float, Field(gt=0.0, le=86_400.0)] = 300.0
    # Stable id for the frozen evaluation window (no intra-period retrain).
    evaluation_period_id: str = "default"
    # Optional cron-like hint for an external scheduler (not executed here).
    retrain_schedule: str | None = None
    enable_scheduled_retrain_hook: bool = False
    # When True, use the synthetic fixture market feed (tests / smoke).
    use_fixture_market_data: bool = False
    commission_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    spread_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 2.0
    slippage_bps: Annotated[float, Field(ge=0.0, le=1000.0)] = 1.0
    initial_cash: Annotated[float, Field(gt=0.0)] = 100_000.0


class PaperApprovalGateSettings(BaseModel):
    """Configurable paper-trading approval thresholds.

    Defaults below are **EXAMPLE** conservative illustrations only — not
    investment advice and not hard-coded business rules. Override per deployment.
    """

    model_config = ConfigDict(extra="forbid")

    # EXAMPLE default: require non-negative OOS return
    min_oos_return: float = 0.0
    # EXAMPLE default: max drawdown must be <= 25%
    max_drawdown: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    # EXAMPLE default: turnover fraction cap
    max_turnover: Annotated[float, Field(gt=0.0)] = 2.0
    # EXAMPLE default: at least this many evaluated market periods (e.g. folds/days)
    min_evaluated_market_periods: Annotated[int, Field(ge=1)] = 3
    # EXAMPLE default: require evidence across at least this many regimes
    min_regimes_evaluated: Annotated[int, Field(ge=1)] = 2
    # EXAMPLE default: minimum excess return vs required benchmark
    min_excess_return_vs_benchmark: float = 0.0
    require_transaction_costs_included: bool = True
    require_no_critical_audit_findings: bool = True
    require_feature_manifest_match: bool = True
    require_model_metadata: bool = True
    require_regression_tests_passed: bool = True
    require_benchmark_comparison: bool = True


class LiveApprovalGateSettings(BaseModel):
    """Additional live-tier approval thresholds (on top of paper gates).

    Defaults are **EXAMPLE** illustrations only — override explicitly before any
    live consideration. Live broker remains unimplemented/stubbed.
    """

    model_config = ConfigDict(extra="forbid")

    # EXAMPLE default: 30 calendar days of prospective paper
    min_paper_duration_days: Annotated[float, Field(ge=0.0)] = 30.0
    # EXAMPLE default: at least 50 paper trades
    min_paper_trades: Annotated[int, Field(ge=0)] = 50
    # EXAMPLE default: paper Sharpe must be >= 0.5
    min_paper_sharpe: float = 0.5
    # EXAMPLE default: paper max drawdown <= 20%
    max_paper_drawdown: Annotated[float, Field(gt=0.0, le=1.0)] = 0.20
    # EXAMPLE default: |paper slippage − assumed| <= 5 bps
    max_slippage_difference_bps: Annotated[float, Field(ge=0.0)] = 5.0
    require_no_unresolved_reconciliation_errors: bool = True
    require_no_kill_switch_investigation: bool = True
    require_manual_approval_record: bool = True
    require_explicit_live_capital_limit: bool = True
    require_successful_broker_sandbox_test: bool = True


class ApprovalSettings(BaseModel):
    """Formal model-approval policy (hashed manifests + configurable gates)."""

    model_config = ConfigDict(extra="forbid")

    # When True, HMAC signature is required (needs RL_TRADER_MANIFEST_SIGNING_KEY).
    require_manifest_signature: bool = False
    # Persist approval rows into storage.model_approvals when a DB URL is available.
    persist_approvals: bool = True
    paper: PaperApprovalGateSettings = Field(default_factory=PaperApprovalGateSettings)
    live: LiveApprovalGateSettings = Field(default_factory=LiveApprovalGateSettings)


class MonitoringSettings(BaseModel):
    """Logging and audit settings — never log secret values."""

    model_config = ConfigDict(extra="forbid")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False
    audit_enabled: bool = True


class DatabaseSettings(BaseModel):
    """Persistence settings (SQLAlchemy / SQLite).

    Never configure the database to store broker API keys or other secrets —
    the storage layer rejects secret-like column names and sanitizes config
    snapshots before insert.
    """

    model_config = ConfigDict(extra="forbid")

    url: str = "sqlite:///./reports/rl_trader.db"
    echo: bool = False
    # When True, paper trading dual-writes decisions/orders into SQL (file
    # ledgers remain intact).
    persist_paper_trading: bool = False


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

    Defaults keep live **disabled**. Rollout stage defaults to 0 (shadow only).
    Capital ceilings are EXAMPLE illustrations — humans set them in config; the
    system must never raise them programmatically.

    Account identifiers are masked in ``repr`` / safe logging helpers.
    """

    model_config = ConfigDict(extra="forbid")

    acknowledgement: str = ""
    account_id: str = ""
    max_live_capital: Annotated[float, Field(gt=0.0)] | None = None
    max_order_value: Annotated[float, Field(gt=0.0)] | None = None
    approved_model_manifest: Path = Path("models/approved/manifest.json")

    # --- Rollout stage (config-driven; default Stage 0 = no live orders) ---
    # 0: shadow decisions only
    # 1: tiny fixed capital, long-only (EXAMPLE capital below)
    # 2: gradual capital increase only after completed human/config review
    # 3: expanded ticker universe only after additional review
    rollout_stage: Annotated[int, Field(ge=0, le=3)] = 0

    # Approved account allowlist — account_id must be listed.
    approved_account_allowlist: tuple[str, ...] = ()

    # Manual runtime arming (separate from config completeness).
    # activate()/deactivate() on ManualLiveController flips this via config copy.
    manual_activated: bool = False

    # EXAMPLE stage capital ceilings (human-configured only — never auto-raised).
    # Stage 1 EXAMPLE tiny capital when max_live_capital is unset for that stage.
    stage1_capital_limit: Annotated[float, Field(gt=0.0)] | None = 1_000.0  # EXAMPLE
    stage2_capital_limit: Annotated[float, Field(gt=0.0)] | None = 5_000.0  # EXAMPLE
    stage3_capital_limit: Annotated[float, Field(gt=0.0)] | None = 25_000.0  # EXAMPLE

    # Human/config review acknowledgements required before stages 2 and 3.
    stage2_review_acknowledged: bool = False
    stage3_review_acknowledged: bool = False
    # Stage 3 expanded universe (empty = not reviewed / not allowed).
    stage3_expanded_universe: tuple[str, ...] = ()

    # Stage 1+ policy: long-only (shorts blocked).
    long_only: bool = True
    # Safety toggles (defaults strict).
    require_market_hours: bool = True
    require_account_reconciliation: bool = True
    # EXAMPLE stale-data threshold (seconds).
    max_data_age_seconds: Annotated[float, Field(gt=0.0, le=86_400.0)] = 60.0
    # Strict position cap as fraction of equity (EXAMPLE).
    max_position_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.10  # EXAMPLE
    # Shadow decision log directory (Stage 0 / dry path).
    shadow_state_dir: Path = Path("reports/live_trading/shadow")

    def __repr__(self) -> str:
        return (
            "LiveTradingSettings("
            f"acknowledgement={'<set>' if self.acknowledgement else '<unset>'}, "
            f"account_id={mask_account_id(self.account_id)!r}, "
            f"max_live_capital={self.max_live_capital!r}, "
            f"max_order_value={self.max_order_value!r}, "
            f"rollout_stage={self.rollout_stage}, "
            f"manual_activated={self.manual_activated}, "
            f"approved_model_manifest={self.approved_model_manifest!r})"
        )
