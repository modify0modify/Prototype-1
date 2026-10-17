"""Configuration loading and validation (research / backtest / paper / live)."""

from rl_trader.config.constants import (
    LIVE_TRADING_CONFIRMATION_PHRASE,
    SUPPORTED_TICKERS,
)
from rl_trader.config.live_gates import (
    ActivationReport,
    GuardCheck,
    LiveTradingBlockedError,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)
from rl_trader.config.masking import mask_account_id, redact_secret
from rl_trader.config.models import (
    ApprovalSettings,
    BacktestSettings,
    BenchmarkSettings,
    BrokerSettings,
    DatabaseSettings,
    DataSettings,
    EnvironmentSettings,
    FeatureSettings,
    FundamentalValuationSettings,
    LiveApprovalGateSettings,
    LiveTradingSettings,
    MonitoringSettings,
    PaperApprovalGateSettings,
    PaperTradingSettings,
    RiskSettings,
    TrainingSettings,
)
from rl_trader.config.settings import (
    AppSettings,
    RuntimeMode,
    Settings,
    get_settings,
    load_yaml_config,
)

__all__ = [
    "LIVE_TRADING_CONFIRMATION_PHRASE",
    "SUPPORTED_TICKERS",
    "ActivationReport",
    "AppSettings",
    "ApprovalSettings",
    "BacktestSettings",
    "BenchmarkSettings",
    "BrokerSettings",
    "DataSettings",
    "DatabaseSettings",
    "EnvironmentSettings",
    "FeatureSettings",
    "FundamentalValuationSettings",
    "GuardCheck",
    "LiveApprovalGateSettings",
    "LiveTradingBlockedError",
    "LiveTradingSettings",
    "MonitoringSettings",
    "PaperApprovalGateSettings",
    "PaperTradingSettings",
    "RiskSettings",
    "RuntimeMode",
    "Settings",
    "TrainingSettings",
    "assert_live_trading_allowed",
    "evaluate_live_activation",
    "get_settings",
    "is_live_trading_blocked",
    "load_yaml_config",
    "mask_account_id",
    "redact_secret",
]
