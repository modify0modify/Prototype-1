"""Deterministic benchmark strategies and shared evaluation harness."""

from rl_trader.benchmarks.harness import (
    StrategyRunResult,
    resolve_env_settings_for_benchmarks,
    run_all_benchmarks,
    run_strategy,
)
from rl_trader.benchmarks.metrics import BenchmarkMetrics, compute_metrics
from rl_trader.benchmarks.panel import PricePanel
from rl_trader.benchmarks.report import write_benchmark_report
from rl_trader.benchmarks.strategies import (
    BuyAndHoldSpyStrategy,
    CashOnlyStrategy,
    CrossSectionalMomentumStrategy,
    EqualWeightBuyAndHoldStrategy,
    InverseVolatilityStrategy,
    MeanReversionStrategy,
    MovingAverageTrendStrategy,
    build_default_strategies,
)
from rl_trader.benchmarks.strategy import BaseStrategy, Strategy
from rl_trader.benchmarks.synthetic import make_synthetic_price_panel

__all__ = [
    "BaseStrategy",
    "BenchmarkMetrics",
    "BuyAndHoldSpyStrategy",
    "CashOnlyStrategy",
    "CrossSectionalMomentumStrategy",
    "EqualWeightBuyAndHoldStrategy",
    "InverseVolatilityStrategy",
    "MeanReversionStrategy",
    "MovingAverageTrendStrategy",
    "PricePanel",
    "Strategy",
    "StrategyRunResult",
    "build_default_strategies",
    "compute_metrics",
    "make_synthetic_price_panel",
    "resolve_env_settings_for_benchmarks",
    "run_all_benchmarks",
    "run_strategy",
    "write_benchmark_report",
]
