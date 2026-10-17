"""OOS backtesting helpers (research). Prefer :mod:`rl_trader.benchmarks`."""

from rl_trader.backtesting.benchmarks import BuyAndHoldBenchmark, EqualWeightBenchmark
from rl_trader.backtesting.metrics import EvalResult, evaluate_equity_curve
from rl_trader.benchmarks import (
    run_all_benchmarks,
    write_benchmark_report,
)

__all__ = [
    "BuyAndHoldBenchmark",
    "EqualWeightBenchmark",
    "EvalResult",
    "evaluate_equity_curve",
    "run_all_benchmarks",
    "write_benchmark_report",
]
