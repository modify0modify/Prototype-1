"""OOS backtesting and benchmarks (research)."""

from rl_trader.backtesting.benchmarks import BuyAndHoldBenchmark, EqualWeightBenchmark
from rl_trader.backtesting.metrics import EvalResult, evaluate_equity_curve

__all__ = [
    "BuyAndHoldBenchmark",
    "EqualWeightBenchmark",
    "EvalResult",
    "evaluate_equity_curve",
]
