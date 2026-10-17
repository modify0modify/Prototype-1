"""Backtest metrics — thin re-export / compatibility layer."""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_trader.benchmarks.metrics import BenchmarkMetrics, compute_metrics

# Backward-compatible stub name used by evaluate_agent script.
EvalResult = BenchmarkMetrics


def evaluate_equity_curve(curve: Any) -> BenchmarkMetrics:
    """Compute metrics from an equity curve array (or stub empty curve)."""
    if curve is None:
        return BenchmarkMetrics(
            strategy="stub",
            cumulative_return=0.0,
            annualised_return=0.0,
            annualised_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            turnover=0.0,
            transaction_costs=0.0,
            number_of_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            exposure=0.0,
            worst_day=0.0,
            best_day=0.0,
            starting_capital=0.0,
            ending_capital=0.0,
            n_days=0,
        )
    equity = np.asarray(curve, dtype=np.float64).reshape(-1)
    if equity.size < 2:
        start = float(equity[0]) if equity.size else 0.0
        return compute_metrics(
            strategy="equity_curve",
            equity=equity if equity.size else np.array([0.0]),
            daily_returns=np.array([]),
            turnovers=np.array([]),
            costs=np.array([]),
            n_trades=0,
            exposures=np.array([]),
            starting_capital=start if start > 0 else 1.0,
        )
    rets = equity[1:] / equity[:-1] - 1.0
    return compute_metrics(
        strategy="equity_curve",
        equity=equity,
        daily_returns=rets,
        turnovers=np.zeros_like(rets),
        costs=np.zeros_like(rets),
        n_trades=0,
        exposures=np.ones_like(rets),
        starting_capital=float(equity[0]),
    )
