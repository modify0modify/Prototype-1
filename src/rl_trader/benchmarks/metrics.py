"""Comparison metrics for benchmark / backtest equity curves."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Full comparison metric set for one strategy run."""

    strategy: str
    cumulative_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    turnover: float
    transaction_costs: float
    number_of_trades: int
    win_rate: float
    profit_factor: float
    exposure: float
    worst_day: float
    best_day: float
    starting_capital: float
    ending_capital: float
    n_days: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return asdict(self)


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if den == 0.0 or not np.isfinite(den) or not np.isfinite(num):
        return default
    return float(num / den)


def compute_metrics(
    *,
    strategy: str,
    equity: np.ndarray,
    daily_returns: np.ndarray,
    turnovers: np.ndarray,
    costs: np.ndarray,
    n_trades: int,
    exposures: np.ndarray,
    starting_capital: float,
    trading_days_per_year: int = 252,
) -> BenchmarkMetrics:
    """Compute comparison metrics from aligned daily series."""
    equity = np.asarray(equity, dtype=np.float64)
    rets = np.asarray(daily_returns, dtype=np.float64)
    if equity.size < 1:
        raise ValueError("equity curve is empty")
    ending = float(equity[-1])
    cum = _safe_div(ending, starting_capital, default=1.0) - 1.0
    n = int(rets.size)
    if n == 0:
        return BenchmarkMetrics(
            strategy=strategy,
            cumulative_return=cum,
            annualised_return=0.0,
            annualised_volatility=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            turnover=0.0,
            transaction_costs=float(np.sum(costs)) if costs.size else 0.0,
            number_of_trades=int(n_trades),
            win_rate=0.0,
            profit_factor=0.0,
            exposure=0.0,
            worst_day=0.0,
            best_day=0.0,
            starting_capital=float(starting_capital),
            ending_capital=ending,
            n_days=0,
        )

    finite = rets[np.isfinite(rets)]
    mean_r = float(np.mean(finite)) if finite.size else 0.0
    std_r = float(np.std(finite, ddof=0)) if finite.size else 0.0
    downside = finite[finite < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if downside.size else 0.0
    ann = trading_days_per_year
    ann_ret = (1.0 + mean_r) ** ann - 1.0 if np.isfinite(mean_r) else 0.0
    ann_vol = std_r * float(np.sqrt(ann))
    sharpe = _safe_div(mean_r, std_r) * float(np.sqrt(ann))
    sortino = _safe_div(mean_r, downside_std) * float(np.sqrt(ann))

    peaks = np.maximum.accumulate(equity)
    dd = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_dd = float(np.max(dd)) if dd.size else 0.0
    calmar = _safe_div(ann_ret, max_dd) if max_dd > 0 else 0.0

    wins = finite[finite > 0]
    losses = finite[finite < 0]
    win_rate = _safe_div(float(wins.size), float(finite.size))
    gross_profit = float(np.sum(wins)) if wins.size else 0.0
    gross_loss = float(-np.sum(losses)) if losses.size else 0.0
    profit_factor = _safe_div(gross_profit, gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    if not np.isfinite(profit_factor):
        profit_factor = 0.0 if gross_profit <= 0 else 1e6

    return BenchmarkMetrics(
        strategy=strategy,
        cumulative_return=float(cum),
        annualised_return=float(ann_ret),
        annualised_volatility=float(ann_vol),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_dd),
        calmar=float(calmar),
        turnover=float(np.sum(turnovers)) if turnovers.size else 0.0,
        transaction_costs=float(np.sum(costs)) if costs.size else 0.0,
        number_of_trades=int(n_trades),
        win_rate=float(win_rate),
        profit_factor=float(profit_factor),
        exposure=float(np.mean(exposures)) if exposures.size else 0.0,
        worst_day=float(np.min(finite)) if finite.size else 0.0,
        best_day=float(np.max(finite)) if finite.size else 0.0,
        starting_capital=float(starting_capital),
        ending_capital=ending,
        n_days=n,
    )
