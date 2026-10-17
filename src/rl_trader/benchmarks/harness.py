"""Shared evaluation harness for deterministic benchmarks.

Every strategy is run on the same price panel, dates, starting capital,
long+cash position rules, and the portfolio-env cost / execution simulator
so transaction-cost assumptions match the RL environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rl_trader.benchmarks.metrics import BenchmarkMetrics, compute_metrics
from rl_trader.benchmarks.panel import PricePanel
from rl_trader.benchmarks.strategies import build_default_strategies
from rl_trader.benchmarks.strategy import BaseStrategy
from rl_trader.config.models import (
    BenchmarkSettings,
    EnvironmentSettings,
)
from rl_trader.environment.actions import project_target_weights
from rl_trader.environment.execution_sim import portfolio_nav, simulate_rebalance


def _sanitize_long_weights(raw: np.ndarray) -> np.ndarray:
    """Non-negative weights with sum ≤ 1 (no per-asset cap)."""
    w = np.maximum(np.asarray(raw, dtype=np.float64).reshape(-1), 0.0)
    total = float(np.sum(w))
    if total > 1.0:
        w = w / total
    return w


@dataclass(frozen=True, slots=True)
class StrategyRunResult:
    """One strategy evaluation on the shared harness."""

    metrics: BenchmarkMetrics
    equity_curve: np.ndarray
    timestamps: np.ndarray
    daily_returns: np.ndarray
    assumptions: dict[str, Any]


def resolve_env_settings_for_benchmarks(
    *,
    environment: EnvironmentSettings,
    benchmarks: BenchmarkSettings,
    initial_cash: float | None = None,
) -> EnvironmentSettings:
    """Build the EnvironmentSettings used for simulated fills.

    When ``benchmarks.use_environment_costs`` is True (default), commission /
    spread / slippage / impact / max_weight / delay match the RL env. Starting
    cash may still be overridden (e.g. from ``backtest.initial_cash``).
    """
    cash = float(initial_cash if initial_cash is not None else environment.initial_cash)
    if benchmarks.use_environment_costs:
        return environment.model_copy(update={"initial_cash": cash})
    # Fallback: still use env shape but caller may have synced bps elsewhere.
    return environment.model_copy(update={"initial_cash": cash})


def run_strategy(
    strategy: BaseStrategy,
    panel: PricePanel,
    env_settings: EnvironmentSettings,
    *,
    trading_days_per_year: int = 252,
) -> StrategyRunResult:
    """Evaluate one strategy with env-matching cost / execution simulation."""
    strategy.prepare(panel.prices, panel.symbols)
    lookback = max(1, int(strategy.required_lookback))
    start_t = lookback - 1
    if start_t >= panel.n_bars - 1:
        raise ValueError(
            f"{strategy.name}: required_lookback={lookback} leaves no tradeable bars "
            f"(n_bars={panel.n_bars})"
        )

    cash = float(env_settings.initial_cash)
    shares = np.zeros(panel.n_assets, dtype=np.float64)
    equity: list[float] = [cash]
    eq_timestamps: list[Any] = [panel.timestamps[start_t]]
    daily_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    exposures: list[float] = []
    n_trades = 0
    # Match PortfolioEnv: delay>=1 queues targets so close-informed decisions
    # do not fill at the same bar's close.
    from collections import deque

    pending: deque[np.ndarray | None] = deque()
    for _ in range(max(0, int(env_settings.execution_delay_bars))):
        pending.append(None)

    for t in range(start_t, panel.n_bars - 1):
        raw = strategy.target_weights(t)
        if strategy.enforce_max_weight:
            target = project_target_weights(raw, max_weight=env_settings.max_weight)
        else:
            # Buy-and-hold: entry already respected max_weight; allow drift above cap.
            target = _sanitize_long_weights(raw)
        pending.append(target)
        delayed = pending.popleft()
        exec_weights = (
            delayed
            if delayed is not None
            else np.zeros(panel.n_assets, dtype=np.float64)
        )
        prices_t = panel.prices[t]
        nav_before = portfolio_nav(cash, shares, prices_t)
        fill = simulate_rebalance(
            cash=cash,
            shares=shares,
            prices=prices_t,
            target_weights=exec_weights,
            settings=env_settings,
            enforce_max_weight=strategy.enforce_max_weight,
        )
        cash = fill.cash_after
        shares = fill.approved_shares
        turnover = (
            float(np.sum(np.abs(fill.trade_notional)) / nav_before)
            if nav_before > 0
            else 0.0
        )
        n_trades += int(np.sum(np.abs(fill.trades_shares) > 1e-12))
        prices_next = panel.prices[t + 1]
        nav_end = portfolio_nav(cash, shares, prices_next)
        period_return = (
            (nav_end - nav_before) / nav_before if nav_before > 0 else -1.0
        )
        exposure = float(np.sum(fill.approved_weights))

        equity.append(nav_end)
        eq_timestamps.append(panel.timestamps[t + 1])
        daily_returns.append(period_return)
        turnovers.append(turnover)
        costs.append(float(fill.costs.total))
        exposures.append(exposure)

    eq_arr = np.asarray(equity, dtype=np.float64)
    metrics = compute_metrics(
        strategy=strategy.name,
        equity=eq_arr,
        daily_returns=np.asarray(daily_returns, dtype=np.float64),
        turnovers=np.asarray(turnovers, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64),
        n_trades=n_trades,
        exposures=np.asarray(exposures, dtype=np.float64),
        starting_capital=float(env_settings.initial_cash),
        trading_days_per_year=trading_days_per_year,
    )
    assumptions = {
        "initial_cash": env_settings.initial_cash,
        "max_weight": env_settings.max_weight,
        "commission_bps": env_settings.commission_bps,
        "bid_ask_spread_bps": env_settings.bid_ask_spread_bps,
        "slippage_bps": env_settings.slippage_bps,
        "market_impact_coeff": env_settings.market_impact_coeff,
        "execution_delay_bars": env_settings.execution_delay_bars,
        "position_rules": "long_plus_cash_v1",
        "cost_model": "portfolio_env.simulate_rebalance",
        "symbols": list(panel.symbols),
        "start": str(panel.timestamps[start_t]),
        "end": str(panel.timestamps[-1]),
        "required_lookback": lookback,
    }
    return StrategyRunResult(
        metrics=metrics,
        equity_curve=eq_arr,
        timestamps=np.asarray(eq_timestamps, dtype=object),
        daily_returns=np.asarray(daily_returns, dtype=np.float64),
        assumptions=assumptions,
    )


def run_all_benchmarks(
    panel: PricePanel,
    *,
    env_settings: EnvironmentSettings,
    benchmark_settings: BenchmarkSettings | None = None,
    strategies: list[BaseStrategy] | None = None,
) -> list[StrategyRunResult]:
    """Run the default (or provided) suite on one shared panel."""
    settings = benchmark_settings or BenchmarkSettings()
    suite = (
        strategies
        if strategies is not None
        else build_default_strategies(settings, max_weight=env_settings.max_weight)
    )
    results: list[StrategyRunResult] = []
    for strategy in suite:
        results.append(
            run_strategy(
                strategy,
                panel,
                env_settings,
                trading_days_per_year=settings.trading_days_per_year,
            )
        )
    return results
