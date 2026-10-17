"""Evaluate a continuous-action policy on an episode with env cost model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from rl_trader.benchmarks.metrics import BenchmarkMetrics, compute_metrics
from rl_trader.config.models import EnvironmentSettings
from rl_trader.environment.actions import project_target_weights
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.portfolio_env import PortfolioEnv


class WeightPolicy(Protocol):
    """Emits target weights (SB3-compatible predict)."""

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        ...


@dataclass(frozen=True, slots=True)
class TradeLogRow:
    """One rebalance / mark event for fold trade logs."""

    bar: int
    timestamp: str
    nav_before: float
    nav_after: float
    turnover: float
    cost: float
    period_return: float
    target_weights: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyEvalResult:
    """Agent evaluation on one fold test window."""

    metrics: BenchmarkMetrics
    equity_curve: np.ndarray
    drawdowns: np.ndarray
    timestamps: np.ndarray
    daily_returns: np.ndarray
    turnovers: np.ndarray
    costs: np.ndarray
    trade_log: tuple[TradeLogRow, ...]
    regime_labels: np.ndarray


class EqualWeightPolicy:
    """Deterministic equal-weight policy (CI / synthetic smoke without SB3 train)."""

    def __init__(self, n_assets: int, max_weight: float) -> None:
        raw = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
        self._action = project_target_weights(raw, max_weight=max_weight).astype(
            np.float32
        )

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        _ = observation, deterministic
        return self._action.copy(), None


def evaluate_policy_on_episode(
    policy: WeightPolicy,
    episode: EpisodeData,
    env_settings: EnvironmentSettings,
    *,
    trading_days_per_year: int = 252,
    strategy_name: str = "agent",
    seed: int = 0,
    regime_labels: np.ndarray | None = None,
) -> PolicyEvalResult:
    """Roll ``PortfolioEnv`` with ``policy``; collect equity, costs, trade log."""
    env = PortfolioEnv(episode, env_settings, seed=seed)
    obs, _info = env.reset(seed=seed)
    equity: list[float] = [float(env_settings.initial_cash)]
    timestamps: list[Any] = [episode.timestamps[0]]
    daily_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    trades: list[TradeLogRow] = []
    done = False
    bar = 0
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        nav_before = float(env.unwrapped._nav())  # type: ignore[attr-defined]
        obs, _reward, terminated, truncated, info = env.step(action)
        nav_after = float(info.get("portfolio_value", nav_before))
        period_return = float(info.get("return_", 0.0))
        if period_return == 0.0 and nav_before > 0:
            period_return = (nav_after - nav_before) / nav_before
        cost = float(info.get("costs", 0.0))
        turnover = float(cost / nav_before) if nav_before > 0 else 0.0
        # Prefer notional turnover from trade shares when available.
        raw_trades = info.get("trades")
        if raw_trades is not None and bar < episode.n_bars:
            trade_arr = np.asarray(raw_trades, dtype=np.float64)
            prices = episode.prices[min(bar, episode.n_bars - 1)]
            notional = float(np.sum(np.abs(trade_arr * prices)))
            turnover = notional / nav_before if nav_before > 0 else 0.0
        equity.append(nav_after)
        ts = episode.timestamps[min(bar + 1, episode.n_bars - 1)]
        timestamps.append(ts)
        daily_returns.append(period_return)
        turnovers.append(turnover)
        costs.append(cost)
        trades.append(
            TradeLogRow(
                bar=bar,
                timestamp=str(ts)[:10],
                nav_before=nav_before,
                nav_after=nav_after,
                turnover=turnover,
                cost=cost,
                period_return=period_return,
                target_weights=tuple(float(x) for x in action),
            )
        )
        done = bool(terminated or truncated)
        bar += 1

    eq = np.asarray(equity, dtype=np.float64)
    peaks = np.maximum.accumulate(eq)
    dd = np.where(peaks > 0, (peaks - eq) / peaks, 0.0)
    rets = np.asarray(daily_returns, dtype=np.float64)
    metrics = compute_metrics(
        strategy=strategy_name,
        equity=eq,
        daily_returns=rets,
        turnovers=np.asarray(turnovers, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64),
        n_trades=sum(1 for t in trades if t.turnover > 1e-12),
        exposures=np.ones_like(rets),
        starting_capital=float(env_settings.initial_cash),
        trading_days_per_year=trading_days_per_year,
    )
    if regime_labels is None:
        labels = np.array(["unknown"] * max(rets.size, 1), dtype=object)
    else:
        labels = np.asarray(regime_labels, dtype=object)
        if labels.shape[0] != rets.shape[0] and rets.size:
            labels = np.resize(labels, rets.shape[0])
    return PolicyEvalResult(
        metrics=metrics,
        equity_curve=eq,
        drawdowns=dd,
        timestamps=np.asarray(timestamps, dtype=object),
        daily_returns=rets,
        turnovers=np.asarray(turnovers, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64),
        trade_log=tuple(trades),
        regime_labels=labels,
    )
