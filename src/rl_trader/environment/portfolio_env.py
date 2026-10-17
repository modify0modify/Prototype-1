"""Gymnasium daily portfolio trading environment (research simulation).

Agents emit continuous long-only target weights. The environment simulates
fills, costs, and risk-adjusted rewards. No broker adapters are used.

Look-ahead safety: at decision index ``t`` the observation uses only
``features[t]`` and marks at ``prices[t]``. The step then advances to ``t+1``
to realize the period return from ``prices[t] → prices[t+1]``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np

from rl_trader.config.models import EnvironmentSettings
from rl_trader.environment.actions import project_target_weights
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.execution_sim import portfolio_nav, simulate_rebalance
from rl_trader.environment.reward import RewardBreakdown, compute_reward
from rl_trader.environment.spaces import (
    ObservationLayout,
    build_observation_layout,
    make_action_space,
    make_observation_space,
    pack_observation,
)
from rl_trader.environment.step_log import StepLogRecord, as_float_tuple
from rl_trader.environment.termination import TerminationReason, check_termination


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Daily stock / portfolio allocation environment (Gymnasium API)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        episode: EpisodeData,
        settings: EnvironmentSettings | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.episode = episode
        self.settings = settings or EnvironmentSettings()
        self.layout: ObservationLayout = build_observation_layout(
            episode, recent_returns_lookback=self.settings.recent_returns_lookback
        )
        self.action_space = make_action_space(
            episode.n_assets, self.settings.max_weight
        )
        self.observation_space = make_observation_space(self.layout)

        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._cash = 0.0
        self._shares = np.zeros(episode.n_assets, dtype=np.float64)
        self._peak_nav = 0.0
        self._recent_returns: deque[float] = deque(
            maxlen=self.settings.recent_returns_lookback
        )
        self._pending_weights: deque[np.ndarray | None] = deque()
        self._step_logs: list[StepLogRecord] = []
        self._observation_id = 0
        self._terminated = False

    @property
    def step_logs(self) -> list[StepLogRecord]:
        """Per-step audit records for the current episode."""
        return list(self._step_logs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset cash / positions to the start of the episode panel."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _ = options
        self._t = 0
        self._cash = float(self.settings.initial_cash)
        self._shares = np.zeros(self.episode.n_assets, dtype=np.float64)
        self._peak_nav = self._cash
        self._recent_returns.clear()
        for _ in range(self.settings.recent_returns_lookback):
            self._recent_returns.append(0.0)
        self._pending_weights.clear()
        for _ in range(self.settings.execution_delay_bars):
            self._pending_weights.append(None)
        self._step_logs.clear()
        self._observation_id = 0
        self._terminated = False
        obs = self._build_observation()
        info = {
            "timestamp": self._timestamp_str(self._t),
            "observation_id": self._observation_id,
            "portfolio_value": self._nav(),
            "termination_reason": TerminationReason.NONE.value,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulated trading day and advance the clock."""
        if self._terminated:
            raise RuntimeError("episode already terminated; call reset()")

        decision_t = self._t
        requested = project_target_weights(
            action, max_weight=self.settings.max_weight
        )
        # Optional execution delay: trade the delayed target at today's prices.
        self._pending_weights.append(requested)
        delayed = self._pending_weights.popleft()
        exec_weights = (
            delayed
            if delayed is not None
            else np.zeros(self.episode.n_assets, dtype=np.float64)
        )

        prices_t = self.episode.prices[decision_t]
        nav_before = self._nav_at(prices_t)
        pre_trade_positions = self._position_weights(prices_t, nav_before)

        fill = simulate_rebalance(
            cash=self._cash,
            shares=self._shares,
            prices=prices_t,
            target_weights=exec_weights,
            settings=self.settings,
        )
        self._cash = fill.cash_after
        self._shares = fill.approved_shares
        nav_after_trade = self._nav_at(prices_t)
        turnover = (
            float(np.sum(np.abs(fill.trade_notional)) / nav_before)
            if nav_before > 0
            else 0.0
        )
        cost_frac = fill.costs.total / nav_before if nav_before > 0 else 0.0

        next_t = decision_t + 1
        if next_t >= self.episode.n_bars:
            nav_end = nav_after_trade
            # Full-step return vs pre-trade NAV (costs already in cash).
            period_return = (
                (nav_end - nav_before) / nav_before if nav_before > 0 else -1.0
            )
            obs_t = self.episode.n_bars - 1
            terminated = True
            term_reason = TerminationReason.DATA_EXHAUSTED
        else:
            prices_next = self.episode.prices[next_t]
            nav_end = self._nav_at(prices_next)
            period_return = (
                (nav_end - nav_before) / nav_before if nav_before > 0 else -1.0
            )
            self._t = next_t
            self._peak_nav = max(self._peak_nav, nav_end)
            obs_t = self._t
            terminated = False
            term_reason = TerminationReason.NONE

        self._recent_returns.append(float(period_return))
        drawdown = self._drawdown(nav_end)
        obs = self._build_observation_at(obs_t)

        if not terminated:
            term_reason = check_termination(
                next_index=self._t + 1,
                n_bars=self.episode.n_bars,
                portfolio_value=nav_end,
                drawdown=drawdown,
                observation=obs,
                settings=self.settings,
            )
            # End-of-panel is handled when the next advance fails; only safety
            # conditions terminate early here.
            if term_reason in {
                TerminationReason.INVALID_PORTFOLIO_VALUE,
                TerminationReason.MAX_DRAWDOWN,
                TerminationReason.NON_FINITE_OBSERVATION,
            }:
                terminated = True
            else:
                term_reason = TerminationReason.NONE

        reward_bd = compute_reward(
            portfolio_return=period_return,
            drawdown=drawdown,
            turnover=turnover,
            transaction_cost_frac=cost_frac,
            weights=fill.approved_weights,
            recent_returns=np.asarray(self._recent_returns, dtype=np.float64),
            settings=self.settings,
        )
        info = self._finish_step(
            action=action,
            pre_trade_positions=pre_trade_positions,
            requested=requested,
            fill_approved=fill.approved_weights,
            trades=fill.trades_shares,
            costs_total=fill.costs.total,
            costs_breakdown={
                "commission": fill.costs.commission,
                "spread": fill.costs.spread,
                "slippage": fill.costs.slippage,
                "market_impact": fill.costs.market_impact,
            },
            nav_end=nav_end,
            period_return=period_return,
            drawdown=drawdown,
            reward_bd=reward_bd,
            term_reason=term_reason,
            terminated=terminated,
            log_timestamp_t=obs_t,
            rejected=fill.rejected,
            reject_reason=fill.reject_reason,
        )
        self._terminated = terminated
        return obs, float(reward_bd.total), terminated, False, info

    def _finish_step(
        self,
        *,
        action: np.ndarray,
        pre_trade_positions: np.ndarray,
        requested: np.ndarray,
        fill_approved: np.ndarray,
        trades: np.ndarray,
        costs_total: float,
        costs_breakdown: dict[str, float],
        nav_end: float,
        period_return: float,
        drawdown: float,
        reward_bd: RewardBreakdown,
        term_reason: TerminationReason,
        terminated: bool,
        log_timestamp_t: int,
        rejected: bool,
        reject_reason: str | None,
    ) -> dict[str, Any]:
        self._observation_id += 1
        log = StepLogRecord(
            timestamp=self._timestamp_str(log_timestamp_t),
            observation_id=self._observation_id,
            action=as_float_tuple(action),
            pre_trade_positions=as_float_tuple(pre_trade_positions),
            requested_positions=as_float_tuple(requested),
            approved_simulated_positions=as_float_tuple(fill_approved),
            trades=as_float_tuple(trades),
            costs=float(costs_total),
            portfolio_value=float(nav_end),
            return_=float(period_return),
            drawdown=float(drawdown),
            reward=float(reward_bd.total),
            termination_reason=term_reason.value if terminated else "",
            rejected_trade=rejected,
            reject_reason=reject_reason,
        )
        self._step_logs.append(log)
        return {
            **log.to_dict(),
            "reward_breakdown": {
                "portfolio_return": reward_bd.portfolio_return,
                "drawdown_penalty": reward_bd.drawdown_penalty,
                "turnover_penalty": reward_bd.turnover_penalty,
                "transaction_cost_penalty": reward_bd.transaction_cost_penalty,
                "concentration_penalty": reward_bd.concentration_penalty,
                "volatility_penalty": reward_bd.volatility_penalty,
            },
            "costs_breakdown": costs_breakdown,
        }

    def _nav(self) -> float:
        idx = min(self._t, self.episode.n_bars - 1)
        return self._nav_at(self.episode.prices[idx])

    def _nav_at(self, prices: np.ndarray) -> float:
        return portfolio_nav(self._cash, self._shares, prices)

    def _drawdown(self, nav: float) -> float:
        if self._peak_nav <= 0.0 or not np.isfinite(nav):
            return 1.0
        return max(0.0, (self._peak_nav - nav) / self._peak_nav)

    def _position_weights(self, prices: np.ndarray, nav: float) -> np.ndarray:
        if nav <= 0.0 or not np.isfinite(nav):
            return np.zeros(self.episode.n_assets, dtype=np.float64)
        return np.asarray((self._shares * prices) / nav, dtype=np.float64)

    def _build_observation(self) -> np.ndarray:
        return self._build_observation_at(self._t)

    def _build_observation_at(self, t: int) -> np.ndarray:
        prices = self.episode.prices[t]
        nav = self._nav_at(prices)
        cash_frac = self._cash / nav if nav > 0 else 1.0
        positions = self._position_weights(prices, nav)
        drawdown = self._drawdown(nav)
        recent = np.asarray(self._recent_returns, dtype=np.float64)
        return pack_observation(
            self.layout,
            market=self.episode.market_features[t],
            fundamental=self.episode.fundamental_features[t],
            regime=self.episode.regime_features[t],
            cash_frac=float(cash_frac),
            positions=positions,
            recent_returns=recent,
            drawdown=float(drawdown),
        )

    def _timestamp_str(self, t: int) -> str:
        value = self.episode.timestamps[t]
        if hasattr(value, "isoformat"):
            return str(value.isoformat())
        return str(value)
