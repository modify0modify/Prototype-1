"""Unit tests for the Gymnasium PortfolioEnv (offline fixtures only)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_trader.config import get_settings
from rl_trader.config.models import EnvironmentSettings
from rl_trader.environment import (
    EpisodeData,
    PortfolioEnv,
    TerminationReason,
    project_target_weights,
)
from rl_trader.environment.costs import estimate_trade_costs
from rl_trader.environment.execution_sim import simulate_rebalance
from rl_trader.environment.reward import compute_reward
from rl_trader.environment.synthetic import make_synthetic_episode


def _settings(**overrides: object) -> EnvironmentSettings:
    base: dict[str, object] = {
        "initial_cash": 100_000.0,
        "max_weight": 0.5,
        "commission_bps": 10.0,
        "bid_ask_spread_bps": 4.0,
        "slippage_bps": 5.0,
        "market_impact_coeff": 0.05,
        "execution_delay_bars": 0,
        "recent_returns_lookback": 5,
        "drawdown_penalty": 0.1,
        "turnover_penalty": 0.05,
        "transaction_cost_penalty": 1.0,
        "concentration_penalty": 0.01,
        "volatility_penalty": 0.01,
        "max_drawdown_terminate": 0.5,
        "min_portfolio_value": 1.0,
    }
    base.update(overrides)
    return EnvironmentSettings(**base)  # type: ignore[arg-type]


def test_project_target_weights_caps_and_cash_residual() -> None:
    w = project_target_weights(np.array([0.9, 0.9]), max_weight=0.5)
    assert np.all(w <= 0.5 + 1e-9)
    assert float(np.sum(w)) <= 1.0 + 1e-9
    assert np.allclose(w, [0.5, 0.5])


def test_project_scales_when_sum_exceeds_one() -> None:
    w = project_target_weights(np.array([0.8, 0.8]), max_weight=1.0)
    assert np.isclose(float(np.sum(w)), 1.0)
    assert np.allclose(w, [0.5, 0.5])


def test_observation_space_shape() -> None:
    episode = make_synthetic_episode(
        n_bars=10, n_assets=3, n_market=4, n_fundamental=2, n_regime=3
    )
    env = PortfolioEnv(episode, _settings(recent_returns_lookback=5, max_weight=0.4))
    # 4 market + 2 fund + 3 regime + cash + 3 positions + 5 recent returns + drawdown
    assert env.observation_space.shape == (19,)
    assert env.action_space.shape == (3,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (19,)
    assert np.isfinite(obs).all()
    assert info["portfolio_value"] == pytest.approx(100_000.0)


def test_check_env_passes() -> None:
    episode = make_synthetic_episode(n_bars=20, n_assets=2)
    env = PortfolioEnv(episode, _settings(max_weight=0.5))
    check_env(env, skip_render_check=True)


def test_costs_reduce_nav_on_rebalance() -> None:
    settings = _settings(commission_bps=50.0, slippage_bps=50.0, bid_ask_spread_bps=20.0)
    prices = np.array([100.0, 100.0])
    fill = simulate_rebalance(
        cash=100_000.0,
        shares=np.zeros(2),
        prices=prices,
        target_weights=np.array([0.4, 0.4]),
        settings=settings,
    )
    nav = fill.cash_after + float(np.dot(fill.approved_shares, prices))
    assert fill.costs.total > 0.0
    assert nav < 100_000.0


def test_estimate_trade_costs_components() -> None:
    settings = _settings()
    costs = estimate_trade_costs(
        np.array([10_000.0, -5_000.0]),
        portfolio_value=100_000.0,
        settings=settings,
    )
    assert costs.commission > 0
    assert costs.spread > 0
    assert costs.slippage > 0
    assert costs.market_impact > 0
    assert costs.total == pytest.approx(
        costs.commission + costs.spread + costs.slippage + costs.market_impact
    )


def test_reward_penalties_reduce_raw_return() -> None:
    settings = _settings(
        drawdown_penalty=1.0,
        turnover_penalty=1.0,
        transaction_cost_penalty=1.0,
        concentration_penalty=1.0,
        volatility_penalty=1.0,
    )
    bd = compute_reward(
        portfolio_return=0.01,
        drawdown=0.05,
        turnover=0.1,
        transaction_cost_frac=0.001,
        weights=np.array([0.5, 0.5]),
        recent_returns=np.array([0.01, -0.01, 0.02, 0.0, 0.0]),
        settings=settings,
    )
    assert bd.total < bd.portfolio_return


def test_termination_data_exhausted() -> None:
    episode = make_synthetic_episode(n_bars=5, n_assets=2, price_path="flat")
    env = PortfolioEnv(
        episode,
        _settings(
            max_weight=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
            bid_ask_spread_bps=0.0,
            market_impact_coeff=0.0,
        ),
    )
    env.reset(seed=0)
    action = np.array([0.0, 0.0], dtype=np.float32)
    terminated = False
    last_reason = ""
    for _ in range(10):
        _, _, terminated, _, info = env.step(action)
        last_reason = info["termination_reason"]
        if terminated:
            break
    assert terminated
    assert last_reason == TerminationReason.DATA_EXHAUSTED.value


def test_termination_max_drawdown() -> None:
    episode = make_synthetic_episode(
        n_bars=20,
        n_assets=1,
        price_path="crash",
        n_market=1,
        n_fundamental=1,
        n_regime=1,
    )
    env = PortfolioEnv(
        episode,
        _settings(
            max_weight=1.0,
            max_drawdown_terminate=0.05,
            commission_bps=0.0,
            slippage_bps=0.0,
            bid_ask_spread_bps=0.0,
            market_impact_coeff=0.0,
            drawdown_penalty=0.0,
            turnover_penalty=0.0,
            transaction_cost_penalty=0.0,
        ),
    )
    env.reset(seed=0)
    terminated = False
    reason = ""
    for _ in range(20):
        _, _, terminated, _, info = env.step(np.array([1.0], dtype=np.float32))
        reason = info["termination_reason"]
        if terminated:
            break
    assert terminated
    assert reason == TerminationReason.MAX_DRAWDOWN.value


def test_termination_invalid_portfolio_value() -> None:
    episode = make_synthetic_episode(
        n_bars=5, n_assets=1, n_market=1, n_fundamental=0, n_regime=0
    )
    env = PortfolioEnv(episode, _settings(min_portfolio_value=1e12, max_weight=1.0))
    env.reset(seed=0)
    env._cash = 0.5
    env._shares[:] = 0.0
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert info["termination_reason"] == TerminationReason.INVALID_PORTFOLIO_VALUE.value


def test_lookahead_observation_uses_current_features_only() -> None:
    episode = make_synthetic_episode(n_bars=8, n_assets=2)
    env = PortfolioEnv(
        episode,
        _settings(
            max_weight=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
            bid_ask_spread_bps=0.0,
            market_impact_coeff=0.0,
        ),
    )
    obs0, _ = env.reset(seed=0)
    n_m = episode.n_market
    assert np.allclose(obs0[:n_m], episode.market_features[0])
    assert not np.allclose(obs0[:n_m], episode.market_features[1])
    obs1, _, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(obs1[:n_m], episode.market_features[1])


def test_step_log_fields_present() -> None:
    episode = make_synthetic_episode(n_bars=6, n_assets=2)
    env = PortfolioEnv(episode, _settings())
    env.reset(seed=1)
    _, _, _, _, info = env.step(np.array([0.2, 0.2], dtype=np.float32))
    for key in (
        "timestamp",
        "observation_id",
        "action",
        "pre_trade_positions",
        "requested_positions",
        "approved_simulated_positions",
        "trades",
        "costs",
        "portfolio_value",
        "return",
        "drawdown",
        "reward",
        "termination_reason",
    ):
        assert key in info
    assert len(env.step_logs) == 1


def test_execution_delay_defers_trades() -> None:
    episode = make_synthetic_episode(n_bars=10, n_assets=2, price_path="flat")
    env = PortfolioEnv(
        episode,
        _settings(
            execution_delay_bars=1,
            max_weight=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
            bid_ask_spread_bps=0.0,
            market_impact_coeff=0.0,
        ),
    )
    env.reset(seed=0)
    _, _, _, _, info0 = env.step(np.array([0.5, 0.0], dtype=np.float32))
    assert sum(abs(t) for t in info0["trades"]) == pytest.approx(0.0)
    _, _, _, _, info1 = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert sum(abs(t) for t in info1["trades"]) > 0.0


def test_cash_constraint_scales_buys() -> None:
    settings = _settings(
        max_weight=1.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        bid_ask_spread_bps=0.0,
        market_impact_coeff=0.0,
    )
    fill = simulate_rebalance(
        cash=100.0,
        shares=np.array([0.0, 0.0]),
        prices=np.array([1000.0, 1000.0]),
        target_weights=np.array([0.5, 0.5]),
        settings=settings,
    )
    assert fill.cash_after >= -1e-8
    assert float(np.sum(fill.approved_shares * np.array([1000.0, 1000.0]))) <= 100.0 + 1e-6


def test_rejected_invalid_prices() -> None:
    settings = _settings()
    fill = simulate_rebalance(
        cash=10_000.0,
        shares=np.array([0.0]),
        prices=np.array([np.nan]),
        target_weights=np.array([0.5]),
        settings=settings,
    )
    assert fill.rejected
    assert fill.reject_reason == "invalid_prices_or_cash"


def test_non_finite_observation_terminates() -> None:
    episode = make_synthetic_episode(
        n_bars=6, n_assets=1, n_market=1, n_fundamental=0, n_regime=0
    )
    episode.market_features[1, 0] = np.nan
    env = PortfolioEnv(
        episode,
        _settings(
            max_weight=1.0,
            commission_bps=0.0,
            slippage_bps=0.0,
            bid_ask_spread_bps=0.0,
            market_impact_coeff=0.0,
        ),
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert terminated
    assert info["termination_reason"] == TerminationReason.NON_FINITE_OBSERVATION.value


def test_episode_validation_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="prices shape"):
        EpisodeData(
            timestamps=np.array(
                [date(2023, 1, 3), date(2023, 1, 3) + timedelta(days=1)], dtype=object
            ),
            prices=np.ones((2, 3)),
            market_features=np.zeros((2, 1)),
            fundamental_features=np.zeros((2, 1)),
            regime_features=np.zeros((2, 1)),
            symbols=("A", "B"),
        )


def test_training_yaml_loads_environment(monkeypatch: object) -> None:
    for key in ("APP_MODE", "LIVE_TRADING_ENABLED", "BROKER_ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    root = Path(__file__).resolve().parents[2]
    settings = get_settings(config_file=root / "configs" / "training.yaml")
    assert settings.environment.max_weight == 0.25
    assert settings.environment.commission_bps == 1.0
    assert settings.environment.drawdown_penalty == 0.1
