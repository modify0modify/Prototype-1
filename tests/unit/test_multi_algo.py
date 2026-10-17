"""Tests for multi-algorithm agent interface, wiring, and comparison hygiene."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from rl_trader.agents.compare import compare_algorithms
from rl_trader.agents.interface import SB3PortfolioAgent
from rl_trader.agents.random_control import evaluate_random_policy
from rl_trader.agents.registry import SUPPORTED_ALGORITHMS, get_algorithm_spec, list_algorithms
from rl_trader.agents.splits import split_episode_by_fractions
from rl_trader.agents.trainer import (
    cost_assumptions_dict,
    reward_params_dict,
    train_agent,
)
from rl_trader.config.models import EnvironmentSettings, FeatureSettings, TrainingSettings
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.portfolio.targets import TargetPortfolio


def _env() -> EnvironmentSettings:
    return EnvironmentSettings(
        initial_cash=10_000.0,
        max_weight=0.5,
        commission_bps=1.0,
        bid_ask_spread_bps=2.0,
        slippage_bps=1.0,
        market_impact_coeff=0.0,
        drawdown_penalty=0.1,
        turnover_penalty=0.01,
        transaction_cost_penalty=1.0,
    )


def _splits():
    episode = make_synthetic_episode(
        n_bars=40, n_assets=2, n_market=2, n_fundamental=1, n_regime=1, seed=2
    )
    return split_episode_by_fractions(episode, train_frac=0.6, val_frac=0.2)


def _tiny(algorithm: str, **overrides: object) -> TrainingSettings:
    base: dict[str, object] = {
        "algorithm": algorithm,
        "model_id": f"{algorithm}-unit",
        "total_timesteps": 128,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "n_steps": 64,
        "n_epochs": 1,
        "buffer_size": 1_000,
        "learning_starts": 32,
        "gamma": 0.99,
        "seed": 0,
        "device": "cpu",
        "eval_freq": 64,
        "n_eval_episodes": 1,
        "early_stopping_patience": 2,
        "min_evals_before_stop": 1,
        "normalize_features": True,
        "evaluate_test_after_training": True,
        "train_start": date(2023, 1, 3),
        "train_end": date(2023, 1, 20),
        "val_start": date(2023, 1, 21),
        "val_end": date(2023, 1, 28),
        "test_start": date(2023, 1, 29),
        "test_end": date(2023, 2, 5),
    }
    base.update(overrides)
    return TrainingSettings(**base)  # type: ignore[arg-type]


def test_registry_lists_ppo_sac_td3() -> None:
    assert list_algorithms() == ("ppo", "sac", "td3")
    assert SUPPORTED_ALGORITHMS == ("ppo", "sac", "td3")
    for name in SUPPORTED_ALGORITHMS:
        spec = get_algorithm_spec(name)
        assert spec.name == name
        assert spec.model_filename.endswith(".zip")


def test_registry_rejects_unknown() -> None:
    with pytest.raises(KeyError, match="unsupported"):
        get_algorithm_spec("a2c")


def test_identical_cost_and_reward_wiring() -> None:
    env = _env()
    costs = cost_assumptions_dict(env)
    rewards = reward_params_dict(env)
    # Same snapshots would be embedded in every algorithm's manifest.
    assert costs["commission_bps"] == env.commission_bps
    assert rewards["drawdown_penalty"] == env.drawdown_penalty
    for algo in SUPPORTED_ALGORITHMS:
        hparams = get_algorithm_spec(algo).hyperparameters(_tiny(algo))
        assert hparams["algorithm"] == algo
        assert "learning_rate" in hparams


def test_val_only_selection_and_test_once(tmp_path: Path) -> None:
    splits = _splits()
    env = _env()
    result = train_agent(
        splits,
        training=_tiny("sac", evaluate_test_after_training=True),
        environment=env,
        features=FeatureSettings(),
        output_dir=tmp_path / "sac",
        algorithm="sac",
    )
    assert result.manifest.model_selected_on == "validation"
    assert result.access_guard.fitted_scaler_on == "train"
    assert result.test_mean_reward is not None
    assert result.access_guard.test_eval_completed is True
    assert result.manifest.algorithm_hyperparameters["algorithm"] == "sac"
    assert (result.output_dir / "algorithm_hyperparameters.json").exists()
    assert (result.output_dir / "sac_model.zip").exists()


def test_sb3_agent_predict_target_continuous(tmp_path: Path) -> None:
    splits = _splits()
    result = train_agent(
        splits,
        training=_tiny("td3", evaluate_test_after_training=False, total_timesteps=96),
        environment=_env(),
        output_dir=tmp_path / "td3",
        algorithm="td3",
    )
    agent = SB3PortfolioAgent.load(result.output_dir, "td3")
    shape = agent.model.observation_space.shape
    obs = np.zeros(shape, dtype=np.float32)
    target = agent.predict_target(obs)
    assert isinstance(target, TargetPortfolio)
    assert target.weights.shape == (splits.train.n_assets,)
    assert np.all(target.weights >= -1e-6)
    # Action Box high is max_weight
    assert float(np.max(target.weights)) <= 0.5 + 1e-5


def test_random_policy_control_runs() -> None:
    splits = _splits()
    env = _env()
    control = evaluate_random_policy(
        splits,
        environment=env,
        training=_tiny("ppo"),
        split="test",
        n_episodes=2,
        seed=0,
        profit_threshold=0.5,  # lenient for short synthetic paths
    )
    assert control.split == "test"
    assert control.n_episodes == 2
    assert control.starting_cash == env.initial_cash
    assert control.cost_assumptions == cost_assumptions_dict(env)
    assert control.reward_parameters == reward_params_dict(env)
    # Soft check: control should not scream profit on this short noisy fixture
    # with costs; if it does, profit_flag documents the anomaly.
    assert isinstance(control.profit_flag, bool)


def test_compare_algorithms_smoke(tmp_path: Path) -> None:
    splits = _splits()
    report = compare_algorithms(
        splits,
        training=_tiny("ppo", model_id="cmp", total_timesteps=96, eval_freq=48),
        environment=_env(),
        features=FeatureSettings(),
        algorithms=("ppo", "sac"),
        output_dir=tmp_path / "cmp",
        random_episodes=1,
    )
    assert report.to_dict()["winner_declared"] is False
    assert set(report.algorithms) == {"ppo", "sac"}
    assert len(report.rows) == 2
    for row in report.rows:
        assert row.model_selected_on == "validation"
        assert row.test_used_for_selection is False
        assert row.test_mean_reward is not None
    # Shared wiring
    assert report.shared_cost_assumptions == cost_assumptions_dict(_env())
    assert "profit_flag" in report.random_control
    assert (tmp_path / "cmp" / "latest.md").exists()
    text = (tmp_path / "cmp" / "latest.md").read_text(encoding="utf-8")
    assert "Winner declared" in text
    assert "No algorithm is declared best" in text
