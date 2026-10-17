"""Random-policy control to sanity-check the trading environment.

A uniform random policy over the continuous long/cash action Box should not
systematically generate large profits if the env / cost model are wired
correctly. This control uses the same cost assumptions and reward definition
as trained agents.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from gymnasium import spaces

from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import SplitEpisodes
from rl_trader.agents.trainer import cost_assumptions_dict, reward_params_dict
from rl_trader.config.models import EnvironmentSettings, TrainingSettings
from rl_trader.environment.portfolio_env import PortfolioEnv


@dataclass(frozen=True, slots=True)
class RandomControlResult:
    """Summary of a random-policy roll-out on one split."""

    split: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_episode_return: float
    mean_final_nav: float
    starting_cash: float
    profit_flag: bool
    notes: str
    cost_assumptions: dict[str, Any]
    reward_parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_random_policy(
    splits: SplitEpisodes,
    *,
    environment: EnvironmentSettings,
    training: TrainingSettings,
    split: str = "test",
    n_episodes: int = 3,
    seed: int | None = None,
    scaler: FeatureScaler | None = None,
    profit_threshold: float = 0.05,
) -> RandomControlResult:
    """Roll out a seeded random policy on ``split`` (default: test).

    ``profit_flag`` is True when mean final NAV exceeds starting cash by more
    than ``profit_threshold`` (relative) — a soft sanity alarm, not a claim
    about markets.
    """
    if split == "train":
        episode = splits.train
    elif split == "validation":
        episode = splits.validation
    elif split == "test":
        episode = splits.test
    else:
        raise ValueError(f"unknown split {split!r}")

    if training.normalize_features:
        if scaler is None:
            scaler = FeatureScaler().fit(splits.train, split="train")
        episode = scaler.transform(episode)

    rng_seed = training.seed if seed is None else seed
    rewards: list[float] = []
    episode_returns: list[float] = []
    final_navs: list[float] = []

    for ep_i in range(n_episodes):
        env = PortfolioEnv(episode, environment, seed=rng_seed + ep_i)
        _obs, _info = env.reset(seed=rng_seed + ep_i)
        done = False
        ep_reward = 0.0
        start_nav = float(environment.initial_cash)
        step_i = 0
        info: dict[str, Any] = {}
        while not done:
            box = cast(spaces.Box, env.action_space)
            low = np.asarray(box.low, dtype=np.float64)
            high = np.asarray(box.high, dtype=np.float64)
            local = np.random.default_rng(rng_seed + 10_000 * ep_i + step_i)
            action = local.uniform(low=low, high=high).astype(np.float32)
            _obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = bool(terminated or truncated)
            step_i += 1
        end_nav = float(info.get("portfolio_value", start_nav))
        rewards.append(ep_reward)
        episode_returns.append((end_nav / start_nav) - 1.0 if start_nav > 0 else 0.0)
        final_navs.append(end_nav)

    mean_nav = float(np.mean(final_navs))
    start_cash = float(environment.initial_cash)
    relative_gain = (mean_nav / start_cash) - 1.0 if start_cash > 0 else 0.0
    profit_flag = relative_gain > profit_threshold
    notes = (
        "Random continuous actions over the env Box; same costs/rewards as RL agents. "
        "Large systematic profits would suggest an env implementation bug — "
        "not evidence of a tradable edge."
    )
    if profit_flag:
        notes += (
            f" WARNING: mean final NAV exceeded starting cash by "
            f"{relative_gain:.2%} (threshold={profit_threshold:.2%})."
        )

    return RandomControlResult(
        split=split,
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards, ddof=0)),
        mean_episode_return=float(np.mean(episode_returns)),
        mean_final_nav=mean_nav,
        starting_cash=start_cash,
        profit_flag=profit_flag,
        notes=notes,
        cost_assumptions=cost_assumptions_dict(environment),
        reward_parameters=reward_params_dict(environment),
    )
