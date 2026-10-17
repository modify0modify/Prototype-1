#!/usr/bin/env python3
"""Compare PPO / SAC / TD3 on identical splits, costs, and rewards.

Selects each algorithm's candidate using validation only; reports test once.
Includes a random-policy control. Does **not** declare a winning algorithm.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from rl_trader.agents.compare import compare_algorithms
from rl_trader.agents.registry import SUPPORTED_ALGORITHMS
from rl_trader.agents.splits import split_episode_by_dates, split_episode_by_fractions
from rl_trader.config import get_settings
from rl_trader.config.models import TrainingSettings
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.monitoring import configure_logging, get_logger
from rl_trader.utils.seeds import set_global_seeds


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()  # type: ignore[no-any-return]
    return date.fromisoformat(str(value)[:10])


def _daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _align(training: TrainingSettings, splits: object) -> TrainingSettings:
    from rl_trader.agents.splits import SplitEpisodes

    assert isinstance(splits, SplitEpisodes)
    return training.model_copy(
        update={
            "train_start": _as_date(splits.train.timestamps[0]),
            "train_end": _as_date(splits.train.timestamps[-1]),
            "val_start": _as_date(splits.validation.timestamps[0]),
            "val_end": _as_date(splits.validation.timestamps[-1]),
            "test_start": _as_date(splits.test.timestamps[0]),
            "test_end": _as_date(splits.test.timestamps[-1]),
        }
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for multi-algorithm comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument(
        "--algorithms",
        type=str,
        default="ppo,sac,td3",
        help="Comma-separated algorithms to compare",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Tiny synthetic episode (smoke compare; no network)",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.compare_agents")

    algos = tuple(a.strip().lower() for a in args.algorithms.split(",") if a.strip())
    for algo in algos:
        if algo not in SUPPORTED_ALGORITHMS:
            logger.error(
                "unsupported algorithm %r; supported=%s", algo, SUPPORTED_ALGORITHMS
            )
            return 2

    training = settings.training
    if args.timesteps is not None:
        training = training.model_copy(update={"total_timesteps": args.timesteps})

    set_global_seeds(training.seed)

    if args.fixture:
        n_assets = max(2, min(3, len(settings.data.universe)))
        symbols = tuple(settings.data.universe[:n_assets])
        raw = make_synthetic_episode(
            n_bars=48,
            n_assets=n_assets,
            n_market=2,
            n_fundamental=1,
            n_regime=1,
            seed=training.seed,
        )
        episode = EpisodeData(
            timestamps=raw.timestamps,
            prices=raw.prices,
            market_features=raw.market_features,
            fundamental_features=raw.fundamental_features,
            regime_features=raw.regime_features,
            symbols=symbols,
        )
        splits = split_episode_by_fractions(episode, train_frac=0.6, val_frac=0.2)
        training = _align(training, splits).model_copy(
            update={
                "model_id": "compare",
                "total_timesteps": args.timesteps or 256,
                "n_steps": 64,
                "batch_size": 32,
                "n_epochs": 1,
                "buffer_size": 2_000,
                "learning_starts": 64,
                "eval_freq": 64,
                "early_stopping_patience": 3,
                "min_evals_before_stop": 1,
                "evaluate_test_after_training": True,
            }
        )
        logger.info(
            "compare fixture bars=%d assets=%s algos=%s timesteps=%d",
            episode.n_bars,
            episode.symbols,
            algos,
            training.total_timesteps,
        )
    else:
        stamps = (
            _daterange(training.train_start, training.train_end)
            + _daterange(training.val_start, training.val_end)
            + _daterange(training.test_start, training.test_end)
        )
        raw = make_synthetic_episode(
            n_bars=len(stamps),
            n_assets=len(settings.data.universe),
            seed=training.seed,
        )
        episode = EpisodeData(
            timestamps=np.array(stamps, dtype=object),
            prices=raw.prices,
            market_features=raw.market_features,
            fundamental_features=raw.fundamental_features,
            regime_features=raw.regime_features,
            symbols=tuple(settings.data.universe),
        )
        splits = split_episode_by_dates(episode, training)
        training = training.model_copy(update={"model_id": "compare"})

    out_dir = args.output_dir or training.comparison_dir
    report = compare_algorithms(
        splits,
        training=training,
        environment=settings.environment,
        features=settings.features,
        algorithms=algos,
        output_dir=out_dir,
        random_episodes=2,
    )
    logger.info(
        "comparison written under %s (winner_declared=False) random_profit_flag=%s",
        out_dir,
        report.random_control.get("profit_flag"),
    )
    for row in report.rows:
        logger.info(
            "%s val=%s test=%s selected_on=%s",
            row.algorithm,
            row.best_val_mean_reward,
            row.test_mean_reward,
            row.model_selected_on,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
