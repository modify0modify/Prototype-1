#!/usr/bin/env python3
"""Train a PPO portfolio agent (Stable-Baselines3).

Reads settings from YAML/config. Fits feature scalers on the training window
only; early-stops / selects on validation; never uses the test set for fitting
or model selection. Use ``--fixture`` for a tiny offline smoke run.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from rl_trader.agents.ppo_trainer import train_ppo
from rl_trader.agents.splits import (
    SplitEpisodes,
    split_episode_by_dates,
    split_episode_by_fractions,
)
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


def _align_settings_to_split(
    training: TrainingSettings, splits: SplitEpisodes
) -> TrainingSettings:
    """Rewrite inclusive date windows to match the materialized split panels."""
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


def _daterange(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _episode_for_configured_dates(
    training: TrainingSettings,
    universe: tuple[str, ...],
    seed: int,
) -> EpisodeData:
    """Build an offline synthetic panel whose timestamps match config windows."""
    stamps = (
        _daterange(training.train_start, training.train_end)
        + _daterange(training.val_start, training.val_end)
        + _daterange(training.test_start, training.test_end)
    )
    if len(stamps) < 6:
        raise ValueError("configured train/val/test windows are too short")
    raw = make_synthetic_episode(
        n_bars=len(stamps),
        n_assets=len(universe),
        n_market=3,
        n_fundamental=2,
        n_regime=2,
        seed=seed,
    )
    return EpisodeData(
        timestamps=np.array(stamps, dtype=object),
        prices=raw.prices,
        market_features=raw.market_features,
        fundamental_features=raw.fundamental_features,
        regime_features=raw.regime_features,
        symbols=tuple(universe),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for PPO training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research.yaml"),
        help="YAML config path",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        help="Override training.algorithm (only 'ppo' supported)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use a tiny synthetic episode (smoke training; no network)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override training.total_timesteps",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override training.model_id / output folder name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override checkpoint output directory",
    )
    args = parser.parse_args(argv)

    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.train_agent")

    training = settings.training
    updates: dict[str, object] = {}
    if args.algorithm is not None:
        updates["algorithm"] = args.algorithm.lower()
    if args.timesteps is not None:
        updates["total_timesteps"] = args.timesteps
    if args.model_id is not None:
        updates["model_id"] = args.model_id
    if updates:
        training = training.model_copy(update=updates)

    if training.algorithm != "ppo":
        logger.error("unsupported algorithm %r (only ppo)", training.algorithm)
        return 2

    set_global_seeds(training.seed)

    if args.fixture:
        n_assets = max(2, min(4, len(settings.data.universe)))
        symbols = tuple(settings.data.universe[:n_assets])
        raw = make_synthetic_episode(
            n_bars=60,
            n_assets=n_assets,
            n_market=3,
            n_fundamental=2,
            n_regime=2,
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
        training = _align_settings_to_split(training, splits)
        if args.timesteps is None and args.model_id is None:
            training = training.model_copy(
                update={
                    "total_timesteps": 256,
                    "n_steps": 64,
                    "batch_size": 32,
                    "n_epochs": 1,
                    "eval_freq": 64,
                    "early_stopping_patience": 3,
                    "min_evals_before_stop": 1,
                    "model_id": "ppo-smoke-fixture",
                }
            )
        elif args.timesteps is None:
            training = training.model_copy(
                update={
                    "total_timesteps": 256,
                    "n_steps": 64,
                    "batch_size": 32,
                    "n_epochs": 1,
                    "eval_freq": 64,
                    "early_stopping_patience": 3,
                    "min_evals_before_stop": 1,
                }
            )
        logger.info(
            "fixture smoke episode bars=%d assets=%s", episode.n_bars, episode.symbols
        )
    else:
        episode = _episode_for_configured_dates(
            training, settings.data.universe, training.seed
        )
        splits = split_episode_by_dates(episode, training)
        logger.info(
            "date-split train=%s..%s val=%s..%s test=%s..%s bars=%d",
            training.train_start,
            training.train_end,
            training.val_start,
            training.val_end,
            training.test_start,
            training.test_end,
            episode.n_bars,
        )

    out_dir = args.output_dir or (training.checkpoint_dir / training.model_id)
    logger.info(
        "starting PPO training model_id=%s timesteps=%d out=%s",
        training.model_id,
        training.total_timesteps,
        out_dir,
    )
    result = train_ppo(
        splits,
        training=training,
        environment=settings.environment,
        features=settings.features,
        full_episode_dates=(
            _as_date(splits.train.timestamps[0]),
            _as_date(splits.test.timestamps[-1]),
        ),
        output_dir=out_dir,
    )
    logger.info(
        "training complete out=%s best_val=%s test_report=%s selected_on=%s",
        result.output_dir,
        result.best_val_mean_reward,
        result.test_mean_reward,
        result.manifest.model_selected_on,
    )
    if result.access_guard.fitted_scaler_on != "train":
        logger.error("scaler was not fitted on train")
        return 1
    if result.manifest.model_selected_on != "validation":
        logger.error("model must be selected on validation only")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
