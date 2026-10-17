#!/usr/bin/env python3
"""Run walk-forward validation on a labelled synthetic panel (offline).

Fits scalers per training window, evaluates the agent on each non-overlapping
test fold, runs benchmark strategies with the shared cost model, and writes
artifacts under ``reports/walk_forward/``. Research only — no live edge claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_trader.config import get_settings
from rl_trader.config.models import BacktestSettings
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.monitoring import configure_logging, get_logger
from rl_trader.utils.seeds import set_global_seeds
from rl_trader.validation import run_walk_forward


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for walk-forward validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research.yaml"),
        help="YAML config path",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Compact synthetic smoke windows (recommended for CI)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip SB3 training; evaluate equal-weight policy (fast CI)",
    )
    parser.add_argument("--folds", type=int, default=None, help="Max folds")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_walk_forward")
    set_global_seeds(settings.seed)

    backtest = settings.backtest
    if args.fixture:
        backtest = BacktestSettings(
            initial_cash=backtest.initial_cash,
            commission_bps=settings.environment.commission_bps,
            slippage_bps=settings.environment.slippage_bps,
            walk_forward_dir=backtest.walk_forward_dir,
            walk_forward_folds=args.folds or 3,
            train_bars=24,
            val_bars=8,
            test_bars=8,
            step_bars=8,
            window_mode=backtest.window_mode,
            retrain_mode=backtest.retrain_mode,
            max_dd_threshold=backtest.max_dd_threshold,
            algorithm=backtest.algorithm,
            train_timesteps=args.timesteps or 128,
            train_agent=not args.no_train,
            trading_days_per_year=backtest.trading_days_per_year,
        )
    else:
        updates: dict[str, object] = {}
        if args.folds is not None:
            updates["walk_forward_folds"] = args.folds
        if args.timesteps is not None:
            updates["train_timesteps"] = args.timesteps
        if args.no_train:
            updates["train_agent"] = False
        if updates:
            backtest = backtest.model_copy(update=updates)

    n_assets = max(2, min(3, len(settings.data.universe)))
    need = (
        backtest.train_bars
        + backtest.val_bars
        + backtest.test_bars
        + max(0, backtest.walk_forward_folds - 1) * backtest.step_bars
    )
    raw = make_synthetic_episode(
        n_bars=max(need + 2, 60),
        n_assets=n_assets,
        n_market=2,
        n_fundamental=1,
        n_regime=2,
        seed=settings.seed,
        price_path="drift",
    )
    symbols = tuple(settings.data.universe[:n_assets])
    episode = EpisodeData(
        timestamps=raw.timestamps,
        prices=raw.prices,
        market_features=raw.market_features,
        fundamental_features=raw.fundamental_features,
        regime_features=raw.regime_features,
        symbols=symbols,
    )
    logger.info(
        "walk_forward SYNTHETIC panel bars=%d assets=%s folds<=%d "
        "train/val/test/step=%d/%d/%d/%d train_agent=%s mode=%s",
        episode.n_bars,
        episode.symbols,
        backtest.walk_forward_folds,
        backtest.train_bars,
        backtest.val_bars,
        backtest.test_bars,
        backtest.step_bars,
        backtest.train_agent,
        backtest.window_mode,
    )

    result = run_walk_forward(
        episode,
        backtest=backtest,
        environment=settings.environment,
        training=settings.training,
        features=settings.features,
        benchmarks=settings.benchmarks,
        output_dir=args.output_dir,
    )
    logger.info(
        "walk_forward complete out=%s folds=%d summary=%s aggregate=%s",
        result.output_dir,
        len(result.fold_rows),
        result.summary_csv,
        result.aggregate_path,
    )
    for row in result.fold_rows:
        logger.info(
            "fold=%d ret=%.6g sharpe=%.4f under_bh=%s dd_flag=%s unstable=%s",
            row.fold_id,
            row.agent_cumulative_return,
            row.agent_sharpe,
            row.underperforms_buy_hold,
            row.exceeds_max_dd_threshold,
            row.unstable,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
