#!/usr/bin/env python3
"""Run deterministic benchmark strategies on a shared evaluation harness.

Uses the portfolio-env cost / execution simulator. Does **not** train an RL agent.
Synthetic panels are used when no processed price panel is supplied — not market data,
not investment advice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_trader.benchmarks import (
    make_synthetic_price_panel,
    resolve_env_settings_for_benchmarks,
    run_all_benchmarks,
    write_benchmark_report,
)
from rl_trader.config import get_settings
from rl_trader.monitoring import configure_logging, get_logger


def main() -> int:
    """Evaluate the baseline suite and write JSON/Markdown reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research.yaml"),
        help="YAML config with environment + benchmarks sections",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=180,
        help="Synthetic panel length when no external panel is loaded",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override benchmarks.output_dir",
    )
    args = parser.parse_args()

    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_benchmarks")

    symbols = tuple(settings.data.universe)
    if settings.benchmarks.spy_symbol.upper() not in {s.upper() for s in symbols}:
        symbols = (settings.benchmarks.spy_symbol.upper(),) + symbols

    panel = make_synthetic_price_panel(
        n_bars=args.n_bars,
        symbols=symbols,
        seed=settings.seed,
    )
    env_settings = resolve_env_settings_for_benchmarks(
        environment=settings.environment,
        benchmarks=settings.benchmarks,
        initial_cash=settings.backtest.initial_cash,
    )
    results = run_all_benchmarks(
        panel,
        env_settings=env_settings,
        benchmark_settings=settings.benchmarks,
    )
    out_dir = args.output_dir or settings.benchmarks.output_dir
    json_path, md_path = write_benchmark_report(results, Path(out_dir))
    logger.info(
        "benchmarks complete strategies=%d json=%s md=%s (no RL training)",
        len(results),
        json_path,
        md_path,
    )
    for result in results:
        m = result.metrics
        logger.info(
            "%s cum=%.4f sharpe=%.3f mdd=%.3f costs=%.2f trades=%d",
            m.strategy,
            m.cumulative_return,
            m.sharpe,
            m.max_drawdown,
            m.transaction_costs,
            m.number_of_trades,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
