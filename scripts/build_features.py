#!/usr/bin/env python3
"""Build look-ahead-safe features from cleaned OHLCV parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rl_trader.config import get_settings
from rl_trader.features import run_feature_pipeline
from rl_trader.monitoring import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Build causal feature matrices")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research.yaml"),
        help="YAML config path",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol (default: all non-benchmark names in universe)",
    )
    return parser


def _find_cleaned(processed_dir: Path, symbol: str) -> Path:
    matches = sorted(processed_dir.glob(f"ohlcv/{symbol}_*.parquet"))
    if not matches:
        matches = sorted(processed_dir.glob(f"**/{symbol}_*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"no cleaned OHLCV for {symbol} under {processed_dir}; "
            "run scripts/download_data.py first"
        )
    return matches[-1]


def main(argv: list[str] | None = None) -> int:
    """Entry: ``python scripts/build_features.py --config configs/research.yaml``."""
    args = build_parser().parse_args(argv)
    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.build_features")

    universe = list(settings.data.universe)
    benchmark = settings.features.benchmark_symbol.upper()
    symbols = [args.symbol.upper()] if args.symbol else [s for s in universe if s != benchmark]
    if not symbols:
        symbols = universe[:]

    bench_path = _find_cleaned(settings.data.processed_dir, benchmark)
    bench = pd.read_parquet(bench_path)
    logger.info("benchmark=%s path=%s", benchmark, bench_path)

    failures = 0
    for symbol in symbols:
        path = _find_cleaned(settings.data.processed_dir, symbol)
        ohlcv = pd.read_parquet(path)
        result = run_feature_pipeline(
            ohlcv,
            settings.features,
            symbol=symbol,
            benchmark_ohlcv=bench,
            valuation=settings.fundamentals,
        )
        logger.info(
            "%s features=%s rows=%s missing_warn=%s manifest=%s quality=%s",
            symbol,
            result.feature_path,
            result.quality.n_rows,
            len(result.quality.warnings),
            result.manifest_path,
            result.quality_path,
        )
        if not result.quality.ok:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
