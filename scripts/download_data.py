#!/usr/bin/env python3
"""Download historical daily OHLCV for the configured universe."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from rl_trader.config import get_settings
from rl_trader.data import run_data_pipeline
from rl_trader.monitoring import configure_logging, get_logger


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for the download pipeline."""
    parser = argparse.ArgumentParser(description="Download and validate daily OHLCV data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research.yaml"),
        help="YAML config path (default: configs/research.yaml)",
    )
    parser.add_argument("--start", type=_parse_date, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_date, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated override universe (e.g. SPY,QQQ,PLTR)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore local cache and re-fetch from the provider",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python scripts/download_data.py --config configs/research.yaml``."""
    args = build_parser().parse_args(argv)
    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.download_data")

    symbols = None
    if args.symbols:
        symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())

    logger.info(
        "starting download provider=%s mode=%s universe=%s",
        settings.data.provider,
        settings.app_mode.value,
        symbols or settings.data.universe,
    )
    result = run_data_pipeline(
        settings.data,
        start=args.start,
        end=args.end,
        symbols=symbols,
        use_cache=not args.no_cache,
    )
    for item in result.results:
        status = "ok" if item.validation.ok else "FAILED"
        logger.info(
            "%s %s raw=%s cleaned=%s report=%s errors=%s",
            item.symbol,
            status,
            item.raw_path,
            item.cleaned_path,
            item.quality_report_path,
            item.validation.errors,
        )
    if not result.ok:
        logger.error("data pipeline completed with validation failures")
        return 1
    logger.info("data pipeline completed successfully for %s symbols", len(result.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
