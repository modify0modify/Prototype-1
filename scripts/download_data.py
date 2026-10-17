#!/usr/bin/env python3
"""Download historical market data (research stub)."""

from __future__ import annotations

from pathlib import Path

from rl_trader.config import get_settings
from rl_trader.data import StubDownloader
from rl_trader.monitoring import configure_logging, get_logger


def main() -> int:
    """Entry point — wires settings to the stub downloader."""
    settings = get_settings(config_file=Path("configs/research.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.download_data")
    logger.info("download_data stub (mode=%s) — no network calls", settings.mode.value)
    _ = StubDownloader()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
