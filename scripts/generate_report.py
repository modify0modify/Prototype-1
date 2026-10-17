#!/usr/bin/env python3
"""Generate a placeholder report under reports/ (stub)."""

from __future__ import annotations

from pathlib import Path

from rl_trader.config import get_settings
from rl_trader.monitoring import configure_logging, get_logger


def main() -> int:
    """Write a short markdown placeholder report."""
    settings = get_settings(config_file=Path("configs/research.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.generate_report")

    out = Path("reports/backtests/stub_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Stub backtest report\n\nNo full model evaluation yet.\n",
        encoding="utf-8",
    )
    logger.info("wrote %s (mode=%s)", out, settings.mode.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
