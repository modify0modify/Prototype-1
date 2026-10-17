#!/usr/bin/env python3
"""Run walk-forward validation (research stub)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from rl_trader.config import get_settings
from rl_trader.monitoring import configure_logging, get_logger
from rl_trader.validation import build_walk_forward_plan


def main() -> int:
    """Build a stub walk-forward plan and log fold count."""
    settings = get_settings(config_file=Path("configs/research.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_walk_forward")
    plan = build_walk_forward_plan(date(2020, 1, 1), date(2024, 1, 1), n_folds=3)
    logger.info("walk_forward stub folds=%s", len(plan.folds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
