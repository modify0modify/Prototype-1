#!/usr/bin/env python3
"""Run paper trading loop stub (risk → paper broker; never live)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_trader.agents import StubAgent
from rl_trader.brokers import is_live_trading_blocked
from rl_trader.config import get_settings
from rl_trader.execution import PaperExecutor
from rl_trader.monitoring import configure_logging, get_logger


def main() -> int:
    """One paper step through the risk engine."""
    settings = get_settings(config_file=Path("configs/paper.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_paper_trader")

    if not is_live_trading_blocked(settings):
        logger.error("refusing to run: live trading unexpectedly allowed")
        return 1

    agent = StubAgent(n_assets=4)
    target = agent.predict_target(np.zeros(4, dtype=np.float32))
    fill = PaperExecutor().execute(target)
    logger.info(
        "paper fill mode=%s decision=%s",
        fill.mode,
        fill.metadata.get("decision"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
