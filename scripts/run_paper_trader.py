#!/usr/bin/env python3
"""Paper trading entrypoint — gated until cost model is wired.

Exits non-zero by default. Research/simulation should use PortfolioEnv +
walk-forward / backtests; do not enable paper trading while critical audit
findings remain.
"""

from __future__ import annotations

from pathlib import Path

from rl_trader.brokers import is_live_trading_blocked
from rl_trader.brokers.paper import PaperTradingDisabledError
from rl_trader.config import get_settings
from rl_trader.execution import PaperExecutor
from rl_trader.monitoring import configure_logging, get_logger


def main() -> int:
    """Refuse paper trading while the incomplete broker gate is active."""
    settings = get_settings(config_file=Path("configs/paper.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_paper_trader")

    if not is_live_trading_blocked(settings):
        logger.error("refusing to run: live trading unexpectedly allowed")
        return 1

    try:
        PaperExecutor(allow_incomplete=False)
    except PaperTradingDisabledError as exc:
        logger.error("paper trading disabled: %s", exc)
        return 2

    logger.error("unreachable: paper broker must remain gated")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
