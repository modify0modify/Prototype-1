"""Minimal CLI entrypoints for skeleton smoke checks."""

from __future__ import annotations

from rl_trading import __version__
from rl_trading.config import get_settings
from rl_trading.live import is_live_trading_blocked
from rl_trading.logging import configure_logging, get_logger
from rl_trading.rl import set_global_seeds


def run_smoke() -> int:
    """Load config, confirm paper defaults, confirm live is blocked."""
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trading.smoke")
    set_global_seeds(settings.seed)

    settings.assert_live_blocked_by_default()
    live_blocked = is_live_trading_blocked(settings)

    logger.info(
        "rl-trading smoke ok version=%s mode=%s live_blocked=%s",
        __version__,
        settings.mode.value,
        live_blocked,
    )
    if not settings.is_paper_default:
        logger.warning("defaults are not paper-safe; check env overrides")
    if not live_blocked:
        raise SystemExit("smoke failed: live trading unexpectedly allowed")
    return 0


def main() -> None:
    """Module entrypoint: ``python -m rl_trading.cli``."""
    raise SystemExit(run_smoke())


if __name__ == "__main__":
    main()
