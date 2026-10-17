#!/usr/bin/env python3
"""Prospective paper-trading service entrypoint.

Pipeline: approved model → features → agent targets → risk → PaperBroker.

Never enables live trading. Use ``--dry-run`` to generate decisions without
creating simulated orders. Use ``--loop`` for a deployable continuous service
with SIGTERM/SIGINT graceful shutdown (finishes the current cycle, then stops).
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from types import FrameType

from rl_trader.brokers import is_live_trading_blocked
from rl_trader.config import RuntimeMode, get_settings
from rl_trader.monitoring import configure_logging, get_logger
from rl_trader.paper_trading.locks import ProcessLockError
from rl_trader.paper_trading.model_loader import ApprovedModelError
from rl_trader.paper_trading.service import (
    PaperTradingService,
    StaleMarketDataError,
    build_service_from_settings,
)

_STOP = False


def _request_stop(signum: int, frame: FrameType | None) -> None:
    """Signal handler — cooperative stop after the current cycle."""
    global _STOP
    _ = frame
    _STOP = True
    logger = get_logger("rl_trader.scripts.run_paper_trader")
    logger.warning("shutdown signal received (%s) — finishing current cycle", signum)


def main(argv: list[str] | None = None) -> int:
    """Run one prospective paper cycle, or a loop until signalled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper.yaml"),
        help="YAML config (default: configs/paper.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate risk decisions without creating simulated orders",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force fixture market data for a local smoke cycle",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run cycles until SIGTERM/SIGINT (deploy mode)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=60.0,
        help="Sleep between cycles when --loop (default 60)",
    )
    args = parser.parse_args(argv)

    settings = get_settings(config_file=args.config)
    configure_logging(
        settings.log_level,
        json_logs=settings.log_json,
        log_dir=settings.monitoring.log_dir,
        log_filename=settings.monitoring.log_filename,
        max_bytes=settings.monitoring.log_max_bytes,
        backup_count=settings.monitoring.log_backup_count,
    )
    logger = get_logger("rl_trader.scripts.run_paper_trader")

    if settings.app_mode is not RuntimeMode.PAPER:
        logger.error("refusing to run: app_mode=%s (paper required)", settings.app_mode.value)
        return 1
    if settings.live_trading_enabled:
        logger.error("refusing to run: live_trading_enabled must be false")
        return 1
    if not is_live_trading_blocked(settings):
        logger.error("refusing to run: live trading unexpectedly allowed")
        return 1
    if settings.broker.environment != "paper":
        logger.error("refusing to run: broker.environment=%s", settings.broker.environment)
        return 1

    if args.demo or settings.paper_trading.use_fixture_market_data:
        from rl_trader.paper_trading.demo_model import provision_demo_approved_model

        demo_root = Path(settings.paper_trading.state_dir) / "demo_model"
        manifest = provision_demo_approved_model(
            demo_root,
            evaluation_period_id=settings.paper_trading.evaluation_period_id,
        )
        settings = settings.model_copy(
            update={
                "paper_trading": settings.paper_trading.model_copy(
                    update={"use_fixture_market_data": True}
                ),
                "live": settings.live.model_copy(
                    update={"approved_model_manifest": manifest}
                ),
            }
        )
        logger.info("demo mode: provisioned stub approved model at %s", manifest)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    service: PaperTradingService = build_service_from_settings(
        settings, dry_run=args.dry_run
    )
    exit_code = 0
    try:
        model = service.start()
        logger.info(
            "paper service ready model_id=%s version=%s period=%s dry_run=%s loop=%s",
            model.model_id,
            model.model_version,
            model.evaluation_period_id,
            args.dry_run,
            args.loop,
        )
        while True:
            if _STOP:
                logger.info("graceful shutdown before next cycle")
                break
            result = service.run_cycle()
            if result.shutdown:
                logger.error("safe shutdown: %s", result.risk_reason)
                exit_code = 3
                break
            if result.skipped and result.details.get("idempotent"):
                logger.info("idempotent skip cycle_id=%s", result.cycle_id)
            else:
                logger.info(
                    "cycle complete id=%s decision=%s reason=%s orders=%s dry_run=%s",
                    result.cycle_id,
                    result.risk_decision,
                    result.risk_reason,
                    result.order_ids,
                    result.dry_run,
                )
            if not args.loop:
                break
            # Interruptible sleep between cycles.
            deadline = time.monotonic() + max(0.0, float(args.interval_seconds))
            while time.monotonic() < deadline:
                if _STOP:
                    break
                time.sleep(min(0.5, deadline - time.monotonic()))
        return exit_code
    except ProcessLockError as exc:
        logger.error("lock error: %s", exc)
        return 4
    except ApprovedModelError as exc:
        logger.error("approved model gate failed: %s", exc)
        return 2
    except StaleMarketDataError as exc:
        logger.error("stale data: %s", exc)
        return 3
    except Exception:
        logger.exception("paper trader failed")
        return 1
    finally:
        logger.info("stopping paper service (release lock, flush broker)")
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
