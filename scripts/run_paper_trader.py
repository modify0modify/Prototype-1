#!/usr/bin/env python3
"""Prospective paper-trading service entrypoint.

Pipeline: approved model → features → agent targets → risk → PaperBroker.

Never enables live trading. Use ``--dry-run`` to generate decisions without
creating simulated orders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main(argv: list[str] | None = None) -> int:
    """Run one prospective paper cycle (or dry-run)."""
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
    args = parser.parse_args(argv)

    settings = get_settings(config_file=args.config)
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_paper_trader")

    if settings.app_mode is not RuntimeMode.PAPER:
        logger.error("refusing to run: app_mode=%s (paper required)", settings.app_mode.value)
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

    service: PaperTradingService = build_service_from_settings(
        settings, dry_run=args.dry_run
    )
    try:
        model = service.start()
        logger.info(
            "paper service ready model_id=%s version=%s period=%s dry_run=%s",
            model.model_id,
            model.model_version,
            model.evaluation_period_id,
            args.dry_run,
        )
        result = service.run_cycle()
        if result.shutdown:
            logger.error("safe shutdown: %s", result.risk_reason)
            return 3
        if result.skipped and result.details.get("idempotent"):
            logger.info("idempotent skip cycle_id=%s", result.cycle_id)
            return 0
        logger.info(
            "cycle complete id=%s decision=%s reason=%s orders=%s dry_run=%s",
            result.cycle_id,
            result.risk_decision,
            result.risk_reason,
            result.order_ids,
            result.dry_run,
        )
        return 0
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
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
