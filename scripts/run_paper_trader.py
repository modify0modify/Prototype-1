#!/usr/bin/env python3
"""Paper trading entrypoint — simulated broker only (no live credentials).

By default prints account status after connecting the local PaperBroker.
Pass ``--demo`` to run a one-symbol buy/sell cycle against injected quotes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from rl_trader.brokers import (
    OrderRequest,
    OrderSide,
    OrderType,
    Quote,
    is_live_trading_blocked,
)
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config import get_settings
from rl_trader.monitoring import configure_logging, get_logger


def _demo_cycle(broker: PaperBroker) -> int:
    now = datetime.now(UTC)
    broker.set_quote(
        Quote(symbol="DEMO", bid=99.9, ask=100.1, last=100.0, timestamp=now)
    )
    order = broker.submit_order(
        OrderRequest(
            symbol="DEMO",
            side=OrderSide.BUY,
            qty=10.0,
            order_type=OrderType.MARKET,
            idempotency_key="demo-buy-1",
        )
    )
    account = broker.get_account()
    positions = broker.get_positions()
    logger = get_logger("rl_trader.scripts.run_paper_trader")
    logger.info(
        "demo cycle: order=%s status=%s cash=%.2f positions=%d",
        order.order_id,
        order.status,
        account.cash,
        len(positions),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Connect the paper broker; optionally run a local demo cycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a one-shot simulated buy against an injected DEMO quote",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/interim/paper_broker"),
        help="Persistent paper ledger directory",
    )
    args = parser.parse_args(argv)

    settings = get_settings(config_file=Path("configs/paper.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.run_paper_trader")

    if not is_live_trading_blocked(settings):
        logger.error("refusing to run: live trading unexpectedly allowed")
        return 1

    cfg = PaperBrokerConfig(data_dir=args.data_dir)
    broker = PaperBroker(config=cfg)
    broker.connect()
    try:
        account = broker.get_account()
        logger.info(
            "paper broker connected account_id=%s cash=%.2f equity=%.2f",
            account.account_id,
            account.cash,
            account.equity,
        )
        if args.demo:
            return _demo_cycle(broker)
        logger.info("pass --demo to run a local simulated order cycle")
        return 0
    finally:
        broker.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
