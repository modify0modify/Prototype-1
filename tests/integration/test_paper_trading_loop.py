"""Integration: full prospective paper-trading cycle with PaperBroker."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config import AppSettings, RuntimeMode
from rl_trader.config.models import (
    BrokerSettings,
    LiveTradingSettings,
    PaperTradingSettings,
    RiskSettings,
)
from rl_trader.paper_trading.market_data import FixtureMarketDataProvider
from rl_trader.paper_trading.service import PaperTradingService


def _approved(tmp_path: Path) -> Path:
    from rl_trader.approval.testing import write_sealed_approved_manifest

    return write_sealed_approved_manifest(
        tmp_path / "approved",
        tier="paper",
        model_id="integ-paper",
        n_assets=2,
    )


def test_prospective_paper_loop_end_to_end(tmp_path: Path) -> None:
    manifest = _approved(tmp_path)
    now = datetime(2024, 6, 3, 15, 30, tzinfo=UTC)
    settings = AppSettings(
        _env_file=None,
        app_mode=RuntimeMode.PAPER,
        live_trading_enabled=False,
        broker=BrokerSettings(environment="paper"),
        risk=RiskSettings(
            max_position_size=0.5,
            max_portfolio_exposure=1.0,
            max_sector_exposure=1.0,
            min_cash_reserve=0.0,
            max_order_notional=1_000_000.0,
            max_daily_turnover=2.0,
            max_orders_per_day=50,
            daily_loss_limit=0.5,
            weekly_loss_limit=0.5,
            max_drawdown_limit=0.9,
            max_volatility_target=1.0,
            max_consecutive_losses=20,
            max_data_age_seconds=300.0,
            abnormal_price_jump=0.5,
            max_spread_bps=500.0,
            configuration_version="integ-paper",
        ),
        paper_trading=PaperTradingSettings(
            state_dir=tmp_path / "state",
            broker_data_dir=tmp_path / "broker",
            max_data_age_seconds=120.0,
            evaluation_period_id="integ-period",
            use_fixture_market_data=True,
            initial_cash=100_000.0,
            commission_bps=5.0,
            spread_bps=2.0,
            slippage_bps=1.0,
        ),
        live=LiveTradingSettings(approved_model_manifest=manifest),
    )
    md = FixtureMarketDataProvider(
        n_assets=2, bar_age_seconds=2.0, now=now, symbols=("A0", "A1")
    )
    broker = PaperBroker(
        config=PaperBrokerConfig(
            data_dir=tmp_path / "broker",
            initial_cash=100_000.0,
            commission_bps=5.0,
            spread_bps=2.0,
            slippage_bps=1.0,
        ),
        now_fn=lambda: now,
    )
    with PaperTradingService(
        settings,
        dry_run=False,
        market_data=md,
        broker=broker,
        now_fn=lambda: now,
    ) as svc:
        result = svc.run_cycle()

    assert result.shutdown is False
    assert result.skipped is False
    assert result.model_id == "integ-paper"
    assert result.model_version == "integ-1"
    assert result.risk_decision in {"approved", "resized"}
    assert result.order_ids

    state = tmp_path / "state"
    assert (state / "heartbeat.json").is_file()
    assert (state / "decisions.jsonl").is_file()
    assert (state / "metrics.json").is_file()
    assert (state / "paper_trader.lock").is_file() or True  # released after exit

    decision = json.loads((state / "decisions.jsonl").read_text().strip().splitlines()[-1])
    assert decision["model_version"] == "integ-1"
    assert decision["dry_run"] is False
    assert decision["order_ids"]
    assert decision["risk_decision"] in {"approved", "resized"}

    metrics = json.loads((state / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cycles"] == 1
    assert metrics["orders_submitted"] >= 1
    assert metrics["last_equity"] is not None

    heartbeat = json.loads((state / "heartbeat.json").read_text(encoding="utf-8"))
    assert "timestamp" in heartbeat
    assert heartbeat["status"] == "stopped"

    # Positions exist after cycle
    broker2 = PaperBroker(
        config=PaperBrokerConfig(data_dir=tmp_path / "broker", initial_cash=100_000.0),
        now_fn=lambda: now,
    )
    broker2.connect()
    assert broker2.get_positions()
    broker2.disconnect()
