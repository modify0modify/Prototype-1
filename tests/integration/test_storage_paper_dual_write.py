"""Integration: paper trading dual-writes to SQL while keeping file ledgers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config import AppSettings, RuntimeMode
from rl_trader.config.models import (
    BrokerSettings,
    DatabaseSettings,
    LiveTradingSettings,
    PaperTradingSettings,
    RiskSettings,
)
from rl_trader.paper_trading.market_data import FixtureMarketDataProvider
from rl_trader.paper_trading.service import PaperTradingService
from rl_trader.storage import OrderRow, create_db_engine, session_scope, upgrade_head


def _manifest(tmp_path: Path) -> Path:
    from rl_trader.approval.testing import write_sealed_approved_manifest

    return write_sealed_approved_manifest(
        tmp_path / "approved",
        tier="paper",
        model_id="db-paper",
        n_assets=2,
    )


def test_paper_dual_write_and_file_ledger(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'paper.db'}"
    upgrade_head(db_url)
    manifest = _manifest(tmp_path)
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    settings = AppSettings(
        _env_file=None,
        app_mode=RuntimeMode.PAPER,
        live_trading_enabled=False,
        broker=BrokerSettings(environment="paper"),
        database=DatabaseSettings(url=db_url, persist_paper_trading=True),
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
            configuration_version="db-paper-v1",
        ),
        paper_trading=PaperTradingSettings(
            state_dir=tmp_path / "state",
            broker_data_dir=tmp_path / "broker",
            max_data_age_seconds=120.0,
            evaluation_period_id="db-period",
            use_fixture_market_data=True,
            initial_cash=100_000.0,
        ),
        live=LiveTradingSettings(approved_model_manifest=manifest),
    )
    md = FixtureMarketDataProvider(n_assets=2, bar_age_seconds=1.0, now=now)
    broker = PaperBroker(
        config=PaperBrokerConfig(data_dir=tmp_path / "broker", initial_cash=100_000.0),
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

    assert result.order_ids
    # File ledger still written
    assert (tmp_path / "broker" / "orders.json").is_file()
    assert (tmp_path / "state" / "decisions.jsonl").is_file()

    engine = create_db_engine(db_url, create_schema=False)
    with session_scope(engine) as session:
        n_orders = session.scalar(select(func.count()).select_from(OrderRow))
        assert int(n_orders or 0) >= 1
        order = session.scalars(select(OrderRow)).first()
        assert order is not None
        assert order.risk_decision_id is not None
        assert order.agent_decision_id is not None
        assert order.configuration_version_id is not None
