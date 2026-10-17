"""Integration: paper trading dual-writes to SQL while keeping file ledgers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    save_checkpoint_metadata,
)
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config import AppSettings, RuntimeMode
from rl_trader.config.models import (
    BrokerSettings,
    DatabaseSettings,
    LiveTradingSettings,
    PaperTradingSettings,
    RiskSettings,
)
from rl_trader.features.engineering import FeatureSpec
from rl_trader.paper_trading.market_data import FixtureMarketDataProvider
from rl_trader.paper_trading.service import PaperTradingService
from rl_trader.storage import OrderRow, create_db_engine, session_scope, upgrade_head


def _manifest(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    specs = (
        FeatureSpec(name="returns_1d", lookback=1),
        FeatureSpec(name="vol_20d", lookback=20),
    )
    save_checkpoint_metadata(
        ckpt,
        CheckpointMetadata(
            model_id="db-paper",
            algorithm="stub",
            feature_specs=specs,
            training_period=TrainingPeriod(
                start=datetime(2023, 1, 3).date(),
                end=datetime(2023, 6, 30).date(),
            ),
            seed=3,
            package_version="db-1",
        ),
    )
    (ckpt / "feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_specs": [
                    {
                        "name": s.name,
                        "lookback": s.lookback,
                        "version": "1",
                        "causal_only": True,
                    }
                    for s in specs
                ]
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "approved": True,
                "model_id": "db-paper",
                "model_version": "db-1",
                "algorithm": "stub",
                "n_assets": 2,
                "checkpoint_dir": str(ckpt),
            }
        ),
        encoding="utf-8",
    )
    return path


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
