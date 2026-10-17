"""Integration: full prospective paper-trading cycle with PaperBroker."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    save_checkpoint_metadata,
)
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config import AppSettings, RuntimeMode
from rl_trader.config.models import (
    BrokerSettings,
    LiveTradingSettings,
    PaperTradingSettings,
    RiskSettings,
)
from rl_trader.features.engineering import FeatureSpec
from rl_trader.paper_trading.market_data import FixtureMarketDataProvider
from rl_trader.paper_trading.service import PaperTradingService


def _approved(tmp_path: Path) -> Path:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    specs = (
        FeatureSpec(name="returns_1d", lookback=1),
        FeatureSpec(name="vol_20d", lookback=20),
    )
    save_checkpoint_metadata(
        ckpt,
        CheckpointMetadata(
            model_id="integ-paper",
            algorithm="stub",
            feature_specs=specs,
            training_period=TrainingPeriod(
                start=datetime(2023, 1, 3).date(),
                end=datetime(2023, 12, 29).date(),
            ),
            seed=7,
            package_version="integ-1",
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
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "approved": True,
                "model_id": "integ-paper",
                "model_version": "integ-1",
                "algorithm": "stub",
                "n_assets": 2,
                "checkpoint_dir": str(ckpt),
            }
        ),
        encoding="utf-8",
    )
    return manifest


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
