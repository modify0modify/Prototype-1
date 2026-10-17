"""Unit tests for the prospective paper-trading service."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
from rl_trader.paper_trading.demo_model import provision_demo_approved_model
from rl_trader.paper_trading.locks import ProcessLock, ProcessLockError
from rl_trader.paper_trading.market_data import (
    FixtureMarketDataProvider,
    StaleMarketDataError,
    assert_snapshot_fresh,
)
from rl_trader.paper_trading.model_loader import ApprovedModelError, load_approved_model
from rl_trader.paper_trading.service import PaperTradingService


def _specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(name="returns_1d", lookback=1),
        FeatureSpec(name="vol_20d", lookback=20),
    )


def _write_stub_approved(root: Path, *, approved: bool = True, n_assets: int = 2) -> Path:
    from rl_trader.approval.testing import (
        write_sealed_approved_manifest,
        write_sealed_unapproved_manifest,
    )

    if approved:
        return write_sealed_approved_manifest(
            root, tier="paper", model_id="unit-paper-model", n_assets=n_assets
        )
    return write_sealed_unapproved_manifest(root, model_id="unit-paper-model")


def _settings(tmp_path: Path, manifest: Path, **pt_overrides: object) -> AppSettings:
    pt: dict[str, object] = {
        "state_dir": tmp_path / "state",
        "broker_data_dir": tmp_path / "broker",
        "max_data_age_seconds": 120.0,
        "evaluation_period_id": "unit-period",
        "use_fixture_market_data": True,
        "initial_cash": 100_000.0,
    }
    pt.update(pt_overrides)
    return AppSettings(
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
            max_orders_per_day=100,
            daily_loss_limit=0.5,
            weekly_loss_limit=0.5,
            max_drawdown_limit=0.9,
            max_volatility_target=1.0,
            max_consecutive_losses=50,
            max_data_age_seconds=300.0,
            abnormal_price_jump=0.5,
            max_spread_bps=500.0,
            configuration_version="paper-unit",
        ),
        paper_trading=PaperTradingSettings(**pt),  # type: ignore[arg-type]
        live=LiveTradingSettings(approved_model_manifest=manifest),
    )


def test_paper_mode_gate_rejects_live(tmp_path: Path) -> None:
    manifest = _write_stub_approved(tmp_path / "m")
    settings = _settings(tmp_path, manifest)
    settings = settings.model_copy(update={"app_mode": RuntimeMode.LIVE})
    svc = PaperTradingService(settings, acquire_lock=False)
    with pytest.raises(RuntimeError, match="app_mode=paper"):
        svc.start()


def test_approved_model_only_load(tmp_path: Path) -> None:
    bad = _write_stub_approved(tmp_path / "bad", approved=False)
    with pytest.raises(ApprovedModelError, match="not paper/live"):
        load_approved_model(bad, evaluation_period_id="p")
    good = _write_stub_approved(tmp_path / "good")
    bundle = load_approved_model(good, evaluation_period_id="p")
    assert bundle.model_id == "unit-paper-model"
    assert bundle.model_version == "0.1.0-demo"
    assert bundle.algorithm == "stub"
    assert bundle.content_hash
    assert bundle.approval_status == "paper"


def test_stale_data_shutdown(tmp_path: Path) -> None:
    manifest = _write_stub_approved(tmp_path / "m")
    settings = _settings(tmp_path, manifest, max_data_age_seconds=5.0)
    clock = {"t": datetime(2024, 6, 3, 15, 0, tzinfo=UTC)}

    def now_fn() -> datetime:
        return clock["t"]

    md = FixtureMarketDataProvider(
        n_assets=2,
        bar_age_seconds=60.0,
        now=clock["t"],
        symbols=("A0", "A1"),
    )
    broker = PaperBroker(
        config=PaperBrokerConfig(data_dir=tmp_path / "broker", initial_cash=100_000.0),
        now_fn=now_fn,
    )
    svc = PaperTradingService(
        settings,
        dry_run=True,
        market_data=md,
        broker=broker,
        now_fn=now_fn,
        acquire_lock=True,
    )
    svc.start()
    result = svc.run_cycle()
    svc.stop()
    assert result.shutdown is True
    assert "stale" in result.risk_reason.lower()
    alerts = (tmp_path / "state" / "alerts.jsonl").read_text(encoding="utf-8")
    assert "stale_or_incomplete_data_shutdown" in alerts


def test_dry_run_creates_no_orders(tmp_path: Path) -> None:
    manifest = _write_stub_approved(tmp_path / "m")
    settings = _settings(tmp_path, manifest, max_data_age_seconds=120.0)
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    md = FixtureMarketDataProvider(n_assets=2, bar_age_seconds=1.0, now=now)
    broker = PaperBroker(
        config=PaperBrokerConfig(data_dir=tmp_path / "broker", initial_cash=100_000.0),
        now_fn=lambda: now,
    )
    svc = PaperTradingService(
        settings,
        dry_run=True,
        market_data=md,
        broker=broker,
        now_fn=lambda: now,
    )
    svc.start()
    result = svc.run_cycle()
    svc.stop()
    assert result.dry_run is True
    assert result.order_ids == []
    assert result.skipped is False
    assert broker.get_open_orders() == []
    # No fills file / positions from orders
    assert broker.get_positions() == []
    decisions = (tmp_path / "state" / "decisions.jsonl").read_text(encoding="utf-8")
    assert '"dry_run": true' in decisions
    assert "unit-paper-model" in decisions


def test_process_lock_prevents_duplicate_process(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "paper_trader.lock"
    with ProcessLock(lock_path):
        second = ProcessLock(lock_path)
        with pytest.raises(ProcessLockError, match="already running"):
            second.acquire()


def test_idempotency_on_restart_skips_duplicate_cycle(tmp_path: Path) -> None:
    manifest = _write_stub_approved(tmp_path / "m")
    settings = _settings(tmp_path, manifest)
    now = datetime(2024, 6, 3, 16, 0, tzinfo=UTC)
    md = FixtureMarketDataProvider(n_assets=2, bar_age_seconds=1.0, now=now)

    def _run(*, dry_run: bool) -> object:
        broker = PaperBroker(
            config=PaperBrokerConfig(data_dir=tmp_path / "broker", initial_cash=100_000.0),
            now_fn=lambda: now,
        )
        svc = PaperTradingService(
            settings,
            dry_run=dry_run,
            market_data=md,
            broker=broker,
            now_fn=lambda: now,
        )
        svc.start()
        result = svc.run_cycle()
        svc.stop()
        return result

    first = _run(dry_run=False)
    assert first.skipped is False  # type: ignore[union-attr]
    assert first.order_ids  # type: ignore[union-attr]
    n_orders = len(json.loads((tmp_path / "broker" / "orders.json").read_text())["orders"])

    second = _run(dry_run=False)
    assert second.skipped is True  # type: ignore[union-attr]
    assert second.details.get("idempotent") is True  # type: ignore[union-attr]
    n_orders_after = len(json.loads((tmp_path / "broker" / "orders.json").read_text())["orders"])
    assert n_orders_after == n_orders


def test_risk_reject_reason_persisted(tmp_path: Path) -> None:
    manifest = _write_stub_approved(tmp_path / "m")
    # Tiny max position + exposure forces reject/resize; use kill via daily loss context
    # by configuring absurdly tight daily loss and injecting via custom risk — simpler:
    # use zero max_orders so any... actually orders_submitted starts at 0.
    # Force reject with max_position and resize disabled path: invalid via kill switch pre-trip.
    settings = _settings(tmp_path, manifest)
    settings = settings.model_copy(
        update={
            "risk": settings.risk.model_copy(
                update={
                    "max_position_size": 0.01,
                    "max_portfolio_exposure": 0.02,
                    "min_cash_reserve": 0.99,
                }
            )
        }
    )
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    md = FixtureMarketDataProvider(n_assets=2, bar_age_seconds=1.0, now=now)
    broker = PaperBroker(
        config=PaperBrokerConfig(data_dir=tmp_path / "broker", initial_cash=100_000.0),
        now_fn=lambda: now,
    )
    svc = PaperTradingService(
        settings,
        dry_run=False,
        market_data=md,
        broker=broker,
        now_fn=lambda: now,
    )
    svc.start()
    # Trip kill switch so review rejects with a durable reason.
    svc.risk_engine.kill_switch.trip("unit-test-trip")
    result = svc.run_cycle()
    svc.stop()
    assert result.risk_decision == "rejected"
    assert result.risk_reason
    doc = json.loads((tmp_path / "state" / "decisions.jsonl").read_text().strip().splitlines()[-1])
    assert doc["risk_decision"] == "rejected"
    assert doc["risk_reason"]
    assert doc["model_version"] == "9"
    assert doc["rejected_orders"]
    assert doc["rejected_orders"][0]["reason"]


def test_assert_snapshot_fresh_helper() -> None:
    now = datetime(2024, 6, 3, 12, 0, tzinfo=UTC)
    md = FixtureMarketDataProvider(n_assets=2, bar_age_seconds=1.0, now=now)
    snap = md.latest()
    assert_snapshot_fresh(snap, max_age_seconds=10.0, now=now)
    with pytest.raises(StaleMarketDataError):
        assert_snapshot_fresh(snap, max_age_seconds=0.1, now=now + timedelta(seconds=5))


def test_provision_demo_model(tmp_path: Path) -> None:
    path = provision_demo_approved_model(tmp_path / "demo")
    bundle = load_approved_model(path, evaluation_period_id="paper-default")
    assert bundle.algorithm == "stub"
    assert path.is_file()
