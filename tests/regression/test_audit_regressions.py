"""Regression tests for final production-readiness audit findings."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config import get_settings
from rl_trader.config.models import (
    BrokerSettings,
    LiveTradingSettings,
    PaperTradingSettings,
    RiskSettings,
)
from rl_trader.config.settings import AppSettings, RuntimeMode
from rl_trader.execution.paper import PaperExecutor
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import RiskDecision, RiskEngine, RiskReviewContext
from rl_trader.risk.decisions import DecisionRecord, RiskResult


def test_paper_yaml_disables_fixture_market_data_by_default(monkeypatch: object) -> None:
    """Controlled paper must not silently use synthetic fixtures."""
    for key in ("APP_MODE", "LIVE_TRADING_ENABLED", "BROKER_ENVIRONMENT"):
        monkeypatch.delenv(key, raising=False)  # type: ignore[attr-defined]
    settings = get_settings(config_file=Path("configs/paper.yaml"))
    assert settings.paper_trading.use_fixture_market_data is False


def test_service_refuses_silent_fixture_fallback(tmp_path: Path) -> None:
    """start() must not invent FixtureMarketData when fixtures are disabled."""
    from rl_trader.approval.testing import write_sealed_approved_manifest
    from rl_trader.paper_trading.service import PaperTradingService

    manifest = write_sealed_approved_manifest(
        tmp_path / "m", tier="paper", model_id="audit-model", n_assets=2
    )
    settings = AppSettings(
        _env_file=None,
        app_mode=RuntimeMode.PAPER,
        live_trading_enabled=False,
        broker=BrokerSettings(environment="paper"),
        risk=RiskSettings(
            max_position_size=0.5,
            max_portfolio_exposure=1.0,
            max_sector_exposure=1.0,
            daily_loss_limit=0.5,
            max_drawdown_limit=0.9,
            configuration_version="audit",
        ),
        paper_trading=PaperTradingSettings(
            state_dir=tmp_path / "state",
            broker_data_dir=tmp_path / "broker",
            use_fixture_market_data=False,
            evaluation_period_id="audit",
        ),
        live=LiveTradingSettings(approved_model_manifest=manifest),
    )
    svc = PaperTradingService(settings, acquire_lock=False)
    with pytest.raises(RuntimeError, match="refusing silent fixture fallback"):
        svc.start()


def test_paper_broker_rejects_risk_rejected_targets(tmp_path: Path) -> None:
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    broker = PaperBroker(
        PaperBrokerConfig(data_dir=tmp_path / "b", initial_cash=10_000.0),
        now_fn=lambda: now,
    )
    broker.connect()
    rejected = RiskResult(
        decision=RiskDecision.REJECTED,
        portfolio=None,
        reason="kill_switch",
        record=DecisionRecord(
            decision_status=RiskDecision.REJECTED,
            original_order=(0.5,),
            approved_order=None,
            violated_rules=("kill_switch",),
            reason="kill_switch",
            timestamp=now,
            configuration_version="a",
            model_identifier="m",
        ),
    )
    with pytest.raises(ValueError, match="rejected"):
        broker.submit_target(rejected)


def test_paper_executor_never_calls_broker_on_risk_reject(tmp_path: Path) -> None:
    """Risk rejection must short-circuit before broker submit_target."""
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)

    class _SpyBroker(PaperBroker):
        def __init__(self) -> None:
            super().__init__(
                PaperBrokerConfig(data_dir=tmp_path / "spy", initial_cash=1_000.0),
                now_fn=lambda: now,
            )
            self.submit_target_calls = 0
            self.connect()

        def submit_target(self, risk_result: RiskResult):  # type: ignore[override]
            self.submit_target_calls += 1
            return super().submit_target(risk_result)

    spy = _SpyBroker()
    eng = RiskEngine(
        settings=RiskSettings(
            max_position_size=0.1,
            max_portfolio_exposure=0.2,
            max_sector_exposure=0.2,
            daily_loss_limit=0.05,
            max_drawdown_limit=0.2,
            configuration_version="audit",
        )
    )
    eng.kill_switch.trip("audit")
    executor = PaperExecutor(risk_engine=eng, broker=spy, auto_connect=False)
    with pytest.raises(ValueError, match="risk rejected"):
        executor.execute(TargetPortfolio(weights=np.array([0.1, 0.1])))
    assert spy.submit_target_calls == 0


def test_approval_persist_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging

    from rl_trader.approval.evidence import example_passing_paper_evidence
    from rl_trader.approval.gates import ApprovalGateReport
    from rl_trader.approval.service import VerifiedApproval, maybe_persist_verified

    verified = VerifiedApproval(
        manifest={"model_id": "log-me", "approver": "a", "code_commit_id": "c", "notes": ""},
        content_hash="abc",
        evidence=example_passing_paper_evidence(),
        gate_report=ApprovalGateReport(tier="paper", allowed=True, results=()),
        tier="paper",
    )

    def _boom(_url: str) -> None:
        raise OSError("simulated db unavailable")

    monkeypatch.setattr("rl_trader.storage.migrate.upgrade_head", _boom)
    caplog.set_level(logging.ERROR, logger="rl_trader.approval")
    maybe_persist_verified(
        verified,
        database_url=f"sqlite:///{tmp_path / 'unused.db'}",
        persist=True,
    )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "failed to persist approval" in text
    assert "log-me" in text


def test_risk_review_context_required_for_sizing() -> None:
    """Oversized orders are resized — agent cannot bypass notional caps."""
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    eng = RiskEngine(
        settings=RiskSettings(
            max_position_size=0.5,
            max_portfolio_exposure=1.0,
            max_sector_exposure=1.0,
            max_order_notional=1_000.0,
            min_cash_reserve=0.0,
            max_daily_turnover=2.0,
            max_orders_per_day=50,
            daily_loss_limit=0.5,
            max_drawdown_limit=0.9,
            max_volatility_target=1.0,
            max_consecutive_losses=50,
            max_data_age_seconds=300.0,
            abnormal_price_jump=0.5,
            max_spread_bps=500.0,
            configuration_version="audit",
        )
    )
    ctx = RiskReviewContext(
        symbols=("A", "B"),
        prices=np.array([100.0, 100.0]),
        prior_prices=np.array([100.0, 100.0]),
        price_timestamps=(now - timedelta(seconds=1), now - timedelta(seconds=1)),
        now=now,
        portfolio_value=100_000.0,
        cash=100_000.0,
        current_weights=np.array([0.0, 0.0]),
        day_pnl_frac=0.0,
        week_pnl_frac=0.0,
        peak_nav=100_000.0,
        current_nav=100_000.0,
        realized_volatility=0.1,
        bid_ask_spread_bps=np.array([5.0, 5.0]),
        sector_by_symbol={"A": "x", "B": "x"},
        market_is_open=True,
        orders_submitted_today=0,
        last_order_fingerprint=None,
        pending_order_ids=(),
        consecutive_losses=0,
        database_available=True,
        account_confirmed=True,
        trading_mode="paper",
        model_id="a",
        config_version="audit",
    )
    result = eng.review(TargetPortfolio(weights=np.array([0.5, 0.0])), context=ctx)
    assert result.portfolio is not None
    notional = abs(float(result.portfolio.weights[0])) * 100_000.0
    assert notional <= 1_000.0 + 1e-6


def test_order_router_detects_paper_path_calling_live() -> None:
    """Miswired paper adapter that invokes live must fail closed."""
    from rl_trader.brokers.types import Order, OrderRequest, OrderSide, OrderStatus, OrderType
    from rl_trader.live_trading.routing import LiveRoutingError, OrderRouter, RoutingMode

    class _Live:
        mode = "live"

        def __init__(self) -> None:
            self.calls = 0

        def submit_order(self, request: OrderRequest) -> Order:
            self.calls += 1
            return Order(
                order_id="live-1",
                symbol=request.symbol,
                side=request.side,
                qty=request.qty,
                order_type=request.order_type,
                status=OrderStatus.ACCEPTED,
            )

    class _LeakyPaper:
        mode = "paper"

        def __init__(self) -> None:
            self.live_ref: object | None = None

        def submit_order(self, request: OrderRequest) -> Order:
            assert self.live_ref is not None
            return self.live_ref.submit_order(request)  # type: ignore[no-any-return]

    live = _Live()
    paper = _LeakyPaper()
    router = OrderRouter(paper=paper, live=live)
    paper.live_ref = router._live
    req = OrderRequest(symbol="SPY", side=OrderSide.BUY, qty=1.0, order_type=OrderType.MARKET)
    with pytest.raises(LiveRoutingError, match="separation violated"):
        router.submit(RoutingMode.PAPER, req)
    assert live.calls == 1
    assert router.live_submit_calls == 1


def test_live_executor_never_calls_broker_on_risk_reject(tmp_path: Path) -> None:
    """Parity with paper: LiveExecutor must short-circuit before submit_target."""
    from rl_trader.approval.testing import write_sealed_approved_manifest
    from rl_trader.config import LIVE_TRADING_CONFIRMATION_PHRASE
    from rl_trader.config.models import DataSettings
    from rl_trader.execution.live import LiveExecutor
    from rl_trader.live_trading.activation import ManualLiveController

    manifest = write_sealed_approved_manifest(
        tmp_path / "m", tier="live", model_id="audit-live", n_assets=2
    )
    settings = AppSettings(
        _env_file=None,
        app_mode=RuntimeMode.LIVE,
        live_trading_enabled=True,
        data=DataSettings(survivorship_bias_acknowledged=True),
        broker=BrokerSettings(environment="live"),
        risk=RiskSettings(
            max_position_size=0.1,
            max_portfolio_exposure=0.2,
            daily_loss_limit=0.02,
            max_drawdown_limit=0.1,
            configuration_version="audit",
        ),
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT999",
            max_live_capital=10_000.0,
            max_order_value=500.0,
            approved_model_manifest=manifest,
            approved_account_allowlist=("LIVEACCT999",),
            rollout_stage=0,
            manual_activated=False,
            shadow_state_dir=tmp_path / "shadow",
        ),
    )
    ctrl = ManualLiveController(settings)
    settings = ctrl.activate(LIVE_TRADING_CONFIRMATION_PHRASE)
    eng = RiskEngine(settings=settings.risk)
    eng.kill_switch.trip("audit")
    executor = LiveExecutor(settings, risk_engine=eng, manual=ctrl)

    calls = {"n": 0}
    original = executor.broker.submit_target

    def _spy(result: RiskResult):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(result)

    executor.broker.submit_target = _spy  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="live broker not called"):
        executor.execute(TargetPortfolio(weights=np.array([0.1, 0.1])))
    assert calls["n"] == 0


def test_compose_demo_flag_is_labelled_smoke_only() -> None:
    """Default compose must not pretend to be controlled paper without --demo label."""
    text = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "--demo" in text
    assert "synthetic" in text.lower() or "smoke" in text.lower()
    assert "controlled paper" in text.lower()
