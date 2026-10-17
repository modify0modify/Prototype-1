"""Prospective paper-trading service loop.

Order of operations (one cycle):

1. Confirm operating mode is paper
2. Load only an approved model candidate (frozen for the evaluation period)
3. Verify the feature manifest
4. Obtain latest market data
5. Validate timestamps and data freshness
6. Update features (via market provider snapshot)
7. Load latest simulated account state
8. Ask agent for target weights
9. Send targets to risk engine
10. Convert approved weights into orders
11. Submit to PaperBroker (skipped on ``dry_run``)
12. Reconcile expected vs actual paper positions
13. Store all decisions
14. Update performance metrics
15. Send alerts for failures or risk events
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import OrderStatus, Quote
from rl_trader.config.models import PaperTradingSettings, RiskSettings
from rl_trader.config.settings import AppSettings, RuntimeMode
from rl_trader.monitoring import get_logger
from rl_trader.paper_trading.alerts import AlertSink
from rl_trader.paper_trading.decisions import CycleDecisionRecord, DecisionStore
from rl_trader.paper_trading.heartbeat import HeartbeatWriter
from rl_trader.paper_trading.locks import ProcessLock, ProcessLockError
from rl_trader.paper_trading.market_data import (
    FixtureMarketDataProvider,
    MarketDataProvider,
    MarketSnapshot,
    StaleMarketDataError,
    assert_snapshot_fresh,
)
from rl_trader.paper_trading.metrics import MetricsStore
from rl_trader.paper_trading.model_loader import (
    ApprovedModelBundle,
    load_approved_model,
)
from rl_trader.paper_trading.retrain import RetrainHookResult, maybe_schedule_retrain
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.context import RiskReviewContext
from rl_trader.risk.decisions import RiskDecision
from rl_trader.risk.engine import RiskEngine, RiskResult

__all__ = [
    "PaperTradingCycleResult",
    "PaperTradingService",
    "StaleMarketDataError",
]


@dataclass(slots=True)
class PaperTradingCycleResult:
    """Outcome of a single prospective paper cycle."""

    cycle_id: str
    skipped: bool
    dry_run: bool
    risk_decision: str
    risk_reason: str
    model_id: str
    model_version: str
    order_ids: list[str] = field(default_factory=list)
    shutdown: bool = False
    alerts: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class PaperTradingService:
    """Orchestrates the approved-model → risk → PaperBroker path."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        dry_run: bool = False,
        market_data: MarketDataProvider | None = None,
        broker: PaperBroker | None = None,
        risk_engine: RiskEngine | None = None,
        model: ApprovedModelBundle | None = None,
        now_fn: Callable[[], datetime] | None = None,
        acquire_lock: bool = True,
    ) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._acquire_lock = acquire_lock
        self.pt = settings.paper_trading
        self.state_dir = Path(self.pt.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.lock = ProcessLock(self.state_dir / self.pt.lock_filename)
        self.heartbeat = HeartbeatWriter(self.state_dir / self.pt.heartbeat_filename)
        self.decisions = DecisionStore(self.state_dir / self.pt.decisions_filename)
        self.metrics = MetricsStore(self.state_dir / self.pt.metrics_filename)
        self.alerts = AlertSink(self.state_dir / self.pt.alerts_filename)
        self.logger = get_logger("rl_trader.paper_trading")

        self.risk_engine = risk_engine or RiskEngine(settings=settings.risk)
        self._model = model
        self._market_data = market_data
        self._broker = broker
        self._owns_broker = broker is None
        self._lock_held = False
        self._model_frozen = False
        self.last_retrain_hook: RetrainHookResult | None = None
        self._db_sink: Any | None = None
        self._db_engine: Any | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> ApprovedModelBundle:
        """Confirm paper mode, acquire lock, load frozen approved model."""
        self._confirm_paper_mode()
        if self._acquire_lock:
            try:
                self.lock.acquire()
                self._lock_held = True
            except ProcessLockError:
                self.alerts.emit(
                    "error",
                    "process_lock_held",
                    now=self._now(),
                    context={"lock": str(self.lock.path)},
                )
                raise

        now = self._now()
        self.heartbeat.beat(now=now, status="starting")

        if self._model is None:
            self._model = load_approved_model(
                self.settings.live.approved_model_manifest,
                evaluation_period_id=self.pt.evaluation_period_id,
                approval_settings=self.settings.approval,
            )
        self._model_frozen = True

        if self._market_data is None:
            n_assets = getattr(self._model.agent, "n_assets", len(self.settings.data.universe))
            self._market_data = FixtureMarketDataProvider(
                n_assets=int(n_assets),
                bar_age_seconds=5.0,
                now=now,
                symbols=tuple(self.settings.data.universe[: int(n_assets)]),
            )

        if self._broker is None:
            self._broker = PaperBroker(
                config=PaperBrokerConfig(
                    data_dir=Path(self.pt.broker_data_dir),
                    initial_cash=self.pt.initial_cash,
                    commission_bps=self.pt.commission_bps,
                    spread_bps=self.pt.spread_bps,
                    slippage_bps=self.pt.slippage_bps,
                    rng_seed=self.settings.seed,
                ),
                now_fn=self._now_fn,
            )
        if not self._broker.connected:
            self._broker.connect()

        # Retrain is never applied in-loop; optional external hook only.
        self.last_retrain_hook = maybe_schedule_retrain(
            enabled=self.pt.enable_scheduled_retrain_hook,
            schedule=self.pt.retrain_schedule,
            evaluation_period_id=self.pt.evaluation_period_id,
            model_id=self._model.model_id,
            now=now,
        )

        self._init_db_sink()

        self.heartbeat.beat(
            now=now,
            status="ready",
            details={
                "model_id": self._model.model_id,
                "model_version": self._model.model_version,
                "evaluation_period_id": self._model.evaluation_period_id,
                "dry_run": self.dry_run,
                "model_frozen": self._model_frozen,
            },
        )
        return self._model

    def stop(self) -> None:
        """Flush broker state, release lock, write stopped heartbeat."""
        if self._broker is not None and self._broker.connected and self._owns_broker:
            self._broker.disconnect()
        self.heartbeat.beat(now=self._now(), status="stopped")
        if self._lock_held:
            self.lock.release()
            self._lock_held = False

    def __enter__(self) -> PaperTradingService:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        _ = args
        self.stop()

    # ------------------------------------------------------------------ #
    # Cycle
    # ------------------------------------------------------------------ #

    def run_cycle(self) -> PaperTradingCycleResult:
        """Execute one full prospective paper cycle."""
        if self._model is None or self._broker is None or self._market_data is None:
            raise RuntimeError("service not started; call start() first")
        if not self._model_frozen:
            raise RuntimeError("model must remain frozen during the evaluation period")

        model = self._model
        broker = self._broker
        now = self._now()
        self.heartbeat.beat(now=now, status="cycle_running", details={"model_id": model.model_id})

        try:
            snapshot = self._market_data.latest()
            assert_snapshot_fresh(
                snapshot,
                max_age_seconds=self.pt.max_data_age_seconds,
                now=now,
            )
        except StaleMarketDataError as exc:
            self.alerts.emit(
                "critical",
                "stale_or_incomplete_data_shutdown",
                now=now,
                context={"error": str(exc)},
            )
            self.heartbeat.beat(now=now, status="shutdown_stale_data", details={"error": str(exc)})
            return PaperTradingCycleResult(
                cycle_id="",
                skipped=True,
                dry_run=self.dry_run,
                risk_decision="shutdown",
                risk_reason=str(exc),
                model_id=model.model_id,
                model_version=model.model_version,
                shutdown=True,
                alerts=["stale_or_incomplete_data_shutdown"],
            )

        cycle_id = self._cycle_id(model, snapshot)
        if self.decisions.has_cycle(cycle_id):
            self.logger.info("skipping duplicate cycle_id=%s (restart idempotency)", cycle_id)
            return PaperTradingCycleResult(
                cycle_id=cycle_id,
                skipped=True,
                dry_run=self.dry_run,
                risk_decision="skipped_duplicate",
                risk_reason="cycle_already_recorded",
                model_id=model.model_id,
                model_version=model.model_version,
                details={"idempotent": True},
            )

        # Inject quotes for the paper broker.
        for i, sym in enumerate(snapshot.symbols):
            broker.set_quote(
                Quote(
                    symbol=sym,
                    bid=float(snapshot.bids[i]),
                    ask=float(snapshot.asks[i]),
                    last=float(snapshot.prices[i]),
                    timestamp=snapshot.price_timestamps[i],
                )
            )

        account = broker.get_account()
        positions = broker.get_positions()
        current_weights = self._current_weights(
            symbols=snapshot.symbols,
            prices=snapshot.prices,
            positions={p.symbol: p.qty for p in positions},
            equity=account.equity,
        )

        raw_target = model.agent.predict_target(snapshot.observation)
        target = TargetPortfolio(
            weights=np.asarray(raw_target.weights, dtype=np.float64).reshape(-1),
            as_of=snapshot.as_of.isoformat(),
        )

        context = self._build_risk_context(
            snapshot=snapshot,
            account_cash=account.cash,
            portfolio_value=account.equity,
            current_weights=current_weights,
            model_id=model.model_id,
            pending=tuple(o.order_id for o in broker.get_open_orders()),
        )
        risk_result = self.risk_engine.review(
            target,
            context,
            model_id=model.model_id,
        )
        # Attach symbols for PaperBroker.submit_target
        risk_result = RiskResult(
            decision=risk_result.decision,
            portfolio=risk_result.portfolio,
            reason=risk_result.reason,
            record=risk_result.record,
            details={
                **dict(risk_result.details or {}),
                "symbols": list(snapshot.symbols),
                "model_version": model.model_version,
                "cycle_id": cycle_id,
            },
        )

        order_ids: list[str] = []
        rejected_orders: list[dict[str, Any]] = []
        alert_names: list[str] = []
        reconcile: dict[str, Any] = {}

        if risk_result.decision is RiskDecision.REJECTED:
            alert_names.append("risk_rejected")
            self.alerts.emit(
                "warning",
                "risk_rejected",
                now=now,
                context={
                    "reason": risk_result.reason,
                    "violated_rules": list(risk_result.record.violated_rules),
                    "model_id": model.model_id,
                    "model_version": model.model_version,
                    "cycle_id": cycle_id,
                },
            )
            rejected_orders.append(
                {
                    "stage": "risk",
                    "reason": risk_result.reason,
                    "violated_rules": list(risk_result.record.violated_rules),
                }
            )
        elif self.dry_run:
            self.logger.info(
                "dry-run: decision=%s — no simulated orders created",
                risk_result.decision.value,
            )
        else:
            fill = broker.submit_target(risk_result)
            order_ids = list(fill.metadata.get("order_ids") or [])
            for oid in order_ids:
                order = broker.get_order_status(oid)
                if order.status is OrderStatus.REJECTED:
                    rejected_orders.append(
                        {
                            "stage": "broker",
                            "order_id": oid,
                            "symbol": order.symbol,
                            "reason": order.reject_reason or "rejected",
                            "status": str(order.status),
                        }
                    )
                    alert_names.append("order_rejected")
                    self.alerts.emit(
                        "warning",
                        "order_rejected",
                        now=now,
                        context={
                            "order_id": oid,
                            "symbol": order.symbol,
                            "reason": order.reject_reason,
                            "model_id": model.model_id,
                            "model_version": model.model_version,
                        },
                    )

            expected = (
                np.asarray(risk_result.portfolio.weights, dtype=np.float64).reshape(-1)
                if risk_result.portfolio is not None
                else current_weights
            )
            reconcile = self._reconcile_positions(
                broker=broker,
                symbols=snapshot.symbols,
                prices=snapshot.prices,
                expected_weights=expected,
            )
            if not reconcile.get("ok", False):
                alert_names.append("reconcile_mismatch")
                self.alerts.emit(
                    "warning",
                    "reconcile_mismatch",
                    now=now,
                    context=reconcile,
                )

        account_after = broker.get_account()
        approved_w = (
            list(np.asarray(risk_result.portfolio.weights, dtype=np.float64).reshape(-1))
            if risk_result.portfolio is not None
            else None
        )
        record = CycleDecisionRecord(
            cycle_id=cycle_id,
            model_id=model.model_id,
            model_version=model.model_version,
            evaluation_period_id=model.evaluation_period_id,
            as_of=snapshot.as_of.isoformat(),
            dry_run=self.dry_run,
            risk_decision=risk_result.decision.value,
            risk_reason=risk_result.reason,
            violated_rules=list(risk_result.record.violated_rules),
            target_weights=list(np.asarray(target.weights, dtype=np.float64).reshape(-1)),
            approved_weights=approved_w,
            order_ids=order_ids,
            rejected_orders=rejected_orders,
            reconcile=reconcile,
            alerts=alert_names,
            metadata={
                "algorithm": model.algorithm,
                "checkpoint_dir": str(model.checkpoint_dir),
                "feature_specs": [
                    {"name": s.name, "lookback": s.lookback, "version": s.version}
                    for s in model.feature_specs
                ],
                "risk_record": risk_result.record.to_dict(),
            },
        )
        self.decisions.append(record)
        if self._db_sink is not None:
            try:
                db_ids = self._db_sink.persist_cycle(
                    record,
                    broker=broker,
                    commission_bps=self.pt.commission_bps,
                    spread_bps=self.pt.spread_bps,
                    slippage_bps=self.pt.slippage_bps,
                )
                self.logger.info("dual-wrote cycle to database ids=%s", db_ids)
            except Exception:
                self.logger.exception("database dual-write failed (file ledger intact)")
                self.alerts.emit(
                    "error",
                    "database_dual_write_failed",
                    now=now,
                    context={"cycle_id": cycle_id},
                )
        self.metrics.update_from_cycle(
            now=now,
            risk_decision=risk_result.decision.value,
            dry_run=self.dry_run,
            orders_submitted=0 if self.dry_run else len(order_ids),
            equity=account_after.equity,
            cash=account_after.cash,
            model_id=model.model_id,
        )
        self.heartbeat.beat(
            now=now,
            status="cycle_complete",
            details={
                "cycle_id": cycle_id,
                "decision": risk_result.decision.value,
                "dry_run": self.dry_run,
            },
        )
        return PaperTradingCycleResult(
            cycle_id=cycle_id,
            skipped=False,
            dry_run=self.dry_run,
            risk_decision=risk_result.decision.value,
            risk_reason=risk_result.reason,
            model_id=model.model_id,
            model_version=model.model_version,
            order_ids=order_ids,
            alerts=alert_names,
            details={"reconcile": reconcile, "rejected_orders": rejected_orders},
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _init_db_sink(self) -> None:
        """Optionally open SQL dual-write when configured (file ledgers stay primary)."""
        db = self.settings.database
        if not db.persist_paper_trading:
            return
        from rl_trader.storage.db import create_db_engine
        from rl_trader.storage.migrate import upgrade_head
        from rl_trader.storage.paper_sink import PaperTradingDBSink
        from rl_trader.storage.repository import StorageRepository

        upgrade_head(db.url)
        engine = create_db_engine(db.url, echo=db.echo, create_schema=False)
        repo = StorageRepository(engine)
        cfg_id = repo.upsert_configuration_version(
            self.settings.risk.configuration_version,
            {
                "risk": self.settings.risk.model_dump(mode="json"),
                "paper_trading": self.pt.model_dump(mode="json"),
                "broker": {"environment": self.settings.broker.environment},
            },
        )
        assert self._model is not None
        ckpt_id = repo.record_checkpoint(
            model_id=self._model.model_id,
            algorithm=self._model.algorithm,
            checkpoint_path=str(self._model.checkpoint_dir),
            model_version=self._model.model_version,
            package_version=(
                self._model.metadata.package_version if self._model.metadata else ""
            ),
        )
        self._db_engine = engine
        self._db_sink = PaperTradingDBSink(
            engine,
            model_checkpoint_id=ckpt_id,
            configuration_version_id=cfg_id,
        )

    def _confirm_paper_mode(self) -> None:
        if self.settings.app_mode is not RuntimeMode.PAPER:
            raise RuntimeError(
                f"paper trading service requires app_mode=paper "
                f"(got {self.settings.app_mode.value})"
            )
        if self.settings.live_trading_enabled:
            raise RuntimeError("live_trading_enabled must be false for paper service")
        if self.settings.broker.environment != "paper":
            raise RuntimeError("broker.environment must be 'paper'")

    def _now(self) -> datetime:
        ts = self._now_fn()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC)

    def _cycle_id(self, model: ApprovedModelBundle, snapshot: MarketSnapshot) -> str:
        bar = snapshot.feature_as_of.astimezone(UTC).isoformat()
        return f"{model.evaluation_period_id}:{model.model_id}:{bar}"

    def _current_weights(
        self,
        *,
        symbols: tuple[str, ...],
        prices: np.ndarray,
        positions: dict[str, float],
        equity: float,
    ) -> np.ndarray:
        if equity <= 0:
            return np.zeros(len(symbols), dtype=np.float64)
        weights = np.zeros(len(symbols), dtype=np.float64)
        for i, sym in enumerate(symbols):
            qty = float(positions.get(sym, 0.0))
            weights[i] = (qty * float(prices[i])) / equity
        return weights

    def _build_risk_context(
        self,
        *,
        snapshot: MarketSnapshot,
        account_cash: float,
        portfolio_value: float,
        current_weights: np.ndarray,
        model_id: str,
        pending: tuple[str, ...],
    ) -> RiskReviewContext:
        risk: RiskSettings = self.settings.risk
        spreads = (
            (snapshot.asks - snapshot.bids) / np.maximum(snapshot.prices, 1e-12) * 10_000.0
        )
        return RiskReviewContext(
            symbols=snapshot.symbols,
            prices=snapshot.prices,
            price_timestamps=snapshot.price_timestamps,
            now=snapshot.as_of,
            portfolio_value=portfolio_value,
            cash=account_cash,
            current_weights=current_weights,
            prior_prices=snapshot.prior_prices,
            bid_ask_spread_bps=spreads,
            market_is_open=True,
            pending_order_ids=pending,
            trading_mode="paper",
            model_id=model_id,
            config_version=risk.configuration_version,
            peak_nav=max(portfolio_value, self.metrics.metrics.peak_equity or portfolio_value),
            current_nav=portfolio_value,
        )

    def _reconcile_positions(
        self,
        *,
        broker: PaperBroker,
        symbols: tuple[str, ...],
        prices: np.ndarray,
        expected_weights: np.ndarray,
    ) -> dict[str, Any]:
        ledger = broker.reconcile()
        account = broker.get_account()
        positions = {p.symbol: p.qty for p in broker.get_positions()}
        actual = self._current_weights(
            symbols=symbols,
            prices=prices,
            positions=positions,
            equity=account.equity,
        )
        expected = np.asarray(expected_weights, dtype=np.float64).reshape(-1)
        if expected.shape != actual.shape:
            return {
                "ok": False,
                "reason": "shape_mismatch",
                "ledger": ledger,
                "expected": expected.tolist(),
                "actual": actual.tolist(),
            }
        abs_err = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
        # Tolerant: costs / partial fills / cash residual.
        ok = abs_err <= 0.15 and bool(ledger.get("ok", False))
        return {
            "ok": ok,
            "max_abs_weight_error": abs_err,
            "expected_weights": expected.tolist(),
            "actual_weights": actual.tolist(),
            "ledger": ledger,
            "cash": account.cash,
            "equity": account.equity,
        }


def build_service_from_settings(
    settings: AppSettings,
    *,
    dry_run: bool = False,
    market_data: MarketDataProvider | None = None,
) -> PaperTradingService:
    """Factory used by the CLI."""
    pt: PaperTradingSettings = settings.paper_trading
    md = market_data
    if md is None and pt.use_fixture_market_data:
        md = FixtureMarketDataProvider(
            n_assets=min(2, len(settings.data.universe)) or 2,
            symbols=tuple(settings.data.universe[:2]),
        )
    return PaperTradingService(settings, dry_run=dry_run, market_data=md)
