"""Optional dual-write from paper trading into SQL storage.

File ledgers under ``reports/paper_trading`` and ``data/interim/paper_broker``
remain authoritative for the paper broker process. When
``database.persist_paper_trading`` is enabled, decisions / orders / fills /
alerts are also recorded in SQLite without replacing those files.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.engine import Engine

from rl_trader.brokers.paper import PaperBroker
from rl_trader.brokers.types import Order, OrderStatus
from rl_trader.paper_trading.decisions import CycleDecisionRecord
from rl_trader.storage.repository import StorageRepository


class PaperTradingDBSink:
    """Persist a completed paper cycle into the relational store."""

    def __init__(
        self,
        engine: Engine,
        *,
        model_checkpoint_id: uuid.UUID | None = None,
        configuration_version_id: uuid.UUID | None = None,
    ) -> None:
        self.repo = StorageRepository(engine)
        self.model_checkpoint_id = model_checkpoint_id
        self.configuration_version_id = configuration_version_id

    def persist_cycle(
        self,
        record: CycleDecisionRecord,
        *,
        broker: PaperBroker | None = None,
        commission_bps: float = 0.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> dict[str, Any]:
        """Write agent/risk decisions, orders, fills, snapshot, metrics, alerts."""
        as_of = datetime.fromisoformat(record.as_of)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)

        agent_id = self.repo.record_agent_decision(
            model_id=record.model_id,
            model_version=record.model_version,
            cycle_id=record.cycle_id,
            as_of=as_of,
            target_weights=record.target_weights,
            model_checkpoint_id=self.model_checkpoint_id,
        )
        risk_id = self.repo.record_risk_decision(
            model_id=record.model_id,
            decision=record.risk_decision,
            reason=record.risk_reason,
            violated_rules=record.violated_rules,
            approved_weights=record.approved_weights,
            agent_decision_id=agent_id,
            configuration_version_id=self.configuration_version_id,
            details={
                "cycle_id": record.cycle_id,
                "dry_run": record.dry_run,
                "alerts": record.alerts,
            },
        )

        order_db_ids: list[str] = []
        if broker is not None and not record.dry_run:
            for oid in record.order_ids:
                try:
                    order = broker.get_order_status(oid)
                except KeyError:
                    continue
                db_id = self._persist_order(
                    order,
                    agent_decision_id=agent_id,
                    risk_decision_id=risk_id,
                    dry_run=False,
                    commission_bps=commission_bps,
                    spread_bps=spread_bps,
                    slippage_bps=slippage_bps,
                )
                order_db_ids.append(str(db_id))

        # Persist risk-stage rejects as synthetic order rows when no broker orders.
        for rej in record.rejected_orders:
            if rej.get("stage") != "risk":
                continue
            self.repo.record_order(
                broker_order_id=f"risk-reject:{record.cycle_id}",
                symbol="_PORTFOLIO",
                side="buy",
                qty=0.0,
                status="rejected",
                reject_reason=str(rej.get("reason") or record.risk_reason),
                idempotency_key=f"risk-reject:{record.cycle_id}",
                dry_run=record.dry_run,
                model_checkpoint_id=self.model_checkpoint_id,
                agent_decision_id=agent_id,
                risk_decision_id=risk_id,
                configuration_version_id=self.configuration_version_id,
                created_at=as_of,
            )

        snapshot_id = None
        if broker is not None and broker.connected:
            account = broker.get_account()
            positions = [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_price": p.avg_price,
                    "market_price": p.market_price,
                    "market_value": p.market_value,
                }
                for p in broker.get_positions()
            ]
            snapshot_id = self.repo.record_portfolio_snapshot(
                as_of=as_of,
                cash=account.cash,
                equity=account.equity,
                buying_power=account.buying_power,
                positions=positions,
            )

        for name in record.alerts:
            self.repo.record_alert(
                level="warning",
                message=name,
                context={"cycle_id": record.cycle_id, "model_id": record.model_id},
                created_at=as_of,
            )

        metrics_id = self.repo.record_performance_metrics(
            as_of=as_of,
            metrics={
                "cycle_id": record.cycle_id,
                "risk_decision": record.risk_decision,
                "orders": len(record.order_ids),
                "dry_run": record.dry_run,
            },
            model_id=record.model_id,
        )

        return {
            "agent_decision_id": str(agent_id),
            "risk_decision_id": str(risk_id),
            "order_ids": order_db_ids,
            "snapshot_id": None if snapshot_id is None else str(snapshot_id),
            "metrics_id": str(metrics_id),
        }

    def _persist_order(
        self,
        order: Order,
        *,
        agent_decision_id: uuid.UUID,
        risk_decision_id: uuid.UUID,
        dry_run: bool,
        commission_bps: float,
        spread_bps: float,
        slippage_bps: float,
    ) -> uuid.UUID:
        db_id = self.repo.record_order(
            broker_order_id=order.order_id,
            symbol=order.symbol,
            side=str(order.side),
            qty=order.qty,
            status=str(order.status),
            order_type=str(order.order_type),
            idempotency_key=order.idempotency_key,
            limit_price=order.limit_price,
            filled_qty=order.filled_qty,
            avg_fill_price=order.avg_fill_price,
            reject_reason=order.reject_reason,
            dry_run=dry_run,
            model_checkpoint_id=self.model_checkpoint_id,
            agent_decision_id=agent_decision_id,
            risk_decision_id=risk_decision_id,
            configuration_version_id=self.configuration_version_id,
            created_at=order.created_at,
        )
        if order.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED} and order.filled_qty:
            px = float(order.avg_fill_price or 0.0)
            notional = abs(px * float(order.filled_qty))
            # Attribute stored commission; estimate spread/slippage from bps if needed.
            commission = float(order.commission)
            spread_cost = notional * (spread_bps / 10_000.0)
            slippage_cost = notional * (slippage_bps / 10_000.0)
            self.repo.record_fill(
                order_id=db_id,
                qty=float(order.filled_qty),
                price=px,
                filled_at=order.updated_at or order.created_at or datetime.now(UTC),
                commission=commission,
                spread_cost=spread_cost,
                slippage_cost=slippage_cost,
            )
        return db_id
