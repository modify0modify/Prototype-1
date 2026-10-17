"""Reconciliation queries over the trading database."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine

from rl_trader.storage.db import session_scope
from rl_trader.storage.models import (
    FillRow,
    OrderRow,
    PortfolioSnapshotRow,
    PositionRow,
)


@dataclass(frozen=True, slots=True)
class ReconcileFinding:
    """One reconciliation issue or OK check."""

    check: str
    ok: bool
    detail: str
    metrics: dict[str, Any]


def orders_without_risk_link(engine: Engine) -> ReconcileFinding:
    """Orders (non-dry-run) missing a risk_decision_id."""
    with session_scope(engine) as session:
        n = session.scalar(
            select(func.count())
            .select_from(OrderRow)
            .where(
                OrderRow.risk_decision_id.is_(None),
                OrderRow.dry_run.is_(False),
            )
        )
    count = int(n or 0)
    return ReconcileFinding(
        check="orders_without_risk_link",
        ok=count == 0,
        detail=f"{count} non-dry-run orders missing risk_decision_id",
        metrics={"count": count},
    )


def orders_without_agent_link(engine: Engine) -> ReconcileFinding:
    """Orders missing an agent_decision_id."""
    with session_scope(engine) as session:
        n = session.scalar(
            select(func.count())
            .select_from(OrderRow)
            .where(
                OrderRow.agent_decision_id.is_(None),
                OrderRow.dry_run.is_(False),
            )
        )
    count = int(n or 0)
    return ReconcileFinding(
        check="orders_without_agent_link",
        ok=count == 0,
        detail=f"{count} non-dry-run orders missing agent_decision_id",
        metrics={"count": count},
    )


def orders_without_configuration_link(engine: Engine) -> ReconcileFinding:
    """Orders missing configuration_version_id."""
    with session_scope(engine) as session:
        n = session.scalar(
            select(func.count())
            .select_from(OrderRow)
            .where(
                OrderRow.configuration_version_id.is_(None),
                OrderRow.dry_run.is_(False),
            )
        )
    count = int(n or 0)
    return ReconcileFinding(
        check="orders_without_configuration_link",
        ok=count == 0,
        detail=f"{count} non-dry-run orders missing configuration_version_id",
        metrics={"count": count},
    )


def filled_orders_without_fills(engine: Engine) -> ReconcileFinding:
    """Orders marked filled/partially_filled that have zero fill rows."""
    with session_scope(engine) as session:
        fill_counts = (
            select(FillRow.order_id, func.count().label("n"))
            .group_by(FillRow.order_id)
            .subquery()
        )
        q = (
            select(func.count())
            .select_from(OrderRow)
            .outerjoin(fill_counts, OrderRow.id == fill_counts.c.order_id)
            .where(
                OrderRow.status.in_(("filled", "partially_filled")),
                func.coalesce(fill_counts.c.n, 0) == 0,
            )
        )
        n = session.scalar(q)
    count = int(n or 0)
    return ReconcileFinding(
        check="filled_orders_without_fills",
        ok=count == 0,
        detail=f"{count} filled/partial orders with no fill rows",
        metrics={"count": count},
    )


def fill_qty_vs_order_filled_qty(engine: Engine) -> ReconcileFinding:
    """Sum(fill.qty) should match order.filled_qty within tolerance."""
    mismatches = 0
    with session_scope(engine) as session:
        orders = session.scalars(
            select(OrderRow).where(OrderRow.filled_qty > 0)
        ).all()
        for order in orders:
            total = session.scalar(
                select(func.coalesce(func.sum(FillRow.qty), 0.0)).where(
                    FillRow.order_id == order.id
                )
            )
            if abs(float(total or 0.0) - float(order.filled_qty)) > 1e-6:
                mismatches += 1
    return ReconcileFinding(
        check="fill_qty_vs_order_filled_qty",
        ok=mismatches == 0,
        detail=f"{mismatches} orders where sum(fills.qty) != filled_qty",
        metrics={"mismatches": mismatches},
    )


def duplicate_idempotency_keys(engine: Engine) -> ReconcileFinding:
    """Idempotency keys must be unique (schema enforces; query for reporting)."""
    with session_scope(engine) as session:
        rows = session.execute(
            text(
                "SELECT idempotency_key, COUNT(*) AS n FROM orders "
                "WHERE idempotency_key IS NOT NULL "
                "GROUP BY idempotency_key HAVING n > 1"
            )
        ).all()
    count = len(rows)
    return ReconcileFinding(
        check="duplicate_idempotency_keys",
        ok=count == 0,
        detail=f"{count} duplicated idempotency keys",
        metrics={"count": count, "keys": [r[0] for r in rows]},
    )


def orphan_fills(engine: Engine) -> ReconcileFinding:
    """Fills whose order_id does not exist (FK should prevent; defensive)."""
    with session_scope(engine) as session:
        n = session.scalar(
            select(func.count())
            .select_from(FillRow)
            .outerjoin(OrderRow, FillRow.order_id == OrderRow.id)
            .where(OrderRow.id.is_(None))
        )
    count = int(n or 0)
    return ReconcileFinding(
        check="orphan_fills",
        ok=count == 0,
        detail=f"{count} orphan fill rows",
        metrics={"count": count},
    )


def position_snapshot_cash_vs_equity(engine: Engine) -> ReconcileFinding:
    """Latest snapshot: cash + sum(position market_value) ≈ equity (when MV present)."""
    with session_scope(engine) as session:
        snap = session.scalar(
            select(PortfolioSnapshotRow).order_by(PortfolioSnapshotRow.as_of.desc()).limit(1)
        )
        if snap is None:
            return ReconcileFinding(
                check="position_snapshot_cash_vs_equity",
                ok=True,
                detail="no portfolio snapshots",
                metrics={},
            )
        positions = session.scalars(
            select(PositionRow).where(PositionRow.snapshot_id == snap.id)
        ).all()
        mv_sum = 0.0
        used_mv = True
        for p in positions:
            if p.market_value is None:
                used_mv = False
                mv_sum += float(p.qty) * float(p.avg_price)
            else:
                mv_sum += float(p.market_value)
        implied = float(snap.cash) + mv_sum
        err = abs(implied - float(snap.equity))
        # Loose tolerance: costs / marks.
        ok = err <= max(1.0, 0.02 * abs(float(snap.equity)))
    return ReconcileFinding(
        check="position_snapshot_cash_vs_equity",
        ok=ok,
        detail=(
            f"snapshot {snap.id}: cash+mv={implied:.4f} equity={snap.equity:.4f} "
            f"err={err:.4f} used_market_value={used_mv}"
        ),
        metrics={
            "cash": snap.cash,
            "equity": snap.equity,
            "implied_equity": implied,
            "abs_error": err,
        },
    )


def rejected_orders_have_reason(engine: Engine) -> ReconcileFinding:
    """Rejected orders must store a reject_reason."""
    with session_scope(engine) as session:
        n = session.scalar(
            select(func.count())
            .select_from(OrderRow)
            .where(
                OrderRow.status == "rejected",
                (OrderRow.reject_reason.is_(None)) | (OrderRow.reject_reason == ""),
            )
        )
    count = int(n or 0)
    return ReconcileFinding(
        check="rejected_orders_have_reason",
        ok=count == 0,
        detail=f"{count} rejected orders missing reject_reason",
        metrics={"count": count},
    )


RECONCILE_CHECKS = (
    orders_without_risk_link,
    orders_without_agent_link,
    orders_without_configuration_link,
    filled_orders_without_fills,
    fill_qty_vs_order_filled_qty,
    duplicate_idempotency_keys,
    orphan_fills,
    position_snapshot_cash_vs_equity,
    rejected_orders_have_reason,
)


def run_all_reconciliations(engine: Engine) -> list[ReconcileFinding]:
    """Execute every reconciliation check."""
    return [fn(engine) for fn in RECONCILE_CHECKS]
