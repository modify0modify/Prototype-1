"""High-level persistence helpers for trading entities."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from rl_trader.storage.db import session_scope
from rl_trader.storage.models import (
    AgentDecisionRow,
    AlertRow,
    ConfigurationVersionRow,
    FeatureManifestRow,
    FillRow,
    KillSwitchEventRow,
    MarketDataMetadataRow,
    ModelApprovalRow,
    ModelCheckpointRow,
    ModelRunRow,
    OrderRow,
    PerformanceMetricsRow,
    PortfolioSnapshotRow,
    PositionRow,
    RiskDecisionRow,
    new_uuid,
)
from rl_trader.storage.secrets import sanitize_config_for_storage


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


def _utc(ts: datetime | None = None) -> datetime:
    if ts is None:
        return datetime.now(UTC)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


class StorageRepository:
    """Transactional writers / readers over the trading schema."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # --- configuration -------------------------------------------------

    def upsert_configuration_version(
        self,
        version_key: str,
        config: dict[str, Any],
    ) -> uuid.UUID:
        """Store a sanitized configuration snapshot; return existing id if key exists."""
        clean = sanitize_config_for_storage(config)
        payload = _dumps(clean)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with session_scope(self.engine) as session:
            existing = session.scalar(
                select(ConfigurationVersionRow).where(
                    ConfigurationVersionRow.version_key == version_key
                )
            )
            if existing is not None:
                return existing.id
            row = ConfigurationVersionRow(
                id=new_uuid(),
                version_key=version_key,
                config_hash=digest,
                config_json=payload,
            )
            session.add(row)
            session.flush()
            return row.id

    # --- model lineage -------------------------------------------------

    def record_feature_manifest(
        self,
        *,
        name: str,
        version: str,
        feature_specs: list[dict[str, Any]],
        notes: str = "",
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            existing = session.scalar(
                select(FeatureManifestRow).where(
                    FeatureManifestRow.name == name,
                    FeatureManifestRow.version == version,
                )
            )
            if existing is not None:
                return existing.id
            row = FeatureManifestRow(
                id=new_uuid(),
                name=name,
                version=version,
                feature_specs_json=_dumps(feature_specs),
                notes=notes,
            )
            session.add(row)
            session.flush()
            return row.id

    def record_model_run(
        self,
        *,
        run_key: str,
        algorithm: str,
        seed: int = 0,
        feature_manifest_id: uuid.UUID | None = None,
        hyperparameters: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            existing = session.scalar(
                select(ModelRunRow).where(ModelRunRow.run_key == run_key)
            )
            if existing is not None:
                return existing.id
            row = ModelRunRow(
                id=new_uuid(),
                run_key=run_key,
                algorithm=algorithm,
                seed=seed,
                feature_manifest_id=feature_manifest_id,
                hyperparameters_json=_dumps(hyperparameters or {}),
                status=status,
            )
            session.add(row)
            session.flush()
            return row.id

    def record_checkpoint(
        self,
        *,
        model_id: str,
        algorithm: str,
        checkpoint_path: str,
        model_version: str = "1",
        package_version: str = "",
        model_run_id: uuid.UUID | None = None,
        feature_manifest_id: uuid.UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = ModelCheckpointRow(
                id=new_uuid(),
                model_run_id=model_run_id,
                model_id=model_id,
                algorithm=algorithm,
                checkpoint_path=checkpoint_path,
                package_version=package_version,
                model_version=model_version,
                feature_manifest_id=feature_manifest_id,
                metadata_json=_dumps(sanitize_config_for_storage(metadata or {})),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_approval(
        self,
        *,
        checkpoint_id: uuid.UUID,
        model_id: str,
        approved: bool,
        approved_by: str = "",
        manifest_path: str = "",
        notes: str = "",
        approved_at: datetime | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = ModelApprovalRow(
                id=new_uuid(),
                checkpoint_id=checkpoint_id,
                model_id=model_id,
                approved=approved,
                approved_at=_utc(approved_at) if approved else None,
                approved_by=approved_by,
                manifest_path=manifest_path,
                notes=notes,
            )
            session.add(row)
            session.flush()
            return row.id

    # --- decisions / orders --------------------------------------------

    def record_agent_decision(
        self,
        *,
        model_id: str,
        model_version: str,
        cycle_id: str,
        as_of: datetime,
        target_weights: list[float],
        model_checkpoint_id: uuid.UUID | None = None,
        observation_hash: str | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = AgentDecisionRow(
                id=new_uuid(),
                model_checkpoint_id=model_checkpoint_id,
                model_id=model_id,
                model_version=model_version,
                cycle_id=cycle_id,
                as_of=_utc(as_of),
                target_weights_json=_dumps(target_weights),
                observation_hash=observation_hash,
            )
            session.add(row)
            session.flush()
            return row.id

    def record_risk_decision(
        self,
        *,
        model_id: str,
        decision: str,
        reason: str,
        violated_rules: list[str],
        approved_weights: list[float] | None = None,
        agent_decision_id: uuid.UUID | None = None,
        configuration_version_id: uuid.UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = RiskDecisionRow(
                id=new_uuid(),
                agent_decision_id=agent_decision_id,
                configuration_version_id=configuration_version_id,
                model_id=model_id,
                decision=decision,
                reason=reason,
                violated_rules_json=_dumps(violated_rules),
                approved_weights_json=(
                    None if approved_weights is None else _dumps(approved_weights)
                ),
                details_json=_dumps(details or {}),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_order(
        self,
        *,
        broker_order_id: str,
        symbol: str,
        side: str,
        qty: float,
        status: str,
        order_type: str = "market",
        idempotency_key: str | None = None,
        limit_price: float | None = None,
        filled_qty: float = 0.0,
        avg_fill_price: float | None = None,
        reject_reason: str | None = None,
        dry_run: bool = False,
        model_checkpoint_id: uuid.UUID | None = None,
        agent_decision_id: uuid.UUID | None = None,
        risk_decision_id: uuid.UUID | None = None,
        configuration_version_id: uuid.UUID | None = None,
        created_at: datetime | None = None,
    ) -> uuid.UUID:
        """Insert an order (including rejected). Duplicate idempotency keys reuse the row."""
        with session_scope(self.engine) as session:
            if idempotency_key:
                existing = session.scalar(
                    select(OrderRow).where(OrderRow.idempotency_key == idempotency_key)
                )
                if existing is not None:
                    return existing.id
            row = OrderRow(
                id=new_uuid(),
                broker_order_id=broker_order_id,
                idempotency_key=idempotency_key,
                model_checkpoint_id=model_checkpoint_id,
                agent_decision_id=agent_decision_id,
                risk_decision_id=risk_decision_id,
                configuration_version_id=configuration_version_id,
                symbol=symbol.upper(),
                side=side,
                order_type=order_type,
                qty=float(qty),
                limit_price=limit_price,
                filled_qty=float(filled_qty),
                avg_fill_price=avg_fill_price,
                status=status,
                reject_reason=reject_reason,
                dry_run=dry_run,
                created_at=_utc(created_at),
                updated_at=_utc(created_at),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_fill(
        self,
        *,
        order_id: uuid.UUID,
        qty: float,
        price: float,
        filled_at: datetime,
        commission: float = 0.0,
        spread_cost: float = 0.0,
        slippage_cost: float = 0.0,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = FillRow(
                id=new_uuid(),
                order_id=order_id,
                qty=float(qty),
                price=float(price),
                filled_at=_utc(filled_at),
                commission=float(commission),
                spread_cost=float(spread_cost),
                slippage_cost=float(slippage_cost),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_portfolio_snapshot(
        self,
        *,
        as_of: datetime,
        cash: float,
        equity: float,
        buying_power: float,
        positions: list[dict[str, Any]],
        currency: str = "USD",
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            snap = PortfolioSnapshotRow(
                id=new_uuid(),
                as_of=_utc(as_of),
                cash=float(cash),
                equity=float(equity),
                buying_power=float(buying_power),
                currency=currency,
            )
            session.add(snap)
            session.flush()
            for pos in positions:
                session.add(
                    PositionRow(
                        id=new_uuid(),
                        snapshot_id=snap.id,
                        symbol=str(pos["symbol"]).upper(),
                        qty=float(pos["qty"]),
                        avg_price=float(pos.get("avg_price", 0.0)),
                        market_price=(
                            None
                            if pos.get("market_price") is None
                            else float(pos["market_price"])
                        ),
                        market_value=(
                            None
                            if pos.get("market_value") is None
                            else float(pos["market_value"])
                        ),
                        as_of=_utc(as_of),
                    )
                )
            return snap.id

    def record_performance_metrics(
        self,
        *,
        as_of: datetime,
        metrics: dict[str, Any],
        model_id: str | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = PerformanceMetricsRow(
                id=new_uuid(),
                as_of=_utc(as_of),
                cycles=int(metrics.get("cycles", 0)),
                approved=int(metrics.get("approved", 0)),
                resized=int(metrics.get("resized", 0)),
                rejected=int(metrics.get("rejected", 0)),
                orders_submitted=int(metrics.get("orders_submitted", 0)),
                last_equity=(
                    None
                    if metrics.get("last_equity") is None
                    else float(metrics["last_equity"])
                ),
                last_cash=(
                    None if metrics.get("last_cash") is None else float(metrics["last_cash"])
                ),
                peak_equity=(
                    None
                    if metrics.get("peak_equity") is None
                    else float(metrics["peak_equity"])
                ),
                model_id=model_id,
                metrics_json=_dumps(metrics),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_alert(
        self,
        *,
        level: str,
        message: str,
        context: dict[str, Any] | None = None,
        source: str = "paper_trading",
        created_at: datetime | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = AlertRow(
                id=new_uuid(),
                level=level,
                message=message,
                source=source,
                context_json=_dumps(sanitize_config_for_storage(context or {})),
                created_at=_utc(created_at),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_kill_switch_event(
        self,
        *,
        reason: str,
        active: bool = True,
        details: dict[str, Any] | None = None,
        cleared_at: datetime | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = KillSwitchEventRow(
                id=new_uuid(),
                reason=reason,
                active=active,
                details_json=_dumps(details or {}),
                cleared_at=None if cleared_at is None else _utc(cleared_at),
            )
            session.add(row)
            session.flush()
            return row.id

    def record_market_data_metadata(
        self,
        *,
        symbol: str,
        source: str,
        bar_count: int,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        schema_version: str = "1",
        checksum: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        with session_scope(self.engine) as session:
            row = MarketDataMetadataRow(
                id=new_uuid(),
                symbol=symbol.upper(),
                source=source,
                start_ts=None if start_ts is None else _utc(start_ts),
                end_ts=None if end_ts is None else _utc(end_ts),
                bar_count=bar_count,
                schema_version=schema_version,
                checksum=checksum,
                extra_json=_dumps(extra or {}),
            )
            session.add(row)
            session.flush()
            return row.id
