"""SQLAlchemy ORM models for the trading system storage layer.

All primary keys are immutable UUIDs. Timestamps are stored as timezone-aware
UTC. **No secret columns** (API keys, tokens, passwords) are permitted — see
:mod:`rl_trader.storage.secrets`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rl_trader.storage.db import Base
from rl_trader.storage.types import UtcDateTime


def _utc_now() -> datetime:
    return datetime.now(UTC)


def new_uuid() -> uuid.UUID:
    """Allocate a new immutable primary key."""
    return uuid.uuid4()


class MarketDataMetadataRow(Base):
    """Metadata describing a stored / cached market-data panel."""

    __tablename__ = "market_data_metadata"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(64), default="unknown")
    start_ts: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    end_ts: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    bar_count: Mapped[int] = mapped_column(Integer, default=0)
    schema_version: Mapped[str] = mapped_column(String(64), default="1")
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    extra_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)


class FeatureManifestRow(Base):
    """Versioned feature definitions used by models."""

    __tablename__ = "feature_manifests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64), default="1")
    feature_specs_json: Mapped[str] = mapped_column(Text)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_feature_manifest_name_ver"),)


class ModelRunRow(Base):
    """One training / evaluation run."""

    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    run_key: Mapped[str] = mapped_column(String(128), unique=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer, default=0)
    training_start: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    training_end: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    hyperparameters_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(32), default="completed")
    feature_manifest_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("feature_manifests.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    feature_manifest: Mapped[FeatureManifestRow | None] = relationship()
    checkpoints: Mapped[list[ModelCheckpointRow]] = relationship(back_populates="model_run")


class ModelCheckpointRow(Base):
    """Persisted checkpoint metadata (path only — never weight blobs / secrets)."""

    __tablename__ = "model_checkpoints"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    model_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_runs.id"), nullable=True
    )
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    algorithm: Mapped[str] = mapped_column(String(64))
    checkpoint_path: Mapped[str] = mapped_column(String(512))
    package_version: Mapped[str] = mapped_column(String(64), default="")
    model_version: Mapped[str] = mapped_column(String(64), default="1")
    feature_manifest_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("feature_manifests.id"), nullable=True
    )
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    model_run: Mapped[ModelRunRow | None] = relationship(back_populates="checkpoints")
    feature_manifest: Mapped[FeatureManifestRow | None] = relationship()
    approvals: Mapped[list[ModelApprovalRow]] = relationship(back_populates="checkpoint")


class ModelApprovalRow(Base):
    """Approval record for a checkpoint used by paper / live gates."""

    __tablename__ = "model_approvals"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    checkpoint_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_checkpoints.id")
    )
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    approved: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)
    approved_by: Mapped[str] = mapped_column(String(128), default="")
    manifest_path: Mapped[str] = mapped_column(String(512), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    checkpoint: Mapped[ModelCheckpointRow] = relationship(back_populates="approvals")


class ConfigurationVersionRow(Base):
    """Immutable configuration snapshot (secrets stripped before insert)."""

    __tablename__ = "configuration_versions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    version_key: Mapped[str] = mapped_column(String(128), unique=True)
    config_hash: Mapped[str] = mapped_column(String(128))
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)


class AgentDecisionRow(Base):
    """Agent-proposed target weights for one cycle."""

    __tablename__ = "agent_decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    model_checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_checkpoints.id"), nullable=True
    )
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    model_version: Mapped[str] = mapped_column(String(64), default="")
    cycle_id: Mapped[str] = mapped_column(String(256), index=True)
    as_of: Mapped[datetime] = mapped_column(UtcDateTime())
    target_weights_json: Mapped[str] = mapped_column(Text)
    observation_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    checkpoint: Mapped[ModelCheckpointRow | None] = relationship()
    risk_decisions: Mapped[list[RiskDecisionRow]] = relationship(back_populates="agent_decision")


class RiskDecisionRow(Base):
    """Risk engine approve / resize / reject outcome."""

    __tablename__ = "risk_decisions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    agent_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("agent_decisions.id"), nullable=True
    )
    configuration_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("configuration_versions.id"), nullable=True
    )
    model_id: Mapped[str] = mapped_column(String(128), index=True)
    decision: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(Text, default="")
    violated_rules_json: Mapped[str] = mapped_column(Text, default="[]")
    approved_weights_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    agent_decision: Mapped[AgentDecisionRow | None] = relationship(back_populates="risk_decisions")
    configuration_version: Mapped[ConfigurationVersionRow | None] = relationship()
    orders: Mapped[list[OrderRow]] = relationship(back_populates="risk_decision")


class OrderRow(Base):
    """Broker order including rejected ones; unique idempotency key."""

    __tablename__ = "orders"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_orders_idempotency_key"),
        UniqueConstraint("broker_order_id", name="uq_orders_broker_order_id"),
        Index("ix_orders_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    broker_order_id: Mapped[str] = mapped_column(String(128))
    idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True)

    model_checkpoint_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("model_checkpoints.id"), nullable=True
    )
    agent_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("agent_decisions.id"), nullable=True
    )
    risk_decision_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("risk_decisions.id"), nullable=True
    )
    configuration_version_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("configuration_versions.id"), nullable=True
    )

    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    order_type: Mapped[str] = mapped_column(String(16), default="market")
    qty: Mapped[float] = mapped_column(Float)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    model_checkpoint: Mapped[ModelCheckpointRow | None] = relationship()
    agent_decision: Mapped[AgentDecisionRow | None] = relationship()
    risk_decision: Mapped[RiskDecisionRow | None] = relationship(back_populates="orders")
    configuration_version: Mapped[ConfigurationVersionRow | None] = relationship()
    fills: Mapped[list[FillRow]] = relationship(back_populates="order")


class FillRow(Base):
    """Fill event with transaction costs stored separately."""

    __tablename__ = "fills"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    order_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), ForeignKey("orders.id"))
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    filled_at: Mapped[datetime] = mapped_column(UtcDateTime())
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    spread_cost: Mapped[float] = mapped_column(Float, default=0.0)
    slippage_cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    order: Mapped[OrderRow] = relationship(back_populates="fills")


class PortfolioSnapshotRow(Base):
    """Point-in-time portfolio cash / equity snapshot."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    as_of: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    cash: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    buying_power: Mapped[float] = mapped_column(Float, default=0.0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    positions: Mapped[list[PositionRow]] = relationship(back_populates="snapshot")


class PositionRow(Base):
    """Position lot tied to a portfolio snapshot (and optional live mark)."""

    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "symbol", name="uq_positions_snapshot_symbol"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("portfolio_snapshots.id"), nullable=True
    )
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    qty: Mapped[float] = mapped_column(Float)
    avg_price: Mapped[float] = mapped_column(Float, default=0.0)
    market_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    as_of: Mapped[datetime] = mapped_column(UtcDateTime())
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)

    snapshot: Mapped[PortfolioSnapshotRow | None] = relationship(back_populates="positions")


class PerformanceMetricsRow(Base):
    """Persisted performance metrics snapshots."""

    __tablename__ = "performance_metrics"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    as_of: Mapped[datetime] = mapped_column(UtcDateTime(), index=True)
    cycles: Mapped[int] = mapped_column(Integer, default=0)
    approved: Mapped[int] = mapped_column(Integer, default=0)
    resized: Mapped[int] = mapped_column(Integer, default=0)
    rejected: Mapped[int] = mapped_column(Integer, default=0)
    orders_submitted: Mapped[int] = mapped_column(Integer, default=0)
    last_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_cash: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)


class AlertRow(Base):
    """Operational / risk alert events."""

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    level: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64), default="paper_trading")
    context_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)


class KillSwitchEventRow(Base):
    """Kill-switch trip / clear audit trail."""

    __tablename__ = "kill_switch_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=new_uuid)
    reason: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)
    cleared_at: Mapped[datetime | None] = mapped_column(UtcDateTime(), nullable=True)


class AuditEventRow(Base):
    """Legacy append-only audit events (kept for AuditRecorder compatibility)."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime(), default=_utc_now)
    event_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)


REQUIRED_TABLES: tuple[str, ...] = (
    "market_data_metadata",
    "feature_manifests",
    "model_runs",
    "model_checkpoints",
    "model_approvals",
    "agent_decisions",
    "risk_decisions",
    "orders",
    "fills",
    "positions",
    "portfolio_snapshots",
    "performance_metrics",
    "alerts",
    "kill_switch_events",
    "configuration_versions",
)


def table_names() -> set[str]:
    """Return registered ORM table names."""
    return set(Base.metadata.tables.keys())


def row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a mapped instance to a plain dict (CSV / JSON helpers)."""
    mapper = getattr(type(row), "__mapper__", None)
    if mapper is None:
        raise TypeError(f"not a mapped instance: {type(row)!r}")
    out: dict[str, Any] = {}
    for col in mapper.columns:
        val = getattr(row, col.key)
        if isinstance(val, datetime):
            out[col.key] = val.isoformat()
        elif isinstance(val, uuid.UUID):
            out[col.key] = str(val)
        else:
            out[col.key] = val
    return out
