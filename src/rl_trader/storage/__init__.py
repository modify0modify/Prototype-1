"""Storage helpers (SQLAlchemy / SQLite + Alembic migrations).

Secrets (API keys, tokens, passwords) must never be persisted — see
:mod:`rl_trader.storage.secrets`.
"""

from rl_trader.storage.db import Base, create_db_engine, session_scope
from rl_trader.storage.export import export_all_csv, export_table_csv
from rl_trader.storage.migrate import upgrade_head
from rl_trader.storage.models import (
    REQUIRED_TABLES,
    AgentDecisionRow,
    AlertRow,
    AuditEventRow,
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
)
from rl_trader.storage.reconcile import ReconcileFinding, run_all_reconciliations
from rl_trader.storage.repository import StorageRepository
from rl_trader.storage.secrets import assert_no_secret_columns, sanitize_config_for_storage

__all__ = [
    "REQUIRED_TABLES",
    "AgentDecisionRow",
    "AlertRow",
    "AuditEventRow",
    "Base",
    "ConfigurationVersionRow",
    "FeatureManifestRow",
    "FillRow",
    "KillSwitchEventRow",
    "MarketDataMetadataRow",
    "ModelApprovalRow",
    "ModelCheckpointRow",
    "ModelRunRow",
    "OrderRow",
    "PerformanceMetricsRow",
    "PortfolioSnapshotRow",
    "PositionRow",
    "ReconcileFinding",
    "RiskDecisionRow",
    "StorageRepository",
    "assert_no_secret_columns",
    "create_db_engine",
    "export_all_csv",
    "export_table_csv",
    "run_all_reconciliations",
    "sanitize_config_for_storage",
    "session_scope",
    "upgrade_head",
]
