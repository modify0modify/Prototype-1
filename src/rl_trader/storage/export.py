"""CSV export helpers for storage tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine

from rl_trader.storage.db import session_scope
from rl_trader.storage.models import (
    REQUIRED_TABLES,
    row_to_dict,
    table_names,
)

# Map logical export names → ORM classes (lazy import inside).
_EXPORTABLE = (
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


def _orm_for_table(table: str) -> type[Any]:
    from rl_trader.storage import models as m

    mapping: dict[str, type[Any]] = {
        "market_data_metadata": m.MarketDataMetadataRow,
        "feature_manifests": m.FeatureManifestRow,
        "model_runs": m.ModelRunRow,
        "model_checkpoints": m.ModelCheckpointRow,
        "model_approvals": m.ModelApprovalRow,
        "agent_decisions": m.AgentDecisionRow,
        "risk_decisions": m.RiskDecisionRow,
        "orders": m.OrderRow,
        "fills": m.FillRow,
        "positions": m.PositionRow,
        "portfolio_snapshots": m.PortfolioSnapshotRow,
        "performance_metrics": m.PerformanceMetricsRow,
        "alerts": m.AlertRow,
        "kill_switch_events": m.KillSwitchEventRow,
        "configuration_versions": m.ConfigurationVersionRow,
    }
    if table not in mapping:
        raise KeyError(f"unknown export table: {table}")
    return mapping[table]


def export_table_csv(engine: Engine, table: str, path: Path) -> int:
    """Export one table to CSV; return row count."""
    cls = _orm_for_table(table)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with session_scope(engine) as session:
        rows = list(session.scalars(select(cls)).all())
        dicts = [row_to_dict(r) for r in rows]
    if not dicts:
        path.write_text("", encoding="utf-8")
        return 0
    fieldnames = list(dicts[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dicts)
    return len(dicts)


def export_all_csv(engine: Engine, directory: Path) -> dict[str, int]:
    """Export all required trading tables to ``directory/*.csv``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    present = table_names()
    counts: dict[str, int] = {}
    for name in _EXPORTABLE:
        if name not in present and name not in REQUIRED_TABLES:
            continue
        counts[name] = export_table_csv(engine, name, directory / f"{name}.csv")
    return counts
