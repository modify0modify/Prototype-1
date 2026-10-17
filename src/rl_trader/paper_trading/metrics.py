"""Paper-trading performance metrics persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PerformanceMetrics:
    """Rolling paper-trading metrics snapshot."""

    cycles: int = 0
    approved: int = 0
    resized: int = 0
    rejected: int = 0
    dry_runs: int = 0
    orders_submitted: int = 0
    last_equity: float | None = None
    last_cash: float | None = None
    last_model_id: str | None = None
    last_updated: str | None = None
    peak_equity: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceMetrics:
        """Restore from JSON."""
        return cls(
            cycles=int(data.get("cycles", 0)),
            approved=int(data.get("approved", 0)),
            resized=int(data.get("resized", 0)),
            rejected=int(data.get("rejected", 0)),
            dry_runs=int(data.get("dry_runs", 0)),
            orders_submitted=int(data.get("orders_submitted", 0)),
            last_equity=(
                None if data.get("last_equity") is None else float(data["last_equity"])
            ),
            last_cash=None if data.get("last_cash") is None else float(data["last_cash"]),
            last_model_id=data.get("last_model_id"),
            last_updated=data.get("last_updated"),
            peak_equity=(
                None if data.get("peak_equity") is None else float(data["peak_equity"])
            ),
            extras=dict(data.get("extras") or {}),
        )


class MetricsStore:
    """Load/save performance metrics JSON."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.metrics = self._load()

    def _load(self) -> PerformanceMetrics:
        if not self.path.is_file():
            return PerformanceMetrics()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return PerformanceMetrics()
        return PerformanceMetrics.from_dict(data)

    def save(self) -> None:
        """Persist current metrics."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.metrics.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def update_from_cycle(
        self,
        *,
        now: datetime,
        risk_decision: str,
        dry_run: bool,
        orders_submitted: int,
        equity: float | None,
        cash: float | None,
        model_id: str,
    ) -> PerformanceMetrics:
        """Apply one cycle outcome and persist."""
        m = self.metrics
        m.cycles += 1
        key = risk_decision.lower()
        if key == "approved":
            m.approved += 1
        elif key == "resized":
            m.resized += 1
        elif key == "rejected":
            m.rejected += 1
        if dry_run:
            m.dry_runs += 1
        m.orders_submitted += int(orders_submitted)
        m.last_model_id = model_id
        m.last_updated = now.isoformat()
        if equity is not None:
            m.last_equity = float(equity)
            if m.peak_equity is None or equity > m.peak_equity:
                m.peak_equity = float(equity)
        if cash is not None:
            m.last_cash = float(cash)
        self.save()
        return m
