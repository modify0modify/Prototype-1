"""Durable decision / order ledger for the paper-trading service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


@dataclass(slots=True)
class CycleDecisionRecord:
    """One prospective paper cycle decision (always persisted)."""

    cycle_id: str
    model_id: str
    model_version: str
    evaluation_period_id: str
    as_of: str
    dry_run: bool
    risk_decision: str
    risk_reason: str
    violated_rules: list[str] = field(default_factory=list)
    target_weights: list[float] = field(default_factory=list)
    approved_weights: list[float] | None = None
    order_ids: list[str] = field(default_factory=list)
    rejected_orders: list[dict[str, Any]] = field(default_factory=list)
    reconcile: dict[str, Any] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return asdict(self)


class DecisionStore:
    """Append-only JSONL decision store with cycle idempotency lookup."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._seen: set[str] | None = None

    def _ensure_loaded(self) -> set[str]:
        if self._seen is not None:
            return self._seen
        seen: set[str] = set()
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(doc, dict) and doc.get("cycle_id"):
                    seen.add(str(doc["cycle_id"]))
        self._seen = seen
        return seen

    def has_cycle(self, cycle_id: str) -> bool:
        """True if ``cycle_id`` was already recorded (restart safety)."""
        return cycle_id in self._ensure_loaded()

    def append(self, record: CycleDecisionRecord) -> None:
        """Persist one decision; updates the in-memory idempotency set."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), default=_json_default) + "\n")
        self._ensure_loaded().add(record.cycle_id)

    def load_all(self) -> list[dict[str, Any]]:
        """Return all stored decision documents."""
        if not self.path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if isinstance(doc, dict):
                out.append(doc)
        return out
