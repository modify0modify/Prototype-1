"""Decision / order audit recording stubs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.engine import Engine

from rl_trader.storage.db import AuditEventRow, session_scope


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """In-memory audit event before persistence."""

    event_type: str
    payload: dict[str, Any]
    created_at: datetime | None = None


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


class AuditRecorder:
    """Append-only audit writer backed by SQLAlchemy."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record(self, event_type: str, payload: dict[str, Any]) -> AuditEvent:
        """Persist an audit event and return the in-memory representation."""
        created = datetime.now(UTC)
        event = AuditEvent(event_type=event_type, payload=payload, created_at=created)
        row = AuditEventRow(
            created_at=created,
            event_type=event_type,
            payload_json=json.dumps(payload, default=_json_default),
        )
        with session_scope(self._engine) as session:
            session.add(row)
        return event
