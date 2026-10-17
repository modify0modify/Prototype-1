"""Heartbeat timestamp writer for the paper-trading service."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class HeartbeatWriter:
    """Persists last-alive timestamps under the paper state directory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def beat(
        self,
        *,
        now: datetime,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Write / overwrite the heartbeat document."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": now.isoformat(),
            "status": status,
            "details": dict(details or {}),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def read(self) -> dict[str, Any] | None:
        """Return the last heartbeat document, if any."""
        if not self.path.is_file():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
