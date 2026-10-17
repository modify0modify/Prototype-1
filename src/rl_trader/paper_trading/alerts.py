"""Lightweight alert sink for paper-trading failures and risk events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rl_trader.monitoring import get_logger


@dataclass(frozen=True, slots=True)
class Alert:
    """Single alert event."""

    level: str
    message: str
    timestamp: datetime
    context: dict[str, Any]


class AlertSink:
    """Log + append JSONL alerts (no external webhook required)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.logger = get_logger("rl_trader.paper_trading.alerts")
        self.events: list[Alert] = []

    def emit(
        self,
        level: str,
        message: str,
        *,
        now: datetime,
        context: dict[str, Any] | None = None,
    ) -> Alert:
        """Record an alert to memory, log, and disk."""
        alert = Alert(
            level=level,
            message=message,
            timestamp=now,
            context=dict(context or {}),
        )
        self.events.append(alert)
        log_fn = self.logger.error if level.lower() in {"error", "critical"} else self.logger.warning
        log_fn("%s | %s", message, alert.context)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "level": alert.level,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "context": alert.context,
                    }
                )
                + "\n"
            )
        return alert
