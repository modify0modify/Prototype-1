"""Structured logging setup.

Uses the stdlib logging module with a consistent JSON-capable formatter hook.
Call :func:`configure_logging` once at process start.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure root logging for the process.

    Parameters
    ----------
    level:
        Log level name (DEBUG, INFO, WARNING, ERROR).
    json_logs:
        When True, emit one JSON object per line; otherwise human-readable text.
    """
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        )
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the ``rl_trading`` namespace."""
    return logging.getLogger(name)
