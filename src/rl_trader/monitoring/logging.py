"""Structured logging setup (stdout + optional rotating files)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
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


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = False,
    log_dir: Path | str | None = None,
    log_filename: str = "rl_trader.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure root logging for the process.

    Always attaches stdout. When ``log_dir`` is set, also writes a rotating
    file log (size-based rotation — suitable for container + host deploys).
    """
    root = logging.getLogger()
    root.handlers.clear()
    formatter: logging.Formatter
    if json_logs:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            directory / log_filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the ``rl_trader`` namespace."""
    return logging.getLogger(name)
