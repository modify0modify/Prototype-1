"""Programmatic Alembic migration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def alembic_config(database_url: str | None = None) -> Config:
    """Build Alembic config pointing at the repo ``alembic.ini``."""
    root = _repo_root()
    ini = root / "alembic.ini"
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(root / "alembic"))
    url = database_url or os.environ.get("DATABASE_URL") or "sqlite:///./reports/rl_trader.db"
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def upgrade_head(database_url: str | None = None) -> None:
    """Apply all migrations to head."""
    command.upgrade(alembic_config(database_url), "head")


def downgrade(revision: str = "-1", database_url: str | None = None) -> None:
    """Downgrade one or more revisions."""
    command.downgrade(alembic_config(database_url), revision)


def current(database_url: str | None = None) -> None:
    """Print current revision."""
    command.current(alembic_config(database_url))
