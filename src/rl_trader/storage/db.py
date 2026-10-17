"""SQLAlchemy / SQLite persistence stubs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""


class AuditEventRow(Base):
    """Persisted decision / order audit event (skeleton table)."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    event_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)


def ensure_sqlite_parent(database_url: str) -> None:
    """Create parent directory for SQLite file URLs when needed."""
    prefix = "sqlite:///"
    if database_url.startswith(prefix) and not database_url.startswith("sqlite:///:memory:"):
        path = Path(database_url.removeprefix(prefix))
        if str(path) not in {":memory:", ""}:
            path.parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine and ensure schema exists."""
    ensure_sqlite_parent(database_url)
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Provide a transactional session scope."""
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
