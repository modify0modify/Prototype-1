"""SQLAlchemy engine / session helpers for SQLite (and compatible URLs)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base shared by all storage models."""


def ensure_sqlite_parent(database_url: str) -> None:
    """Create parent directory for SQLite file URLs when needed."""
    prefix = "sqlite:///"
    if database_url.startswith(prefix) and not database_url.startswith("sqlite:///:memory:"):
        path = Path(database_url.removeprefix(prefix))
        if str(path) not in {":memory:", ""}:
            path.parent.mkdir(parents=True, exist_ok=True)


def _enable_sqlite_fks(engine: Engine) -> None:
    """Enforce foreign keys on SQLite connections."""

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: object, _connection_record: object) -> None:
        if engine.dialect.name != "sqlite":
            return
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_db_engine(
    database_url: str,
    *,
    echo: bool = False,
    create_schema: bool = False,
) -> Engine:
    """Create a SQLAlchemy engine.

    Prefer Alembic migrations (``make migrate``) for durable databases.
    Pass ``create_schema=True`` only for ephemeral tests that skip migrations.
    """
    ensure_sqlite_parent(database_url)
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        # Needed for :memory: shared across connections in tests when using StaticPool.
        connect_args["check_same_thread"] = False
    engine = create_engine(database_url, future=True, echo=echo, connect_args=connect_args)
    _enable_sqlite_fks(engine)
    if create_schema:
        # Import models so metadata is fully populated.
        import rl_trader.storage.models  # noqa: F401

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
