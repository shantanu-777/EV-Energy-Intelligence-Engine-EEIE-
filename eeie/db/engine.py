"""SQLAlchemy engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from eeie.config import get_settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def create_engine_from_settings(database_url: str | None = None) -> Engine:
    """Build an `Engine` from settings (or an explicit URL for tests)."""
    url = database_url or get_settings().database_url
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def get_engine() -> Engine:
    """Return the process-wide engine, lazily constructed."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine_from_settings()
        _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def SessionLocal() -> Session:  # noqa: N802 - mirrors common SQLAlchemy idiom
    """Return a new session bound to the process-wide engine."""
    get_engine()
    assert _session_factory is not None
    return _session_factory()


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yields a session, closes on teardown."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and one-off jobs."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
