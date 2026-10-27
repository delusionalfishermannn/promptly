"""Engine/session management.

The MVP runs on SQLite (zero infrastructure); Postgres is a URL change away because
the schema avoids dialect-specific features (``JSON`` → ``JSONB`` via ``with_variant``).
Alembic owns production migrations; ``create_all`` is used for local bootstrap/tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_settings
from db.base import Base

_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def _make_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    url = url or settings.database_url
    kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        engine = create_engine(url, **kwargs)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:  # pragma: no cover
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

        return engine
    kwargs["pool_size"] = 10
    kwargs["max_overflow"] = 20
    return create_engine(url, **kwargs)


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        get_settings().ensure_dirs()
        _ENGINE = _make_engine()
    return _ENGINE


def get_session_factory() -> sessionmaker[Session]:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _SESSION_FACTORY


def get_db() -> Iterator[Session]:
    """FastAPI dependency — commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for workers/CLI/demo code."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(create_all: bool = True) -> None:
    if create_all:
        Base.metadata.create_all(bind=get_engine())


def drop_db() -> None:
    Base.metadata.drop_all(bind=get_engine())


def reset_engine(url: str | None = None) -> None:
    """Test helper: dispose the pool and rebuild against a new URL."""
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        _ENGINE.dispose()
    _ENGINE = None
    _SESSION_FACTORY = None
    if url:
        get_settings().database_url = url
