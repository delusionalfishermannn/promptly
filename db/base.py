"""SQLAlchemy declarative base + shared column helpers."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.clock import utcnow

# JSON on SQLite/MySQL, JSONB on Postgres (indexable, faster, still a superset).
JSONType: Any = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy declarative convention
        dict[str, Any]: JSONType,
        list[Any]: JSONType,
        datetime: DateTime(timezone=True),
    }


def pk(prefix: str) -> Any:
    """Prefixed UUID primary key generated client-side (no DB sequence needed)."""
    return mapped_column(
        String(48), primary_key=True, default=lambda: f"{prefix}_{uuid.uuid4().hex}"
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class EmbeddingType(TypeDecorator):
    """A float vector stored as JSON for portability.

    Postgres deployments add a ``pgvector`` column + HNSW index in the migration
    (see ``db/migrations/versions/0001_initial.py``); ``market_intel.similarity``
    transparently uses SQL ``<=>`` when pgvector is available and otherwise does an
    exact NumPy cosine scan, which is entirely adequate at MVP scale.
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: list[float] | None, dialect: Any) -> Any:
        if value is None:
            return None
        return [float(v) for v in value]

    def process_result_value(self, value: Any, dialect: Any) -> list[float] | None:
        if value is None:
            return None
        return [float(v) for v in value]
