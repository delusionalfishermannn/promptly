"""Time helpers — always timezone-aware UTC internally, ISO-8601 on the wire."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def days_ago(days: float) -> datetime:
    return utcnow() - timedelta(days=days)


def window_start(days: int) -> datetime:
    return utcnow() - timedelta(days=days)
