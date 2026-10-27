"""Mock official-API connector with the *identical* interface to a real one.

Purpose: the demo and the test suite must exercise the full ingestion path (pagination,
cursors, rate limits, quota accounting, partial failures) with no keys and no network.
Swap ``MOCK`` for ``REDDIT_API`` in the source config and nothing else changes.

Fixtures live in ``demo/data/mock_api_fixture.json`` and are synthetic (no third-party
content), so they carry ``PermissionBasis.SYNTHETIC``.
"""

from __future__ import annotations

import contextlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar

from connectors.base import Connector, ConnectorResult, RateLimiter, SignalRecord
from core.clock import parse_iso, utcnow
from core.constants import ConnectorKind, PermissionBasis, SignalKind
from core.errors import DependencyUnavailable

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "demo" / "data" / "mock_api_fixture.json"


class MockOfficialAPIConnector(Connector):
    """Paginated, cursor-based, rate-limited fake of an official content API."""

    source_type: ClassVar[str] = "mock"
    connector_kind: ClassVar[str] = str(ConnectorKind.MOCK)
    display_name: ClassVar[str] = "Mock official API (synthetic fixture)"
    allowed_bases: ClassVar[set[str]] = {str(PermissionBasis.SYNTHETIC)}
    requires_secret_env: ClassVar[tuple[str, ...]] = ()
    tos_notes: ClassVar[str] = (
        "Synthetic fixture data. Mirrors the interface of an official API so the real "
        "connector can be dropped in without changing orchestration."
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.fixture_path = Path(self.config.get("fixture_path") or DEFAULT_FIXTURE)
        self.page_size = int(self.config.get("page_size", 25))
        self.limiter = RateLimiter(per_second=float(self.config.get("rate_per_second", 5.0)))
        self.fail_after_pages = int(self.config.get("fail_after_pages", 0))

    def validate_config(self) -> list[str]:
        issues: list[str] = []
        if not self.fixture_path.exists():
            issues.append(f"fixture not found: {self.fixture_path}")
        if self.page_size < 1 or self.page_size > 500:
            issues.append("page_size must be between 1 and 500")
        return issues

    def _load(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DependencyUnavailable(f"cannot read mock fixture: {exc}") from exc
        items = payload.get("items") if isinstance(payload, dict) else payload
        return [item for item in (items or []) if isinstance(item, dict)]

    def run(
        self,
        *,
        since: Any = None,
        limit: int = 500,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        result = ConnectorResult()
        cursor = cursor or {}
        offset = int(cursor.get("offset", 0))
        items = self._load()
        # Simulate recency: fixtures may carry relative "days_ago" values.
        now = utcnow()
        pages = 0
        while offset < len(items) and len(result.records) < limit:
            if self.fail_after_pages and pages >= self.fail_after_pages:
                result.partial = True
                result.stats["stopped_reason"] = "simulated_upstream_error"
                break
            self.limiter.acquire()
            page = items[offset : offset + self.page_size]
            pages += 1
            for item in page:
                record = self._to_record(item, now)
                if record is None:
                    result.add_reject(offset, "missing text")
                elif since is not None and record.occurred_at < since:
                    result.stats["skipped_old"] = result.stats.get("skipped_old", 0) + 1
                else:
                    result.records.append(record)
                offset += 1
                if len(result.records) >= limit:
                    break
        result.cursor = {"offset": offset, "exhausted": offset >= len(items)}
        result.stats.update(
            {
                "pages": pages,
                "items_seen": offset,
                "rows_out": len(result.records),
                "rows_rejected": len(result.rejects),
            }
        )
        return result

    def _to_record(self, item: dict[str, Any], now: Any) -> SignalRecord | None:
        text = str(item.get("text") or item.get("title") or item.get("selftext") or "").strip()
        if not text:
            return None
        occurred = now - timedelta(days=float(item.get("days_ago", 0)))
        if item.get("occurred_at"):
            with contextlib.suppress(ValueError):
                occurred = parse_iso(str(item["occurred_at"]))
        kind = str(item.get("kind") or SignalKind.QUESTION)
        return SignalRecord(
            text=" ".join(text.split()),
            external_id=str(item.get("id") or "") or None,
            kind=kind,
            occurred_at=occurred,
            volume=float(item.get("volume", item.get("score", 1)) or 1.0),
            url=item.get("url"),
            segment_hint=item.get("segment"),
            metadata={
                "mock": True,
                "subreddit": item.get("subreddit"),
                "tags": item.get("tags", []),
            },
            raw=item,
        )
