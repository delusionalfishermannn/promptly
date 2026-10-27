"""Manual upload connectors (CSV + JSON) — the day-1 data path with zero integrations.

Design goals
------------
* **Lenient** on column names (``question``, ``query``, ``ticket_subject``, ...),
  strict on provenance and size limits.
* **Lossless rejects** — every skipped row is reported with a reason so the operator
  can fix the export rather than silently losing demand signal.
* **PII aware** — flagged emails/phones are redacted by default; the operator can
  opt out only for sources they have declared as containing personal information
  (AU Privacy Act APP 11 minimisation, recorded on ``data_sources``).
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime
from typing import Any, ClassVar

from connectors.base import Connector, ConnectorResult, SignalRecord
from core.clock import parse_iso, utcnow
from core.config import get_settings
from core.constants import ConnectorKind, PermissionBasis, SignalKind

TEXT_COLUMNS = (
    "text",
    "question",
    "query",
    "title",
    "subject",
    "body",
    "content",
    "message",
    "keyword",
    "term",
    "phrase",
    "comment",
    "review",
    "prompt_request",
    "ticket_subject",
    "search_query",
    "feedback",
)
DATE_COLUMNS = ("occurred_at", "created_at", "date", "timestamp", "ts", "time", "submitted_at")
VOLUME_COLUMNS = ("volume", "count", "search_volume", "impressions", "frequency", "n", "qty")
KIND_COLUMNS = ("kind", "type", "category", "event_type")
SEGMENT_COLUMNS = ("segment", "buyer_segment", "audience", "persona", "customer_type")
URL_COLUMNS = ("url", "link", "permalink", "source_url")
ID_COLUMNS = ("id", "external_id", "ticket_id", "keyword_id")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?61|0)[\s\-]?(?:\d[\s\-]?){8,10}(?!\d)")
CARD_RE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")

KIND_ALIASES: dict[str, str] = {
    "question": str(SignalKind.QUESTION),
    "q": str(SignalKind.QUESTION),
    "faq": str(SignalKind.QUESTION),
    "ticket": str(SignalKind.SUPPORT_TICKET),
    "support": str(SignalKind.SUPPORT_TICKET),
    "support_ticket": str(SignalKind.SUPPORT_TICKET),
    "keyword": str(SignalKind.AD_KEYWORD),
    "ad_keyword": str(SignalKind.AD_KEYWORD),
    "comment": str(SignalKind.COMMENT),
    "review": str(SignalKind.REVIEW),
    "search": str(SignalKind.SEARCH_QUERY),
    "search_query": str(SignalKind.SEARCH_QUERY),
    "view": str(SignalKind.STORE_EVENT),
    "click": str(SignalKind.STORE_EVENT),
    "purchase": str(SignalKind.STORE_EVENT),
}


def redact_pii(text: str) -> tuple[str, bool]:
    """Minimise personal information by default (APP 11 / data minimisation)."""
    found = False
    for regex, replacement in (
        (EMAIL_RE, "[email]"),
        (PHONE_RE, "[phone]"),
        (CARD_RE, "[card]"),
    ):
        new_text, n = regex.subn(replacement, text)
        if n:
            found = True
            text = new_text
    return text, found


def _pick(row: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for candidate in candidates:
        if candidate in lowered and str(lowered[candidate]).strip():
            return lowered[candidate]
    return None


def _parse_date(value: Any) -> datetime:
    if value in (None, ""):
        return utcnow()
    text = str(value).strip()
    if text.isdigit():
        number = int(text)
        if len(text) == 13:  # epoch millis
            return datetime.fromtimestamp(number / 1000, tz=utcnow().tzinfo)
        if len(text) == 10:  # epoch seconds
            return datetime.fromtimestamp(number, tz=utcnow().tzinfo)
    for parser in (parse_iso, lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))):
        try:
            parsed = parser(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=utcnow().tzinfo)
        except (ValueError, TypeError):
            continue
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=utcnow().tzinfo)
        except ValueError:
            continue
    return utcnow()


class _UploadConnector(Connector):
    connector_kind: ClassVar[str] = str(ConnectorKind.UPLOAD)
    allowed_bases: ClassVar[set[str]] = {
        str(PermissionBasis.OPERATOR_PROVIDED),
        str(PermissionBasis.FIRST_PARTY_ANALYTICS),
        str(PermissionBasis.PERMISSIVE_LICENSE),
        str(PermissionBasis.SYNTHETIC),
        str(PermissionBasis.PUBLIC_DOMAIN),
    }
    tos_notes: ClassVar[str] = (
        "You must have the right to use the uploaded export. Do not upload data obtained "
        "by scraping or in breach of another platform's terms."
    )

    # Rows are supplied by the API/ingestion service; ``run`` is for re-processing a
    # previously stored batch, so uploads are always explicit operator actions.
    def __init__(
        self, config: dict[str, Any] | None = None, rows: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(config)
        self.rows = rows or []
        self.redact = bool((config or {}).get("redact_pii", True))

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        return self.parse_rows(self.rows[:limit])

    # ------------------------------------------------------------------ parsing
    def parse_rows(self, rows: list[dict[str, Any]]) -> ConnectorResult:
        settings = get_settings()
        result = ConnectorResult()
        if len(rows) > settings.max_upload_rows:
            result.error = (
                f"too many rows ({len(rows)} > {settings.max_upload_rows}); split the export"
            )
            return result
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                result.add_reject(index, "row is not an object")
                continue
            text_value = _pick(row, TEXT_COLUMNS)
            if not text_value:
                result.add_reject(index, "no text column found", str(row)[:120])
                continue
            text = " ".join(str(text_value).split())
            if len(text) < 8:
                result.add_reject(index, "text too short", text)
                continue
            if len(text) > 20_000:
                result.add_reject(index, "text too long", text[:120])
                continue
            if self.redact:
                text, had_pii = redact_pii(text)
                if had_pii:
                    result.stats["pii_redacted"] = result.stats.get("pii_redacted", 0) + 1
            if text in seen:
                result.stats["duplicates_in_batch"] = result.stats.get("duplicates_in_batch", 0) + 1
                continue
            seen.add(text)

            kind_raw = str(_pick(row, KIND_COLUMNS) or "").strip().lower()
            kind = KIND_ALIASES.get(kind_raw, str(SignalKind.OTHER))
            volume_raw = _pick(row, VOLUME_COLUMNS)
            try:
                volume = float(volume_raw) if volume_raw not in (None, "") else 1.0
            except (TypeError, ValueError):
                volume = 1.0
            url = _pick(row, URL_COLUMNS)
            record = SignalRecord(
                text=text,
                external_id=str(_pick(row, ID_COLUMNS) or "") or None,
                kind=kind,
                occurred_at=_parse_date(_pick(row, DATE_COLUMNS)),
                volume=max(volume, 0.0),
                url=str(url) if url else None,
                segment_hint=str(_pick(row, SEGMENT_COLUMNS) or "") or None,
                metadata={"upload_row": index},
                raw={
                    k: v
                    for k, v in row.items()
                    if k in (TEXT_COLUMNS + DATE_COLUMNS + VOLUME_COLUMNS)
                },
            )
            result.records.append(record)
        result.stats.update(
            {
                "rows_in": len(rows),
                "rows_out": len(result.records),
                "rows_rejected": len(result.rejects),
            }
        )
        return result


class CsvUploadConnector(_UploadConnector):
    source_type: ClassVar[str] = "manual_csv"
    display_name: ClassVar[str] = "CSV upload"

    def parse_bytes(self, data: bytes, *, encoding: str = "utf-8-sig") -> ConnectorResult:
        text = data.decode(encoding, errors="replace")
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        return self.parse_rows([dict(row) for row in reader])


class JsonUploadConnector(_UploadConnector):
    source_type: ClassVar[str] = "manual_json"
    display_name: ClassVar[str] = "JSON upload"

    def parse_bytes(self, data: bytes, *, encoding: str = "utf-8") -> ConnectorResult:
        text = data.decode(encoding, errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return ConnectorResult(error=f"invalid JSON: {exc.msg} at line {exc.lineno}")
        if isinstance(payload, dict):
            for key in ("rows", "data", "items", "signals", "records", "results"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            return ConnectorResult(error="JSON must be an array of objects (or {rows: [...]})")
        return self.parse_rows([row for row in payload if isinstance(row, dict)])


class StorefrontUploadConnector(_UploadConnector):
    """First-party storefront analytics export (views/clicks/conversions CSV)."""

    source_type: ClassVar[str] = "storefront"
    display_name: ClassVar[str] = "Storefront analytics export"
    allowed_bases: ClassVar[set[str]] = {
        str(PermissionBasis.FIRST_PARTY_ANALYTICS),
        str(PermissionBasis.OPERATOR_PROVIDED),
    }
