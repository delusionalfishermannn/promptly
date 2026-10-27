"""Connector registry — the single place that maps a source type to an implementation."""

from __future__ import annotations

from typing import Any

from connectors.base import Connector
from connectors.mock_api import MockOfficialAPIConnector
from connectors.official import (
    GoogleAdsConnector,
    RedditConnector,
    SearchConsoleConnector,
    StackExchangeConnector,
    SupportTicketConnector,
    YouTubeConnector,
)
from connectors.upload import CsvUploadConnector, JsonUploadConnector, StorefrontUploadConnector

_REGISTRY: dict[str, type[Connector]] = {
    CsvUploadConnector.source_type: CsvUploadConnector,
    JsonUploadConnector.source_type: JsonUploadConnector,
    StorefrontUploadConnector.source_type: StorefrontUploadConnector,
    SupportTicketConnector.source_type: SupportTicketConnector,
    MockOfficialAPIConnector.source_type: MockOfficialAPIConnector,
    RedditConnector.source_type: RedditConnector,
    StackExchangeConnector.source_type: StackExchangeConnector,
    YouTubeConnector.source_type: YouTubeConnector,
    SearchConsoleConnector.source_type: SearchConsoleConnector,
    GoogleAdsConnector.source_type: GoogleAdsConnector,
}


def register(connector_cls: type[Connector]) -> None:
    _REGISTRY[connector_cls.source_type] = connector_cls


def get_connector(source_type: str, config: dict[str, Any] | None = None) -> Connector:
    if source_type not in _REGISTRY:
        raise KeyError(f"no connector for source_type '{source_type}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[source_type](config or {})


def list_connectors() -> list[dict[str, Any]]:
    out = []
    for source_type, cls in sorted(_REGISTRY.items()):
        try:
            instance = cls({})
            out.append(instance.readiness())
        except Exception as exc:
            out.append({"source_type": source_type, "ready": False, "error": str(exc)})
    return out
