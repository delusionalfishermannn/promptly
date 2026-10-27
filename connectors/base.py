"""Connector interface + provenance contract.

Every data path into PromptFoundry goes through a Connector, and every connector
must declare how the data may legally be held. ``IngestionService`` refuses to run a
source whose ``permission_basis`` is not in the connector's ``allowed_bases``.

Hard rules enforced here (not just documented):
* No HTML scraping, no headless browsers, no paywall/login/CAPTCHA circumvention.
  Official HTTP APIs, user exports, first-party analytics, permissive corpora and
  synthetic data only.
* Secrets never live in the database: ``config`` may reference environment variable
  *names* (``secret_env``), which are resolved at call time and never persisted.
* Reads are bounded (``limit``), time-boxed, and rate-limited; partial failure returns
  a ``PARTIAL`` result with per-row rejects rather than losing a whole batch.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from core.clock import utcnow
from core.constants import ConnectorKind, PermissionBasis, SignalKind


@dataclass
class SignalRecord:
    """One normalised-enough signal emitted by a connector (pre-processing)."""

    text: str
    external_id: str | None = None
    kind: str = str(SignalKind.OTHER)
    occurred_at: datetime = field(default_factory=utcnow)
    volume: float = 1.0
    url: str | None = None
    segment_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorResult:
    records: list[SignalRecord] = field(default_factory=list)
    rejects: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    cursor: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    partial: bool = False

    @property
    def ok(self) -> bool:
        return not self.error

    def add_reject(self, index: int | str, reason: str, excerpt: str = "") -> None:
        self.rejects.append({"row": index, "reason": reason, "excerpt": excerpt[:200]})


class Connector(ABC):
    """Base class for all connectors."""

    source_type: ClassVar[str]
    connector_kind: ClassVar[str] = str(ConnectorKind.OFFICIAL_API)
    display_name: ClassVar[str] = ""
    allowed_bases: ClassVar[set[str]] = {str(PermissionBasis.OFFICIAL_API_TOS)}
    requires_secret_env: ClassVar[tuple[str, ...]] = ()
    tos_notes: ClassVar[str] = ""
    docs_url: ClassVar[str] = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    # ---------------------------------------------------------------- contract
    @abstractmethod
    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult: ...

    def validate_config(self) -> list[str]:
        """Return human-readable config problems (empty list == valid)."""
        return []

    def is_ready(self) -> bool:
        """True when credentials/config are present. Never raises."""
        import os

        missing = [env for env in self.requires_secret_env if not os.environ.get(env)]
        return not missing and not self.validate_config()

    def readiness(self) -> dict[str, Any]:
        import os

        missing = [env for env in self.requires_secret_env if not os.environ.get(env)]
        return {
            "source_type": self.source_type,
            "display_name": self.display_name or self.source_type,
            "ready": not missing and not self.validate_config(),
            "missing_secret_env": missing,
            "config_issues": self.validate_config(),
            "allowed_permission_bases": sorted(self.allowed_bases),
            "tos_notes": self.tos_notes,
            "docs_url": self.docs_url,
        }


class RateLimiter:
    """Simple token-bucket to respect provider quotas (default 1 req/sec)."""

    def __init__(self, per_second: float = 1.0, burst: int = 3) -> None:
        self.rate = max(per_second, 0.01)
        self.capacity = max(burst, 1)
        self.tokens = float(self.capacity)
        self.updated = time.monotonic()

    def acquire(self, tokens: float = 1.0) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            time.sleep(min(1.0, (tokens - self.tokens) / self.rate))


def check_permission_basis(connector: Connector, basis: str) -> None:
    if basis not in connector.allowed_bases:
        allowed = ", ".join(sorted(connector.allowed_bases))
        raise ValueError(
            f"permission_basis '{basis}' is not permitted for {connector.source_type} "
            f"(allowed: {allowed})"
        )
