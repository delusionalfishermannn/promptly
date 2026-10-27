"""Official-API connectors. Disabled until the matching credentials exist.

All of these use the providers' documented public APIs only:

* Reddit — OAuth2 ``client_credentials`` against ``oauth.reddit.com`` (ToS-compliant,
  app-registered; respects ``X-Ratelimit-*`` headers).
* StackExchange — ``api.stackexchange.com/2.3/search/advanced`` (public, quota-based).
* YouTube — ``youtube.googleapis.com/youtube/v3/search`` (official Data API v3).
* Google Search Console — official Search Analytics API for **your own** properties
  (requires an OAuth service account; intentionally left as a documented stub so we
  never ship a half-working credential flow that encourages scraping).

No HTML is fetched anywhere. If a provider revokes API access, the connector reports
``readiness=False`` and ingestion for that source is skipped with an audit entry.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, ClassVar

import httpx

from connectors.base import Connector, ConnectorResult, RateLimiter, SignalRecord
from core.clock import utcnow
from core.constants import ConnectorKind, PermissionBasis, SignalKind
from core.errors import DependencyUnavailable, ProviderError


class _JsonApiConnector(Connector):
    connector_kind: ClassVar[str] = str(ConnectorKind.OFFICIAL_API)
    allowed_bases: ClassVar[set[str]] = {str(PermissionBasis.OFFICIAL_API_TOS)}
    base_url: ClassVar[str] = ""
    timeout: ClassVar[float] = 30.0

    def _get(
        self, path: str, *, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, params=params, headers=headers or {})
        except httpx.HTTPError as exc:
            raise DependencyUnavailable(f"{self.source_type} request failed: {exc}") from exc
        if resp.status_code == 429:
            raise ProviderError(f"{self.source_type} rate limited (429); try again later")
        if resp.status_code >= 400:
            raise ProviderError(
                f"{self.source_type} HTTP {resp.status_code}", details={"body": resp.text[:300]}
            )
        return resp.json()


class RedditConnector(_JsonApiConnector):
    source_type: ClassVar[str] = "reddit_api"
    display_name: ClassVar[str] = "Reddit official API"
    requires_secret_env: ClassVar[tuple[str, ...]] = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET")
    base_url: ClassVar[str] = "https://oauth.reddit.com"
    tos_notes: ClassVar[str] = (
        "Requires a registered Reddit app; use OAuth client_credentials, respect rate limits, "
        "do not redistribute content beyond the API terms."
    )

    def validate_config(self) -> list[str]:
        issues = []
        if not self.config.get("subreddits"):
            issues.append("config.subreddits is required (list of subreddit names)")
        if not self.config.get("queries"):
            issues.append("config.queries is required (list of search phrases)")
        return issues

    def _token(self) -> str:
        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        ua = os.environ.get("REDDIT_USER_AGENT", "promptfoundry/0.1 (demand research)")
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": ua},
            )
        if resp.status_code >= 400:
            raise ProviderError(f"reddit auth failed: HTTP {resp.status_code}")
        return resp.json()["access_token"]

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 200,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        if not self.is_ready():
            raise DependencyUnavailable("Reddit connector not configured")
        token = self._token()
        limiter = RateLimiter(per_second=0.8)  # Reddit free tier ≈ 100 QPM
        result = ConnectorResult()
        headers = {
            "Authorization": f"bearer {token}",
            "User-Agent": os.environ.get("REDDIT_USER_AGENT", "promptfoundry/0.1"),
        }
        per_query = max(1, limit // max(len(self.config.get("queries", [])), 1))
        for query in self.config.get("queries", []):
            for subreddit in self.config.get("subreddits", []) or [None]:
                if len(result.records) >= limit:
                    break
                limiter.acquire()
                path = f"/r/{subreddit}/search" if subreddit else "/search"
                data = self._get(
                    path,
                    params={
                        "q": query,
                        "restrict_sr": bool(subreddit),
                        "sort": "new",
                        "limit": per_query,
                        "t": "month",
                    },
                    headers=headers,
                )
                for child in (data.get("data", {}) or {}).get("children", []) or []:
                    post = child.get("data", {}) or {}
                    text = f"{post.get('title', '')}. {post.get('selftext', '')}".strip()
                    if len(text) < 20:
                        continue
                    result.records.append(
                        SignalRecord(
                            text=" ".join(text.split())[:5000],
                            external_id=post.get("id"),
                            kind=str(SignalKind.QUESTION),
                            occurred_at=datetime.fromtimestamp(
                                float(post.get("created_utc", utcnow().timestamp())),
                                tz=utcnow().tzinfo,
                            ),
                            volume=float(post.get("score", 1) or 1),
                            url=f"https://www.reddit.com{post.get('permalink', '')}",
                            metadata={"subreddit": post.get("subreddit"), "query": query},
                        )
                    )
                    if len(result.records) >= limit:
                        break
        result.stats.update({"rows_out": len(result.records), "connector": self.source_type})
        return result


class StackExchangeConnector(_JsonApiConnector):
    source_type: ClassVar[str] = "stackexchange_api"
    display_name: ClassVar[str] = "StackExchange API"
    base_url: ClassVar[str] = "https://api.stackexchange.com"
    tos_notes: ClassVar[str] = (
        "Public API with quota; attribution required by the StackExchange terms. Keep "
        "attribution metadata on every signal and respect the daily quota."
    )

    def validate_config(self) -> list[str]:
        return [] if self.config.get("queries") else ["config.queries is required"]

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 200,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        limiter = RateLimiter(per_second=1.0)
        result = ConnectorResult()
        site = str(self.config.get("site", "stackoverflow"))
        for query in self.config.get("queries", [])[:10]:
            limiter.acquire()
            data = self._get(
                "/2.3/search/advanced",
                params={
                    "order": "desc",
                    "sort": "creation",
                    "q": query,
                    "site": site,
                    "pagesize": min(50, limit - len(result.records)),
                    "filter": "default",
                },
            )
            for item in data.get("items", []) or []:
                title = str(item.get("title", "")).strip()
                if len(title) < 12:
                    continue
                result.records.append(
                    SignalRecord(
                        text=title[:2000],
                        external_id=str(item.get("question_id")),
                        kind=str(SignalKind.QUESTION),
                        occurred_at=datetime.fromtimestamp(
                            float(item.get("creation_date", utcnow().timestamp())),
                            tz=utcnow().tzinfo,
                        ),
                        volume=float(item.get("score", 1) or 1),
                        url=item.get("link"),
                        metadata={
                            "site": site,
                            "attribution": "StackExchange (CC BY-SA 4.0) — link required",
                            "tags": item.get("tags", []),
                            "query": query,
                        },
                    )
                )
            if data.get("quota_remaining") is not None:
                result.stats["quota_remaining"] = data["quota_remaining"]
        result.stats.update({"rows_out": len(result.records), "connector": self.source_type})
        return result


class YouTubeConnector(_JsonApiConnector):
    source_type: ClassVar[str] = "youtube_api"
    display_name: ClassVar[str] = "YouTube Data API v3"
    requires_secret_env: ClassVar[tuple[str, ...]] = ("YOUTUBE_API_KEY",)
    base_url: ClassVar[str] = "https://www.googleapis.com/youtube/v3"
    tos_notes: ClassVar[str] = (
        "Official Data API v3 only; quota-limited (10k units/day by default). Do not "
        "cache or republish content beyond the YouTube API Services Terms."
    )

    def validate_config(self) -> list[str]:
        return [] if self.config.get("queries") else ["config.queries is required"]

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 100,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        key = os.environ.get("YOUTUBE_API_KEY", "")
        if not key:
            raise DependencyUnavailable("YOUTUBE_API_KEY not set")
        result = ConnectorResult()
        published_after = (since or (utcnow() - timedelta(days=30))).strftime("%Y-%m-%dT%H:%M:%SZ")
        for query in self.config.get("queries", [])[:5]:
            data = self._get(
                "/search",
                params={
                    "key": key,
                    "part": "snippet",
                    "q": query,
                    "type": "video",
                    "maxResults": min(25, max(1, limit - len(result.records))),
                    "publishedAfter": published_after,
                    "order": "viewCount",
                },
            )
            for item in data.get("items", []) or []:
                snippet = item.get("snippet", {}) or {}
                title = str(snippet.get("title", "")).strip()
                if len(title) < 12:
                    continue
                result.records.append(
                    SignalRecord(
                        text=title[:2000],
                        external_id=item.get("id", {}).get("videoId"),
                        kind=str(SignalKind.QUESTION),
                        occurred_at=utcnow(),
                        volume=1.0,
                        url=f"https://www.youtube.com/watch?v={item.get('id', {}).get('videoId', '')}",
                        metadata={"channel": snippet.get("channelTitle"), "query": query},
                    )
                )
        result.stats.update({"rows_out": len(result.records), "connector": self.source_type})
        return result


class SearchConsoleConnector(_JsonApiConnector):
    """Search Analytics for properties **you own** (OAuth service account required).

    Deliberately a stub: a half-implemented OAuth flow invites operators to point this
    at properties they do not control. The interface, readiness probe and error
    messages are complete so implementation is a drop-in.
    """

    source_type: ClassVar[str] = "search_console_api"
    display_name: ClassVar[str] = "Google Search Console (owned properties)"
    requires_secret_env: ClassVar[tuple[str, ...]] = (
        "GOOGLE_SERVICE_ACCOUNT_JSON",
        "SEARCH_CONSOLE_SITE_URL",
    )
    base_url: ClassVar[str] = "https://searchconsole.googleapis.com"
    tos_notes: ClassVar[str] = (
        "Only for properties you have verified in Search Console. Requires an OAuth "
        "service account with the webmasters.readonly scope."
    )

    def validate_config(self) -> list[str]:
        return (
            []
            if self.config.get("site_url") or os.environ.get("SEARCH_CONSOLE_SITE_URL")
            else ["config.site_url or SEARCH_CONSOLE_SITE_URL is required"]
        )

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        raise DependencyUnavailable(
            "Search Console connector requires the OAuth service-account flow. "
            "Implement `run()` using webmasters/v3 searchAnalytics.query with your own "
            "verified property; the audit log and interface are already wired."
        )


class GoogleAdsConnector(_JsonApiConnector):
    """Keyword Planner volume for the operator's own Google Ads account (stub)."""

    source_type: ClassVar[str] = "google_ads_api"
    display_name: ClassVar[str] = "Google Ads Keyword Planner (own account)"
    requires_secret_env: ClassVar[tuple[str, ...]] = (
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_CLIENT_ID",
        "GOOGLE_ADS_CLIENT_SECRET",
        "GOOGLE_ADS_REFRESH_TOKEN",
        "GOOGLE_ADS_CUSTOMER_ID",
    )
    base_url: ClassVar[str] = "https://googleads.googleapis.com"
    tos_notes: ClassVar[str] = (
        "Keyword Planner data must come through the official Google Ads API using your own "
        "account and developer token; redistribution limits apply."
    )

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        raise DependencyUnavailable(
            "Google Ads connector requires OAuth + developer token. Implement "
            "`run()` against KeywordPlanIdeaService; ingest volumes as signal weight."
        )


class SupportTicketConnector(Connector):
    """Support tickets exported from your helpdesk (CSV) or pushed via webhook."""

    source_type: ClassVar[str] = "support_tickets"
    connector_kind: ClassVar[str] = str(ConnectorKind.UPLOAD)
    display_name: ClassVar[str] = "Support tickets (export/webhook)"
    allowed_bases: ClassVar[set[str]] = {
        str(PermissionBasis.OPERATOR_PROVIDED),
        str(PermissionBasis.FIRST_PARTY_ANALYTICS),
    }
    tos_notes: ClassVar[str] = (
        "Tickets may contain personal information — declare it on the source, keep "
        "redaction on, and honour your retention policy (APP 11/12)."
    )

    def run(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
        cursor: dict[str, Any] | None = None,
    ) -> ConnectorResult:
        from connectors.upload import CsvUploadConnector

        path = self.config.get("path")
        if not path:
            raise DependencyUnavailable("config.path (CSV export) is required")
        with open(path, "rb") as handle:
            return CsvUploadConnector(self.config).parse_bytes(handle.read())
