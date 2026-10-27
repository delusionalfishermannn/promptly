"""Storefront adapters — official APIs only, mock by default.

* ``MockStorefront`` — fully offline: writes the listing/deliverable to the artifact
  store and returns a deterministic fake URL. Used by the demo and tests.
* ``GumroadStorefront`` — Gumroad API v2 (``POST /v2/products``,
  ``PUT /v2/products/:id``). Requires ``GUMROAD_ACCESS_TOKEN``.
* ``LemonSqueezyStorefront`` — Lemon Squeezy JSON:API (``POST /v1/products``).
  Requires ``LEMONSQUEEZY_API_KEY`` and a store id.

If the configured adapter lacks credentials, :func:`get_storefront` falls back to the
mock adapter and the publish flow records that fact in the audit log — no silent
half-configuration, and **no scraping of any marketplace**.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

from core.artifacts import get_store
from core.clock import utcnow
from core.config import get_settings
from core.errors import DependencyUnavailable
from observability.logging import get_logger
from publishing.packager import PackResult

log = get_logger("publishing.storefront")


@dataclass
class PublishResult:
    ok: bool
    external_id: str = ""
    url: str = ""
    error: str = ""
    fingerprint: str = ""
    mode: str = "mock"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "external_id": self.external_id,
            "url": self.url,
            "error": self.error,
            "fingerprint": self.fingerprint,
            "mode": self.mode,
            "detail": self.detail,
        }


class StorefrontAdapter(ABC):
    name: ClassVar[str] = "base"

    @abstractmethod
    def publish(
        self, *, product: Any, version: Any, listing: dict[str, Any], pack: PackResult
    ) -> PublishResult: ...

    @abstractmethod
    def unpublish(self, *, product: Any, version: Any) -> PublishResult: ...


class MockStorefront(StorefrontAdapter):
    """Offline adapter: artifacts are the deliverable; the URL is a stable fake."""

    name = "mock"

    def publish(
        self, *, product: Any, version: Any, listing: dict[str, Any], pack: PackResult
    ) -> PublishResult:
        payload = {
            "product_id": product.id,
            "version_id": version.id,
            "listing": listing,
            "pack": pack.to_dict(),
            "published_at": utcnow().isoformat(),
            "mode": "mock",
            "note": "Mock storefront — no external listing was created.",
        }
        ref = get_store().put_json(payload, kind="listing", subject_id=version.id)
        fingerprint = hashlib.sha256(
            json.dumps(
                {"product": product.id, "version": version.version, "listing": listing},
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return PublishResult(
            ok=True,
            external_id=f"mock_{fingerprint[:16]}",
            url=f"https://storefront.mock/{product.slug}/{version.version}",
            fingerprint=fingerprint,
            mode="mock",
            detail={"artifact_uri": ref.uri},
        )

    def unpublish(self, *, product: Any, version: Any) -> PublishResult:
        return PublishResult(
            ok=True,
            external_id=f"mock_{product.slug}",
            mode="mock",
            detail={"note": "mock unpublish is a no-op"},
        )


class GumroadStorefront(StorefrontAdapter):
    """Gumroad API v2. Docs: https://gumroad.com/api (official only)."""

    name = "gumroad"
    base_url = "https://api.gumroad.com/v2"

    def _token(self) -> str:
        token = get_settings().gumroad_access_token
        if not token:
            raise DependencyUnavailable("GUMROAD_ACCESS_TOKEN is not set")
        return token

    def publish(
        self, *, product: Any, version: Any, listing: dict[str, Any], pack: PackResult
    ) -> PublishResult:
        try:
            token = self._token()
        except DependencyUnavailable as exc:
            return PublishResult(ok=False, error=str(exc), mode="gumroad")
        payload = {
            "access_token": token,
            "name": f"{listing['title']} (v{version.version})",
            "description": listing["description"],
            "price": int(round(listing["price_aud"] * 100)),
            "currency": listing.get("currency", "AUD").lower(),
            "tags": ",".join(listing.get("seo_tags", [])[:5]),
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(f"{self.base_url}/products", data=payload)
            data = resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            return PublishResult(ok=False, error=f"gumroad request failed: {exc}", mode="gumroad")
        if resp.status_code >= 400 or not data.get("success", True):
            return PublishResult(
                ok=False,
                error=f"gumroad HTTP {resp.status_code}: {str(data)[:200]}",
                mode="gumroad",
            )
        product_data = data.get("product", {})
        return PublishResult(
            ok=True,
            external_id=str(product_data.get("id", "")),
            url=str(product_data.get("short_url", "")),
            fingerprint=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            mode="gumroad",
            detail={"note": "created as a Gumroad product; upload the pack ZIP to it manually"},
        )

    def unpublish(self, *, product: Any, version: Any) -> PublishResult:
        try:
            token = self._token()
        except DependencyUnavailable as exc:
            return PublishResult(ok=False, error=str(exc), mode="gumroad")
        external_id = (product.listing or {}).get("external_id") or ""
        if not external_id:
            return PublishResult(
                ok=False, error="no external_id recorded for this product", mode="gumroad"
            )
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.delete(
                    f"{self.base_url}/products/{external_id}", data={"access_token": token}
                )
            return PublishResult(ok=resp.status_code < 400, external_id=external_id, mode="gumroad")
        except httpx.HTTPError as exc:
            return PublishResult(ok=False, error=str(exc), mode="gumroad")


class LemonSqueezyStorefront(StorefrontAdapter):
    """Lemon Squeezy JSON:API (official only)."""

    name = "lemon_squeezy"
    base_url = "https://api.lemonsqueezy.com/v1"

    def publish(
        self, *, product: Any, version: Any, listing: dict[str, Any], pack: PackResult
    ) -> PublishResult:
        api_key = get_settings().lemon_squeezy_api_key
        store_id = get_settings().lemon_squeezy_store_id
        if not api_key or not store_id:
            return PublishResult(
                ok=False,
                error="LEMONSQUEEZY_API_KEY and LEMONSQUEEZY_STORE_ID are required",
                mode="lemon_squeezy",
            )
        payload = {
            "data": {
                "type": "products",
                "attributes": {
                    "name": f"{listing['title']} (v{version.version})",
                    "description": listing["description"],
                    "price": int(round(listing["price_aud"] * 100)),
                    "status": "draft",
                },
                "relationships": {"store": {"data": {"type": "stores", "id": str(store_id)}}},
            }
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{self.base_url}/products",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/vnd.api+json",
                        "Content-Type": "application/vnd.api+json",
                    },
                    json=payload,
                )
            data = resp.json() if resp.content else {}
        except httpx.HTTPError as exc:
            return PublishResult(ok=False, error=str(exc), mode="lemon_squeezy")
        if resp.status_code >= 400:
            return PublishResult(
                ok=False,
                error=f"lemon squeezy HTTP {resp.status_code}: {str(data)[:200]}",
                mode="lemon_squeezy",
            )
        item = data.get("data", {})
        return PublishResult(
            ok=True,
            external_id=str(item.get("id", "")),
            url=str(item.get("attributes", {}).get("buy_link", "")),
            fingerprint=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            mode="lemon_squeezy",
            detail={"note": "created as a draft product"},
        )

    def unpublish(self, *, product: Any, version: Any) -> PublishResult:
        api_key = get_settings().lemon_squeezy_api_key
        external_id = (product.listing or {}).get("external_id") or ""
        if not api_key or not external_id:
            return PublishResult(ok=False, error="missing key or external id", mode="lemon_squeezy")
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.patch(
                    f"{self.base_url}/products/{external_id}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/vnd.api+json",
                        "Content-Type": "application/vnd.api+json",
                    },
                    json={
                        "data": {
                            "type": "products",
                            "id": str(external_id),
                            "attributes": {"status": "draft"},
                        }
                    },
                )
            return PublishResult(
                ok=resp.status_code < 400, external_id=external_id, mode="lemon_squeezy"
            )
        except httpx.HTTPError as exc:
            return PublishResult(ok=False, error=str(exc), mode="lemon_squeezy")


ADAPTERS: dict[str, type[StorefrontAdapter]] = {
    "mock": MockStorefront,
    "gumroad": GumroadStorefront,
    "lemon_squeezy": LemonSqueezyStorefront,
}


def get_storefront(adapter: str | None = None) -> StorefrontAdapter:
    settings = get_settings()
    choice = (adapter or settings.storefront_adapter or "mock").lower()
    if choice == "gumroad" and not settings.gumroad_access_token:
        log.warning("gumroad adapter without credentials — falling back to mock")
        return MockStorefront()
    if choice == "lemon_squeezy" and not settings.lemon_squeezy_api_key:
        log.warning("lemon squeezy adapter without credentials — falling back to mock")
        return MockStorefront()
    return ADAPTERS.get(choice, MockStorefront)()


def list_adapters() -> list[dict[str, Any]]:
    settings = get_settings()
    return [
        {"name": "mock", "ready": True, "requires": []},
        {
            "name": "gumroad",
            "ready": bool(settings.gumroad_access_token),
            "requires": ["GUMROAD_ACCESS_TOKEN"],
        },
        {
            "name": "lemon_squeezy",
            "ready": bool(settings.lemon_squeezy_api_key),
            "requires": ["LEMONSQUEEZY_API_KEY", "LEMONSQUEEZY_STORE_ID"],
        },
    ]
