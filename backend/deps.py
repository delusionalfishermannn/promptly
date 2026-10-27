"""Request context, authentication, RBAC dependencies, rate limiting.

Two credential types are supported:

* **Session token** (browser/CLI): ``Authorization: Bearer <token>`` — opaque, stored
  as a SHA-256 digest, TTL-bounded, revocable.
* **API key** (machine clients): ``X-API-Key: <key>`` — same digest-only storage, with
  optional scopes and expiry.

Every authenticated request yields a :class:`RequestContext` that carries the DB
session, the user, roles, effective permissions and request metadata. Routers declare
``Depends(require(Permission.X))``; a denied check is audited, not just refused.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, Header, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import utcnow
from core.config import get_settings
from core.constants import ActorType
from core.errors import AuthenticationError, PermissionDenied, RateLimited
from core.rbac import Permission, has_permission, permissions_for
from core.security import token_digest
from db.models import ApiKey, AuthSession, User
from db.repository import record_audit, roles_for_user
from db.session import get_db
from observability.logging import actor_var, request_id_var


@dataclass
class RequestContext:
    session: Session
    user: User | None = None
    roles: list[str] = field(default_factory=list)
    permissions: set[str] = field(default_factory=set)
    request_id: str = "-"
    ip: str = ""
    user_agent: str = ""
    credential: str = "none"  # session | api_key | bootstrap
    scopes: list[str] = field(default_factory=list)

    @property
    def actor_id(self) -> str:
        return self.user.id if self.user else "system"

    @property
    def actor_label(self) -> str:
        return self.user.email if self.user else "system"

    def can(self, permission: Permission | str) -> bool:
        return has_permission(self.roles, permission)

    def audit(
        self,
        action: str,
        *,
        subject_type: str = "",
        subject_id: str = "",
        decision: str = "",
        reason: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        record_audit(
            self.session,
            action=action,
            actor_type=str(ActorType.USER) if self.user else str(ActorType.SYSTEM),
            actor_id=self.actor_id,
            actor_label=self.actor_label,
            subject_type=subject_type,
            subject_id=subject_id,
            decision=decision,
            reason=reason,
            detail=detail or {},
            request_id=self.request_id,
            ip=self.ip,
        )


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _resolve_session(session: Session, token: str) -> User:
    digest = token_digest(token)
    row = session.scalar(select(AuthSession).where(AuthSession.token_hash == digest))
    if row is None or row.revoked_at is not None:
        raise AuthenticationError("invalid or revoked session token")
    expires = (
        row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=utcnow().tzinfo)
    )
    if expires <= utcnow():
        raise AuthenticationError("session token expired")
    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("user is inactive")
    return user


def _resolve_api_key(session: Session, key: str) -> tuple[User, list[str]]:
    digest = token_digest(key)
    row = session.scalar(select(ApiKey).where(ApiKey.key_hash == digest))
    if row is None or row.revoked_at is not None:
        raise AuthenticationError("invalid or revoked API key")
    if row.expires_at is not None:
        expires = (
            row.expires_at
            if row.expires_at.tzinfo
            else row.expires_at.replace(tzinfo=utcnow().tzinfo)
        )
        if expires <= utcnow():
            raise AuthenticationError("API key expired")
    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("API key owner is inactive")
    row.last_used_at = utcnow()
    session.flush()
    return user, list(row.scopes or [])


def get_context(
    request: Request,
    session: Annotated[Session, Depends(get_db)],
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> RequestContext:
    request_id = getattr(request.state, "request_id", "-")
    ip = _client_ip(request)
    user_agent = request.headers.get("user-agent", "")[:240]
    ctx = RequestContext(
        session=session,
        request_id=request_id,
        ip=ip,
        user_agent=user_agent,
    )

    token = ""
    credential = "none"
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        credential = "session"
    elif x_api_key:
        token = x_api_key.strip()
        credential = "api_key"

    if token:
        if credential == "api_key":
            user, scopes = _resolve_api_key(session, token)
            ctx.scopes = scopes
        else:
            user = _resolve_session(session, token)
        ctx.user = user
        ctx.credential = credential
        ctx.roles = roles_for_user(session, user.id)
        ctx.permissions = permissions_for(ctx.roles)
    actor_var.set(ctx.actor_label)
    request_id_var.set(request_id)
    return ctx


def require(permission: Permission | str) -> Callable[..., RequestContext]:
    """Dependency factory enforcing a permission, auditing denials."""

    def dependency(context: Annotated[RequestContext, Depends(get_context)]) -> RequestContext:
        if context.user is None:
            raise AuthenticationError("authentication required")
        if not context.can(permission):
            context.audit(
                "authz.denied",
                decision="denied",
                reason=f"missing permission {permission}",
                detail={"permission": str(permission), "roles": context.roles},
            )
            context.session.commit()
            raise PermissionDenied(f"missing permission: {permission}")
        return context

    return dependency


def require_authenticated(
    context: Annotated[RequestContext, Depends(get_context)],
) -> RequestContext:
    if context.user is None:
        raise AuthenticationError("authentication required")
    return context


# --------------------------------------------------------------------------- #
# Simple in-process rate limiting (per actor/IP). For multi-instance deployments
# swap the backend for Redis (documented in DESIGN.md §7).
# --------------------------------------------------------------------------- #
_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_WINDOW = 60.0


def rate_limit(
    request: Request,
    context: Annotated[RequestContext, Depends(get_context)],
) -> RequestContext:
    settings = get_settings()
    limit = max(settings.rate_limit_per_minute, 1)
    key = context.actor_id if context.user else (context.ip or "anonymous")
    now = time.monotonic()
    bucket = _BUCKETS[key]
    while bucket and now - bucket[0] > _RATE_WINDOW:
        bucket.popleft()
    if len(bucket) >= limit:
        retry_after = int(_RATE_WINDOW - (now - bucket[0])) + 1
        raise RateLimited(
            f"rate limit exceeded ({limit}/min)", details={"retry_after_seconds": retry_after}
        )
    bucket.append(now)
    return context


def reset_rate_limits() -> None:
    _BUCKETS.clear()


require_rate_limited = rate_limit
