"""Typed application errors mapped to HTTP status codes by the API layer."""

from __future__ import annotations

from typing import Any


class PromptFoundryError(Exception):
    """Base class for all expected, user-facing errors."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ValidationError(PromptFoundryError):
    status_code = 422
    code = "validation_error"


class NotFound(PromptFoundryError):
    status_code = 404
    code = "not_found"


class Conflict(PromptFoundryError):
    status_code = 409
    code = "conflict"


class AuthenticationError(PromptFoundryError):
    status_code = 401
    code = "unauthenticated"


class PermissionDenied(PromptFoundryError):
    status_code = 403
    code = "permission_denied"


class RateLimited(PromptFoundryError):
    status_code = 429
    code = "rate_limited"


class BudgetExceeded(PromptFoundryError):
    """Hard spend circuit breaker tripped."""

    status_code = 402
    code = "budget_exceeded"


class GovernanceBlocked(PromptFoundryError):
    status_code = 409
    code = "governance_blocked"


class PublishBlocked(PromptFoundryError):
    status_code = 409
    code = "publish_blocked"


class ProviderError(PromptFoundryError):
    status_code = 502
    code = "provider_error"


class DependencyUnavailable(PromptFoundryError):
    status_code = 503
    code = "dependency_unavailable"
