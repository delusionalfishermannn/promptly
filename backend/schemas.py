"""Pydantic request/response contracts for the API.

Response models are intentionally *loose* (``dict[str, Any]`` payloads inside typed
envelopes) because domain objects are already validated and versioned; write models are
strict, because they are the surface an attacker or an integrator touches.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=5, max_length=320)
    password: str = Field(min_length=10, max_length=200)
    display_name: str = Field(default="", max_length=160)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str
    password: str


class TokenResponse(BaseModel):
    token: str
    token_type: str = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_at: str
    user: dict[str, Any]


class ApiKeyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="default", max_length=120)
    scopes: list[str] = Field(default_factory=list)
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class RoleAssignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["admin", "approver", "operator", "analyst", "viewer"]


# --------------------------------------------------------------------------- #
# Sources & ingestion
# --------------------------------------------------------------------------- #


class SourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=2, max_length=160)
    source_type: Literal[
        "manual_csv",
        "manual_json",
        "storefront",
        "support_tickets",
        "mock",
        "reddit_api",
        "stackexchange_api",
        "youtube_api",
        "search_console_api",
        "google_ads_api",
    ]
    permission_basis: Literal[
        "operator_provided",
        "first_party_analytics",
        "official_api_tos",
        "permissive_license",
        "synthetic",
        "public_domain",
    ]
    permission_notes: str = Field(default="", max_length=2000)
    config: dict[str, Any] = Field(default_factory=dict)
    license: str = Field(default="", max_length=120)
    contains_personal_information: bool = False
    retention_days: int = Field(default=365, ge=1, le=3650)
    schedule_cron: str | None = None
    enabled: bool = True


class RunSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=500, ge=1, le=2000)


# --------------------------------------------------------------------------- #
# Discovery / opportunities
# --------------------------------------------------------------------------- #


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_signals: int = Field(default=2000, ge=10, le=20000)
    min_cluster_size: int | None = Field(default=None, ge=2, le=50)
    eps: float | None = Field(default=None, gt=0.0, lt=1.0)


class LaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_candidates: int = Field(default=4, ge=2, le=6)
    primary_family: Literal["openai", "anthropic", "google", "generic"] = "generic"
    evaluate_holdout: bool = True
    run_gate: bool = True
    price_aud: float | None = Field(default=None, ge=0, le=5000)


# --------------------------------------------------------------------------- #
# Products / versions / evaluation
# --------------------------------------------------------------------------- #


class ProductUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, max_length=200)
    positioning: str | None = None
    description: str | None = None
    price_aud: float | None = Field(default=None, ge=0, le=5000)
    tags: list[str] | None = None
    license: str | None = None


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_id: str | None = None
    stage: Literal["screen", "offline", "finalist", "online", "holdout"] = "finalist"
    model: str | None = None
    judge: bool = True
    votes: int = Field(default=2, ge=1, le=5)
    robustness: bool = True
    max_cases: int | None = Field(default=None, ge=1, le=200)
    budget_cap_aud: float = Field(default=1.5, gt=0, le=50)


class SuiteLoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str | None = None
    path: str | None = None


class CalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    judge_model: str | None = None
    benchmark_id: str | None = None
    items: list[dict[str, Any]] = Field(min_length=1, max_length=200)


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #


class GateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_injection: bool = True
    request_approval: bool = True
    eval_run_id: str | None = None


class ApprovalDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(default="", max_length=2000)
    conditions: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #


class PublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    price_aud: float | None = Field(default=None, ge=0, le=5000)


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to_version_id: str
    reason: str = Field(default="", max_length=2000)


# --------------------------------------------------------------------------- #
# Experimentation
# --------------------------------------------------------------------------- #


class ExperimentArmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)
    version_id: str | None = None
    variant_id: str | None = None
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    is_control: bool = False


class ExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str
    name: str = Field(min_length=3, max_length=200)
    arms: list[ExperimentArmSpec] = Field(min_length=2, max_length=6)
    hypothesis: str = ""
    kind: Literal["ab_test", "bandit"] = "ab_test"
    metric_primary: str = "conversion"
    traffic_fraction: float = Field(default=0.1, gt=0.0, le=1.0)
    min_sample: int = Field(default=200, ge=10, le=100000)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    guardrails: dict[str, Any] = Field(default_factory=dict)


class AssignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_key: str = Field(min_length=1, max_length=120)
    channel: str = Field(default="direct", max_length=48)
    force_arm: str | None = None


class ConversionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exposure_id: str
    event_type: Literal["view", "click", "checkout_started", "purchase", "refund"] = "purchase"
    value_aud: float = Field(default=0.0, ge=0, le=100000)
    order_ref: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sale_id: str
    amount_aud: float = Field(gt=0, le=100000)
    reason_code: Literal[
        "not_as_described", "quality", "duplicate", "changed_mind", "support", "other"
    ] = "other"
    note: str = Field(default="", max_length=2000)


class StorefrontEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_type: Literal["view", "click", "checkout_started", "purchase"]
    product_id: str | None = None
    topic_id: str | None = None
    path: str = ""
    query: str = ""
    session_key: str = Field(default="", max_length=80)
    meta: dict[str, Any] = Field(default_factory=dict)


class ExperimentDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    promote_winning_arm: bool = False


class StopExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=3, max_length=2000)


# --------------------------------------------------------------------------- #
# Ops
# --------------------------------------------------------------------------- #


class FlagUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: dict[str, Any]

    @field_validator("value")
    @classmethod
    def _non_empty(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("flag value must not be empty")
        return value


class AlertAcknowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str = Field(default="", max_length=500)


class SupportTicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str
    subject: str = Field(min_length=3, max_length=240)
    body: str = Field(default="", max_length=4000)
    minutes_spent: int = Field(default=0, ge=0, le=600)


class Envelope(BaseModel):
    """Standard list envelope."""

    items: list[dict[str, Any]]
    count: int
    next_cursor: str | None = None


class StatusResponse(BaseModel):
    status: str
    detail: dict[str, Any] = Field(default_factory=dict)
