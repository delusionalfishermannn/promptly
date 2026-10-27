"""PromptFoundry relational schema (SQLAlchemy 2.0 declarative).

Conventions
-----------
* Prefixed string primary keys (``usr_...``, ``opp_...``) — safe in URLs and logs,
  no DB round-trip needed to generate them.
* Statuses are plain strings mirroring :mod:`core.constants` (portable + easy to
  extend without a migration on SQLite).
* Money is integer **AUD micros**; embeddings are JSON/pgvector lists; free-form
  structured data is JSON/JSONB.
* Nothing in this module imports from ``backend``/``workers`` — the schema is the
  bottom of the dependency graph.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from core.clock import utcnow
from db.base import Base, EmbeddingType, JSONType, TimestampMixin, pk

# --------------------------------------------------------------------------- #
# Identity, RBAC, audit
# --------------------------------------------------------------------------- #


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = pk("usr")
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_service_account: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    profile: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)


class Role(Base, TimestampMixin):
    __tablename__ = "roles"

    id: Mapped[str] = pk("rol")
    name: Mapped[str] = mapped_column(String(48), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(400), default="", nullable=False)


class UserRole(Base):
    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )
    granted_by: Mapped[str | None] = mapped_column(String(48))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    id: Mapped[str] = pk("key")
    user_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    prefix: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    scopes: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuthSession(Base):
    """Browser/CLI session token; only the SHA-256 digest is persisted."""

    __tablename__ = "auth_sessions"

    id: Mapped[str] = pk("ses")
    user_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    ip: Mapped[str] = mapped_column(String(64), default="", nullable=False)


class AuditLog(Base):
    """Append-only. Every state change and every privileged read is recorded."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = pk("aud")
    actor_type: Mapped[str] = mapped_column(String(24), default="system", nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(48), index=True)
    actor_label: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    action: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    subject_type: Mapped[str] = mapped_column(String(48), default="", nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    ip: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


# --------------------------------------------------------------------------- #
# Data sources, ingestion, signals
# --------------------------------------------------------------------------- #


class DataSource(Base, TimestampMixin):
    __tablename__ = "data_sources"

    id: Mapped[str] = pk("src")
    name: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    connector_kind: Mapped[str] = mapped_column(String(32), default="upload", nullable=False)
    permission_basis: Mapped[str] = mapped_column(String(48), nullable=False)
    permission_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    license: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    jurisdiction: Mapped[str] = mapped_column(String(8), default="AU", nullable=False)
    contains_personal_information: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    retention_days: Mapped[int] = mapped_column(Integer, default=365, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    # ^ non-secret only. Secrets are referenced by env-var NAME, never stored.
    schedule_cron: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str | None] = mapped_column(String(48))
    status: Mapped[str] = mapped_column(String(24), default="active", nullable=False)


class ConnectorRun(Base, TimestampMixin):
    __tablename__ = "connector_runs"

    id: Mapped[str] = pk("run")
    source_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(24), default="queued", nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rows_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    triggered_by: Mapped[str] = mapped_column(String(48), default="manual", nullable=False)
    cursor: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)


class RawSignal(Base, TimestampMixin):
    __tablename__ = "raw_signals"

    id: Mapped[str] = pk("raw")
    source_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[str | None] = mapped_column(String(48), index=True)
    external_id: Mapped[str | None] = mapped_column(String(200), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(String(400))
    __table_args__ = (
        UniqueConstraint("source_id", "payload_hash", name="uq_raw_signal_dedupe"),
        Index("ix_raw_signals_source_collected", "source_id", "collected_at"),
    )


class NormalizedSignal(Base, TimestampMixin):
    __tablename__ = "normalized_signals"

    id: Mapped[str] = pk("sig")
    raw_signal_id: Mapped[str | None] = mapped_column(
        String(48), ForeignKey("raw_signals.id", ondelete="SET NULL"), index=True
    )
    source_id: Mapped[str | None] = mapped_column(String(48), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    lang: Mapped[str] = mapped_column(String(8), default="en", nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="other", nullable=False, index=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    intent_strength: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    buyer_segment: Mapped[str] = mapped_column(String(24), default="unknown", nullable=False)
    urgency: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    complexity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    sentiment: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    pain: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    velocity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    recency_decay: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    quality: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType)
    embedding_model: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    embedding_ref: Mapped[str | None] = mapped_column(String(400))
    topic_id: Mapped[str | None] = mapped_column(
        String(48), ForeignKey("topics.id", ondelete="SET NULL"), index=True
    )
    __table_args__ = (
        Index("ix_normalized_signals_topic_kind", "topic_id", "kind"),
        Index("ix_normalized_signals_time", "collected_at"),
    )


# --------------------------------------------------------------------------- #
# Topics (jobs-to-be-done)
# --------------------------------------------------------------------------- #


class Topic(Base, TimestampMixin):
    __tablename__ = "topics"

    id: Mapped[str] = pk("top")
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    niche: Mapped[str] = mapped_column(String(48), default="general", nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    jtbd: Mapped[str] = mapped_column(Text, default="", nullable=False)
    buyer_segments: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    keywords: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType)
    embedding_model: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    drift_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="active", nullable=False, index=True)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    artifact_uri: Mapped[str | None] = mapped_column(String(400))


class TopicMembership(Base):
    __tablename__ = "topic_memberships"

    id: Mapped[str] = pk("mem")
    topic_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    signal_id: Mapped[str] = mapped_column(
        String(48),
        ForeignKey("normalized_signals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    distance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    __table_args__ = (UniqueConstraint("topic_id", "signal_id", name="uq_topic_signal"),)


class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"

    id: Mapped[str] = pk("trn")
    topic_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    window_days: Mapped[int] = mapped_column(Integer, default=28, nullable=False)
    signal_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    velocity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    demand_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    drift_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    centroid_embedding: Mapped[list[float] | None] = mapped_column(EmbeddingType)
    top_terms: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)


# --------------------------------------------------------------------------- #
# Opportunities
# --------------------------------------------------------------------------- #


class Opportunity(Base, TimestampMixin):
    __tablename__ = "opportunities"

    id: Mapped[str] = pk("opp")
    topic_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(240), nullable=False)
    jtbd: Mapped[str] = mapped_column(Text, default="", nullable=False)
    buyer_segment: Mapped[str] = mapped_column(String(24), default="unknown", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="new", nullable=False, index=True)
    rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)
    positioning: Mapped[str] = mapped_column(Text, default="", nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), default="low", nullable=False)
    regulated_domain: Mapped[str] = mapped_column(String(24), default="none", nullable=False)
    price_hypothesis_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(String(400))


class OpportunityScore(Base):
    __tablename__ = "opportunity_scores"

    id: Mapped[str] = pk("scr")
    opportunity_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    demand: Mapped[float] = mapped_column(Float, nullable=False)
    supply_proxy: Mapped[float] = mapped_column(Float, nullable=False)
    monetization: Mapped[float] = mapped_column(Float, nullable=False)
    build_cost: Mapped[float] = mapped_column(Float, nullable=False)
    risk: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    composite: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    composite_lower: Mapped[float] = mapped_column(Float, nullable=False)
    composite_upper: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(48), default="deterministic_v1", nullable=False)
    weights: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class EvidenceItem(Base):
    __tablename__ = "evidence_items"

    id: Mapped[str] = pk("evd")
    opportunity_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("opportunities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    signal_id: Mapped[str | None] = mapped_column(String(48), index=True)
    source_id: Mapped[str | None] = mapped_column(String(48), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="signal", nullable=False)
    quote: Mapped[str] = mapped_column(Text, default="", nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


# --------------------------------------------------------------------------- #
# Prompt products / registry
# --------------------------------------------------------------------------- #


class PromptProduct(Base, TimestampMixin):
    __tablename__ = "prompt_products"

    id: Mapped[str] = pk("prd")
    opportunity_id: Mapped[str | None] = mapped_column(
        String(48), ForeignKey("opportunities.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    slug: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    niche: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    buyer: Mapped[str] = mapped_column(String(24), default="unknown", nullable=False)
    positioning: Mapped[str] = mapped_column(Text, default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="draft", nullable=False, index=True)
    current_version_id: Mapped[str | None] = mapped_column(String(48))
    price_aud_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="AUD", nullable=False)
    tags: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    license: Mapped[str] = mapped_column(
        String(120), default="commercial-single-seat", nullable=False
    )
    regulated_domain: Mapped[str] = mapped_column(String(24), default="none", nullable=False)
    risk_tier: Mapped[str] = mapped_column(String(16), default="low", nullable=False)
    owner_id: Mapped[str | None] = mapped_column(String(48))
    listing: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)


class PromptVersion(Base, TimestampMixin):
    __tablename__ = "prompt_versions"

    id: Mapped[str] = pk("ver")
    product_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("prompt_products.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="draft", nullable=False, index=True)
    changelog: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ir: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    ir_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    model_matrix: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    metrics_summary: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(String(400))
    signature: Mapped[str | None] = mapped_column(String(200))
    signature_alg: Mapped[str] = mapped_column(String(32), default="hmac-sha256", nullable=False)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parent_version_id: Mapped[str | None] = mapped_column(String(48))
    rollback_of: Mapped[str | None] = mapped_column(String(48))
    approved_by: Mapped[str | None] = mapped_column(String(48))
    released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)
    __table_args__ = (
        UniqueConstraint("product_id", "version", name="uq_product_version"),
        Index("ix_prompt_versions_product_status", "product_id", "status"),
    )


class PromptVariant(Base, TimestampMixin):
    __tablename__ = "prompt_variants"

    id: Mapped[str] = pk("var")
    version_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("prompt_versions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_family: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    compiled_text: Mapped[str] = mapped_column(Text, nullable=False)
    compiled_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    ir_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    __table_args__ = (
        UniqueConstraint("version_id", "model_family", name="uq_variant_model_family"),
    )


# --------------------------------------------------------------------------- #
# Benchmarks & evaluation
# --------------------------------------------------------------------------- #


class Benchmark(Base, TimestampMixin):
    __tablename__ = "benchmarks"

    id: Mapped[str] = pk("bmk")
    key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    niche: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(24), default="1.0.0", nullable=False)
    is_holdout: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)


class TestSuite(Base, TimestampMixin):
    __tablename__ = "test_suites"

    id: Mapped[str] = pk("ste")
    benchmark_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("benchmarks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    niche: Mapped[str] = mapped_column(String(120), default="", nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(24), default="1.0.0", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    judge_rubric: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    weights: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    is_holdout: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    min_score: Mapped[float] = mapped_column(Float, default=0.6, nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(String(400))


class TestCase(Base, TimestampMixin):
    __tablename__ = "test_cases"

    id: Mapped[str] = pk("tcs")
    suite_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("test_suites.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_key: Mapped[str] = mapped_column(String(80), nullable=False)
    name: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    input: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    expected_schema: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    gold_output: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    assertions: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    tags: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    difficulty: Mapped[str] = mapped_column(String(16), default="normal", nullable=False)
    is_adversarial: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(48), default="synthetic", nullable=False)
    __table_args__ = (UniqueConstraint("suite_id", "case_key", name="uq_suite_case"),)


class EvaluationRun(Base, TimestampMixin):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = pk("evr")
    version_id: Mapped[str | None] = mapped_column(String(48), index=True)
    variant_id: Mapped[str | None] = mapped_column(String(48), index=True)
    suite_id: Mapped[str | None] = mapped_column(String(48), index=True)
    stage: Mapped[str] = mapped_column(String(16), default="offline", nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), default="running", nullable=False, index=True)
    n_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    n_passed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False, index=True)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    report_uri: Mapped[str | None] = mapped_column(String(400))
    parent_run_id: Mapped[str | None] = mapped_column(String(48))
    is_regression_gate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_under_test: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)
    error: Mapped[str] = mapped_column(Text, default="", nullable=False)


class EvalMetric(Base):
    __tablename__ = "eval_metrics"

    id: Mapped[str] = pk("emt")
    run_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    metric: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    n: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    __table_args__ = (UniqueConstraint("run_id", "metric", name="uq_run_metric"),)


class JudgeVote(Base):
    __tablename__ = "judge_votes"

    id: Mapped[str] = pk("jvt")
    run_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    case_id: Mapped[str | None] = mapped_column(String(48), index=True)
    judge_model: Mapped[str] = mapped_column(String(80), nullable=False)
    candidate_model: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    criterion: Mapped[str] = mapped_column(String(64), default="overall", nullable=False)
    rubric_score: Mapped[float] = mapped_column(Float, nullable=False)
    scale_max: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    verdict: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class JudgeCalibration(Base, TimestampMixin):
    __tablename__ = "judge_calibrations"

    id: Mapped[str] = pk("jcl")
    benchmark_id: Mapped[str | None] = mapped_column(String(48), index=True)
    judge_model: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    kappa: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    agreement: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    n_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bias_correction: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    calib_set: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    artifact_uri: Mapped[str | None] = mapped_column(String(400))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


# --------------------------------------------------------------------------- #
# Governance
# --------------------------------------------------------------------------- #


class SafetyCheck(Base):
    __tablename__ = "safety_checks"

    id: Mapped[str] = pk("sfc")
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str | None] = mapped_column(String(48), index=True)
    check: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(24), default="2026.10", nullable=False)
    findings: Mapped[list[Any]] = mapped_column(JSONType, default=list, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class InjectionTest(Base):
    __tablename__ = "injection_tests"

    id: Mapped[str] = pk("inj")
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    attack_key: Mapped[str] = mapped_column(String(80), nullable=False)
    technique: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="medium", nullable=False)
    payload: Mapped[str] = mapped_column(Text, default="", nullable=False)
    response: Mapped[str] = mapped_column(Text, default="", nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    judge_note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class IpSimilarityCheck(Base):
    __tablename__ = "ip_similarity_checks"

    id: Mapped[str] = pk("ipc")
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    compared_against: Mapped[str] = mapped_column(String(64), nullable=False)
    max_similarity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    shingle_overlap: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    nearest_ref: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    threshold: Mapped[float] = mapped_column(Float, default=0.92, nullable=False)
    decision: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Approval(Base, TimestampMixin):
    __tablename__ = "approvals"

    id: Mapped[str] = pk("apv")
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    requested_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    decided_by: Mapped[str | None] = mapped_column(String(48))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    conditions: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    policy_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_automatic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    __table_args__ = (Index("ix_approvals_subject_status", "subject_type", "subject_id", "status"),)


# --------------------------------------------------------------------------- #
# Experimentation & commerce
# --------------------------------------------------------------------------- #


class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"

    id: Mapped[str] = pk("exp")
    product_id: Mapped[str | None] = mapped_column(String(48), index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, default="", nullable=False)
    kind: Mapped[str] = mapped_column(String(16), default="bandit", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="draft", nullable=False, index=True)
    metric_primary: Mapped[str] = mapped_column(String(48), default="conversion", nullable=False)
    guardrails: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    alpha: Mapped[float] = mapped_column(Float, default=0.05, nullable=False)
    power: Mapped[float] = mapped_column(Float, default=0.8, nullable=False)
    min_sample: Mapped[int] = mapped_column(Integer, default=200, nullable=False)
    traffic_fraction: Mapped[float] = mapped_column(Float, default=0.1, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    decision_reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    report_uri: Mapped[str | None] = mapped_column(String(400))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)


class ExperimentArm(Base):
    __tablename__ = "experiment_arms"

    id: Mapped[str] = pk("arm")
    experiment_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    version_id: Mapped[str | None] = mapped_column(String(48), index=True)
    variant_id: Mapped[str | None] = mapped_column(String(48), index=True)
    weight: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    is_control: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    bandit_params: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    __table_args__ = (UniqueConstraint("experiment_id", "name", name="uq_arm_name"),)


class Exposure(Base):
    __tablename__ = "exposures"

    id: Mapped[str] = pk("expo")
    experiment_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    arm_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    subject_key: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(48), default="unknown", nullable=False)
    assignment_probability: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    __table_args__ = (UniqueConstraint("experiment_id", "subject_key", name="uq_exposure_subject"),)


class Allocation(Base):
    __tablename__ = "allocations"

    id: Mapped[str] = pk("alc")
    experiment_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    exposure_id: Mapped[str | None] = mapped_column(String(48), index=True)
    arm_id: Mapped[str] = mapped_column(String(48), nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), default="thompson", nullable=False)
    propensity: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    reason: Mapped[str] = mapped_column(String(200), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class Conversion(Base):
    __tablename__ = "conversions"

    id: Mapped[str] = pk("cnv")
    experiment_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    exposure_id: Mapped[str | None] = mapped_column(String(48), index=True)
    arm_id: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), default="purchase", nullable=False)
    value_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)


class SaleEvent(Base):
    __tablename__ = "sales_events"

    id: Mapped[str] = pk("sal")
    product_id: Mapped[str | None] = mapped_column(String(48), index=True)
    version_id: Mapped[str | None] = mapped_column(String(48), index=True)
    exposure_id: Mapped[str | None] = mapped_column(String(48), index=True)
    order_ref: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), default="mock", nullable=False)
    amount_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    fee_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="AUD", nullable=False)
    channel: Mapped[str] = mapped_column(String(48), default="unknown", nullable=False)
    country: Mapped[str] = mapped_column(String(2), default="AU", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)


class Refund(Base):
    __tablename__ = "refunds"

    id: Mapped[str] = pk("rfd")
    sale_id: Mapped[str] = mapped_column(
        String(48), ForeignKey("sales_events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    amount_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    reason_code: Mapped[str] = mapped_column(
        String(48), default="other", nullable=False, index=True
    )
    reason_note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id: Mapped[str] = pk("sup")
    product_id: Mapped[str | None] = mapped_column(String(48), index=True)
    channel: Mapped[str] = mapped_column(String(32), default="email", nullable=False)
    subject: Mapped[str] = mapped_column(String(240), default="", nullable=False)
    body: Mapped[str] = mapped_column(Text, default="", nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    minutes_spent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class StorefrontEvent(Base):
    """First-party analytics: our own storefront views/clicks (permitted data)."""

    __tablename__ = "storefront_events"

    id: Mapped[str] = pk("stev")
    product_id: Mapped[str | None] = mapped_column(String(48), index=True)
    topic_id: Mapped[str | None] = mapped_column(String(48), index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    path: Mapped[str] = mapped_column(String(240), default="", nullable=False)
    query: Mapped[str] = mapped_column(Text, default="", nullable=False)
    session_key: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    meta: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)


# --------------------------------------------------------------------------- #
# Ops: artifacts, costs, flags, alerts
# --------------------------------------------------------------------------- #


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = pk("art")
    kind: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    subject_type: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    uri: Mapped[str] = mapped_column(String(400), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(80), default="application/json", nullable=False
    )
    signature: Mapped[str | None] = mapped_column(String(200))
    created_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class CostEvent(Base):
    __tablename__ = "cost_events"

    id: Mapped[str] = pk("cost")
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(80))
    purpose: Mapped[str | None] = mapped_column(String(64))
    amount_micros: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="AUD", nullable=False)
    subject_type: Mapped[str | None] = mapped_column(String(32))
    subject_id: Mapped[str | None] = mapped_column(String(64), index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class RuntimeFlag(Base, TimestampMixin):
    """Kill switches and operator overrides that must not need a redeploy."""

    __tablename__ = "runtime_flags"

    id: Mapped[str] = pk("flg")
    key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_by: Mapped[str] = mapped_column(String(48), default="system", nullable=False)


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[str] = pk("alr")
    kind: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), default="warning", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    subject_type: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    subject_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
