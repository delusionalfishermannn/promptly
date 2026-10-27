"""Small repository helpers shared by services, routers, workers and the demo.

Deliberately thin: explicit functions over a Session, no hidden unit-of-work magic.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import Select, func, select

from core.clock import utcnow
from core.ids import new_id
from db.models import (
    Alert,
    Approval,
    Artifact,
    AuditLog,
    CostEvent,
    PromptProduct,
    PromptVersion,
    Role,
    RuntimeFlag,
    User,
    UserRole,
)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def record_audit(
    session: Any,
    *,
    action: str,
    actor_type: str = "system",
    actor_id: str | None = None,
    actor_label: str = "",
    subject_type: str = "",
    subject_id: str = "",
    decision: str = "",
    reason: str = "",
    detail: dict[str, Any] | None = None,
    request_id: str = "",
    trace_id: str = "",
    ip: str = "",
    commit: bool = False,
) -> AuditLog:
    entry = AuditLog(
        actor_type=actor_type,
        actor_id=actor_id,
        actor_label=actor_label,
        action=action,
        subject_type=subject_type,
        subject_id=subject_id,
        decision=decision,
        reason=reason,
        detail=detail or {},
        request_id=request_id,
        trace_id=trace_id,
        ip=ip,
    )
    session.add(entry)
    session.flush()
    if commit:
        session.commit()
    return entry


def audit_query(
    session: Any,
    *,
    action: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    actor_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[AuditLog]:
    stmt: Select[Any] = select(AuditLog).order_by(AuditLog.created_at.desc())
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if subject_type:
        stmt = stmt.where(AuditLog.subject_type == subject_type)
    if subject_id:
        stmt = stmt.where(AuditLog.subject_id == subject_id)
    if actor_id:
        stmt = stmt.where(AuditLog.actor_id == actor_id)
    return list(session.scalars(stmt.limit(limit).offset(offset)))


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
def record_artifact(
    session: Any,
    ref: Any,
    *,
    kind: str,
    subject_type: str = "",
    subject_id: str = "",
    created_by: str = "system",
    signature: str | None = None,
) -> Artifact:
    row = Artifact(
        kind=kind,
        subject_type=subject_type,
        subject_id=subject_id,
        uri=ref.uri,
        sha256=ref.sha256,
        size_bytes=ref.size_bytes,
        content_type=ref.content_type,
        signature=signature,
        created_by=created_by,
    )
    session.add(row)
    session.flush()
    return row


def artifacts_for(session: Any, subject_id: str, kind: str | None = None) -> list[Artifact]:
    stmt = (
        select(Artifact)
        .where(Artifact.subject_id == subject_id)
        .order_by(Artifact.created_at.desc())
    )
    if kind:
        stmt = stmt.where(Artifact.kind == kind)
    return list(session.scalars(stmt))


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
def raise_alert(
    session: Any,
    *,
    kind: str,
    message: str,
    severity: str = "warning",
    subject_type: str = "",
    subject_id: str = "",
    detail: dict[str, Any] | None = None,
) -> Alert:
    alert = Alert(
        kind=kind,
        severity=severity,
        message=message,
        subject_type=subject_type,
        subject_id=subject_id,
        detail=detail or {},
    )
    session.add(alert)
    session.flush()
    return alert


# --------------------------------------------------------------------------- #
# Runtime flags (kill switches)
# --------------------------------------------------------------------------- #
DEFAULT_FLAGS: dict[str, tuple[dict[str, Any], str]] = {
    "pipeline.enabled": ({"enabled": True}, "Master switch for opportunity discovery"),
    "generation.enabled": ({"enabled": True}, "Master switch for prompt generation"),
    "evaluation.enabled": ({"enabled": True}, "Master switch for evaluation runs"),
    "publishing.auto": (
        {"enabled": False},
        "Auto-publish override; requires PF_AUTO_PUBLISH_ENABLED and a low-risk product",
    ),
    "publishing.kill_switch": ({"engaged": False}, "Immediately halts all publishing"),
    "llm.provider": ({"preferred": "auto"}, "Preferred provider ordering"),
}


def ensure_default_flags(session: Any) -> None:
    existing = {flag.key for flag in session.scalars(select(RuntimeFlag))}
    for key, (value, description) in DEFAULT_FLAGS.items():
        if key not in existing:
            session.add(RuntimeFlag(key=key, value=value, description=description))


def get_flag(session: Any, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    flag = session.scalar(select(RuntimeFlag).where(RuntimeFlag.key == key))
    if flag is None:
        return default if default is not None else DEFAULT_FLAGS.get(key, ({}, ""))[0]
    return flag.value


def set_flag(
    session: Any, key: str, value: dict[str, Any], updated_by: str = "system"
) -> RuntimeFlag:
    flag = session.scalar(select(RuntimeFlag).where(RuntimeFlag.key == key))
    if flag is None:
        flag = RuntimeFlag(key=key, value=value, updated_by=updated_by)
        session.add(flag)
    else:
        flag.value = value
        flag.updated_by = updated_by
        flag.updated_at = utcnow()
    session.flush()
    return flag


def flag_enabled(session: Any, key: str, field: str = "enabled") -> bool:
    return bool(get_flag(session, key).get(field, False))


# --------------------------------------------------------------------------- #
# Users / roles
# --------------------------------------------------------------------------- #
def get_user_by_email(session: Any, email: str) -> User | None:
    return session.scalar(select(User).where(func.lower(User.email) == email.lower()))


def get_user(session: Any, user_id: str) -> User | None:
    return session.get(User, user_id)


def roles_for_user(session: Any, user_id: str) -> list[str]:
    stmt = (
        select(Role.name)
        .join(UserRole, UserRole.role_id == Role.id)
        .where(UserRole.user_id == user_id)
    )
    return [row for row in session.scalars(stmt)]


def assign_role(session: Any, user_id: str, role_name: str, granted_by: str | None = None) -> None:
    role = session.scalar(select(Role).where(Role.name == role_name))
    if role is None:
        raise ValueError(f"unknown role: {role_name}")
    exists = session.scalar(
        select(UserRole).where(UserRole.user_id == user_id, UserRole.role_id == role.id)
    )
    if exists is None:
        session.add(UserRole(user_id=user_id, role_id=role.id, granted_by=granted_by))
        session.flush()


# --------------------------------------------------------------------------- #
# Versions / approvals convenience
# --------------------------------------------------------------------------- #
def latest_version(session: Any, product_id: str) -> PromptVersion | None:
    return session.scalar(
        select(PromptVersion)
        .where(PromptVersion.product_id == product_id)
        .order_by(PromptVersion.created_at.desc())
        .limit(1)
    )


def pending_approvals(session: Any, limit: int = 100) -> list[Approval]:
    return list(
        session.scalars(
            select(Approval)
            .where(Approval.status == "pending")
            .order_by(Approval.requested_at.desc())
            .limit(limit)
        )
    )


def product_by_slug(session: Any, slug: str) -> PromptProduct | None:
    return session.scalar(select(PromptProduct).where(PromptProduct.slug == slug))


# --------------------------------------------------------------------------- #
# Cost helpers
# --------------------------------------------------------------------------- #
def recent_costs(session: Any, limit: int = 50) -> list[CostEvent]:
    return list(
        session.scalars(select(CostEvent).order_by(CostEvent.created_at.desc()).limit(limit))
    )


def new_id_for(prefix: str) -> str:
    return new_id(prefix)


def expiring_soon(expires_at: Any, hours: int = 0) -> bool:
    if expires_at is None:
        return False
    target = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=utcnow().tzinfo)
    return target <= utcnow() + timedelta(hours=hours)
