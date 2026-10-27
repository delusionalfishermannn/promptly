"""Authentication and account management."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import utcnow
from core.config import get_settings
from core.errors import AuthenticationError, Conflict, NotFound, PermissionDenied, ValidationError
from core.rbac import ROLE_PERMISSIONS, Permission, Role, permissions_for
from core.security import hash_password, new_token, token_digest, verify_password
from db.models import ApiKey, AuthSession, User
from db.models import Role as RoleRow
from db.repository import assign_role, get_user_by_email, record_audit, roles_for_user

DEFAULT_ROLES: dict[str, str] = {
    str(Role.ADMIN): "Full control, including users and budgets",
    str(Role.APPROVER): "Reviews and approves/publishes prompt products",
    str(Role.OPERATOR): "Runs pipelines, generates and evaluates products",
    str(Role.ANALYST): "Read access plus experiments and evaluation",
    str(Role.VIEWER): "Read-only catalog/evals",
}

MIN_PASSWORD_LENGTH = 10


def ensure_roles(session: Session) -> None:
    existing = {row.name for row in session.scalars(select(RoleRow))}
    for name, description in DEFAULT_ROLES.items():
        if name not in existing:
            session.add(RoleRow(name=name, description=description))
    session.flush()


def bootstrap(session: Session) -> dict[str, Any]:
    """Idempotent bootstrap: roles + optional admin from env + default flags."""
    from db.repository import ensure_default_flags

    settings = get_settings()
    ensure_roles(session)
    ensure_default_flags(session)
    created_admin = False
    if settings.bootstrap_admin_email and settings.bootstrap_admin_password:
        user = get_user_by_email(session, settings.bootstrap_admin_email)
        if user is None:
            user = User(
                email=settings.bootstrap_admin_email.lower(),
                display_name="Bootstrap Admin",
                password_hash=hash_password(settings.bootstrap_admin_password),
            )
            session.add(user)
            session.flush()
            assign_role(session, user.id, str(Role.ADMIN), granted_by="bootstrap")
            created_admin = True
        elif str(Role.ADMIN) not in roles_for_user(session, user.id):
            assign_role(session, user.id, str(Role.ADMIN), granted_by="bootstrap")
    session.flush()
    return {
        "roles": sorted(DEFAULT_ROLES),
        "admin_bootstrapped": created_admin,
        "admin_email": settings.bootstrap_admin_email or None,
    }


def register_user(
    session: Session,
    *,
    email: str,
    password: str,
    display_name: str = "",
    role: str = str(Role.VIEWER),
    actor_id: str | None = None,
) -> User:
    settings = get_settings()
    if not settings.allow_self_registration and actor_id is None:
        raise PermissionDenied("self-registration is disabled")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise ValidationError("a valid email is required")
    if get_user_by_email(session, email) is not None:
        raise Conflict("email already registered")
    user = User(
        email=email.lower().strip(),
        display_name=display_name or email.split("@")[0],
        password_hash=hash_password(password),
    )
    session.add(user)
    session.flush()
    ensure_roles(session)
    # The first ever user is an admin; otherwise use the requested role (admin-created).
    total_users = len(list(session.scalars(select(User))))
    assigned = str(Role.ADMIN) if total_users == 1 else role
    assign_role(session, user.id, assigned, granted_by=actor_id)
    record_audit(
        session,
        action="auth.register",
        actor_type="user" if actor_id else "system",
        actor_id=actor_id or user.id,
        actor_label=user.email,
        subject_type="user",
        subject_id=user.id,
        decision="created",
        detail={"role": assigned},
    )
    session.flush()
    return user


def authenticate(
    session: Session, *, email: str, password: str, ip: str = "", user_agent: str = ""
) -> dict[str, Any]:
    user = get_user_by_email(session, email)
    if user is None or not verify_password(password, user.password_hash):
        record_audit(
            session,
            action="auth.login",
            actor_type="user",
            actor_label=email[:200],
            decision="denied",
            reason="invalid credentials",
            ip=ip,
        )
        raise AuthenticationError("invalid email or password")
    if not user.is_active:
        raise AuthenticationError("account is disabled")
    settings = get_settings()
    token = new_token("pf_sess")
    expires = utcnow() + timedelta(hours=settings.session_ttl_hours)
    session.add(
        AuthSession(
            user_id=user.id,
            token_hash=token_digest(token),
            issued_at=utcnow(),
            expires_at=expires,
            user_agent=user_agent[:240],
            ip=ip,
        )
    )
    user.last_login_at = utcnow()
    record_audit(
        session,
        action="auth.login",
        actor_type="user",
        actor_id=user.id,
        actor_label=user.email,
        subject_type="user",
        subject_id=user.id,
        decision="allowed",
        ip=ip,
    )
    session.flush()
    roles = roles_for_user(session, user.id)
    return {
        "token": token,
        "token_type": "bearer",
        "expires_at": expires.isoformat(),
        "user": user_payload(user, roles),
    }


def logout(session: Session, *, token: str) -> dict[str, Any]:
    digest = token_digest(token)
    row = session.scalar(select(AuthSession).where(AuthSession.token_hash == digest))
    if row is None:
        raise AuthenticationError("invalid session token")
    row.revoked_at = utcnow()
    session.flush()
    record_audit(
        session,
        action="auth.logout",
        actor_type="user",
        actor_id=row.user_id,
        decision="revoked",
        subject_type="session",
        subject_id=row.id,
    )
    return {"status": "logged_out"}


def user_payload(user: User, roles: list[str]) -> dict[str, Any]:
    permissions = permissions_for(roles)
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "is_active": user.is_active,
        "is_service_account": user.is_service_account,
        "roles": roles,
        "permissions": sorted(permissions),
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


def create_api_key(
    session: Session,
    *,
    user_id: str,
    name: str,
    scopes: list[str] | None = None,
    expires_in_days: int | None = None,
    actor_id: str | None = None,
) -> dict[str, Any]:
    user = session.get(User, user_id)
    if user is None:
        raise NotFound("user not found")
    raw = new_token("pf_key")
    row = ApiKey(
        user_id=user.id,
        name=name[:120],
        prefix=raw[:10],
        key_hash=token_digest(raw),
        scopes=scopes or [],
        expires_at=utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
    )
    session.add(row)
    session.flush()
    record_audit(
        session,
        action="auth.api_key_created",
        actor_type="user",
        actor_id=actor_id or user.id,
        subject_type="api_key",
        subject_id=row.id,
        decision="created",
        detail={"name": row.name, "scopes": row.scopes},
    )
    return {
        "id": row.id,
        "name": row.name,
        "key": raw,  # shown exactly once
        "prefix": row.prefix,
        "scopes": row.scopes,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
    }


def revoke_api_key(session: Session, *, key_id: str, actor_id: str) -> dict[str, Any]:
    row = session.get(ApiKey, key_id)
    if row is None:
        raise NotFound("api key not found")
    row.revoked_at = utcnow()
    session.flush()
    record_audit(
        session,
        action="auth.api_key_revoked",
        actor_id=actor_id,
        subject_type="api_key",
        subject_id=row.id,
        decision="revoked",
    )
    return {"id": row.id, "status": "revoked"}


def list_users(session: Session) -> list[dict[str, Any]]:
    return [
        user_payload(user, roles_for_user(session, user.id))
        for user in session.scalars(select(User))
    ]


def update_user_role(session: Session, *, user_id: str, role: str, actor_id: str) -> dict[str, Any]:
    if role not in {str(item) for item in Role}:
        raise ValidationError(f"unknown role '{role}'")
    user = session.get(User, user_id)
    if user is None:
        raise NotFound("user not found")
    ensure_roles(session)
    assign_role(session, user.id, role, granted_by=actor_id)
    record_audit(
        session,
        action="auth.role_assigned",
        actor_id=actor_id,
        subject_type="user",
        subject_id=user.id,
        decision="granted",
        detail={"role": role},
    )
    return {"user_id": user.id, "roles": roles_for_user(session, user.id)}


def permission_matrix() -> dict[str, list[str]]:
    return {
        str(role): sorted(str(permission) for permission in permissions)
        for role, permissions in ROLE_PERMISSIONS.items()
    }


__all__ = [
    "Permission",
    "authenticate",
    "bootstrap",
    "create_api_key",
    "ensure_roles",
    "list_users",
    "logout",
    "permission_matrix",
    "register_user",
    "revoke_api_key",
    "update_user_role",
    "user_payload",
]
