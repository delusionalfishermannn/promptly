"""Authentication, API keys and user administration."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from backend.deps import RequestContext, rate_limit, require
from backend.schemas import (
    ApiKeyRequest,
    LoginRequest,
    RegisterRequest,
    RoleAssignRequest,
    TokenResponse,
)
from backend.services import auth as auth_service
from core.rbac import Permission
from db.repository import roles_for_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201)
def register(
    payload: RegisterRequest,
    context: Annotated[RequestContext, Depends(rate_limit)],
) -> dict[str, Any]:
    """Self-service registration (viewer role). The first account becomes admin."""
    user = auth_service.register_user(
        context.session,
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
    )
    roles = roles_for_user(context.session, user.id)
    return auth_service.user_payload(user, roles)


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    request: Request,
    context: Annotated[RequestContext, Depends(rate_limit)],
) -> TokenResponse:
    data = auth_service.authenticate(
        context.session,
        email=payload.email,
        password=payload.password,
        ip=context.ip,
        user_agent=context.user_agent,
    )
    return TokenResponse(**data)


@router.post("/logout")
def logout(
    request: Request,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    token = (request.headers.get("authorization") or "")[7:].strip()
    return auth_service.logout(context.session, token=token)


@router.get("/me")
def me(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    assert context.user is not None
    return auth_service.user_payload(context.user, context.roles)


@router.get("/permissions")
def permissions(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    return {
        "me": {"roles": context.roles, "permissions": sorted(context.permissions)},
        "matrix": auth_service.permission_matrix(),
    }


@router.post("/api-keys", status_code=201)
def create_api_key(
    payload: ApiKeyRequest,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    assert context.user is not None
    data = auth_service.create_api_key(
        context.session,
        user_id=context.user.id,
        name=payload.name,
        scopes=payload.scopes,
        expires_in_days=payload.expires_in_days,
        actor_id=context.actor_id,
    )
    context.audit(
        "auth.api_key_created", subject_type="api_key", subject_id=data["id"], decision="created"
    )
    return data


@router.delete("/api-keys/{key_id}")
def revoke_api_key(
    key_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    return auth_service.revoke_api_key(context.session, key_id=key_id, actor_id=context.actor_id)


admin_router = APIRouter(prefix="/admin", tags=["admin"])


@admin_router.get("/users")
def list_users(
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_USERS))],
) -> dict[str, Any]:
    users = auth_service.list_users(context.session)
    return {"items": users, "count": len(users)}


@admin_router.post("/users/{user_id}/role")
def assign_role(
    user_id: str,
    payload: RoleAssignRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_USERS))],
) -> dict[str, Any]:
    return auth_service.update_user_role(
        context.session, user_id=user_id, role=payload.role, actor_id=context.actor_id
    )


@admin_router.post("/api-keys/{user_id}")
def admin_create_api_key(
    user_id: str,
    payload: ApiKeyRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_USERS))],
) -> dict[str, Any]:
    return auth_service.create_api_key(
        context.session,
        user_id=user_id,
        name=payload.name,
        scopes=payload.scopes,
        expires_in_days=payload.expires_in_days,
        actor_id=context.actor_id,
    )
