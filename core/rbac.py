"""Role-based access control: a small, explicit permission matrix.

Roles are coarse (5) and permissions fine-grained so that a policy change does not
require a schema change. Every router declares the permission it needs via
``Depends(require_permission(Permission.X))``; the check is audited.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    APPROVER = "approver"  # human-in-the-loop: may approve/publish
    OPERATOR = "operator"  # runs pipelines, edits products
    ANALYST = "analyst"  # read-mostly + experiments
    VIEWER = "viewer"


class Permission(StrEnum):
    # read
    READ_CATALOG = "read:catalog"
    READ_EVALS = "read:evals"
    READ_COSTS = "read:costs"
    READ_AUDIT = "read:audit"
    # write
    MANAGE_SOURCES = "write:sources"
    RUN_PIPELINES = "write:pipelines"
    MANAGE_PRODUCTS = "write:products"
    RUN_GENERATIONS = "write:generations"
    RUN_EVALUATIONS = "write:evaluations"
    MANAGE_EXPERIMENTS = "write:experiments"
    # governance
    APPROVE = "governance:approve"
    PUBLISH = "governance:publish"
    # admin
    MANAGE_USERS = "admin:users"
    MANAGE_BUDGETS = "admin:budgets"
    ADMIN = "admin:all"


ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: {Permission.ADMIN},
    Role.APPROVER: {
        Permission.READ_CATALOG,
        Permission.READ_EVALS,
        Permission.READ_COSTS,
        Permission.READ_AUDIT,
        Permission.APPROVE,
        Permission.PUBLISH,
        Permission.MANAGE_PRODUCTS,
    },
    Role.OPERATOR: {
        Permission.READ_CATALOG,
        Permission.READ_EVALS,
        Permission.READ_COSTS,
        Permission.MANAGE_SOURCES,
        Permission.RUN_PIPELINES,
        Permission.MANAGE_PRODUCTS,
        Permission.RUN_GENERATIONS,
        Permission.RUN_EVALUATIONS,
        Permission.MANAGE_EXPERIMENTS,
    },
    Role.ANALYST: {
        Permission.READ_CATALOG,
        Permission.READ_EVALS,
        Permission.READ_COSTS,
        Permission.RUN_EVALUATIONS,
        Permission.MANAGE_EXPERIMENTS,
    },
    Role.VIEWER: {Permission.READ_CATALOG, Permission.READ_EVALS},
}

# An admin implicitly holds every permission; keep the matrix readable by
# expanding it once at import time.
ROLE_PERMISSIONS[Role.ADMIN] = set(Permission) | {Permission.ADMIN}


def permissions_for(roles: list[str]) -> set[str]:
    granted: set[str] = set()
    for name in roles:
        try:
            granted |= {str(p) for p in ROLE_PERMISSIONS[Role(name)]}
        except (KeyError, ValueError):
            continue
    return granted


def has_permission(roles: list[str], permission: Permission | str) -> bool:
    granted = permissions_for(roles)
    return str(Permission.ADMIN) in granted or str(permission) in granted
