"""Observability: budgets, costs, audit trail, flags (kill switches), alerts, traces."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import FlagUpdate
from core.budget import budget_status, spend_by_day, spend_summary
from core.config import get_settings
from core.errors import NotFound
from core.rbac import Permission
from db.models import Alert, Artifact, AuditLog, CostEvent, RuntimeFlag
from db.repository import audit_query, get_flag, set_flag
from observability.metrics import render_metrics
from observability.tracing import recent_spans

router = APIRouter(tags=["ops"])


@router.get("/ops/budget")
def budget(
    context: Annotated[RequestContext, Depends(require(Permission.READ_COSTS))],
) -> dict[str, Any]:
    status = budget_status(context.session)
    settings = get_settings()
    return {
        **status.to_dict(),
        "utilisation": status.ratio,
        "enforced": settings.budget_enforced,
        "alert_ratio": settings.budget_alert_ratio,
        "daily_remaining_aud": round(status.daily_remaining_aud, 2),
        "infra_limit_aud": round(settings.budget_weekly_infra_micros / 1_000_000, 2),
        "currency": "AUD",
    }


@router.get("/ops/costs")
def costs(
    context: Annotated[RequestContext, Depends(require(Permission.READ_COSTS))],
    days: Annotated[int, Query(ge=1, le=90)] = 7,
) -> dict[str, Any]:
    return {
        "window_days": days,
        "budget": budget_status(context.session).to_dict(),
        "by_category": spend_summary(context.session, days=days),
        "by_day": spend_by_day(context.session, days=days),
    }


@router.get("/ops/costs/events")
def cost_events(
    context: Annotated[RequestContext, Depends(require(Permission.READ_COSTS))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    rows = list(
        context.session.scalars(
            select(CostEvent).order_by(CostEvent.created_at.desc()).limit(limit)
        )
    )
    items = [
        {
            "id": row.id,
            "category": row.category,
            "provider": row.provider,
            "model": row.model,
            "purpose": row.purpose,
            "amount_aud": round(row.amount_micros / 1_000_000, 6),
            "subject": {"type": row.subject_type, "id": row.subject_id},
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.get("/audit")
def audit(
    context: Annotated[RequestContext, Depends(require(Permission.READ_AUDIT))],
    action: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    rows = audit_query(
        context.session,
        action=action,
        subject_type=subject_type,
        subject_id=subject_id,
        limit=limit,
        offset=offset,
    )
    items = [
        {
            "id": row.id,
            "action": row.action,
            "actor": {"type": row.actor_type, "id": row.actor_id, "label": row.actor_label},
            "subject": {"type": row.subject_type, "id": row.subject_id},
            "decision": row.decision,
            "reason": row.reason,
            "detail": row.detail,
            "request_id": row.request_id,
            "trace_id": row.trace_id,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
    return {"items": items, "count": len(items), "limit": limit, "offset": offset}


@router.get("/ops/flags")
def flags(
    context: Annotated[RequestContext, Depends(require(Permission.READ_COSTS))],
) -> dict[str, Any]:
    rows = list(context.session.scalars(select(RuntimeFlag)))
    return {
        "items": [
            {
                "key": row.key,
                "value": row.value,
                "description": row.description,
                "updated_by": row.updated_by,
                "updated_at": row.updated_at.isoformat(),
            }
            for row in rows
        ],
        "count": len(rows),
    }


@router.put("/ops/flags/{key}")
def update_flag(
    key: str,
    payload: FlagUpdate,
    context: Annotated[RequestContext, Depends(require(Permission.ADMIN))],
) -> dict[str, Any]:
    if key not in {
        "pipeline.enabled",
        "generation.enabled",
        "evaluation.enabled",
        "publishing.auto",
        "publishing.kill_switch",
        "llm.provider",
    }:
        raise NotFound(f"unknown flag '{key}'")
    previous = get_flag(context.session, key)
    flag = set_flag(context.session, key, payload.value, updated_by=context.actor_id)
    context.audit(
        "ops.flag_updated",
        subject_type="runtime_flag",
        subject_id=flag.key,
        decision="updated",
        detail={"previous": previous, "new": payload.value},
    )
    return {"key": flag.key, "value": flag.value, "previous": previous}


@router.get("/ops/alerts")
def alerts(
    context: Annotated[RequestContext, Depends(require(Permission.READ_COSTS))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    unacknowledged_only: bool = True,
) -> dict[str, Any]:
    stmt = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
    if unacknowledged_only:
        stmt = stmt.where(Alert.acknowledged_at.is_(None))
    rows = list(context.session.scalars(stmt))
    items = [
        {
            "id": row.id,
            "kind": row.kind,
            "severity": row.severity,
            "message": row.message,
            "subject": {"type": row.subject_type, "id": row.subject_id},
            "detail": row.detail,
            "acknowledged_at": row.acknowledged_at.isoformat() if row.acknowledged_at else None,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.post("/ops/alerts/{alert_id}/ack")
def acknowledge(
    alert_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_BUDGETS))],
) -> dict[str, Any]:
    from core.clock import utcnow

    alert = context.session.get(Alert, alert_id)
    if alert is None:
        raise NotFound(f"alert {alert_id} not found")
    alert.acknowledged_at = utcnow()
    context.audit(
        "ops.alert_acknowledged", subject_type="alert", subject_id=alert.id, decision="acknowledged"
    )
    return {"id": alert.id, "acknowledged_at": alert.acknowledged_at.isoformat()}


@router.get("/artifacts/{subject_id}")
def artifacts(
    subject_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    rows = list(
        context.session.scalars(
            select(Artifact)
            .where(Artifact.subject_id == subject_id)
            .order_by(Artifact.created_at.desc())
        )
    )
    items = [
        {
            "id": row.id,
            "kind": row.kind,
            "uri": row.uri,
            "sha256": row.sha256,
            "size_bytes": row.size_bytes,
            "signature": row.signature,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.get("/subjects/{subject_id}/audit")
def subject_audit(
    subject_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_AUDIT))],
) -> dict[str, Any]:
    rows = audit_query(context.session, subject_id=subject_id, limit=100)
    return {
        "items": [
            {
                "action": row.action,
                "decision": row.decision,
                "actor_id": row.actor_id,
                "detail": row.detail,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ],
        "count": len(rows),
    }


@router.get("/ops/traces")
def traces(
    context: Annotated[RequestContext, Depends(require(Permission.READ_AUDIT))],
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    spans = recent_spans(limit=limit)
    return {"items": spans, "count": len(spans)}


@router.get("/ops/audit/export")
def export_audit(
    context: Annotated[RequestContext, Depends(require(Permission.READ_AUDIT))],
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> Any:
    """CSV export for compliance reviews (AU Privacy Act access requests)."""
    import csv
    import io

    rows = list(
        context.session.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit))
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "created_at",
            "action",
            "actor_type",
            "actor_id",
            "subject_type",
            "subject_id",
            "decision",
            "reason",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.created_at.isoformat(),
                row.action,
                row.actor_type,
                row.actor_id,
                row.subject_type,
                row.subject_id,
                row.decision,
                row.reason[:300],
            ]
        )
    return Response(content=buffer.getvalue(), media_type="text/csv")


def metrics_response() -> Response:
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")
