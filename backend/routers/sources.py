"""Data sources, connector catalogue and uploads."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import RunSourceRequest, SourceCreate
from backend.services import ingestion
from core.errors import NotFound, ValidationError
from core.rbac import Permission
from db.models import ConnectorRun, DataSource

router = APIRouter(tags=["sources"])


@router.get("/connectors")
def connectors(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    items = ingestion.connector_catalog()
    return {"items": items, "count": len(items)}


@router.post("/sources", status_code=201)
def create_source(
    payload: SourceCreate,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_SOURCES))],
) -> dict[str, Any]:
    source = ingestion.create_source(
        context.session,
        name=payload.name,
        source_type=payload.source_type,
        permission_basis=payload.permission_basis,
        permission_notes=payload.permission_notes,
        config=payload.config,
        license=payload.license,
        contains_personal_information=payload.contains_personal_information,
        retention_days=payload.retention_days,
        schedule_cron=payload.schedule_cron,
        enabled=payload.enabled,
        actor_id=context.actor_id,
    )
    return ingestion.source_summary(context.session, source)


@router.get("/sources")
def list_sources(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    rows = list(context.session.scalars(select(DataSource).order_by(DataSource.created_at.desc())))
    items = [ingestion.source_summary(context.session, row) for row in rows]
    return {"items": items, "count": len(items)}


@router.get("/sources/{source_id}")
def get_source(
    source_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    source = context.session.get(DataSource, source_id)
    if source is None:
        raise NotFound(f"source {source_id} not found")
    return ingestion.source_summary(context.session, source)


@router.post("/sources/{source_id}/run")
def run_source(
    source_id: str,
    payload: RunSourceRequest,
    context: Annotated[RequestContext, Depends(require(Permission.RUN_PIPELINES))],
) -> dict[str, Any]:
    result = ingestion.run_source(
        context.session, source_id=source_id, actor_id=context.actor_id, limit=payload.limit
    )
    return result.to_dict()


@router.post("/sources/{source_id}/upload")
async def upload(
    source_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_SOURCES))],
    file: UploadFile = File(...),
) -> dict[str, Any]:
    content = await file.read()
    if not content:
        raise ValidationError("empty upload")
    result = ingestion.ingest_upload(
        context.session,
        source_id=source_id,
        content=content,
        actor_id=context.actor_id,
        filename=file.filename or "upload.csv",
    )
    return result.to_dict()


@router.get("/runs")
def list_runs(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
    source_id: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    stmt = select(ConnectorRun).order_by(ConnectorRun.created_at.desc()).limit(limit)
    if source_id:
        stmt = stmt.where(ConnectorRun.source_id == source_id)
    rows = list(context.session.scalars(stmt))
    items = [
        {
            "run_id": row.id,
            "source_id": row.source_id,
            "status": row.status,
            "rows_in": row.rows_in,
            "rows_out": row.rows_out,
            "rows_rejected": row.rows_rejected,
            "error": row.error,
            "stats": row.stats,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}
