"""Online experiments, conversions/refunds, first-party storefront events."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import (
    AssignRequest,
    ConversionRequest,
    ExperimentCreate,
    ExperimentDecisionRequest,
    RefundRequest,
    StopExperimentRequest,
    StorefrontEventRequest,
    SupportTicketRequest,
)
from core.constants import EventType
from core.errors import NotFound
from core.rbac import Permission
from db.models import Experiment, StorefrontEvent, SupportTicket
from experimentation import service as experiments
from market_intel.scoring import rank_opportunities

router = APIRouter(tags=["experiments"])


@router.post("/experiments", status_code=201)
def create_experiment(
    payload: ExperimentCreate,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_EXPERIMENTS))],
) -> dict[str, Any]:
    return experiments.create_experiment(
        context.session,
        product_id=payload.product_id,
        name=payload.name,
        arms=[arm.model_dump() for arm in payload.arms],
        hypothesis=payload.hypothesis,
        kind=payload.kind,
        metric_primary=payload.metric_primary,
        traffic_fraction=payload.traffic_fraction,
        min_sample=payload.min_sample,
        alpha=payload.alpha,
        guardrails=payload.guardrails,
        actor_id=context.actor_id,
    )


@router.get("/experiments")
def list_experiments(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    rows = list(
        context.session.scalars(
            select(Experiment).order_by(Experiment.created_at.desc()).limit(limit)
        )
    )
    items = [experiments.experiment_summary(context.session, row.id) for row in rows]
    return {"items": items, "count": len(items)}


@router.get("/experiments/{experiment_id}")
def get_experiment(
    experiment_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    return experiments.experiment_summary(context.session, experiment_id)


@router.post("/experiments/{experiment_id}/assign")
def assign(
    experiment_id: str,
    payload: AssignRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_EXPERIMENTS))],
) -> dict[str, Any]:
    assignment = experiments.assign(
        context.session,
        experiment_id=experiment_id,
        subject_key=payload.subject_key,
        channel=payload.channel,
        force_arm=payload.force_arm,
        actor_id=context.actor_id,
    )
    if assignment is None:
        return {"assigned": False, "reason": "outside traffic rollout"}
    return {"assigned": True, **assignment.to_dict()}


@router.post("/experiments/{experiment_id}/conversions")
def conversion(
    experiment_id: str,
    payload: ConversionRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_EXPERIMENTS))],
) -> dict[str, Any]:
    exposure = None
    from db.models import Exposure

    exposure = context.session.get(Exposure, payload.exposure_id)
    if exposure is None or exposure.experiment_id != experiment_id:
        raise NotFound("exposure does not belong to this experiment")
    value_micros = int(round(payload.value_aud * 1_000_000))
    return experiments.record_conversion(
        context.session,
        exposure_id=payload.exposure_id,
        event_type=payload.event_type,
        value_micros=value_micros,
        order_ref=payload.order_ref,
        meta=payload.meta,
    )


@router.post("/refunds")
def refund(
    payload: RefundRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_EXPERIMENTS))],
) -> dict[str, Any]:
    return experiments.record_refund(
        context.session,
        sale_id=payload.sale_id,
        amount_micros=int(round(payload.amount_aud * 1_000_000)),
        reason_code=payload.reason_code,
        note=payload.note,
    )


@router.get("/experiments/{experiment_id}/analysis")
def analysis(
    experiment_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
) -> dict[str, Any]:
    return experiments.analyse(context.session, experiment_id, actor_id=context.actor_id)


@router.post("/experiments/{experiment_id}/decide")
def decide(
    experiment_id: str,
    payload: ExperimentDecisionRequest,
    context: Annotated[RequestContext, Depends(require(Permission.PUBLISH))],
) -> dict[str, Any]:
    return experiments.apply_decision(
        context.session,
        experiment_id=experiment_id,
        actor_id=context.actor_id,
        promote_winning_arm=payload.promote_winning_arm,
    )


@router.post("/experiments/{experiment_id}/stop")
def stop(
    experiment_id: str,
    payload: StopExperimentRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_EXPERIMENTS))],
) -> dict[str, Any]:
    return experiments.stop_experiment(
        context.session,
        experiment_id=experiment_id,
        reason=payload.reason,
        actor_id=context.actor_id,
    )


@router.post("/storefront/events", status_code=201)
def storefront_event(
    payload: StorefrontEventRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_SOURCES))],
) -> dict[str, Any]:
    """First-party analytics ingestion (our own storefront only)."""
    event = StorefrontEvent(
        product_id=payload.product_id,
        topic_id=payload.topic_id,
        event_type=payload.event_type,
        path=payload.path[:240],
        query=payload.query[:2000],
        session_key=payload.session_key,
        meta=payload.meta,
    )
    context.session.add(event)
    context.session.flush()
    return {"event_id": event.id, "event_type": event.event_type}


@router.post("/support/tickets", status_code=201)
def support_ticket(
    payload: SupportTicketRequest,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_EXPERIMENTS))],
) -> dict[str, Any]:
    return experiments.record_support_ticket(
        context.session,
        product_id=payload.product_id,
        subject=payload.subject,
        body=payload.body,
        minutes_spent=payload.minutes_spent,
    )


@router.get("/storefront/summary")
def storefront_summary(
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
) -> dict[str, Any]:
    rows = list(context.session.scalars(select(StorefrontEvent).limit(5000)))
    by_type: dict[str, int] = {}
    for row in rows:
        by_type[row.event_type] = by_type.get(row.event_type, 0) + 1
    tickets = list(context.session.scalars(select(SupportTicket)))
    return {
        "events": len(rows),
        "by_type": by_type,
        "support_tickets": len(tickets),
        "conversion_rate": round(by_type.get("purchase", 0) / by_type.get("view", 1), 4)
        if by_type.get("view")
        else None,
        "note": "first-party analytics only; no third-party tracking pixels",
    }


@router.get("/opportunities/ranked")
def ranked(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    """Re-rank current opportunities by the conservative lower-bound rule."""
    from db.models import Opportunity

    rows = list(context.session.scalars(select(Opportunity)))
    items = rank_opportunities(
        [
            {
                "opportunity_id": row.id,
                "title": row.title,
                "composite": row.composite_score,
                "composite_lower": row.composite_score - (1 - row.confidence) * 0.1,
                "demand": 0.0,
                "status": row.status,
            }
            for row in rows
        ]
    )
    return {"items": items[:50], "count": len(items), "method": "lower_bound_conservative"}


def _event_types() -> list[str]:
    return [str(item) for item in EventType]
