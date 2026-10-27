"""Discovery pipeline, topics and opportunities."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import DiscoveryRequest, LaunchRequest
from backend.services import orchestration
from core.constants import OpportunityStatus
from core.errors import NotFound
from core.rbac import Permission
from db.models import EvidenceItem, NormalizedSignal, Opportunity, OpportunityScore, Topic
from market_intel.pipeline import run_discovery
from market_intel.topics import drift_report, topic_demand

router = APIRouter(tags=["discovery"])


@router.post("/pipeline/discover")
def discover(
    payload: DiscoveryRequest,
    context: Annotated[RequestContext, Depends(require(Permission.RUN_PIPELINES))],
) -> dict[str, Any]:
    result = run_discovery(
        context.session,
        actor_id=context.actor_id,
        max_signals=payload.max_signals,
        min_cluster_size=payload.min_cluster_size,
        eps=payload.eps,
    )
    context.audit(
        "pipeline.discover",
        subject_type="pipeline",
        decision=(result.clusters_found > 0 and "clustered") or "empty",
        detail={
            "signals": result.signals_considered,
            "clusters": result.clusters_found,
            "opportunities": len(result.opportunities),
        },
    )
    return result.to_dict()


@router.get("/topics")
def list_topics(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    rows = list(context.session.scalars(select(Topic).order_by(Topic.size.desc()).limit(limit)))
    items = [
        {
            "id": row.id,
            "name": row.name,
            "slug": row.slug,
            "size": row.size,
            "keywords": row.keywords,
            "buyer_segments": row.buyer_segments,
            "drift_score": row.drift_score,
            "jtbd": row.jtbd,
            "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.get("/topics/{topic_id}")
def get_topic(
    topic_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    topic = context.session.get(Topic, topic_id)
    if topic is None:
        raise NotFound(f"topic {topic_id} not found")
    signals = list(
        context.session.scalars(
            select(NormalizedSignal)
            .where(NormalizedSignal.topic_id == topic.id)
            .order_by(NormalizedSignal.weight.desc())
            .limit(20)
        )
    )
    return {
        "id": topic.id,
        "name": topic.name,
        "slug": topic.slug,
        "jtbd": topic.jtbd,
        "keywords": topic.keywords,
        "buyer_segments": topic.buyer_segments,
        "size": topic.size,
        "drift_score": topic.drift_score,
        "demand": topic_demand(context.session, topic.id),
        "trend": drift_report(context.session, topic.id, limit=10),
        "top_signals": [
            {
                "id": signal.id,
                "text": signal.text[:400],
                "kind": signal.kind,
                "intent_strength": signal.intent_strength,
                "buyer_segment": signal.buyer_segment,
                "weight": signal.weight,
                "source_id": signal.source_id,
            }
            for signal in signals
        ],
    }


@router.get("/opportunities")
def list_opportunities(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    status: str | None = None,
) -> dict[str, Any]:
    stmt = select(Opportunity).order_by(Opportunity.composite_score.desc()).limit(limit)
    if status:
        stmt = stmt.where(Opportunity.status == status)
    rows = list(context.session.scalars(stmt))
    items = []
    for row in rows:
        score = context.session.scalar(
            select(OpportunityScore)
            .where(OpportunityScore.opportunity_id == row.id)
            .order_by(OpportunityScore.created_at.desc())
            .limit(1)
        )
        topic = context.session.get(Topic, row.topic_id) if row.topic_id else None
        items.append(
            {
                "id": row.id,
                "title": row.title,
                "topic_id": row.topic_id,
                "niche": topic.niche if topic else "",
                "jtbd": topic.jtbd if topic else "",
                "buyer_segment": row.buyer_segment,
                "status": row.status,
                "composite": row.composite_score,
                "confidence": row.confidence,
                "risk_tier": row.risk_tier,
                "regulated_domain": row.regulated_domain,
                "price_hypothesis_aud": round(row.price_hypothesis_micros / 1_000_000, 2),
                "ci": [score.composite_lower, score.composite_upper] if score else None,
                "artifact_uri": row.artifact_uri,
            }
        )
    return {"items": items, "count": len(items)}


@router.get("/opportunities/{opportunity_id}")
def get_opportunity(
    opportunity_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    opportunity = context.session.get(Opportunity, opportunity_id)
    if opportunity is None:
        raise NotFound(f"opportunity {opportunity_id} not found")
    score = context.session.scalar(
        select(OpportunityScore)
        .where(OpportunityScore.opportunity_id == opportunity.id)
        .order_by(OpportunityScore.created_at.desc())
        .limit(1)
    )
    evidence = list(
        context.session.scalars(
            select(EvidenceItem)
            .where(EvidenceItem.opportunity_id == opportunity.id)
            .order_by(EvidenceItem.weight.desc())
            .limit(12)
        )
    )
    topic = context.session.get(Topic, opportunity.topic_id)
    return {
        "id": opportunity.id,
        "title": opportunity.title,
        "jtbd": opportunity.jtbd,
        "buyer_segment": opportunity.buyer_segment,
        "status": opportunity.status,
        "rationale": opportunity.rationale,
        "positioning": opportunity.positioning,
        "risk_tier": opportunity.risk_tier,
        "regulated_domain": opportunity.regulated_domain,
        "composite": opportunity.composite_score,
        "confidence": opportunity.confidence,
        "price_hypothesis_aud": round(opportunity.price_hypothesis_micros / 1_000_000, 2),
        "artifact_uri": opportunity.artifact_uri,
        "topic": {
            "id": topic.id,
            "name": topic.name,
            "keywords": topic.keywords,
            "jtbd": topic.jtbd,
            "size": topic.size,
        }
        if topic
        else None,
        "score": {
            "demand": score.demand,
            "supply_proxy": score.supply_proxy,
            "monetization": score.monetization,
            "build_cost": score.build_cost,
            "risk": score.risk,
            "confidence": score.confidence,
            "composite": score.composite,
            "ci": [score.composite_lower, score.composite_upper],
            "weights": score.weights,
            "inputs": score.inputs,
        }
        if score
        else None,
        "evidence": [
            {
                "quote": item.quote,
                "weight": item.weight,
                "source_id": item.source_id,
                "kind": item.kind,
            }
            for item in evidence
        ],
    }


@router.post("/opportunities/{opportunity_id}/launch")
def launch(
    opportunity_id: str,
    payload: LaunchRequest,
    context: Annotated[RequestContext, Depends(require(Permission.RUN_GENERATIONS))],
) -> dict[str, Any]:
    result = orchestration.launch_product(
        context.session,
        opportunity_id=opportunity_id,
        actor_id=context.actor_id,
        n_candidates=payload.n_candidates,
        primary_family=payload.primary_family,
        evaluate_holdout=payload.evaluate_holdout,
        run_gate=payload.run_gate,
        price_aud=payload.price_aud,
    )
    return result.to_dict()


@router.post("/opportunities/{opportunity_id}/status")
def set_status(
    opportunity_id: str,
    status: str,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_PRODUCTS))],
) -> dict[str, Any]:
    opportunity = context.session.get(Opportunity, opportunity_id)
    if opportunity is None:
        raise NotFound(f"opportunity {opportunity_id} not found")
    if status not in {str(item) for item in OpportunityStatus}:
        raise NotFound(f"unknown status '{status}'")
    opportunity.status = status
    context.audit(
        "opportunity.status", subject_type="opportunity", subject_id=opportunity.id, decision=status
    )
    return {"id": opportunity.id, "status": opportunity.status}
