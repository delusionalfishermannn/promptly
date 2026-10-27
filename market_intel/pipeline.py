"""The discovery pipeline: embed → cluster → topics → score → opportunity cards.

This is the "no external data required" core of PromptFoundry: given only uploaded
exports (or the synthetic demo fixture) it produces ranked, evidenced opportunities.

Idempotent by design:
* signals already embedded are not re-embedded;
* a topic+segment opportunity is updated in place rather than duplicated;
* every run writes a cluster report artifact and an audit entry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.artifacts import get_store
from core.clock import utcnow
from core.config import get_settings
from core.constants import OpportunityStatus
from db.models import (
    EvidenceItem,
    NormalizedSignal,
    Opportunity,
    OpportunityScore,
    PromptProduct,
    SaleEvent,
    Topic,
)
from db.repository import flag_enabled, raise_alert, record_artifact, record_audit
from llm.embeddings import get_embedder
from market_intel.cards import build_card, render_card_markdown
from market_intel.cluster import cluster_vectors
from market_intel.scoring import rank_opportunities, score_opportunity
from market_intel.topics import (
    classify_regulated_domain,
    demand_score,
    dominant_segment,
    upsert_topics,
)
from observability.logging import get_logger
from observability.tracing import span

log = get_logger("market_intel.pipeline")


@dataclass
class DiscoveryResult:
    signals_considered: int = 0
    embeddings_created: int = 0
    clusters_found: int = 0
    noise_ratio: float = 0.0
    topics_created: int = 0
    topics_updated: int = 0
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    cluster_report_uri: str | None = None
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signals_considered": self.signals_considered,
            "embeddings_created": self.embeddings_created,
            "clusters_found": self.clusters_found,
            "noise_ratio": round(self.noise_ratio, 4),
            "topics_created": self.topics_created,
            "topics_updated": self.topics_updated,
            "opportunities": self.opportunities,
            "cluster_report_uri": self.cluster_report_uri,
            "duration_ms": round(self.duration_ms, 2),
            "warnings": self.warnings,
        }


def _catalog_performance(session: Session) -> list[dict[str, Any]]:
    """Internal benchmarks: our own products + realised sales (attribution-safe)."""
    products = list(session.scalars(select(PromptProduct)))
    out: list[dict[str, Any]] = []
    for product in products:
        sales = list(session.scalars(select(SaleEvent).where(SaleEvent.product_id == product.id)))
        revenue = sum(sale.amount_micros for sale in sales)
        out.append(
            {
                "id": product.id,
                "name": product.name,
                "niche": product.niche,
                "tags": product.tags,
                "price_aud": product.price_aud_micros / 1_000_000,
                "sales": len(sales),
                "revenue_aud": revenue / 1_000_000,
                "conversion_rate": 0.0,  # filled below when exposures exist
            }
        )
    return out


def _embed_missing(session: Session, signals: list[NormalizedSignal]) -> tuple[int, str]:
    embedder = get_embedder()
    missing = [s for s in signals if not s.embedding]
    if not missing:
        return 0, embedder.name
    vectors = embedder.embed([s.text for s in missing])
    for signal, vector in zip(missing, vectors, strict=True):
        signal.embedding = vector
        signal.embedding_model = embedder.name
    session.flush()
    return len(missing), embedder.name


def _permitted_density(session: Session) -> float | None:
    """Operator-supplied category density from an allowed export (never scraped)."""
    from db.models import DataSource

    source = session.scalar(
        select(DataSource).where(
            DataSource.source_type == "manual_json", DataSource.enabled.is_(True)
        )
    )
    if source is None:
        return None
    density = (source.config or {}).get("category_density")
    try:
        return float(density) if density is not None else None
    except (TypeError, ValueError):
        return None


def _evidence_for(session: Session, topic_id: str, limit: int = 6) -> list[dict[str, Any]]:
    signals = list(
        session.scalars(
            select(NormalizedSignal)
            .where(NormalizedSignal.topic_id == topic_id)
            .order_by(NormalizedSignal.weight.desc())
            .limit(limit)
        )
    )
    return [
        {
            "signal_id": signal.id,
            "source_id": signal.source_id,
            "quote": signal.text[:400],
            "weight": round(signal.weight, 3),
            "kind": signal.kind,
            "intent_strength": round(signal.intent_strength, 3),
            "collected_at": signal.collected_at.isoformat(),
        }
        for signal in signals
    ]


def run_discovery(
    session: Session,
    *,
    actor_id: str = "system",
    max_signals: int = 3000,
    min_cluster_size: int | None = None,
    eps: float | None = None,
    create_opportunities: bool = True,
    product_weights: dict[str, float] | None = None,
) -> DiscoveryResult:
    started = time.perf_counter()
    settings = get_settings()
    if not flag_enabled(session, "pipeline.enabled"):
        raise RuntimeError("pipeline.enabled flag is off — discovery disabled by operator")

    result = DiscoveryResult()
    with span("pipeline.discovery", max_signals=max_signals):
        signals = list(
            session.scalars(
                select(NormalizedSignal)
                .order_by(NormalizedSignal.collected_at.desc())
                .limit(max_signals)
            )
        )
        signals.reverse()  # chronological for velocity maths
        result.signals_considered = len(signals)
        if len(signals) < max(min_cluster_size or settings.cluster_min_size, 3):
            result.warnings.append(
                f"only {len(signals)} signals — need at least {settings.cluster_min_size} to cluster"
            )
            result.duration_ms = (time.perf_counter() - started) * 1000
            record_audit(
                session,
                action="discovery.run",
                actor_id=actor_id,
                subject_type="pipeline",
                decision="skipped",
                reason=result.warnings[-1],
            )
            return result

        result.embeddings_created, embedder_name = _embed_missing(session, signals)
        vectors = [s.embedding or [] for s in signals]
        cluster_result = cluster_vectors(
            vectors,
            eps=eps if eps is not None else settings.cluster_eps,
            min_size=min_cluster_size or settings.cluster_min_size,
            texts=[s.text for s in signals],
        )
        result.clusters_found = cluster_result.n_clusters
        result.noise_ratio = len(cluster_result.noise_indices) / max(len(signals), 1)
        if cluster_result.params.get("warning"):
            result.warnings.append(
                "high noise ratio: consider raising PF_CLUSTER_EPS or lowering PF_CLUSTER_MIN_SIZE"
            )

        outcomes = upsert_topics(
            session,
            cluster_result=cluster_result,
            signals=signals,
            embedder_name=embedder_name,
        )
        result.topics_created = sum(1 for outcome in outcomes if outcome.created)
        result.topics_updated = sum(1 for outcome in outcomes if not outcome.created)

        catalog = _catalog_performance(session)
        permitted_density = _permitted_density(session)
        scored: list[dict[str, Any]] = []

        for outcome in outcomes:
            topic = session.get(Topic, outcome.topic_id)
            if topic is None:
                continue
            member_signals = [s for s in signals if s.topic_id == topic.id] or list(
                session.scalars(
                    select(NormalizedSignal).where(NormalizedSignal.topic_id == topic.id)
                )
            )
            components = _topic_components(member_signals)
            segment = dominant_segment(member_signals)
            regulated_domain = classify_regulated_domain([s.text for s in member_signals])
            breakdown = score_opportunity(
                topic_id=topic.id,
                topic_name=topic.name,
                segment=segment,
                keywords=list(topic.keywords or []),
                signals=[_signal_row(s) for s in member_signals],
                components=components,
                regulated_domain=regulated_domain,
                catalog=catalog,
                permitted_density=permitted_density,
                weights=product_weights,
            )

            opportunity = session.scalar(
                select(Opportunity).where(
                    Opportunity.topic_id == topic.id,
                    Opportunity.buyer_segment == segment,
                    Opportunity.status.notin_(
                        [str(OpportunityStatus.ARCHIVED), str(OpportunityStatus.REJECTED)]
                    ),
                )
            )
            if opportunity is None:
                opportunity = Opportunity(
                    topic_id=topic.id,
                    title=topic.name,
                    jtbd=topic.jtbd,
                    buyer_segment=segment,
                    status=str(OpportunityStatus.SCORED),
                    rationale="; ".join(breakdown.reasons[:3]),
                    positioning=f"Reusable, model-agnostic prompt kit for {topic.name.lower()}",
                    risk_tier=breakdown.risk_tier,
                    regulated_domain=breakdown.regulated_domain,
                    price_hypothesis_micros=breakdown.price_hypothesis_micros,
                    composite_score=breakdown.composite,
                    confidence=breakdown.confidence,
                    created_by=actor_id,
                )
                session.add(opportunity)
                session.flush()
            else:
                opportunity.title = topic.name
                opportunity.jtbd = topic.jtbd
                opportunity.risk_tier = breakdown.risk_tier
                opportunity.regulated_domain = breakdown.regulated_domain
                opportunity.price_hypothesis_micros = breakdown.price_hypothesis_micros
                opportunity.composite_score = breakdown.composite
                opportunity.confidence = breakdown.confidence
                opportunity.rationale = "; ".join(breakdown.reasons[:3])

            session.add(
                OpportunityScore(
                    opportunity_id=opportunity.id,
                    demand=breakdown.demand,
                    supply_proxy=breakdown.supply_proxy,
                    monetization=breakdown.monetization,
                    build_cost=breakdown.build_cost,
                    risk=breakdown.risk,
                    confidence=breakdown.confidence,
                    composite=breakdown.composite,
                    composite_lower=breakdown.composite_lower,
                    composite_upper=breakdown.composite_upper,
                    method=breakdown.method,
                    weights=breakdown.weights,
                    inputs=breakdown.inputs,
                )
            )

            evidence = _evidence_for(session, topic.id)
            session.query(EvidenceItem).filter(
                EvidenceItem.opportunity_id == opportunity.id
            ).delete()
            for item in evidence:
                session.add(
                    EvidenceItem(
                        opportunity_id=opportunity.id,
                        signal_id=item.get("signal_id"),
                        source_id=item.get("source_id"),
                        kind="signal",
                        quote=item["quote"],
                        weight=item["weight"],
                    )
                )
            session.flush()

            card = build_card(
                opportunity_id=opportunity.id,
                topic={
                    "id": topic.id,
                    "name": topic.name,
                    "slug": topic.slug,
                    "title": topic.name,
                    "jtbd": topic.jtbd,
                    "keywords": list(topic.keywords or []),
                    "size": topic.size,
                    "drift_score": topic.drift_score,
                },
                segment=segment,
                score=breakdown,
                evidence=evidence,
                status=opportunity.status,
            )
            store = get_store()
            ref = store.put_json(card, kind="opportunity_card", subject_id=opportunity.id)
            store.put_text(
                render_card_markdown(card),
                kind="opportunity_card",
                ext="md",
                subject_id=opportunity.id,
            )
            record_artifact(
                session,
                ref,
                kind="opportunity_card",
                subject_type="opportunity",
                subject_id=opportunity.id,
                created_by=actor_id,
            )
            opportunity.artifact_uri = ref.uri
            scored.append(
                {
                    **card["score"],
                    "opportunity_id": opportunity.id,
                    "topic_id": topic.id,
                    "title": topic.name,
                    "buyer_segment": segment,
                    "recommendation": card["recommendation"],
                    "regulated_domain": regulated_domain,
                    "artifact_uri": ref.uri,
                }
            )

        report = {
            "generated_at": utcnow().isoformat(),
            "embedder": embedder_name,
            "cluster": cluster_result.to_dict(),
            "topics": [
                {
                    "topic_id": outcome.topic_id,
                    "name": outcome.topic_name,
                    "created": outcome.created,
                    "size": outcome.size,
                    "drift": outcome.drift,
                    "demand": outcome.demand,
                    "velocity": outcome.velocity,
                }
                for outcome in outcomes
            ],
            "opportunities": scored,
            "catalog_benchmarks": catalog,
            "permitted_density": permitted_density,
        }
        ref = get_store().put_json(report, kind="cluster_report", subject_id="latest")
        record_artifact(
            session, ref, kind="cluster_report", subject_type="pipeline", created_by=actor_id
        )
        result.cluster_report_uri = ref.uri
        result.opportunities = rank_opportunities(scored)
        session.flush()

        if result.noise_ratio > 0.7:
            raise_alert(
                session,
                kind="clustering.noise",
                message=f"{result.noise_ratio:.0%} of signals were noise in the last discovery run",
                subject_type="pipeline",
                detail={"params": cluster_result.params},
            )
        record_audit(
            session,
            action="discovery.run",
            actor_id=actor_id,
            subject_type="pipeline",
            decision="completed",
            detail={
                "signals": result.signals_considered,
                "clusters": result.clusters_found,
                "topics_created": result.topics_created,
                "opportunities": len(scored),
            },
        )
    result.duration_ms = (time.perf_counter() - started) * 1000
    return result


def _signal_row(signal: NormalizedSignal) -> dict[str, Any]:
    return {
        "id": signal.id,
        "weight": signal.weight,
        "intent_strength": signal.intent_strength,
        "pain": signal.pain,
        "recency_decay": signal.recency_decay,
        "quality": signal.quality,
        "urgency": signal.urgency,
        "complexity": signal.complexity,
        "velocity": signal.velocity,
        "buyer_segment": signal.buyer_segment,
        "source_id": signal.source_id,
    }


def _topic_components(signals: list[NormalizedSignal]) -> dict[str, float]:
    components = {
        "volume": sum(s.weight for s in signals),
        "velocity": 0.0,
        "intent": _mean([s.intent_strength for s in signals]),
        "recency": _mean([s.recency_decay for s in signals]),
        "pain": _mean([s.pain for s in signals]),
        "quality": _mean([s.quality for s in signals]),
        "complexity": _mean([s.complexity for s in signals]),
        "n": float(len(signals)),
        "n_sources": float(len({s.source_id for s in signals if s.source_id}) or 1),
        "window_days": 28.0,
    }
    # Velocity from the topic's own trend history when available.
    if len(signals) >= 4:
        ordered = sorted(signals, key=lambda s: s.collected_at)
        half = len(ordered) // 2
        older = sum(s.weight for s in ordered[:half]) or 0.0
        newer = sum(s.weight for s in ordered[half:]) or 0.0
        components["velocity"] = round(((newer - older) / older) if older else 1.0, 4)
    return components


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def current_demand(session: Session, topic_id: str) -> float:
    signals = list(
        session.scalars(select(NormalizedSignal).where(NormalizedSignal.topic_id == topic_id))
    )
    return demand_score(_topic_components(signals))
