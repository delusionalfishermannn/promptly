"""Topic (jobs-to-be-done) naming, upsert and drift tracking.

Clusters are ephemeral; **topics are durable**. On each run we match new centroids
to existing topics by cosine similarity:

* similarity ≥ ``1 - drift_threshold`` → same topic: update centroid with an
  exponential moving average, record drift, add memberships, write a trend snapshot.
* otherwise → new topic.

Drift is defined as ``1 - cos(new_centroid, previous_centroid)`` and is surfaced on
``topics.drift_score`` + every ``trend_snapshots`` row. Sustained drift above the
threshold is an alert signal ("the audience's language changed; re-evaluate").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.clock import utcnow
from core.config import get_settings
from core.ids import slugify
from db.models import (
    NormalizedSignal,
    Topic,
    TopicMembership,
    TrendSnapshot,
)
from llm.embeddings import cosine
from market_intel.cluster import ClusterResult
from market_intel.normalize import WORD_RE

CONCEPT_WORDS = {
    "billing": "billing",
    "email": "email",
    "seo": "SEO content",
    "support": "customer support",
    "career": "career",
    "ads": "advertising",
    "onboarding": "onboarding",
    "product": "product",
    "data": "data",
    "meetings": "meetings",
    "legal": "legal",
    "health": "health",
    "finance": "finance",
}

ARTIFACTS = {
    "email": "email copy",
    "billing": "invoices and reminders",
    "seo": "blog outlines",
    "support": "support replies",
    "career": "resumes and interviews",
    "ads": "ad copy",
    "onboarding": "training material",
    "product": "product specs",
    "data": "analysis summaries",
    "meetings": "meeting notes",
    "legal": "policy drafts",
    "health": "health summaries",
    "finance": "financial summaries",
    "research": "research summaries",
    "sales": "sales follow-ups",
    "ops": "process documentation",
}

# Human-facing product titles. Curated rather than ``.title()``-ed so names read like
# real products ("Invoices & payment reminders", not "Billing Invoices And Reminders").
TOPIC_TITLES = {
    "billing": "Invoices & payment reminders",
    "email": "Outbound email copy",
    "seo": "SEO content outlines",
    "support": "Customer support replies",
    "career": "Résumés & interview prep",
    "ads": "Ad copy & creative briefs",
    "onboarding": "Training & onboarding material",
    "product": "Product specs & acceptance criteria",
    "data": "Analysis & reporting summaries",
    "meetings": "Meeting notes & action items",
    "legal": "Legal document summaries (assistive)",
    "health": "Health information summaries (assistive)",
    "finance": "Finance & bookkeeping summaries",
    "research": "Research & evidence summaries",
    "sales": "Sales follow-ups & proposals",
    "ops": "Process & SOP documentation",
}

SEGMENT_LABEL = {
    "creator": "creators",
    "smb": "small businesses",
    "agency": "agencies",
    "enterprise": "enterprise teams",
    "educator": "educators",
    "student": "students",
    "developer": "developers",
    "unknown": "general buyers",
}


@dataclass
class TopicOutcome:
    topic_id: str
    topic_name: str
    created: bool
    drift: float
    size: int
    signal_ids: list[str]
    centroid: list[float]
    demand: float
    velocity: float


def _dominant_concept(texts: list[str]) -> str:
    from market_intel.normalize import detect_language  # noqa: F401  (kept for symmetry)

    counts: dict[str, int] = {}
    known = set(CONCEPT_WORDS)
    for text in texts:
        for token in WORD_RE.findall(text.lower()):
            if token in known:
                counts[token] = counts.get(token, 0) + 1
    if not counts:
        return "general"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def compose_topic_name(texts: list[str], segment: str) -> tuple[str, str, list[str]]:
    """Deterministic offline naming: human-readable name, JTBD sentence, keywords."""
    concept = _dominant_concept(texts)
    artifact = ARTIFACTS.get(concept, "reusable prompts")
    audience = SEGMENT_LABEL.get(segment, "general buyers")
    if concept == "general":
        # No taxonomy concept dominates: fall back to the strongest distinctive term,
        # and keep the segment as the differentiator so names stay distinguishable.
        ranked = _ranked_terms(texts, 1)
        artifact = f"{ranked[0]} prompts" if ranked else "reusable prompts"
        name = f"{ranked[0].title() if ranked else 'Reusable'} prompts for {audience}"
    else:
        name = f"{TOPIC_TITLES.get(concept, CONCEPT_WORDS[concept].title())} for {audience}"
    singular = {
        "small businesses": "small business",
        "agencies": "agency",
        "creators": "creator",
        "educators": "educator",
        "developers": "developer",
        "students": "student",
        "enterprise teams": "enterprise team",
        "general buyers": "general buyer",
    }.get(audience, audience)
    jtbd = (
        f"When a {singular} needs to produce {artifact} repeatedly, "
        f"they want a reusable, model-agnostic prompt that reliably returns "
        f"structured output they can paste straight into their workflow."
    )
    keywords = _ranked_terms(texts, 6)
    return name[:200], jtbd, keywords


def _ranked_terms(texts: list[str], k: int) -> list[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "for",
        "with",
        "is",
        "are",
        "how",
        "do",
        "i",
        "my",
        "we",
        "you",
        "it",
        "on",
        "that",
        "this",
        "can",
        "what",
        "need",
        "want",
        "best",
        "way",
        "should",
        "which",
        "from",
        "anyone",
        "someone",
    }
    counts: dict[str, int] = {}
    for text in texts:
        tokens = [t for t in WORD_RE.findall(text.lower()) if t not in stop and len(t) > 3]
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def compute_demand_components(
    signals: list[NormalizedSignal], *, window_days: int = 28, now: datetime | None = None
) -> dict[str, float]:
    """Volume, velocity, intent, recency, pain — the raw ingredients of DemandScore."""
    now = now or utcnow()
    recent_cut = now - timedelta(days=window_days / 2)
    volume = sum(s.weight for s in signals)
    recent = [s for s in signals if _aware(s.collected_at) >= recent_cut]
    prior = [s for s in signals if _aware(s.collected_at) < recent_cut]
    recent_volume = sum(s.weight for s in recent)
    prior_volume = sum(s.weight for s in prior)
    velocity = 0.0
    if prior_volume > 0:
        velocity = (recent_volume - prior_volume) / prior_volume
    elif recent_volume > 0:
        velocity = 1.0
    intent = _mean([s.intent_strength for s in signals])
    recency = _mean([s.recency_decay for s in signals])
    pain = _mean([s.pain for s in signals])
    quality = _mean([s.quality for s in signals])
    return {
        "volume": round(volume, 4),
        "recent_volume": round(recent_volume, 4),
        "velocity": round(max(min(velocity, 3.0), -1.0), 4),
        "intent": round(intent, 4),
        "recency": round(recency, 4),
        "pain": round(pain, 4),
        "quality": round(quality, 4),
        "n": float(len(signals)),
        "window_days": float(window_days),
    }


def demand_score(components: dict[str, float]) -> float:
    """Squash unbounded volume with log10; keep the score in [0,1]."""
    volume = components.get("volume", 0.0)
    volume_component = min(math.log10(1 + volume) / math.log10(1 + 100), 1.0)
    velocity = max(min(components.get("velocity", 0.0), 1.5), -0.5)
    velocity_component = (velocity + 0.5) / 2.0
    intent = components.get("intent", 0.0)
    recency = components.get("recency", 0.0)
    quality = components.get("quality", 1.0)
    score = (
        0.40 * volume_component
        + 0.20 * velocity_component
        + 0.22 * intent
        + 0.10 * recency
        + 0.08 * quality
    )
    return round(max(0.0, min(1.0, score)), 4)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=utcnow().tzinfo)


def dominant_segment(signals: list[NormalizedSignal]) -> str:
    scores: dict[str, float] = {}
    for signal in signals:
        if signal.buyer_segment == "unknown":
            continue
        scores[signal.buyer_segment] = scores.get(signal.buyer_segment, 0.0) + max(
            signal.weight, 0.1
        )
    if not scores:
        return "unknown"
    return max(scores.items(), key=lambda kv: kv[1])[0]


def classify_regulated_domain(texts: list[str]) -> str:
    """Cheap, conservative domain classifier used by the governance policy engine."""
    from core.constants import RegulatedDomain

    blob = " ".join(texts).lower()
    patterns = {
        RegulatedDomain.MEDICAL: r"\b(diagnos|symptom|prescri|medication|dosage|triage|patient)\w*",
        RegulatedDomain.MENTAL_HEALTH: r"\b(therapy|therapist|depress|anxiety|mental health|suicid)\w*",
        RegulatedDomain.LEGAL: r"\b(legal advice|attorney|solicitor|lawsuit|litigation|contract clause|liability|visa)\w*",
        RegulatedDomain.FINANCIAL: r"\b(investment advice|portfolio|superannuation|mortgage|loan approval|tax advice|financial advice)\w*",
        RegulatedDomain.TAX: r"\b(tax return|ato\b|deduction|gst\b|bas\b|taxable)\w*",
        RegulatedDomain.IMMIGRATION: r"\b(visa|immigration|permanent residency|migration agent)\w*",
    }
    import re

    for domain, pattern in patterns.items():
        if re.search(pattern, blob):
            return str(domain)
    return str(RegulatedDomain.NONE)


def upsert_topics(
    session: Session,
    *,
    cluster_result: ClusterResult,
    signals: list[NormalizedSignal],
    embedder_name: str,
    drift_threshold: float | None = None,
) -> list[TopicOutcome]:
    """Create/update topics from a clustering result and record trend snapshots."""
    settings = get_settings()
    drift_threshold = drift_threshold if drift_threshold is not None else settings.drift_threshold
    existing = list(session.scalars(select(Topic).where(Topic.status == "active")))
    outcomes: list[TopicOutcome] = []
    used_topics: set[str] = set()

    for cluster in cluster_result.clusters:
        member_signals = [signals[i] for i in cluster.member_indices if i < len(signals)]
        if not member_signals:
            continue
        segment = dominant_segment(member_signals)
        texts = [s.text for s in member_signals]
        name, jtbd, keywords = compose_topic_name(texts, segment)

        match: Topic | None = None
        best_similarity = 0.0
        for topic in existing:
            if topic.id in used_topics or not topic.embedding:
                continue
            similarity = cosine(cluster.centroid, topic.embedding)
            if similarity > best_similarity and similarity >= (1.0 - drift_threshold):
                best_similarity, match = similarity, topic

        components = compute_demand_components(member_signals)
        demand = demand_score(components)
        drift = 0.0
        if match is not None:
            used_topics.add(match.id)
            drift = round(max(0.0, 1.0 - best_similarity), 4)
            # Exponential moving average keeps the centroid stable but adaptive.
            previous = np.asarray(match.embedding or cluster.centroid, dtype=np.float64)
            updated = 0.7 * previous + 0.3 * np.asarray(cluster.centroid, dtype=np.float64)
            norm = np.linalg.norm(updated)
            match.embedding = (updated / norm).tolist() if norm else cluster.centroid
            match.size = (match.size or 0) + len(member_signals)
            match.drift_score = drift
            match.jtbd = match.jtbd or jtbd
            if not match.niche or match.niche == "general":
                match.niche = _dominant_concept([sig.text for sig in member_signals])
            match.buyer_segments = sorted(set(list(match.buyer_segments or []) + [segment]))
            match.keywords = keywords
            match.last_seen_at = utcnow()
            match.embedding_model = embedder_name
            topic = match
            created = False
        else:
            slug_base = slugify(f"{name}-{segment}")
            slug = slug_base
            suffix = 1
            while session.scalar(select(Topic).where(Topic.slug == slug)) is not None:
                suffix += 1
                slug = f"{slug_base}-{suffix}"
            niche_key = (
                _dominant_concept([sig.text for sig in member_signals])
                if member_signals
                else "general"
            )
            topic = Topic(
                name=name,
                slug=slug,
                niche=niche_key,
                description=f"{len(member_signals)} signals clustered around: {', '.join(keywords[:4])}",
                jtbd=jtbd,
                buyer_segments=[segment],
                keywords=keywords,
                embedding=cluster.centroid,
                embedding_model=embedder_name,
                size=len(member_signals),
                drift_score=0.0,
                first_seen_at=utcnow(),
                last_seen_at=utcnow(),
            )
            session.add(topic)
            session.flush()
            existing.append(topic)
            used_topics.add(topic.id)
            created = True

        signal_ids: list[str] = []
        for signal in member_signals:
            signal.topic_id = topic.id
            distance = 1.0 - cosine(
                signal.embedding or cluster.centroid, topic.embedding or cluster.centroid
            )
            exists = session.scalar(
                select(TopicMembership).where(
                    TopicMembership.topic_id == topic.id, TopicMembership.signal_id == signal.id
                )
            )
            if exists is None:
                session.add(
                    TopicMembership(
                        topic_id=topic.id,
                        signal_id=signal.id,
                        distance=distance,
                        weight=signal.weight,
                    )
                )
            signal_ids.append(signal.id)

        session.add(
            TrendSnapshot(
                topic_id=topic.id,
                window_days=int(components.get("window_days", 28)),
                signal_count=len(member_signals),
                volume=components["volume"],
                velocity=components["velocity"],
                demand_score=demand,
                drift_score=drift,
                centroid_embedding=topic.embedding,
                top_terms=keywords[:5],
            )
        )
        session.flush()
        outcomes.append(
            TopicOutcome(
                topic_id=topic.id,
                topic_name=topic.name,
                created=created,
                drift=drift,
                size=topic.size,
                signal_ids=signal_ids,
                centroid=topic.embedding or cluster.centroid,
                demand=demand,
                velocity=components["velocity"],
            )
        )
    return outcomes


def topic_demand(session: Session, topic_id: str, *, window_days: int = 28) -> dict[str, float]:
    signals = list(
        session.scalars(select(NormalizedSignal).where(NormalizedSignal.topic_id == topic_id))
    )
    return compute_demand_components(signals, window_days=window_days)


def topic_signal_count(session: Session, topic_id: str) -> int:
    return int(
        session.scalar(
            select(func.count(NormalizedSignal.id)).where(NormalizedSignal.topic_id == topic_id)
        )
        or 0
    )


def drift_report(session: Session, topic_id: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(TrendSnapshot)
        .where(TrendSnapshot.topic_id == topic_id)
        .order_by(TrendSnapshot.captured_at.desc())
        .limit(limit)
    )
    return [
        {
            "captured_at": row.captured_at.isoformat(),
            "demand_score": row.demand_score,
            "velocity": row.velocity,
            "drift_score": row.drift_score,
            "signal_count": row.signal_count,
            "top_terms": row.top_terms,
        }
        for row in rows
    ]
