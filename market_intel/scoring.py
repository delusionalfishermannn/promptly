"""Opportunity scoring: profit-first, uncertainty-aware, fully explainable.

The composite is a transparent multi-objective function with an explicit
uncertainty penalty and a bootstrap confidence interval:

    profit      = 0.45·demand + 0.35·monetization + 0.20·(1−supply)
    composite   = 0.55·profit + 0.15·(1−risk) + 0.15·(1−build_cost) + 0.15·confidence

Every block is a weighted *average* of sub-scores in [0,1], so the composite is itself
in [0,1] and readable: ~0.35 is "worth investigating", ~0.55 "worth building",
~0.7+ "build now". (An earlier draft summed rather than averaged the profit terms,
which compressed every score below 0.35 and made the gate thresholds unreachable.)

Ranking uses the **lower bound** of the bootstrap interval (conservative choice for a
budget-constrained operator: we prefer well-evidenced mid-size opportunities over
noisy "big" ones). Every weight and input is persisted on ``opportunity_scores`` so a
score can always be re-derived and challenged later.

Bootstrap: resample the topic's signals with replacement (weighted) B times, re-derive
volume/velocity/intent, recompute the composite, and take the 5th/95th percentiles.
Seeded per topic id → deterministic and reproducible in tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from core.config import AUD_MICROS, get_settings
from core.constants import RegulatedDomain, RiskTier
from market_intel.topics import demand_score

DEFAULT_WEIGHTS: dict[str, float] = {
    # profit block (averages to 1.0)
    "demand": 0.45,
    "monetization": 0.35,
    "availability": 0.20,  # = 1 − supply pressure
    "profit": 0.55,  # weight of the profit block in the composite
    # composite block (averages to 1.0)
    "risk": 0.15,  # weight of (1 − risk)
    "build_cost": 0.15,  # weight of (1 − build_cost)
    "confidence": 0.15,  # weight of confidence
}

# Price priors in AUD (whole dollars) by buyer segment — deliberately conservative.
PRICE_PRIORS_AUD: dict[str, float] = {
    "creator": 19.0,
    "smb": 39.0,
    "agency": 79.0,
    "enterprise": 249.0,
    "educator": 19.0,
    "student": 9.0,
    "developer": 29.0,
    "unknown": 29.0,
}

DOMAIN_RISK: dict[str, float] = {
    str(RegulatedDomain.NONE): 0.05,
    str(RegulatedDomain.TAX): 0.60,
    str(RegulatedDomain.FINANCIAL): 0.70,
    str(RegulatedDomain.IMMIGRATION): 0.75,
    str(RegulatedDomain.LEGAL): 0.80,
    str(RegulatedDomain.MENTAL_HEALTH): 0.82,
    str(RegulatedDomain.MEDICAL): 0.85,
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ScoreBreakdown:
    demand: float
    supply_proxy: float
    monetization: float
    build_cost: float
    risk: float
    confidence: float
    composite: float
    composite_lower: float
    composite_upper: float
    price_hypothesis_micros: int
    risk_tier: str
    regulated_domain: str
    supply_known: bool
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    inputs: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    method: str = "profit_first_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "demand": round(self.demand, 4),
            "supply_proxy": round(self.supply_proxy, 4),
            "monetization": round(self.monetization, 4),
            "build_cost": round(self.build_cost, 4),
            "risk": round(self.risk, 4),
            "confidence": round(self.confidence, 4),
            "composite": round(self.composite, 4),
            "composite_lower": round(self.composite_lower, 4),
            "composite_upper": round(self.composite_upper, 4),
            "price_hypothesis_aud": round(self.price_hypothesis_micros / AUD_MICROS, 2),
            "risk_tier": self.risk_tier,
            "regulated_domain": self.regulated_domain,
            "supply_known": self.supply_known,
            "weights": self.weights,
            "inputs": self.inputs,
            "reasons": self.reasons,
            "method": self.method,
        }


def monetization_score(
    *,
    segment: str,
    components: dict[str, float],
    similar_products: list[dict[str, Any]] | None = None,
) -> tuple[float, float, list[str]]:
    """Return (score, price_hypothesis_aud, reasons)."""
    reasons: list[str] = []
    prior = PRICE_PRIORS_AUD.get(segment, PRICE_PRIORS_AUD["unknown"])
    intent = components.get("intent", 0.0)
    pain = components.get("pain", 0.0)
    urgency = _urgency_from_components(components)
    price = prior * (0.85 + 0.45 * intent + 0.25 * pain)
    if similar_products:
        observed = [
            p.get("price_aud", 0.0) for p in similar_products if p.get("price_aud", 0.0) > 0
        ]
        conversions = [
            p.get("conversion_rate", 0.0) for p in similar_products if p.get("sales", 0) > 0
        ]
        if observed:
            benchmark = sorted(observed)[len(observed) // 2]
            price = 0.6 * price + 0.4 * benchmark
            reasons.append(
                f"anchored to {len(observed)} internal product price(s) (median A${benchmark:.0f})"
            )
        if conversions:
            mean_conversion = sum(conversions) / len(conversions)
            reasons.append(f"internal conversion benchmark {mean_conversion:.1%}")
    price = max(5.0, min(price, 1200.0))
    price_norm = math.log1p(price) / math.log1p(1200.0)
    score = _clamp(0.42 * price_norm + 0.28 * intent + 0.18 * pain + 0.12 * urgency)
    reasons.append(
        f"segment '{segment}' supports ~A${price:.0f}; intent={intent:.2f}, pain={pain:.2f}, urgency={urgency:.2f}"
    )
    return score, int(round(price * AUD_MICROS)), reasons


def _urgency_from_components(components: dict[str, float]) -> float:
    velocity = components.get("velocity", 0.0)
    return _clamp(0.5 + 0.5 * velocity)


def supply_proxy(
    *,
    keywords: list[str],
    catalog: list[dict[str, Any]],
    permitted_density: float | None = None,
) -> tuple[float, bool, list[str]]:
    """Competition proxy in [0,1]; ``supply_known=False`` ⇒ wide uncertainty.

    Sources of truth, in order: (1) permitted marketplace/category density explicitly
    supplied by the operator via an allowed export/API, (2) our own catalog saturation.
    We never infer competitor density by scraping.
    """
    reasons: list[str] = []
    keyword_set = {k.lower() for k in keywords if k}
    overlapping = 0
    for product in catalog:
        blob = " ".join(
            [
                str(product.get("niche", "")),
                str(product.get("name", "")),
                " ".join(product.get("tags", []) or []),
            ]
        ).lower()
        if keyword_set and any(keyword in blob for keyword in keyword_set):
            overlapping += 1
    catalog_pressure = _clamp(overlapping / 8.0)
    if overlapping:
        reasons.append(f"{overlapping} existing catalog product(s) overlap this topic")

    if permitted_density is None:
        reasons.append(
            "no permitted competition data — supply treated as unknown (higher uncertainty)"
        )
        return round(0.5 * catalog_pressure + 0.25, 4), False, reasons

    density_score = _clamp(math.log1p(max(permitted_density, 0.0)) / math.log1p(500.0))
    reasons.append(
        f"permitted category density={permitted_density:g} (normalised {density_score:.2f})"
    )
    return round(_clamp(0.45 * density_score + 0.35 * catalog_pressure + 0.2), 4), True, reasons


def build_cost_score(
    *,
    components: dict[str, float],
    regulated_domain: str,
    evidence_count: int,
) -> tuple[float, list[str]]:
    complexity = components.get("complexity", _clamp(components.get("intent", 0.0) * 0.5))
    complexity = _clamp(complexity)
    domain_factor = 0.5 if regulated_domain == str(RegulatedDomain.NONE) else 1.0
    coverage_factor = _clamp(1.0 - evidence_count / 60.0)
    score = _clamp(0.5 * complexity + 0.3 * domain_factor + 0.2 * coverage_factor)
    reasons = [
        f"complexity={complexity:.2f}, regulated_domain={regulated_domain}, evidence={evidence_count}"
    ]
    return score, reasons


def risk_score(
    *,
    regulated_domain: str,
    components: dict[str, float],
    policy_flags: list[str] | None = None,
) -> tuple[float, str, list[str]]:
    base = DOMAIN_RISK.get(regulated_domain, 0.4)
    flags = policy_flags or []
    if "sensitive_personal_data" in flags:
        base += 0.1
    if "profanity" in flags:
        base += 0.02
    score = _clamp(base)
    reasons = [f"regulated_domain={regulated_domain} → base risk {base:.2f}"]
    if flags:
        reasons.append(f"policy flags: {', '.join(flags)}")
    tier = (
        str(RiskTier.PROHIBITED)
        if score >= 0.95
        else str(RiskTier.HIGH)
        if score >= 0.7
        else str(RiskTier.MEDIUM)
        if score >= 0.35
        else str(RiskTier.LOW)
    )
    return score, tier, reasons


def confidence_score(
    *,
    n_signals: int,
    n_sources: int,
    mean_intent: float,
    supply_known: bool,
    mean_quality: float,
) -> float:
    n_factor = 1.0 - math.exp(-n_signals / 25.0)
    diversity = _clamp(n_sources / 3.0)
    supply_factor = 1.0 if supply_known else 0.45
    score = 0.42 * n_factor + 0.18 * diversity + 0.22 * mean_intent + 0.18 * supply_factor
    score *= 0.75 + 0.25 * _clamp(mean_quality)
    return _clamp(score)


def composite_score(
    *,
    demand: float,
    monetization: float,
    supply: float,
    risk: float,
    build_cost: float,
    confidence: float,
    weights: dict[str, float] | None = None,
) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    profit = (
        w["demand"] * demand + w["monetization"] * monetization + w["availability"] * (1.0 - supply)
    )
    value = (
        w["profit"] * profit
        + w["risk"] * (1.0 - risk)
        + w["build_cost"] * (1.0 - build_cost)
        + w["confidence"] * confidence
    )
    return round(_clamp(value), 4)


def bootstrap_interval(
    *,
    signals: list[dict[str, float]],
    weights: dict[str, float],
    fixed: dict[str, float],
    iterations: int = 160,
    seed_key: str = "topic",
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap the composite over resampled signals.

    ``signals`` items provide ``weight, intent_strength, pain, recency_decay, quality``.
    ``fixed`` holds components that do not vary with resampling (monetization, risk,
    build_cost, supply).
    Returns (mean, q05, q95).
    """
    if not signals or iterations <= 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed_key)
    draws: list[float] = []
    n = len(signals)
    for _ in range(iterations):
        sample = [signals[rng.randrange(n)] for _ in range(n)]
        volume = sum(s["weight"] for s in sample) or 0.1
        intent = sum(s["intent_strength"] * s["weight"] for s in sample) / volume
        pain = sum(s["pain"] * s["weight"] for s in sample) / volume
        recency = sum(s["recency_decay"] for s in sample) / n
        quality = sum(s["quality"] for s in sample) / n
        demand = demand_score(
            {
                "volume": volume,
                "velocity": fixed.get("velocity", 0.0),
                "intent": intent,
                "recency": recency,
                "quality": quality,
            }
        )
        monetization, _, _ = monetization_score(
            segment=fixed.get("segment", "unknown"),
            components={"intent": intent, "pain": pain, "velocity": fixed.get("velocity", 0.0)},
            similar_products=fixed.get("similar_products"),
        )
        value = composite_score(
            demand=demand,
            monetization=monetization,
            supply=fixed.get("supply", 0.5),
            risk=fixed.get("risk", 0.2),
            build_cost=fixed.get("build_cost", 0.5),
            confidence=fixed.get("confidence", 0.4),
            weights=weights,
        )
        draws.append(value)
    draws.sort()
    mean = sum(draws) / len(draws)
    low = draws[int(lower_q * (len(draws) - 1))]
    high = draws[int(upper_q * (len(draws) - 1))]
    return round(mean, 4), round(low, 4), round(high, 4)


def score_opportunity(
    *,
    topic_id: str,
    topic_name: str,
    segment: str,
    keywords: list[str],
    signals: list[dict[str, Any]],
    components: dict[str, float],
    regulated_domain: str,
    catalog: list[dict[str, Any]] | None = None,
    permitted_density: float | None = None,
    weights: dict[str, float] | None = None,
    policy_flags: list[str] | None = None,
) -> ScoreBreakdown:
    """Full scoring for one topic/segment, returning an explainable breakdown."""
    settings = get_settings()
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    demand = demand_score(components)
    supply, supply_known, supply_reasons = supply_proxy(
        keywords=keywords, catalog=catalog or [], permitted_density=permitted_density
    )
    monetization, price_micros, money_reasons = monetization_score(
        segment=segment, components=components
    )
    build_cost, cost_reasons = build_cost_score(
        components=components,
        regulated_domain=regulated_domain,
        evidence_count=int(components.get("n", len(signals))),
    )
    risk, risk_tier, risk_reasons = risk_score(
        regulated_domain=regulated_domain, components=components, policy_flags=policy_flags
    )
    confidence = confidence_score(
        n_signals=int(components.get("n", len(signals))),
        n_sources=int(components.get("n_sources", 1)),
        mean_intent=components.get("intent", 0.0),
        supply_known=supply_known,
        mean_quality=components.get("quality", 1.0),
    )
    composite = composite_score(
        demand=demand,
        monetization=monetization,
        supply=supply,
        risk=risk,
        build_cost=build_cost,
        confidence=confidence,
        weights=weights,
    )
    _bootstrap_mean, low, high = bootstrap_interval(
        signals=[
            {
                "weight": float(s.get("weight", 1.0)),
                "intent_strength": float(s.get("intent_strength", 0.0)),
                "pain": float(s.get("pain", 0.0)),
                "recency_decay": float(s.get("recency_decay", 1.0)),
                "quality": float(s.get("quality", 1.0)),
            }
            for s in signals
        ],
        weights=weights,
        fixed={
            "velocity": components.get("velocity", 0.0),
            "segment": segment,
            "supply": supply,
            "risk": risk,
            "build_cost": build_cost,
            "confidence": confidence,
            "similar_products": None,
        },
        seed_key=topic_id,
    )
    # Confidence interval must envelope the point estimate (bootstrap can disagree
    # slightly because of non-linear squashing).
    low = min(low, composite)
    high = max(high, composite)

    reasons = [
        f"demand={demand:.2f} from volume {components.get('volume', 0):.1f}, "
        f"velocity {components.get('velocity', 0):+.2f}, intent {components.get('intent', 0):.2f}",
        *supply_reasons,
        *money_reasons,
        *cost_reasons,
        *risk_reasons,
        f"confidence={confidence:.2f} (n={int(components.get('n', 0))}, "
        f"sources={int(components.get('n_sources', 1))}, supply_known={supply_known})",
        f"composite={composite:.3f} with 90% bootstrap interval [{low:.3f}, {high:.3f}]",
    ]
    return ScoreBreakdown(
        demand=demand,
        supply_proxy=supply,
        monetization=monetization,
        build_cost=build_cost,
        risk=risk,
        confidence=confidence,
        composite=composite,
        composite_lower=low,
        composite_upper=high,
        price_hypothesis_micros=price_micros,
        risk_tier=risk_tier,
        regulated_domain=regulated_domain,
        supply_known=supply_known,
        weights=weights,
        inputs={
            **{k: round(v, 4) for k, v in components.items()},
            "segment": segment,
            "keywords": keywords[:8],
            "n_signals": len(signals),
            "permitted_density": permitted_density,
            "policy_flags": policy_flags or [],
            "settings": {
                "aud_per_usd": settings.aud_per_usd,
                "price_prior_aud": PRICE_PRIORS_AUD.get(segment, PRICE_PRIORS_AUD["unknown"]),
            },
        },
        reasons=reasons,
    )


def rank_opportunities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservative ranking: lower-bound first, then point estimate, then demand."""
    return sorted(
        items,
        key=lambda item: (
            -item.get("composite_lower", 0.0),
            -item.get("composite", 0.0),
            -item.get("demand", 0.0),
        ),
    )
