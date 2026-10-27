"""Opportunity cards — the operator-facing artefact for each ranked opportunity.

A card is deliberately self-contained: scores + interval + evidence quotes + proposed
test strategy + pricing + risks. It is what a human reviews before any generation
budget is spent, and it is stored as a signed artifact for audit.
"""

from __future__ import annotations

from typing import Any

from core.constants import OpportunityStatus, RiskTier
from market_intel.scoring import ScoreBreakdown


def recommendation_for(score: ScoreBreakdown, *, min_lower_bound: float = 0.35) -> str:
    if score.risk_tier == str(RiskTier.PROHIBITED):
        return "reject_prohibited"
    if score.regulated_domain != "none" and score.risk >= 0.7:
        return "assistive_only_requires_compliance_review"
    if score.composite_lower >= min_lower_bound and score.confidence >= 0.45:
        return "proceed_to_generation"
    if score.composite_lower >= min_lower_bound * 0.7:
        return "collect_more_evidence"
    return "deprioritise"


def proposed_test_strategy(score: ScoreBreakdown, *, regulated: bool) -> dict[str, Any]:
    base_cases = 12
    if score.build_cost > 0.6:
        base_cases = 20
    if regulated:
        base_cases += 8
    return {
        "deterministic_cases": base_cases,
        "holdout_ratio": 0.25,
        "judge_criteria": ["task_success", "instruction_following", "structure", "tone", "safety"],
        "robustness": ["ambiguous_input", "long_context", "empty_input", "adversarial_input"],
        "injection_suite": bool(regulated) or True,
        "min_pass_rate": 0.8 if regulated else 0.7,
        "max_cost_per_candidate_aud": 0.35,
        "regression_gate": {"min_absolute_score": 0.6, "max_allowed_drop": 0.03},
    }


def build_card(
    *,
    opportunity_id: str,
    topic: dict[str, Any],
    segment: str,
    score: ScoreBreakdown,
    evidence: list[dict[str, Any]],
    status: str = str(OpportunityStatus.SCORED),
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regulated = score.regulated_domain != "none"
    strategy = proposed_test_strategy(score, regulated=regulated)
    card = {
        "opportunity_id": opportunity_id,
        "status": status,
        "title": topic.get("title") or topic.get("name"),
        "topic": {
            "id": topic.get("id"),
            "name": topic.get("name"),
            "slug": topic.get("slug"),
            "jtbd": topic.get("jtbd"),
            "keywords": topic.get("keywords", []),
            "size": topic.get("size"),
            "drift_score": topic.get("drift_score", 0.0),
        },
        "buyer_segment": segment,
        "score": score.to_dict(),
        "evidence": evidence[:8],
        "recommendation": recommendation_for(score),
        "test_strategy": strategy,
        "pricing": pricing
        or {
            "suggested_price_aud": round(score.price_hypothesis_micros / 1_000_000, 2),
            "currency": "AUD",
            "rationale": "segment prior + intent/pain uplift + internal benchmarks (if any)",
        },
        "risks": [
            {
                "risk": f"regulated domain: {score.regulated_domain}",
                "mitigation": "assistive drafting only, explicit disclaimers, refusal rules, human review",
            }
            if regulated
            else {
                "risk": "demand may be concentrated in a few sources",
                "mitigation": "add permitted sources; require n≥20 before publishing",
            },
            {
                "risk": "supply uncertainty",
                "mitigation": "supply_proxy has high uncertainty until permitted competition data is supplied",
            }
            if not score.supply_known
            else {
                "risk": "competition",
                "mitigation": "differentiate on structure, examples and reliability",
            },
        ],
        "next_actions": [
            "review evidence",
            "generate candidates (Stage 1 screen)",
            "run offline evaluation (Stage 2)",
            "run governance gate",
            "request human approval",
        ],
    }
    return card


def render_card_markdown(card: dict[str, Any]) -> str:
    score = card["score"]
    lines = [
        f"# Opportunity: {card['title']}",
        "",
        f"- **Opportunity ID**: `{card['opportunity_id']}`",
        f"- **Buyer segment**: {card['buyer_segment']}",
        f"- **Recommendation**: `{card['recommendation']}`",
        f"- **Composite score**: {score['composite']:.3f} "
        f"(90% CI {score['composite_lower']:.3f} – {score['composite_upper']:.3f})",
        f"- **Confidence**: {score['confidence']:.2f}  |  **Risk tier**: {score['risk_tier']} "
        f"({score['regulated_domain']})",
        f"- **Suggested price**: A${score['price_hypothesis_aud']:.2f}",
        "",
        "## Score components",
        "",
        "| Component | Value | Notes |",
        "| --- | --- | --- |",
        f"| Demand | {score['demand']:.3f} | volume, velocity, intent, recency |",
        f"| Supply proxy | {score['supply_proxy']:.3f} | {'measured' if score['supply_known'] else 'unknown → high uncertainty'} |",
        f"| Monetization | {score['monetization']:.3f} | segment price prior + intent/pain |",
        f"| Build cost | {score['build_cost']:.3f} | eval complexity + support load |",
        f"| Risk | {score['risk']:.3f} | regulated domain + policy flags |",
        f"| Confidence | {score['confidence']:.3f} | evidence quantity/diversity |",
        "",
        "## Job to be done",
        "",
        card["topic"].get("jtbd") or "—",
        "",
        "## Evidence",
        "",
    ]
    for item in card["evidence"]:
        quote = (item.get("quote") or "").replace("\n", " ")[:220]
        lines.append(
            f"- “{quote}”  \n  — `{item.get('source_id') or 'unknown source'}`, weight {item.get('weight', 0):.2f}"
        )
    if not card["evidence"]:
        lines.append("- (no evidence items attached)")
    lines += [
        "",
        "## Proposed evaluation strategy",
        "",
        f"- Deterministic cases: {card['test_strategy']['deterministic_cases']} "
        f"(holdout {card['test_strategy']['holdout_ratio']:.0%})",
        f"- Judge criteria: {', '.join(card['test_strategy']['judge_criteria'])}",
        f"- Robustness: {', '.join(card['test_strategy']['robustness'])}",
        f"- Min pass rate: {card['test_strategy']['min_pass_rate']:.0%}",
        "",
        "## Risks & mitigations",
        "",
    ]
    for risk in card["risks"]:
        lines.append(f"- **{risk['risk']}** → {risk['mitigation']}")
    lines += ["", "## Why this scored this way", ""]
    for reason in score.get("reasons", []):
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)
