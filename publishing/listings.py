"""Listing copy + pricing suggestions (generated from the IR, reviewed by a human).

Listing claims are governance-checked (``governance.policy.check_policy`` validates the
title/description for unsubstantiable claims). Copy is generated from the product's
own specification, so it cannot drift from what the product actually does.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from core.config import get_settings
from db.models import PromptProduct, PromptVersion
from market_intel.scoring import PRICE_PRIORS_AUD
from prompt_ir.schemas import PromptIR

SEGMENT_LABELS = {
    "creator": "Creators",
    "smb": "Small businesses",
    "agency": "Agencies",
    "enterprise": "Enterprise teams",
    "educator": "Educators",
    "student": "Students",
    "developer": "Developers",
    "professional": "Professionals",
    "analyst": "Analysts",
    "consumer": "Everyday users",
    "unknown": "Teams",
}


def _segment_label(segment: str) -> str:
    return SEGMENT_LABELS.get(segment, (segment or "unknown").replace("_", " ").title())


def suggest_price(
    *,
    segment: str,
    hypothesis_aud: float,
    confidence: float,
    risk: float = 0.1,
    n_competitors_known: bool = False,
) -> dict[str, Any]:
    """Simple, defensible pricing: segment prior → hypothesis → risk/uncertainty band."""
    prior = PRICE_PRIORS_AUD.get(segment, PRICE_PRIORS_AUD["unknown"])
    anchor = hypothesis_aud or prior
    # Discount for high uncertainty (we would rather learn fast) and for risk.
    uncertainty_discount = 1.0 - 0.25 * (1.0 - min(max(confidence, 0.0), 1.0))
    price = anchor * uncertainty_discount * (1.0 - 0.3 * min(max(risk, 0.0), 1.0))
    price = max(5.0, round(price, 2) - 0.01) if price >= 10 else max(5.0, round(price, 2))
    floor = round(max(5.0, price * 0.7), 2)
    ceiling = round(price * 1.45, 2)
    rationale = [
        f"segment prior A${prior:.0f} for '{segment}'",
        f"opportunity hypothesis A${hypothesis_aud:.0f} (confidence {confidence:.2f})",
        "uncertainty discount applied to maximise early learning"
        if confidence < 0.6
        else "confidence supports anchor price",
        "no permitted competitor pricing data"
        if not n_competitors_known
        else "permitted competitor data used",
    ]
    return {
        "price_aud": price,
        "floor_aud": floor,
        "ceiling_aud": ceiling,
        "currency": get_settings().default_currency,
        "rationale": rationale,
        "test_plan": [
            f"Launch at A${price:.2f}",
            f"If conversion < 1% in 200 sessions, test A${floor:.2f}",
            f"If refunds > 8%, raise clarity (not price) first; then test A${ceiling:.2f}",
        ],
    }


def build_listing(
    session: Session,
    *,
    product: PromptProduct,
    version: PromptVersion,
    price_aud: float | None = None,
    score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ir = PromptIR.model_validate(version.ir) if version.ir else None
    segment = product.buyer or (ir.audience.segment if ir else "unknown")
    hypothesis = product.price_aud_micros / 1_000_000
    pricing = suggest_price(
        segment=segment,
        hypothesis_aud=price_aud if price_aud is not None else hypothesis,
        confidence=float((score or {}).get("confidence", 0.5)),
        risk=float((score or {}).get("risk", 0.1)),
        n_competitors_known=bool((score or {}).get("supply_known", False)),
    )
    if price_aud is not None:
        pricing["price_aud"] = price_aud

    deliverable = ir.task.deliverable if ir else "a reusable prompt"
    bullets = []
    if ir:
        bullets.append(f"Produces {deliverable} in one pass — no prompt engineering required.")
        bullets.append(
            f"Strict JSON output contract ({len((ir.io.output_schema or {}).get('properties', {}))} fields) "
            "so it drops straight into your workflow."
        )
        bullets.append(
            f"{len(ir.procedure.steps)}-step procedure with {len(ir.procedure.checklist)} self-checks baked in."
        )
        bullets.append(
            f"Compiled for {len(ir.hints.preferred_families)} model families — switch providers without rewriting."
        )
        if ir.safety.refusal_rules:
            bullets.append(
                "Refuses to invent facts; tells you what information is missing instead."
            )
        bullets.append("Includes a troubleshooting guide, scope boundaries and a changelog.")

    title = product.name
    if ir and len(ir.task.deliverable) < 60 and len(title) < 120:
        title = product.name
    description_lines = [
        product.positioning or (ir.task.objective if ir else ""),
        "",
        "### What you get",
        *[f"- {bullet}" for bullet in bullets],
        "",
        "### Who it's for",
        f"{_segment_label(segment)} who need reliable {deliverable} without rebuilding prompts.",
        "",
        "### What it's not",
        *[f"- {item}" for item in (ir.task.exclusions if ir else [])],
        "",
    ]
    if ir and ir.safety.disclaimers:
        description_lines += ["### Important", *[f"> {text}" for text in ir.safety.disclaimers], ""]

    faqs = [
        {
            "q": "Which models does this work with?",
            "a": "OpenAI, Anthropic and Google families are compiled separately, plus a generic version for local models.",
        },
        {
            "q": "Do I need to know prompt engineering?",
            "a": "No. Paste the prompt as the system message and fill the input tags described in QUICKSTART.md.",
        },
        {
            "q": "What if the output is wrong?",
            "a": "The prompt is designed to state missing information instead of inventing it. "
            "TROUBLESHOOTING.md covers the common failure modes.",
        },
        {
            "q": "Can I resell it?",
            "a": f"No — the licence is {product.license}. You can use it in your own commercial work.",
        },
        {
            "q": "Do you copy other people's prompts?",
            "a": "No. Every prompt is generated from an original specification and checked against our catalog "
            "and any permitted reference corpus before release.",
        },
    ]
    tags = list(dict.fromkeys([*(product.tags or []), *(ir.tags if ir else []), segment]))[:12]
    return {
        "title": title,
        "subtitle": (ir.task.deliverable.capitalize() if ir else "")[:140],
        "bullets": bullets,
        "description": "\n".join(description_lines).strip(),
        "seo_tags": tags,
        "faqs": faqs,
        "price_aud": pricing["price_aud"],
        "pricing": pricing,
        "currency": get_settings().default_currency,
        "compliance_notes": [
            "Listing claims are limited to what the product demonstrably does.",
            "No income, ranking or professional-outcome guarantees.",
        ]
        + (
            [
                f"Regulated domain ({ir.safety.regulated_domain}): assistive only, disclaimers required."
            ]
            if ir and ir.safety.regulated_domain != "none"
            else []
        ),
        "generated_at_version": version.version,
    }


def price_band_aud(competitive: bool) -> tuple[float, float]:
    """Convenience helper for the API: sensible launch band by competitiveness."""
    return (19.0, 49.0) if competitive else (29.0, 89.0)


def expected_revenue(price_aud: float, expected_units: float) -> float:
    return round(max(price_aud, 0) * max(expected_units, 0), 2)


def log_margin(price_aud: float, refund_rate: float = 0.05) -> float:
    """Platform fee + refund drag as a rough margin proxy (no fixed costs modelled)."""
    platform_fee = 0.1 * price_aud
    refunds = refund_rate * price_aud
    return round(price_aud - platform_fee - refunds, 2)
