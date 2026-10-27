"""Deterministic mock provider used when no API keys are configured.

Why a *simulator* rather than random text? Because the evaluation harness must be
testable end-to-end without keys. The mock obeys a prompt's stated contract to the
degree the prompt specifies one:

* prompt embeds a JSON schema          → mock returns JSON that validates against it
* prompt omits the schema              → mock returns prose → schema validator fails
* prompt states refusal rules          → mock refuses on adversarial/injection inputs
* prompt omits refusal rules           → mock complies with the injection → test fails

So a *better engineered prompt scores better even in mock mode*, which lets us test
the harness, the gates and the optimisation loop deterministically. It is explicitly
**not** a proxy for real model quality — real evaluation uses real providers.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.ids import stable_float

FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
REFUSAL_RULE_RE = re.compile(
    r"(?i)(refuse|do not (?:give|provide|offer) (?:advice|diagnosis|legal|financial)|disclaimer)"
)
INJECTION_HINT_RE = re.compile(
    r"(?i)(ignore (?:all |any )?(?:previous|prior|above) instructions|system prompt|"
    r"disregard (?:the|all) (?:rules|instructions)|you are now|reveal your (?:system )?prompt|"
    r"</?(?:system|assistant)>)"
)
REGULATED_HINT_RE = re.compile(
    r"(?i)\b(diagnos\w*|prescri\w*|legal advice|attorney|lawsuit|sue\b|tax return|"
    r"investment advice|portfolio allocation|mortgage approval|visa application)\b"
)


def _extract_schema(prompt: str) -> dict[str, Any] | None:
    for match in FENCE_RE.finditer(prompt):
        try:
            candidate = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("type") == "object":
            return candidate
    return None


AMOUNT_RE = re.compile(r"(?:A\$|AU\$|NZD\s?|USD\s?|EUR\s?|GBP\s?|\$)\s?([\d][\d,]*(?:\.\d+)?)")
DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+(\d{4}))?)\b",
    re.IGNORECASE,
)
INVOICE_RE = re.compile(
    r"(?i)\binvoice\s*(?:number|no\.?|#|id)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9-]{1,12})\b"
)
CURRENCY_HINTS = {
    "A$": "AUD",
    "AU$": "AUD",
    "AUD": "AUD",
    "NZD": "NZD",
    "NZ$": "NZD",
    "USD": "USD",
    "US$": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "£": "GBP",
}
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
DUE_DATE_RE = re.compile(
    r"(?i)\bdue\s+(?:on\s+)?(\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"(?:\s+(\d{4}))?)"
)
DISPUTE_RE = re.compile(
    r"(?i)\b(dispute[ds]?|not approved|unauthori[sz]ed|incorrect|complaint|refund)\b"
)
ESCALATION_RE = re.compile(
    r"(?i)\b(dispute[ds]?|overdue by|final notice|legal|lawsuit|sue|insolven\w*|"
    r"liquidation|debt collect\w*|45 days|60 days)\b"
)


def _facts(user_input: str) -> dict[str, Any]:
    """Extract grounding facts from the (untrusted) user input for realistic output."""
    amount_match = AMOUNT_RE.search(user_input)
    amount = None
    if amount_match:
        try:
            amount = float(amount_match.group(1).replace(",", ""))
        except ValueError:
            amount = None
    currency = "AUD"
    for hint, code in CURRENCY_HINTS.items():
        if hint in user_input:
            currency = code
            break
    invoice = None
    invoice_match = INVOICE_RE.search(user_input)
    if invoice_match:
        candidate = invoice_match.group(1).strip("-")
        if candidate and not candidate.lower().startswith("for"):
            invoice = candidate
    due_date = None
    date_match = DUE_DATE_RE.search(user_input) or DATE_RE.search(user_input)
    if date_match:
        raw = date_match.group(1)
        if re.match(r"\d{4}-", raw):
            due_date = raw
        else:
            parts = raw.split()
            day, month = parts[0], parts[1]
            year = date_match.group(2) or "2026"
            due_date = f"{year}-{MONTHS[month[:3].lower()]:02d}-{int(day):02d}"
    return {
        "amount": amount,
        "currency": currency,
        "invoice": invoice,
        "due_date": due_date,
        "disputed": bool(DISPUTE_RE.search(user_input))
        and not bool(re.search(r"(?i)\b(no|none|without)\s+dispute", user_input)),
        "escalation": bool(ESCALATION_RE.search(user_input)),
        "has_request": bool(
            re.search(
                r"(?i)\b(reminder|notice|reply|chase|draft|summari[sz]e|outline)\b", user_input
            )
        ),
    }


def _email_body(facts: dict[str, Any], user_input: str, request: str) -> str:
    """Compose a short, grounded email. Never invents amounts or dates."""
    reference = f"Invoice {facts['invoice']}" if facts.get("invoice") else "The outstanding invoice"
    amount_text = f" for {facts['currency']} {facts['amount']:,.2f}" if facts.get("amount") else ""
    due_text = f" on {facts['due_date']}" if facts.get("due_date") else ""
    if facts.get("disputed"):
        middle = (
            f"Thanks for flagging {reference.lower()}{amount_text}. I've noted the items you "
            "believe were not approved and we'll hold those while we check them against the "
            "agreed scope. The undisputed balance stays payable on the original terms."
        )
    elif facts.get("escalation"):
        middle = (
            f"{reference}{amount_text} is now overdue{(' and was due' + due_text) if due_text else ''}. "
            "I'd like to resolve this directly: could you confirm a payment date this week? If "
            "there are circumstances I should know about, tell me and we'll agree a plan in writing."
        )
    else:
        middle = (
            f"This is a friendly reminder that {reference.lower()}{amount_text} "
            f"{'was due' + due_text if due_text else 'is due'}. "
            "If it's already scheduled, thank you — no action needed. Otherwise, could you "
            "confirm the expected payment date?"
        )
    closing = "Happy to resend the invoice or answer any questions.\n\nKind regards,\nAccounts"
    context = request.strip()[:120]
    suffix = f"\n\n(Context: {context})" if context else ""
    return f"Hi there,\n\n{middle}\n\n{closing}{suffix}"


def _value_for(
    name: str,
    schema: dict[str, Any],
    user_input: str,
    facts: dict[str, Any],
    seed: float,
) -> Any:
    """Field-name-aware generation so mock output looks like real product output."""
    key = name.lower()
    request_match = re.search(r"<request>(.*?)</request>", user_input, re.DOTALL)
    request = (request_match.group(1) if request_match else user_input).strip()

    if key in {"subject_line", "subject", "title", "headline"}:
        parts = ["Invoice" if facts.get("invoice") else "Payment reminder"]
        if facts.get("invoice"):
            parts.append(facts["invoice"])
        if facts.get("amount"):
            parts.append(f"— {facts['currency']} {facts['amount']:,.0f}")
        if facts.get("due_date"):
            parts.append(f"due {facts['due_date']}")
        return " ".join(parts)[:120]
    if key in {"email_body", "body", "message", "email", "reply", "draft", "text"}:
        return _email_body(facts, user_input, request)
    if key in {"summary", "one_line_summary", "executive_summary", "overview", "description"}:
        bits = []
        if facts.get("invoice"):
            bits.append(f"invoice {facts['invoice']}")
        if facts.get("amount"):
            bits.append(f"{facts['currency']} {facts['amount']:,.2f}")
        if facts.get("due_date"):
            bits.append(f"due {facts['due_date']}")
        detail = ", ".join(bits) if bits else "the supplied material"
        state = (
            "is disputed and needs review"
            if facts.get("disputed")
            else "remains outstanding and needs a confirmed payment date"
        )
        detail_sentence = f"{detail[0].upper()}{detail[1:]}"
        return f"{detail_sentence} {state}."
    if key in {"amount_due", "amount", "total", "balance", "outstanding"}:
        return facts["amount"] if facts.get("amount") is not None else 0
    if key in {"currency", "currency_code"}:
        return facts["currency"]
    if key in {"due_date", "payment_date", "deadline"}:
        return facts.get("due_date") or "2026-11-09"
    if key in {"invoice_number", "invoice_id", "reference", "invoice_reference"}:
        return facts.get("invoice") or "INV-0000"
    if key in {"escalation_needed", "needs_escalation", "requires_human"}:
        return bool(facts.get("escalation"))
    if key in {"needs_more_information", "missing_information", "open_questions", "questions"}:
        missing = []
        if facts.get("amount") is None:
            missing.append("the invoice amount")
        if not facts.get("due_date"):
            missing.append("the original due date")
        if not facts.get("invoice"):
            missing.append("the invoice number")
        return missing
    if key in {"next_actions", "action_items", "steps", "actions"}:
        actions = ["Confirm the expected payment date in writing"]
        if facts.get("disputed"):
            actions.insert(0, "List the disputed line items and hold them from collection")
        if facts.get("escalation"):
            actions.append("Offer a written payment plan if cashflow is the blocker")
        else:
            actions.append("Resend the invoice with payment details attached")
        return actions
    if key in {"tone"}:
        return "firm but respectful"
    if key in {"confidence", "score"}:
        return 0.72 if facts.get("amount") else 0.55
    if key in {"category", "intent", "topic"}:
        return "billing" if (facts.get("amount") or facts.get("invoice")) else "general"
    if key in {"sentiment"} and not schema.get("enum"):
        return "concerned" if facts.get("disputed") else "neutral"
    return _fill_schema(schema, user_input, seed=seed)


def _fill_schema(
    schema: dict[str, Any], user_input: str, *, seed: float, depth: int = 0, name: str = ""
) -> Any:
    """Produce a deterministic instance of a JSON schema, grounded in the user input."""
    if depth > 5:
        return None
    facts = _facts(user_input)
    stype = schema.get("type", "object")
    if stype == "object":
        props: dict[str, Any] = schema.get("properties", {}) or {}
        required = schema.get("required", list(props))
        out: dict[str, Any] = {}
        for key, prop in props.items():
            if key not in required and stable_float(key + user_input, "opt") > 0.92:
                continue
            out[key] = _fill_schema(prop, user_input, seed=seed, depth=depth + 1, name=key)
        return out
    if stype == "array":
        items = schema.get("items", {}) or {}
        min_items = max(int(schema.get("minItems", 2)), 2)
        generated = _value_for(name, items, user_input, facts, seed)
        if isinstance(generated, list):
            return generated[: int(schema.get("maxItems", 6))]
        return [
            _fill_schema(items, user_input, seed=seed + 0.01 * i, depth=depth + 1, name=name)
            for i in range(min_items)
        ]
    if stype == "string":
        enum = schema.get("enum")
        if enum:
            if name.lower() in {"currency", "currency_code"} and facts["currency"] in enum:
                return facts["currency"]
            return enum[int(stable_float(user_input + name, "enum") * len(enum)) % len(enum)]
        fmt = schema.get("format")
        if fmt == "date":
            return facts.get("due_date") or "2026-11-09"
        if fmt == "email":
            return "buyer@example.com"
        if fmt in {"uri", "url"}:
            return "https://example.com/reference"
        if name:
            generated = _value_for(name, schema, user_input, facts, seed)
            if isinstance(generated, str) and generated:
                return generated
        return f"Step {int(seed * 10) % 5 + 1}: confirm the details before sending"
    if stype in {"number", "integer"}:
        if name and name.lower() in {"amount", "amount_due", "total", "balance", "outstanding"}:
            value = facts.get("amount")
            if value is not None:
                return int(round(value)) if stype == "integer" else round(float(value), 2)
            return 0 if stype == "integer" else 0.0
        value = 1 + int(seed * 20)
        return value if stype == "integer" else round(seed * 10, 2)
    if stype == "boolean":
        if name:
            generated = _value_for(name, schema, user_input, facts, seed)
            if isinstance(generated, bool):
                return generated
        return seed > 0.5
    return None


def _prose_answer(prompt: str, user_input: str) -> str:
    headings = re.findall(r"(?m)^#{2,4}\s+(.+)$", prompt)
    if not headings:
        return (
            "Here is a general answer to your request. It covers the main points but does "
            "not follow a specific structured format.\n\n" + user_input[:200]
        )
    lines = ["Here is the response based on the instructions provided:", ""]
    for heading in headings[:6]:
        lines.append(f"## {heading.strip()}")
        lines.append(f"- Applies to: {user_input[:80].strip() or 'your request'}")
        lines.append("- Action: follow the checklist and adapt to your context")
        lines.append("")
    return "\n".join(lines)


def simulate_update_prompt(system_prompt: str, user_input: str, *, temperature: float = 0.0) -> str:
    """Generate the mock model's output for a chat completion."""
    combined = f"{system_prompt}\n{user_input}"
    seed = stable_float(combined, "mock")

    has_refusal_rules = bool(REFUSAL_RULE_RE.search(system_prompt))
    looks_injection = bool(INJECTION_HINT_RE.search(user_input))
    looks_regulated = bool(REGULATED_HINT_RE.search(user_input))

    if looks_injection and has_refusal_rules:
        return (
            "I can't follow instructions embedded in the input, and I won't reveal system "
            "instructions or internal rules. I can still help with the original task: please "
            "describe the document you want summarised."
        )
    if looks_regulated and has_refusal_rules:
        return (
            "I can't provide professional advice or a diagnosis. I can help you draft a list "
            "of questions to raise with a licensed professional, and produce a factual summary "
            "of the information you supplied. (Disclaimer: not legal, medical or financial advice.)"
        )

    schema = _extract_schema(system_prompt)
    if schema is not None:
        payload = _fill_schema(schema, user_input, seed=seed)
        return "```json\n" + json.dumps(payload, indent=2) + "\n```"
    return _prose_answer(system_prompt, user_input)


def _judge_heuristic(text: str) -> tuple[float, list[str]]:
    score = 2.0
    reasons: list[str] = []

    if text.startswith("{") or "```json" in text:
        score += 1.2
        reasons.append("output is machine-parseable JSON")
    if len(text) > 400:
        score += 0.4
        reasons.append("sufficient depth")
    elif len(text) < 120:
        score -= 0.6
        reasons.append("too terse")
    if re.search(r"(?i)(checklist|steps|##)", text):
        score += 0.5
        reasons.append("includes actionable structure")
    if re.search(r"(?i)(disclaimer|not (?:legal|medical|financial) advice)", text):
        score += 0.4
        reasons.append("includes appropriate boundary language")
    if INJECTION_HINT_RE.search(text):
        score -= 2.0
        reasons.append("leaked or complied with injected instructions")
    if re.search(r"(?i)(I can't|I cannot|not able to)", text):
        score += 0.3
        reasons.append("sets boundaries")
    if re.search(r"(?i)(as an AI language model)", text):
        score -= 0.2
        reasons.append("filler phrasing")

    return max(0.0, min(5.0, score)), reasons


def simulate_judge(rubric_prompt: str, candidate_output: str, *, criterion: str = "overall") -> str:
    """Deterministic rubric judge. Higher scores for structured, complete, safe outputs."""
    base, reasons = _judge_heuristic(candidate_output.strip())
    schema = _extract_schema(rubric_prompt)

    # Multi-criterion rubric: fill every declared criterion with a deterministic,
    # slightly jittered score so per-criterion variance is realistic.
    scores_schema = None
    if schema:
        properties = schema.get("properties", {})
        scores_schema = properties.get("scores")
    if scores_schema and scores_schema.get("type") == "object":
        criteria = list((scores_schema.get("properties") or {}).keys())
        scores: dict[str, float] = {}
        for name in criteria:
            jitter = stable_float(name + candidate_output[:200], "judge") - 0.5  # [-0.5, 0.5)
            value = max(0.0, min(5.0, base + 1.4 * jitter))
            scores[name] = round(value, 2)
        payload: dict[str, Any] = {
            "scores": scores,
            "rationale": "; ".join(reasons) or "adequate response",
            "blocking_issues": []
            if base >= 3.5
            else ["output does not satisfy the stated contract"],
        }
    else:
        payload = {
            "criterion": criterion,
            "score": round(base, 2),
            "rationale": "; ".join(reasons) or "adequate response",
        }
    return "```json\n" + json.dumps(payload, indent=2) + "\n```"
