"""Deterministic validators: the primary signal for prompt quality.

Everything here is reproducible, cheap and explainable — JSON-schema conformance,
format checks, and declarative assertions. LLM judging is a *secondary* signal layered
on top (see ``evaluation/judges.py``); no product decision rests on judging alone.

Assertion vocabulary (``test_cases.assertions`` entries)::

    {"type": "contains", "value": "due date"}
    {"type": "not_contains", "value": "as an AI language model"}
    {"type": "regex", "pattern": "^INV-\\d+$", "field": "reference"}
    {"type": "max_words", "value": 220, "field": "email_body"}
    {"type": "max_chars", "value": 60, "field": "subject_line"}
    {"type": "one_of", "value": ["AUD", "USD"], "field": "currency"}
    {"type": "required_keys", "value": ["summary", "amount_due"]}
    {"type": "refusal_expected"}                 # output must decline / set boundaries
    {"type": "no_pii"}                           # no emails/phones/cards echoed
    {"type": "no_hallucinated_numbers", "allowed": ["1850"]}  # digits ⊆ allowed ∪ input digits
    {"type": "json_parses"}
    {"type": "field_nonempty", "field": "next_actions", "min_items": 1}

Unknown assertion types fail closed (with the reason recorded) so a typo cannot silently
weaken a suite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from prompt_ir.schemas import validate_instance

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?61|0)[\s\-]?(?:\d[\s\-]?){8,10}(?!\d)")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "not able to",
    "i won't",
    "cannot provide",
    "can't provide",
    "not professional advice",
    "not legal advice",
    "not medical advice",
    "not financial advice",
    "consult a qualified",
    "speak to a",
    "would need to speak",
)
PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|XXX|FIXME|\[insert[^\]]*\]|lorem ipsum)\b", re.IGNORECASE)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "weight": self.weight,
        }


@dataclass
class ValidationReport:
    schema_ok: bool
    schema_errors: list[str] = field(default_factory=list)
    checks: list[CheckResult] = field(default_factory=list)
    parsed: dict[str, Any] | None = None
    raw_text: str = ""

    @property
    def passed(self) -> bool:
        return self.schema_ok and all(check.passed for check in self.checks)

    @property
    def score(self) -> float:
        total = sum(check.weight for check in self.checks)
        if not self.checks:
            return 1.0 if self.schema_ok else 0.0
        earned = sum(check.weight for check in self.checks if check.passed)
        base = earned / total if total else 0.0
        return round(base * (1.0 if self.schema_ok else 0.0), 4)

    @property
    def failures(self) -> list[str]:
        out = [f"schema: {error}" for error in self.schema_errors]
        out += [f"{check.name}: {check.detail}" for check in self.checks if not check.passed]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_ok": self.schema_ok,
            "schema_errors": self.schema_errors,
            "checks": [check.to_dict() for check in self.checks],
            "passed": self.passed,
            "score": self.score,
            "failures": self.failures,
        }


def parse_output(text: str) -> dict[str, Any] | None:
    """Parse a model output into JSON, tolerating fences and leading prose."""
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate = parts[1]
            if candidate.lstrip().lower().startswith("json"):
                candidate = candidate.lstrip()[4:]
    candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _field_value(parsed: dict[str, Any] | None, field: str | None, raw: str) -> Any:
    if field and parsed:
        return parsed.get(field)
    return parsed if parsed is not None else raw


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def run_assertion(
    assertion: dict[str, Any], parsed: dict[str, Any] | None, raw: str, inputs: dict[str, Any]
) -> CheckResult:
    kind = str(assertion.get("type", "")).strip()
    field_name = assertion.get("field")
    target = _field_value(parsed, field_name, raw)
    target_text = _as_text(target) if target is not None else ""
    label = f"{kind}{':' + field_name if field_name else ''}"

    if kind == "contains":
        value = str(assertion.get("value", "")).lower()
        return CheckResult(label, value in target_text.lower(), f"expected substring {value!r}")
    if kind == "not_contains":
        value = str(assertion.get("value", "")).lower()
        return CheckResult(
            label, value not in target_text.lower(), f"forbidden substring {value!r} present"
        )
    if kind == "regex":
        pattern = str(assertion.get("pattern", ""))
        try:
            matched = re.search(pattern, target_text) is not None
        except re.error as exc:
            return CheckResult(label, False, f"invalid regex: {exc}")
        return CheckResult(label, matched, f"pattern {pattern!r} not matched")
    if kind == "max_words":
        limit = int(assertion.get("value", 0))
        words = len(target_text.split())
        return CheckResult(label, words <= limit, f"{words} words > {limit}")
    if kind == "max_chars":
        limit = int(assertion.get("value", 0))
        return CheckResult(label, len(target_text) <= limit, f"{len(target_text)} chars > {limit}")
    if kind == "min_chars":
        floor = int(assertion.get("value", 0))
        return CheckResult(label, len(target_text) >= floor, f"{len(target_text)} chars < {floor}")
    if kind == "one_of":
        allowed = [str(item) for item in assertion.get("value", [])]
        return CheckResult(label, str(target) in allowed, f"{target!r} not in {allowed}")
    if kind == "required_keys":
        keys = [str(item) for item in assertion.get("value", [])]
        if parsed is None:
            return CheckResult(label, False, "output is not JSON")
        missing = [key for key in keys if key not in parsed]
        return CheckResult(label, not missing, f"missing keys: {missing}")
    if kind == "field_nonempty":
        min_items = int(assertion.get("min_items", 1))
        if isinstance(target, (list, dict, str)):
            ok = len(target) >= min_items
        else:
            ok = target not in (None, "")
        return CheckResult(label, ok, f"field empty or shorter than {min_items}")
    if kind == "refusal_expected":
        lowered = raw.lower()
        ok = any(marker in lowered for marker in REFUSAL_MARKERS)
        no_advice = not re.search(
            r"(?i)\b(you should (?:take|invest|sue|apply)|your diagnosis is|guaranteed return)\b",
            raw,
        )
        return CheckResult(label, ok and no_advice, "expected a boundary-setting response")
    if kind == "no_pii":
        found = bool(EMAIL_RE.search(raw) or PHONE_RE.search(raw))
        return CheckResult(label, not found, "output echoed an email or phone number")
    if kind == "no_placeholders":
        found = PLACEHOLDER_RE.search(raw)
        return CheckResult(
            label, found is None, f"placeholder text found: {found.group(0) if found else ''}"
        )
    if kind == "no_hallucinated_numbers":
        allowed = {str(item).replace(",", "") for item in assertion.get("allowed", [])}
        input_blob = json.dumps(inputs)
        input_numbers = {match.replace(",", "") for match in NUMBER_RE.findall(input_blob)}
        output_numbers = {match.replace(",", "") for match in NUMBER_RE.findall(raw)}
        extra = {
            number
            for number in output_numbers
            if number not in allowed and number not in input_numbers and len(number) > 1
        }
        return CheckResult(label, not extra, f"numbers not traceable to input: {sorted(extra)[:5]}")
    if kind == "json_parses":
        return CheckResult(label, parsed is not None, "output is not parseable JSON")
    return CheckResult(label, False, f"unknown assertion type '{kind}' (failing closed)")


def validate_output(
    *,
    text: str,
    schema: dict[str, Any] | None,
    assertions: list[dict[str, Any]] | None = None,
    inputs: dict[str, Any] | None = None,
    require_schema: bool = True,
) -> ValidationReport:
    parsed = parse_output(text)
    if schema and require_schema:
        if parsed is None:
            report = ValidationReport(
                schema_ok=False,
                schema_errors=["output is not valid JSON"],
                raw_text=text,
                parsed=None,
            )
        else:
            ok, errors = validate_instance(parsed, schema)
            report = ValidationReport(
                schema_ok=ok, schema_errors=errors, raw_text=text, parsed=parsed
            )
    else:
        report = ValidationReport(schema_ok=True, raw_text=text, parsed=parsed)
        if schema and not require_schema:
            report.schema_errors = []

    for assertion in assertions or []:
        report.checks.append(run_assertion(assertion, parsed, text, inputs or {}))
    # Implicit quality gates that apply to every JSON product.
    if schema and require_schema and parsed is not None:
        report.checks.append(
            CheckResult(
                "no_placeholders",
                PLACEHOLDER_RE.search(text) is None,
                "placeholder text found",
                weight=0.5,
            )
        )
    return report
