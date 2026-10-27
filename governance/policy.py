"""Content & safety policy engine.

Default stance (configurable, but safe by default): **do not publish products that
provide professional medical, legal or financial advice**. Assistive drafting is
allowed when the product carries explicit disclaimers, refusal rules and escalation
guidance — and it is always flagged for human review.

Decision precedence: ``BLOCK`` > ``NEEDS_REVIEW`` > ``WARN`` > ``PASS``.

The engine inspects the IR *and* the compiled text for every model family, because
compilation could in principle introduce a phrase the IR did not contain.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.constants import GovernanceDecision, RegulatedDomain, RiskTier
from prompt_ir.schemas import PromptIR

POLICY_VERSION = "2026.10"

PROHIBITED_PATTERNS: list[tuple[str, str]] = [
    (r"\b(child sexual|csam|underage sex)\w*", "csam"),
    (
        r"\b(how to (?:make|build|synthesise|synthesize) (?:a )?(?:bomb|explosive|nerve agent|poison))\b",
        "weapons",
    ),
    (r"\b(3d[- ]print(?:ed)? (?:gun|firearm)|ghost gun|auto ?sear)\b", "weapons"),
    (r"\b(ransomware|keylogger|credential stealer|reverse shell|exploit kit)\b", "malware"),
    (
        r"\b(steal (?:credentials|passwords)|bypass (?:2fa|mfa|authentication)|phishing kit)\b",
        "fraud_credentials",
    ),
    (r"\b(launder money|tax evasion scheme|insider trading scheme)\b", "financial_crime"),
    (r"\b(kill yourself|how to self[- ]harm|suicide method)\b", "self_harm"),
    (
        r"\b(fake reviews?|fabricate (?:evidence|citations)|write a fake (?:reference|doctor'?s note))\b",
        "deception",
    ),
    (
        r"\b(doxx?ing|stalk(?:ing)? someone|spy on (?:my )?(?:partner|employee))\b",
        "harassment_surveillance",
    ),
    (
        r"\b(impersonate (?:a|an) (?:real )?(?:person|police|government official)|deepfake (?:nude|porn))\b",
        "impersonation",
    ),
    (
        r"\b(bypass (?:a )?paywall|scrape (?:behind|past) (?:a )?login|solve captchas? at scale)\b",
        "tos_circumvention",
    ),
    (r"\b(marketplace manipulation|review gating scheme|fake scarcity)\b", "platform_abuse"),
]

DECEPTION_PATTERNS: list[tuple[str, str]] = [
    (r"\bguaranteed (?:results|income|returns|approval|ranking)\b", "guaranteed_outcomes"),
    (r"\b(no (?:risk|downside)|risk[- ]free investment)\b", "risk_free_claims"),
    (r"\bmake \$\d+ (?:per|a) (?:day|week|month)\b", "income_promises"),
    (
        r"\b(undisclosed sponsored|pretend (?:this|it) is (?:real|human))\b",
        "undisclosed_authorship",
    ),
]

ADVICE_VERBS = re.compile(
    r"(?i)\b(you should (?:take|stop taking|invest|sue|sign|apply)|your diagnosis is|"
    r"i diagnose|prescribe|here is your prescription|guaranteed return|this is legal advice|"
    r"this is medical advice|this is financial advice)\b"
)

SENSITIVE_DATA_PATTERNS = re.compile(
    r"(?i)\b(medicare number|tax file number|tfn\b|credit card number|passport number|"
    r"date of birth|home address|driver'?s licen[cs]e)\b"
)

REGULATED_HINTS: dict[str, str] = {
    str(
        RegulatedDomain.MEDICAL
    ): r"\b(diagnos|symptom|prescri|medication|dosage|patient|triage)\w*",
    str(
        RegulatedDomain.MENTAL_HEALTH
    ): r"\b(depress|anxiety|therapy|therapist|mental health|suicid)\w*",
    str(
        RegulatedDomain.LEGAL
    ): r"\b(legal advice|solicitor|attorney|lawsuit|litigation|contract clause|enforceab)\w*",
    str(
        RegulatedDomain.FINANCIAL
    ): r"\b(investment advice|portfolio|superannuation|mortgage|financial advice|credit advice)\w*",
    str(RegulatedDomain.TAX): r"\b(tax return|tax advice|deduction|gst\b|bas\b|ato\b)\w*",
    str(
        RegulatedDomain.IMMIGRATION
    ): r"\b(visa application|immigration advice|migration agent|permanent residency)\w*",
}

ASSISTIVE_MARKERS = (
    "not legal advice",
    "not medical advice",
    "not financial advice",
    "general information",
    "consult a qualified",
    "draft questions",
    "assist",
    "summarise",
    "summarize",
)


@dataclass
class Finding:
    category: str
    severity: str  # info | warn | block
    message: str
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence[:240],
        }


@dataclass
class PolicyResult:
    decision: str
    findings: list[Finding] = field(default_factory=list)
    regulated_domain: str = str(RegulatedDomain.NONE)
    risk_tier: str = str(RiskTier.LOW)
    policy_version: str = POLICY_VERSION
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.decision == str(GovernanceDecision.BLOCK)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "findings": [finding.to_dict() for finding in self.findings],
            "regulated_domain": self.regulated_domain,
            "risk_tier": self.risk_tier,
            "policy_version": self.policy_version,
            "scores": self.scores,
            "blockers": [f.message for f in self.findings if f.severity == "block"],
            "warnings": [f.message for f in self.findings if f.severity == "warn"],
        }


def detect_regulated_domain(text: str, declared: str | None = None) -> str:
    if declared and declared != str(RegulatedDomain.NONE):
        return declared
    for domain, pattern in REGULATED_HINTS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return domain
    return str(RegulatedDomain.NONE)


def check_policy(
    *,
    ir: PromptIR | None = None,
    texts: dict[str, str] | None = None,
    product_meta: dict[str, Any] | None = None,
) -> PolicyResult:
    """Evaluate an IR and/or compiled texts against the policy."""
    findings: list[Finding] = []
    blob_parts: list[str] = list((texts or {}).values())
    if ir is not None:
        blob_parts.append(ir.to_json())
    blob = "\n".join(blob_parts)
    meta = product_meta or {}

    # ---- 1. prohibited content -------------------------------------------
    for pattern, category in PROHIBITED_PATTERNS:
        match = re.search(pattern, blob, re.IGNORECASE)
        if match:
            findings.append(
                Finding(
                    category=f"prohibited:{category}",
                    severity="block",
                    message=f"prohibited content detected ({category})",
                    evidence=match.group(0),
                )
            )

    # ---- 2. deception ----------------------------------------------------
    for pattern, category in DECEPTION_PATTERNS:
        match = re.search(pattern, blob, re.IGNORECASE)
        if match:
            findings.append(
                Finding(
                    category=f"deception:{category}",
                    severity="warn",
                    message=f"potentially misleading claim ({category})",
                    evidence=match.group(0),
                )
            )
    if ADVICE_VERBS.search(blob):
        findings.append(
            Finding(
                category="regulated:direct_advice",
                severity="block",
                message="text contains direct professional-advice language",
                evidence=ADVICE_VERBS.search(blob).group(0) if ADVICE_VERBS.search(blob) else "",
            )
        )

    # ---- 3. regulated domain --------------------------------------------
    # Detect from the *task* (objective, steps, examples), never from exclusions or
    # disclaimers: "do not give legal advice" must not make a product legal-domain.
    declared = ir.safety.regulated_domain if ir else None
    if ir is not None:
        detection_text = "\n".join(
            [
                ir.task.objective,
                ir.task.deliverable,
                " ".join(ir.procedure.steps),
                " ".join(ir.constraints.must_include),
                " ".join(json.dumps(example.input) for example in ir.examples),
                " ".join(
                    json.dumps(example.output)
                    if not isinstance(example.output, str)
                    else example.output
                    for example in ir.examples
                ),
            ]
        )
    else:
        detection_text = blob
    domain = detect_regulated_domain(detection_text, declared)
    if domain != str(RegulatedDomain.NONE):
        disclaimers = list(ir.safety.disclaimers) if ir else []
        refusals = list(ir.safety.refusal_rules) if ir else []
        lowered = blob.lower()
        has_assistive_language = any(marker in lowered for marker in ASSISTIVE_MARKERS)
        if not disclaimers:
            findings.append(
                Finding(
                    category="regulated:missing_disclaimer",
                    severity="block",
                    message=f"regulated domain '{domain}' requires explicit disclaimers",
                )
            )
        if len(refusals) == 0:
            findings.append(
                Finding(
                    category="regulated:missing_refusal_rules",
                    severity="block",
                    message=f"regulated domain '{domain}' requires refusal rules",
                )
            )
        if not has_assistive_language:
            findings.append(
                Finding(
                    category="regulated:not_assistive",
                    severity="warn",
                    message="regulated product should frame output as assistive drafting, not advice",
                )
            )
        findings.append(
            Finding(
                category="regulated:review_required",
                severity="info",
                message=f"regulated domain '{domain}' — assistive-only, human review required before publish",
            )
        )

    # ---- 4. safety scaffolding ------------------------------------------
    if ir is not None:
        if not ir.safety.injection_resistance:
            findings.append(
                Finding(
                    category="safety:no_injection_rule",
                    severity="block",
                    message="missing prompt-injection resistance rule",
                )
            )
        if not ir.safety.disclaimers and domain != str(RegulatedDomain.NONE):
            findings.append(
                Finding(
                    category="safety:no_disclaimer",
                    severity="block",
                    message="regulated product missing disclaimer",
                )
            )
        if not ir.safety.pii_handling:
            findings.append(
                Finding(
                    category="safety:no_pii_rule",
                    severity="warn",
                    message="no explicit personal-data handling rule",
                )
            )
        for index, example in enumerate(ir.examples):
            if example.origin != "synthetic" and not example.rights_note:
                findings.append(
                    Finding(
                        category="ip:unlicensed_example",
                        severity="block",
                        message=f"example #{index} is non-synthetic without a rights note",
                    )
                )
        if ir.provenance.origin not in {
            "synthetic",
            "operator_provided",
            "licensed",
            "public_domain",
        }:
            findings.append(
                Finding(
                    category="ip:unknown_origin",
                    severity="block",
                    message="prompt provenance origin is not permitted",
                )
            )

    # ---- 5. personal data ------------------------------------------------
    if SENSITIVE_DATA_PATTERNS.search(blob):
        findings.append(
            Finding(
                category="privacy:sensitive_identifiers",
                severity="warn",
                message="references sensitive identifiers — ensure minimisation and no retention",
                evidence=SENSITIVE_DATA_PATTERNS.search(blob).group(0)
                if SENSITIVE_DATA_PATTERNS.search(blob)
                else "",
            )
        )

    # ---- 6. listing claims (metadata) ------------------------------------
    for key in ("title", "description"):
        value = str(meta.get(key, ""))
        for pattern, category in DECEPTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                findings.append(
                    Finding(
                        category=f"listing:{category}",
                        severity="block",
                        message=f"listing {key} contains a claim we cannot substantiate ({category})",
                        evidence=value[:120],
                    )
                )

    # ---- decision --------------------------------------------------------
    if any(finding.severity == "block" for finding in findings):
        decision = str(GovernanceDecision.BLOCK)
    elif domain != str(RegulatedDomain.NONE) or any(
        finding.severity == "warn" for finding in findings
    ):
        decision = str(GovernanceDecision.NEEDS_REVIEW)
    else:
        decision = str(GovernanceDecision.PASS)

    block_count = sum(1 for finding in findings if finding.severity == "block")
    warn_count = sum(1 for finding in findings if finding.severity == "warn")
    risk_tier = (
        str(RiskTier.PROHIBITED)
        if decision == str(GovernanceDecision.BLOCK) and block_count > 1
        else str(RiskTier.HIGH)
        if decision == str(GovernanceDecision.BLOCK) or block_count
        else str(RiskTier.MEDIUM)
        if domain != str(RegulatedDomain.NONE) or warn_count
        else str(RiskTier.LOW)
    )
    return PolicyResult(
        decision=decision,
        findings=findings,
        regulated_domain=domain,
        risk_tier=risk_tier,
        scores={"blockers": float(block_count), "warnings": float(warn_count)},
    )


def policy_snapshot() -> dict[str, Any]:
    return {
        "version": POLICY_VERSION,
        "jurisdiction": "AU",
        "defaults": {
            "regulated_domains": "assistive drafting only; disclaimers + refusal rules required",
            "human_approval": "required before publishing",
            "copying": "no third-party prompt text; synthetic or rights-cleared examples only",
        },
        "prohibited_categories": [category for _, category in PROHIBITED_PATTERNS],
        "deception_categories": [category for _, category in DECEPTION_PATTERNS],
    }
