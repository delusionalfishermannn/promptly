"""Niche knowledge base for *original* IR synthesis.

Nothing here is copied from a competitor. Each entry is a first-principles description
of a deliverable's contract: the fields a professional would expect, the steps to
produce them, and the failure modes to guard against. The synthesiser composes these
into a fresh IR for a specific buyer segment and topic keywords.

Adding a niche = adding an entry to ``NICHE_TEMPLATES``. No prompt text is ever loaded
from third-party sources (see DESIGN.md §2.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.constants import RegulatedDomain

_SCHEMA_STRING = {"type": "string"}
_SCHEMA_STRINGS = {"type": "array", "items": {"type": "string"}, "minItems": 1}
_SCHEMA_NUMBER = {"type": "number"}


@dataclass(frozen=True)
class NicheTemplate:
    key: str
    label: str
    deliverable: str
    objective: str
    output_fields: tuple[tuple[str, dict[str, Any]], ...]
    procedure: tuple[str, ...]
    checklist: tuple[str, ...]
    exclusions: tuple[str, ...]
    success_criteria: tuple[str, ...]
    refusal_rules: tuple[str, ...] = ()
    regulated_domain: str = str(RegulatedDomain.NONE)
    disclaimers: tuple[str, ...] = ()
    escalate_when: tuple[str, ...] = ()
    example: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def output_schema(self) -> dict[str, Any]:
        properties = {name: schema for name, schema in self.output_fields}
        # A universal escape hatch: models should say what they are missing, not invent.
        properties.setdefault(
            "needs_more_information",
            {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        )
        return {
            "type": "object",
            "properties": properties,
            "required": [name for name, _ in self.output_fields],
            "additionalProperties": False,
        }

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {"type": "string", "description": "What the user wants produced"},
                "source_material": {
                    "type": "string",
                    "description": "Emails, notes, ticket text or data to work from",
                },
                "context": {
                    "type": "string",
                    "description": "Audience, product or account context",
                },
                "tone": {"type": "string", "description": "Desired tone of voice"},
            },
            "required": ["request", "source_material"],
        }


NICHE_TEMPLATES: dict[str, NicheTemplate] = {
    "billing": NicheTemplate(
        key="billing",
        label="Billing & invoice communication",
        deliverable="a billing email plus a structured payment summary",
        objective=(
            "Turn rough notes about an unpaid or disputed invoice into a courteous, unambiguous "
            "billing message and a machine-readable summary of the amounts, due dates and next actions."
        ),
        output_fields=(
            ("subject_line", {**_SCHEMA_STRING, "maxLength": 120}),
            ("email_body", {**_SCHEMA_STRING, "description": "Plain-text email, <= 220 words"}),
            ("summary", {**_SCHEMA_STRING, "description": "One-sentence internal summary"}),
            ("amount_due", {**_SCHEMA_NUMBER, "minimum": 0}),
            ("currency", {"type": "string", "enum": ["AUD", "USD", "EUR", "GBP", "NZD"]}),
            ("due_date", {"type": "string", "format": "date"}),
            ("next_actions", _SCHEMA_STRINGS),
            ("escalation_needed", {"type": "boolean"}),
        ),
        procedure=(
            "Extract amounts, dates and any dispute points from the source material.",
            "Decide the customer relationship state (first reminder, overdue, disputed, final notice).",
            "Choose a tone that preserves the relationship while being explicit about the amount and deadline.",
            "Draft the subject line and body; keep every factual claim traceable to the input.",
            "Cross-check the numbers in the body against the structured summary fields.",
            "List the next actions and flag escalation only when the input supports it.",
        ),
        checklist=(
            "Amounts and dates in the body match the summary fields exactly.",
            "No fabricated payment terms, legal threats or interest charges.",
            "One clear call to action with a deadline.",
            "Plain English; no jargon or blame.",
        ),
        exclusions=(
            "Debt collection tactics or threats.",
            "Legal advice about contract enforceability.",
            "Inventing invoice numbers, amounts or payment terms.",
        ),
        success_criteria=(
            "Human could send the email with no edits.",
            "Summary fields reconcile to the input.",
            "Tone is firm but relationship-preserving.",
        ),
        example={
            "input": {
                "request": "Second reminder for invoice 1042",
                "source_material": "Invoice 1042, A$1,850, issued 3 Sep, due 17 Sep, client asked for a PO number twice.",
                "context": "Long-standing SMB client, accounts payable team",
                "tone": "friendly but firm",
            },
            "output": {
                "subject_line": "Invoice 1042 — A$1,850 due 17 September (PO reference attached)",
                "email_body": "Hi Sam, ... [illustrative example of a short, specific reminder]",
                "summary": "Second reminder for invoice 1042 (A$1,850, overdue since 17 Sep).",
                "amount_due": 1850,
                "currency": "AUD",
                "due_date": "2026-09-17",
                "next_actions": ["Attach PO reference", "Confirm payment date"],
                "escalation_needed": False,
                "needs_more_information": [],
            },
        },
        tags=("billing", "email", "smb"),
    ),
    "email": NicheTemplate(
        key="email",
        label="Lifecycle email copy",
        deliverable="a lifecycle email with variants and a structured campaign summary",
        objective=(
            "Produce a single lifecycle/outreach email (or 2-3 subject-line variants) grounded in "
            "supplied context, with an explicit value proposition, one CTA and compliance-safe wording."
        ),
        output_fields=(
            (
                "subject_lines",
                {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
            ),
            ("preview_text", {**_SCHEMA_STRING, "maxLength": 140}),
            ("body", _SCHEMA_STRING),
            ("cta", {**_SCHEMA_STRING, "maxLength": 80}),
            ("summary", _SCHEMA_STRING),
            ("personalisation_used", _SCHEMA_STRINGS),
            ("claims_requiring_evidence", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Identify the audience, their current stage and the single desired action.",
            "Extract concrete, verifiable details from the source material to personalise with.",
            "Draft 2-3 distinct subject lines (benefit, curiosity, urgency) without clickbait.",
            "Write the body: context, value, proof, CTA. Keep it under 180 words.",
            "List any claim that needs evidence before sending.",
            "Check against spam-trigger wording and unsubscribe/compliance requirements.",
        ),
        checklist=(
            "Exactly one primary CTA.",
            "No fabricated customer names, metrics or testimonials.",
            "Subject lines under 60 characters where possible.",
            "Every personalisation token has a real value in the input.",
        ),
        exclusions=(
            "Fake urgency or manufactured scarcity.",
            "Unverified performance claims.",
            "Impersonation of a real person or brand.",
        ),
        success_criteria=(
            "Copy is ready for review with no placeholders except intentional tokens.",
            "Claims are flagged rather than invented.",
        ),
        refusal_rules=("Do not write deceptive subject lines or spoof sender identities.",),
        example={
            "input": {
                "request": "Win-back email for lapsed subscribers",
                "source_material": "Subscription lapsed 45 days ago; last used template pack; open rate fell.",
                "context": "Creator newsletter tool, 2,400 subscribers",
                "tone": "warm, concise",
            },
            "output": {
                "subject_lines": ["Your templates are still here", "Pick up where you left off"],
                "preview_text": "A short, specific reason to come back.",
                "body": "...",
                "cta": "Restore my templates",
                "summary": "Win-back email for 45-day lapsed subscribers.",
                "personalisation_used": ["last_template_pack"],
                "claims_requiring_evidence": [],
                "needs_more_information": [],
            },
        },
        tags=("email", "marketing"),
    ),
    "seo": NicheTemplate(
        key="seo",
        label="SEO content structuring",
        deliverable="a search-intent-aligned article outline with metadata",
        objective=(
            "Convert a keyword and source notes into an article outline that satisfies search intent, "
            "with headings, evidence gaps and honest metadata — no invented facts or fake statistics."
        ),
        output_fields=(
            ("title", {**_SCHEMA_STRING, "maxLength": 70}),
            ("meta_description", {**_SCHEMA_STRING, "maxLength": 160}),
            (
                "search_intent",
                {
                    "type": "string",
                    "enum": ["informational", "commercial", "transactional", "navigational"],
                },
            ),
            (
                "outline",
                {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "points": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["heading", "points"],
                    },
                },
            ),
            ("target_keywords", _SCHEMA_STRINGS),
            ("internal_link_ideas", {"type": "array", "items": {"type": "string"}}),
            ("evidence_gaps", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Classify the dominant search intent from the keyword and notes.",
            "Map the questions a searcher needs answered, in the order they ask them.",
            "Build 3-7 H2 sections, each with 2-4 supporting points.",
            "Identify claims that need citations or original data (evidence gaps).",
            "Write title and meta description within length limits, no clickbait.",
            "Suggest 2-4 internal link targets from the supplied site context only.",
        ),
        checklist=(
            "Outline answers the query in the first two sections.",
            "No statistics invented or implied without a source.",
            "Title <= 70 chars, meta <= 160 chars.",
            "Headings are descriptive, not keyword-stuffed.",
        ),
        exclusions=(
            "Guaranteed ranking claims.",
            "Scraped competitor content or copied outlines.",
            "Keyword stuffing and doorway pages.",
        ),
        success_criteria=(
            "A writer could draft the article from the outline without further research "
            "except the flagged gaps.",
        ),
        example={
            "input": {
                "request": "Outline for 'prompt templates for invoicing'",
                "source_material": "We sell prompt packs to Australian small businesses.",
                "context": "Site has pages on invoicing basics and cashflow.",
                "tone": "practical",
            },
            "output": {
                "title": "Prompt Templates for Invoicing: What to Include",
                "meta_description": "A practical structure for invoicing prompts that keep your cashflow predictable.",
                "search_intent": "informational",
                "outline": [
                    {
                        "heading": "Why invoice wording matters",
                        "points": ["Payment timing", "Dispute prevention"],
                    }
                ],
                "target_keywords": ["prompt templates for invoicing", "invoice email template"],
                "internal_link_ideas": ["/invoicing-basics"],
                "evidence_gaps": ["Average payment delay for Australian SMBs (needs a citation)"],
                "needs_more_information": [],
            },
        },
        tags=("seo", "content"),
    ),
    "support": NicheTemplate(
        key="support",
        label="Customer support responses",
        deliverable="a support reply plus a structured resolution record",
        objective=(
            "Turn a raw customer complaint or ticket into an empathetic, accurate reply with a clear "
            "resolution path, and a structured record for the helpdesk."
        ),
        output_fields=(
            ("reply", _SCHEMA_STRING),
            ("summary", _SCHEMA_STRING),
            (
                "issue_category",
                {
                    "type": "string",
                    "enum": ["billing", "bug", "how_to", "feature_request", "account", "other"],
                },
            ),
            ("sentiment", {"type": "string", "enum": ["negative", "neutral", "positive"]}),
            ("severity", {"type": "string", "enum": ["low", "medium", "high", "critical"]}),
            ("resolution_steps", _SCHEMA_STRINGS),
            ("requires_human_escalation", {"type": "boolean"}),
            ("sla_note", {**_SCHEMA_STRING, "maxLength": 160}),
        ),
        procedure=(
            "Read the ticket and restate the customer's problem in one sentence.",
            "Classify issue category, sentiment and severity using the declared enums.",
            "Acknowledge the impact before explaining anything.",
            "Give the shortest correct resolution path; if unknown, say what you need.",
            "State next steps and expected response time without inventing policies.",
            "Flag escalation for refunds, legal threats, data-privacy or safety issues.",
        ),
        checklist=(
            "Acknowledges the specific problem, not a generic apology.",
            "No invented policy, refund entitlement or timeframe.",
            "Enums used exactly as declared.",
            "Escalation flagged for sensitive categories.",
        ),
        exclusions=(
            "Promising refunds or credits without policy input.",
            "Sharing other customers' data.",
            "Admitting legal liability.",
        ),
        success_criteria=(
            "Customer would feel heard and know exactly what happens next.",
            "Agent could send with no edits and log the record as-is.",
        ),
        refusal_rules=(
            "Do not promise refunds, credits or policy exceptions that the input does not authorise.",
        ),
        example={
            "input": {
                "request": "Reply to a frustrated customer",
                "source_material": "Customer says the export failed twice and they lost a morning's work.",
                "context": "SaaS plan: Pro, no known outage",
                "tone": "apologetic, practical",
            },
            "output": {
                "reply": "I'm sorry — losing a morning to a failed export is genuinely frustrating...",
                "summary": "Pro customer: repeated export failure caused data loss; needs workaround.",
                "issue_category": "bug",
                "sentiment": "negative",
                "severity": "high",
                "resolution_steps": ["Reproduce with the same filters", "Offer CSV fallback"],
                "requires_human_escalation": True,
                "sla_note": "Bug triage within one business day.",
                "needs_more_information": ["Export filter settings"],
            },
        },
        tags=("support", "cx"),
    ),
    "career": NicheTemplate(
        key="career",
        label="Career documents",
        deliverable="a tailored resume section and interview prep notes",
        objective=(
            "Rewrite supplied career material for a specific role using only the candidate's real "
            "experience, with quantified achievements where the input supports them."
        ),
        output_fields=(
            ("headline", {**_SCHEMA_STRING, "maxLength": 120}),
            ("summary", {**_SCHEMA_STRING, "maxLength": 600}),
            (
                "bullet_points",
                {"type": "array", "minItems": 3, "maxItems": 8, "items": {"type": "string"}},
            ),
            ("keywords_matched", _SCHEMA_STRINGS),
            ("gaps", {"type": "array", "items": {"type": "string"}}),
            ("interview_questions", {"type": "array", "items": {"type": "string"}}),
            ("claims_to_verify", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Extract the role's requirements from the job description in the input.",
            "Map the candidate's real experience to each requirement.",
            "Rewrite achievements as action + method + measurable result where numbers exist.",
            "Flag gaps instead of inventing experience.",
            "Generate 3-5 likely interview questions targeting the mapped experience.",
            "List every claim the candidate must verify before sending.",
        ),
        checklist=(
            "No invented employers, dates, titles, qualifications or metrics.",
            "Bullets use strong verbs and are under 30 words.",
            "Gaps are named explicitly.",
        ),
        exclusions=(
            "Fabricating experience or credentials.",
            "Writing discriminatory or ageist content.",
            "Guaranteeing interviews or offers.",
        ),
        success_criteria=("Candidate can defend every line in an interview.",),
        example={
            "input": {
                "request": "Tailor my resume to a product manager role",
                "source_material": "5 years ops, led migration project, no PM title.",
                "context": "Mid-size SaaS company",
                "tone": "confident, factual",
            },
            "output": {
                "headline": "Operations leader moving into product",
                "summary": "Ops lead who took a migration from plan to launch...",
                "bullet_points": ["Led a 14-person migration to a new ticketing platform..."],
                "keywords_matched": ["stakeholder management", "migration"],
                "gaps": ["No formal product roadmap ownership"],
                "interview_questions": [
                    "Tell me about a time you prioritised competing requirements."
                ],
                "claims_to_verify": ["Team size in the migration"],
                "needs_more_information": [],
            },
        },
        tags=("career", "resume"),
    ),
    "ads": NicheTemplate(
        key="ads",
        label="Paid ad copy",
        deliverable="ad variants with compliance flags and a structured test plan",
        objective=(
            "Create platform-appropriate ad copy variants grounded in supplied product facts, "
            "with character limits respected and unverifiable claims flagged."
        ),
        output_fields=(
            (
                "variants",
                {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "headline": {"type": "string", "maxLength": 40},
                            "primary_text": {"type": "string", "maxLength": 200},
                            "cta": {"type": "string"},
                        },
                        "required": ["headline", "primary_text", "cta"],
                    },
                },
            ),
            ("angle", {**_SCHEMA_STRING, "maxLength": 120}),
            ("compliance_flags", {"type": "array", "items": {"type": "string"}}),
            ("claims_requiring_evidence", {"type": "array", "items": {"type": "string"}}),
            ("test_plan", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Identify the product's single strongest verifiable benefit from the input.",
            "Choose one persuasion angle (problem, outcome, proof, comparison).",
            "Write 3-5 variants respecting platform character limits.",
            "Flag every claim that needs substantiation for ad policy compliance.",
            "Propose a clean A/B test: one variable per comparison.",
        ),
        checklist=(
            "Character limits respected.",
            "No prohibited claims (health, financial, guaranteed outcomes).",
            "No competitor trademarks or impersonation.",
            "One variable differs between compared variants.",
        ),
        exclusions=(
            "Misleading or unsubstantiated claims.",
            "Before/after body imagery claims.",
            "Targeting protected attributes.",
        ),
        success_criteria=("Variants are policy-safe and testable as described.",),
        refusal_rules=(
            "Do not write ads that promise guaranteed financial, health or legal outcomes.",
        ),
        example={
            "input": {
                "request": "Meta ads for a prompt pack",
                "source_material": "Pack includes 40 tested prompts for SMB invoicing; A$29.",
                "context": "Australian small business owners",
                "tone": "plain, useful",
            },
            "output": {
                "variants": [
                    {
                        "headline": "40 tested invoicing prompts",
                        "primary_text": "Cut invoice admin without guessing at prompts. A$29, instant download.",
                        "cta": "Download",
                    }
                ],
                "angle": "outcome",
                "compliance_flags": ["Price must match checkout page"],
                "claims_requiring_evidence": ["'tested' — link to evaluation summary"],
                "test_plan": ["A/B headline only; keep primary text constant"],
                "needs_more_information": [],
            },
        },
        tags=("ads", "marketing"),
    ),
    "onboarding": NicheTemplate(
        key="onboarding",
        label="Training & onboarding material",
        deliverable="a lesson outline with exercises and assessment rubric",
        objective=(
            "Design a practical onboarding lesson from supplied subject matter, with measurable "
            "learning outcomes, exercises and a rubric."
        ),
        output_fields=(
            ("learning_outcomes", {"type": "array", "minItems": 2, "items": {"type": "string"}}),
            (
                "lesson_plan",
                {
                    "type": "array",
                    "minItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment": {"type": "string"},
                            "content": {"type": "string"},
                            "minutes": {"type": "integer"},
                        },
                        "required": ["segment", "content", "minutes"],
                    },
                },
            ),
            ("exercise", _SCHEMA_STRING),
            (
                "assessment_rubric",
                {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "levels": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["criterion", "levels"],
                    },
                },
            ),
            ("prerequisites", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "State 2-4 observable learning outcomes using action verbs.",
            "Sequence the lesson into short segments with time estimates.",
            "Design one exercise that tests the primary outcome.",
            "Write a rubric with 3 performance levels per criterion.",
            "List prerequisites and required materials from the input only.",
        ),
        checklist=(
            "Outcomes are observable and assessable.",
            "Total time matches the requested duration if given.",
            "Rubric levels are distinguishable.",
        ),
        exclusions=(
            "Accreditation claims.",
            "Content outside the supplied subject matter.",
        ),
        success_criteria=("A facilitator could run the session without further preparation.",),
        example={
            "input": {
                "request": "2-hour onboarding on our CRM",
                "source_material": "CRM has contacts, deals, tasks; sales team of 6.",
                "context": "New hires, mixed experience",
                "tone": "clear, practical",
            },
            "output": {
                "learning_outcomes": ["Create and qualify a contact", "Move a deal through stages"],
                "lesson_plan": [
                    {"segment": "Why pipeline hygiene matters", "content": "...", "minutes": 15}
                ],
                "exercise": "Import a sample contact and assign a task",
                "assessment_rubric": [
                    {"criterion": "Data hygiene", "levels": ["needs work", "meets", "exceeds"]}
                ],
                "prerequisites": ["CRM login"],
                "needs_more_information": [],
            },
        },
        tags=("onboarding", "training"),
    ),
    "product": NicheTemplate(
        key="product",
        label="Product specification",
        deliverable="a product spec with user stories, acceptance criteria and risks",
        objective=(
            "Convert a raw feature idea into a spec with explicit scope, user stories, "
            "acceptance criteria and open questions — no invented metrics or requirements."
        ),
        output_fields=(
            ("problem_statement", _SCHEMA_STRING),
            (
                "scope",
                {
                    "type": "object",
                    "properties": {
                        "in_scope": {"type": "array", "items": {"type": "string"}},
                        "out_of_scope": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["in_scope", "out_of_scope"],
                },
            ),
            (
                "user_stories",
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "story": {"type": "string"},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["story", "acceptance_criteria"],
                    },
                },
            ),
            ("risks", {"type": "array", "items": {"type": "string"}}),
            ("open_questions", {"type": "array", "items": {"type": "string"}}),
            ("success_metrics", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Restate the problem and who has it, using the supplied context only.",
            "Separate in-scope from out-of-scope explicitly.",
            "Write user stories with testable acceptance criteria.",
            "List risks including data, security and accessibility.",
            "List open questions rather than assuming answers.",
            "Propose measurable success metrics if a baseline exists in the input.",
        ),
        checklist=(
            "Every user story has at least one testable acceptance criterion.",
            "Out-of-scope list is non-empty.",
            "No invented baselines or competitor claims.",
        ),
        exclusions=("Committed delivery dates without team input.",),
        success_criteria=("An engineer could estimate the work from the spec.",),
        example={
            "input": {
                "request": "Spec for CSV import of contacts",
                "source_material": "Users currently add contacts one at a time; support tickets ask for bulk import.",
                "context": "B2B SaaS, SMB accounts",
                "tone": "precise",
            },
            "output": {
                "problem_statement": "SMB admins must add contacts one at a time, wasting hours at setup.",
                "scope": {
                    "in_scope": ["CSV upload", "Field mapping"],
                    "out_of_scope": ["API import"],
                },
                "user_stories": [
                    {
                        "story": "As an admin, I can upload a CSV and map fields.",
                        "acceptance_criteria": ["Invalid rows are reported with line numbers"],
                    }
                ],
                "risks": ["Duplicate contacts on re-import"],
                "open_questions": ["Which encodings must be supported?"],
                "success_metrics": ["Median contacts imported per account"],
                "needs_more_information": [],
            },
        },
        tags=("product", "spec"),
    ),
    "data": NicheTemplate(
        key="data",
        label="Data analysis summaries",
        deliverable="a defensible analysis summary with caveats",
        objective=(
            "Summarise supplied data or query results into a decision-ready analysis, separating "
            "observation from interpretation and stating limitations explicitly."
        ),
        output_fields=(
            ("headline", {**_SCHEMA_STRING, "maxLength": 160}),
            ("observations", {"type": "array", "minItems": 1, "items": {"type": "string"}}),
            ("interpretation", _SCHEMA_STRING),
            ("limitations", {"type": "array", "minItems": 1, "items": {"type": "string"}}),
            ("recommended_actions", {"type": "array", "items": {"type": "string"}}),
            ("confidence", {**_SCHEMA_NUMBER, "minimum": 0, "maximum": 1}),
        ),
        procedure=(
            "State what the data can and cannot answer.",
            "Write observations as facts traceable to the input numbers.",
            "Write interpretation separately, explicitly labelled as judgement.",
            "List limitations: sample size, missing fields, time window, confounders.",
            "Recommend at most three actions proportional to confidence.",
        ),
        checklist=(
            "Every number appears in the input.",
            "Interpretation is visibly separated from observation.",
            "Limitations non-empty and specific.",
        ),
        exclusions=(
            "Causal claims from observational data.",
            "Extrapolation beyond the time window.",
        ),
        success_criteria=("A decision-maker can act without over-reading the data.",),
        example={
            "input": {
                "request": "Summarise last month's sales data",
                "source_material": "42 sales, A$3,980 revenue, 7 refunds, top product: invoicing pack.",
                "context": "New storefront, first 60 days",
                "tone": "neutral",
            },
            "output": {
                "headline": "42 sales in month one; invoicing pack dominates",
                "observations": ["42 sales, A$3,980 gross", "7 refunds (16.7%)"],
                "interpretation": "Refund rate is high for a digital product and likely reflects expectation mismatch.",
                "limitations": ["One month of data", "No traffic source breakdown"],
                "recommended_actions": ["Add clearer pre-purchase scope to the listing"],
                "confidence": 0.45,
                "needs_more_information": ["Refund reasons"],
            },
        },
        tags=("data", "analysis"),
    ),
    "meetings": NicheTemplate(
        key="meetings",
        label="Meetings & decisions",
        deliverable="structured meeting notes with decisions and owners",
        objective=(
            "Turn a transcript or rough notes into structured minutes: decisions, action items with "
            "owners and dates, and unresolved questions."
        ),
        output_fields=(
            ("summary", {**_SCHEMA_STRING, "maxLength": 500}),
            (
                "decisions",
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["decision"],
                    },
                },
            ),
            (
                "action_items",
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "owner": {"type": "string"},
                            "due": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            ),
            ("open_questions", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Identify participants and purpose from the input.",
            "Extract decisions explicitly made (not topics discussed).",
            "Extract action items with owner and due date when stated.",
            "List unresolved questions without inventing owners.",
            "Verify no personal or sensitive details are included unnecessarily.",
        ),
        checklist=(
            "Decisions are stated, not implied.",
            "Owners and dates only where the input states them.",
            "No fabricated attendees.",
        ),
        exclusions=("Attributing statements to people not named in the input.",),
        success_criteria=("Attendees can confirm the record in under a minute.",),
        example={
            "input": {
                "request": "Minutes from the pricing meeting",
                "source_material": "Team agreed to trial A$29 for invoicing pack; Sam to update the page by Friday.",
                "context": "Four-person team",
                "tone": "neutral",
            },
            "output": {
                "summary": "Team agreed to trial A$29 pricing for the invoicing pack.",
                "decisions": [
                    {"decision": "Trial A$29", "rationale": "Lower entry price for SMBs"}
                ],
                "action_items": [{"task": "Update pricing page", "owner": "Sam", "due": "Friday"}],
                "open_questions": ["When do we review the trial?"],
                "needs_more_information": [],
            },
        },
        tags=("meetings", "ops"),
    ),
    "legal": NicheTemplate(
        key="legal",
        label="Legal-adjacent drafting (assistive only)",
        deliverable="an assistive policy/contract summary with issues for a qualified professional",
        objective=(
            "Summarise supplied policy or contract text, flag issues and draft non-binding questions "
            "for a qualified lawyer. Never provide legal advice or conclusions."
        ),
        output_fields=(
            ("plain_summary", _SCHEMA_STRING),
            ("key_obligations", {"type": "array", "minItems": 1, "items": {"type": "string"}}),
            ("issues_for_counsel", {"type": "array", "minItems": 1, "items": {"type": "string"}}),
            ("questions_to_ask", {"type": "array", "minItems": 1, "items": {"type": "string"}}),
            ("missing_information", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Summarise the document in plain English without legal conclusions.",
            "List obligations and deadlines stated in the text.",
            "Identify ambiguities and risks as questions, not conclusions.",
            "Draft questions for a qualified legal professional.",
            "Include the mandatory disclaimer verbatim.",
        ),
        checklist=(
            "No legal advice, no 'you should' recommendations on legal risk.",
            "Disclaimer present verbatim.",
            "Every issue is framed as a question.",
        ),
        exclusions=(
            "Legal advice or opinions on enforceability.",
            "Drafting binding clauses without professional review.",
        ),
        success_criteria=(
            "A lawyer could review the flagged list quickly; user understands the doc.",
        ),
        refusal_rules=(
            "Refuse to give legal advice, interpret law, or state whether a clause is enforceable.",
            "Refuse requests to draft documents for signature without professional review.",
        ),
        regulated_domain=str(RegulatedDomain.LEGAL),
        disclaimers=(
            "This is general information, not legal advice. Consult a qualified Australian lawyer "
            "before acting on anything in this document.",
        ),
        escalate_when=(
            "The input involves litigation, deadlines, penalties or personal liability.",
        ),
        example={
            "input": {
                "request": "Summarise this supplier agreement",
                "source_material": "Clause 7: payment within 45 days; Clause 12: unilateral price variation.",
                "context": "SMB buying software",
                "tone": "neutral",
            },
            "output": {
                "plain_summary": "The agreement sets 45-day payment terms and allows the supplier to change prices unilaterally.",
                "key_obligations": ["Pay within 45 days"],
                "issues_for_counsel": ["Unilateral price variation may affect budgeting"],
                "questions_to_ask": ["Can we add a notice period for price changes?"],
                "missing_information": ["Termination clause"],
                "needs_more_information": [],
            },
        },
        tags=("legal", "assistive"),
    ),
    "health": NicheTemplate(
        key="health",
        label="Health information (assistive only)",
        deliverable="a non-diagnostic health information summary with clinician questions",
        objective=(
            "Summarise supplied health information in plain language, prepare questions for a "
            "clinician, and never diagnose, prescribe or triage."
        ),
        output_fields=(
            ("plain_summary", _SCHEMA_STRING),
            (
                "questions_for_clinician",
                {"type": "array", "minItems": 1, "items": {"type": "string"}},
            ),
            ("information_to_gather", {"type": "array", "items": {"type": "string"}}),
            ("red_flags_to_mention", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Restate the supplied information neutrally, without interpretation as diagnosis.",
            "Draft questions the person can ask a clinician.",
            "List information that would help the clinician.",
            "Include the disclaimer verbatim and advise urgent care if red flags are described.",
        ),
        checklist=(
            "No diagnosis, dosage or treatment recommendation.",
            "Disclaimer present verbatim.",
            "Urgent-care advice included when red flags appear.",
        ),
        exclusions=("Diagnosis, triage, dosage or treatment plans.",),
        success_criteria=(
            "User is better prepared for a clinical appointment, not self-treating.",
        ),
        refusal_rules=(
            "Refuse to diagnose, interpret test results, recommend medication or dosage.",
            "Refuse to advise delaying medical care.",
        ),
        regulated_domain=str(RegulatedDomain.MEDICAL),
        disclaimers=(
            "This is general information only, not medical advice or a diagnosis. Speak to a "
            "registered health professional. In an emergency call 000 (Australia).",
        ),
        escalate_when=(
            "Chest pain, breathing difficulty, severe bleeding or thoughts of self-harm.",
        ),
        example={
            "input": {
                "request": "Help me prepare for an appointment",
                "source_material": "Recurring headaches for three weeks, worse in the afternoon.",
                "context": "Adult, no diagnosis yet",
                "tone": "calm",
            },
            "output": {
                "plain_summary": "You have noted recurring afternoon headaches over three weeks.",
                "questions_for_clinician": ["What could cause afternoon-pattern headaches?"],
                "information_to_gather": ["Sleep pattern", "Screen time", "Hydration"],
                "red_flags_to_mention": ["Any vision changes or nausea"],
                "needs_more_information": [],
            },
        },
        tags=("health", "assistive"),
    ),
    "finance": NicheTemplate(
        key="finance",
        label="Financial summaries (assistive only)",
        deliverable="a cashflow/budget summary with decisions to discuss with an adviser",
        objective=(
            "Summarise supplied figures into a plain-language cashflow view with questions for an "
            "accountant. Never give investment, tax or lending advice."
        ),
        output_fields=(
            ("summary", _SCHEMA_STRING),
            (
                "key_figures",
                {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["label", "value"],
                    },
                },
            ),
            ("risks", {"type": "array", "items": {"type": "string"}}),
            (
                "questions_for_adviser",
                {"type": "array", "minItems": 1, "items": {"type": "string"}},
            ),
            ("assumptions", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Restate the figures exactly as supplied.",
            "Compute only simple, shown arithmetic and state assumptions.",
            "Identify cashflow risks without recommending financial products.",
            "Draft questions for a registered adviser or accountant.",
            "Include the disclaimer verbatim.",
        ),
        checklist=(
            "Every figure traceable to the input.",
            "No investment, tax or lending recommendations.",
            "Disclaimer present verbatim.",
        ),
        exclusions=("Investment, tax, superannuation or credit advice.",),
        success_criteria=("Owner understands their position and knows what to ask.",),
        refusal_rules=(
            "Refuse to recommend financial products, tax positions or lending decisions.",
        ),
        regulated_domain=str(RegulatedDomain.FINANCIAL),
        disclaimers=(
            "General information only, not financial, tax or credit advice. Consult a registered "
            "tax agent or licensed adviser before acting.",
        ),
        escalate_when=("Solvency concerns or pending ATO obligations are raised.",),
        example={
            "input": {
                "request": "Summarise this month's cash position",
                "source_material": "Opening A$12,400; income A$6,200; expenses A$9,850.",
                "context": "Sole trader",
                "tone": "plain",
            },
            "output": {
                "summary": "Cash fell by A$3,650 this month.",
                "key_figures": [{"label": "Closing cash", "value": "A$8,750"}],
                "risks": ["Two consecutive months of negative cashflow"],
                "questions_for_adviser": ["Should I set aside GST now or at BAS time?"],
                "assumptions": ["All income received in the period"],
                "needs_more_information": ["GST treatment"],
            },
        },
        tags=("finance", "assistive"),
    ),
    "general": NicheTemplate(
        key="general",
        label="General structured assistant",
        deliverable="a structured, reusable output for the requested task",
        objective=(
            "Produce a structured, immediately usable deliverable for the requested task, using only "
            "the supplied material and stating clearly what is missing."
        ),
        output_fields=(
            ("result", _SCHEMA_STRING),
            ("summary", {**_SCHEMA_STRING, "maxLength": 300}),
            ("steps", {"type": "array", "items": {"type": "string"}}),
            ("assumptions", {"type": "array", "items": {"type": "string"}}),
        ),
        procedure=(
            "Clarify the requested deliverable from the input.",
            "Identify the audience and the decision the output supports.",
            "Draft the deliverable in the declared structure.",
            "List assumptions explicitly.",
            "Flag anything missing rather than inventing it.",
        ),
        checklist=(
            "Output matches the schema.",
            "Assumptions listed.",
            "No invented facts.",
        ),
        exclusions=("Content outside the stated objective.",),
        success_criteria=("Output is usable without further editing.",),
        example={
            "input": {
                "request": "Summarise these notes into a plan",
                "source_material": "Three tasks, one deadline next week.",
                "context": "Solo operator",
                "tone": "direct",
            },
            "output": {
                "result": "A three-step plan sequenced by deadline risk.",
                "summary": "Plan for three tasks ahead of next week's deadline.",
                "steps": [
                    "Clarify acceptance criteria",
                    "Do the riskiest task first",
                    "Buffer a review day",
                ],
                "assumptions": ["One week of working days available"],
                "needs_more_information": [],
            },
        },
        tags=("general",),
    ),
}

# Keyword → template routing (used when the topic keywords are ambiguous).
KEYWORD_ROUTES: dict[str, str] = {
    "billing": "billing",
    "invoice": "billing",
    "payment": "billing",
    "receipt": "billing",
    "email": "email",
    "outreach": "email",
    "newsletter": "email",
    "sequence": "email",
    "seo": "seo",
    "blog": "seo",
    "article": "seo",
    "keyword": "seo",
    "content": "seo",
    "support": "support",
    "ticket": "support",
    "complaint": "support",
    "helpdesk": "support",
    "resume": "career",
    "cv": "career",
    "interview": "career",
    "job": "career",
    "linkedin": "career",
    "ad": "ads",
    "ads": "ads",
    "copy": "ads",
    "headline": "ads",
    "campaign": "ads",
    "onboarding": "onboarding",
    "training": "onboarding",
    "lesson": "onboarding",
    "course": "onboarding",
    "product": "product",
    "spec": "product",
    "roadmap": "product",
    "prd": "product",
    "data": "data",
    "sql": "data",
    "analysis": "data",
    "dashboard": "data",
    "report": "data",
    "meeting": "meetings",
    "standup": "meetings",
    "minutes": "meetings",
    "transcript": "meetings",
    "legal": "legal",
    "contract": "legal",
    "policy": "legal",
    "nda": "legal",
    "compliance": "legal",
    "health": "health",
    "medical": "health",
    "clinical": "health",
    "symptom": "health",
    "finance": "finance",
    "budget": "finance",
    "cashflow": "finance",
    "tax": "finance",
    "accounting": "finance",
}


def route_template(keywords: list[str], text: str = "", default: str = "general") -> str:
    scores: dict[str, float] = {}
    blob = f"{' '.join(keywords)} {text}".lower()
    for token, template_key in KEYWORD_ROUTES.items():
        if token in blob:
            scores[template_key] = scores.get(template_key, 0.0) + 1.0
    if not scores:
        return default
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best if best in NICHE_TEMPLATES else default


def get_template(key: str) -> NicheTemplate:
    return NICHE_TEMPLATES.get(key, NICHE_TEMPLATES["general"])


def list_templates() -> list[dict[str, object]]:
    return [
        {
            "key": template.key,
            "label": template.label,
            "deliverable": template.deliverable,
            "regulated_domain": template.regulated_domain,
            "tags": list(template.tags),
        }
        for template in NICHE_TEMPLATES.values()
    ]
