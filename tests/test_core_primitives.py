"""Pure-function tests: RBAC, hashing, budget breaker, embeddings, prompt IR/compiler."""

from __future__ import annotations

import json

import pytest

from core.budget import assert_within_budget, record_cost, shadow_cost_micros
from core.errors import BudgetExceeded
from core.ids import content_hash, slugify
from core.rbac import ROLE_PERMISSIONS, Permission, Role, has_permission, permissions_for
from prompt_ir.compiler import compile_all, compile_ir
from prompt_ir.schemas import PromptIR, validate_instance


def test_role_permissions_are_monotonic():
    for role in Role:
        perms = permissions_for([str(role)])
        assert isinstance(perms, set)
    assert Permission.PUBLISH.value in ROLE_PERMISSIONS[str(Role.APPROVER)]
    assert has_permission([str(Role.VIEWER)], Permission.PUBLISH) is False
    assert has_permission([str(Role.APPROVER)], Permission.PUBLISH) is True
    # An unknown role grants nothing rather than silently granting everything.
    assert permissions_for(["nope"]) == set()


def test_content_hash_is_stable_and_structured():
    a = content_hash("part-one", "part-two")
    b = content_hash("part-one", "part-two")
    assert a == b and len(a) == 64
    assert content_hash("part-one", "part-three") != a
    assert content_hash("part-one") != content_hash("part-one", "")  # separators matter
    assert (
        slugify("Invoices & payment reminders for SMBs!") == "invoices-payment-reminders-for-smbs"
    )


def test_budget_breaker_trips_and_counts_shadow_cost(session):
    assert_within_budget(session, 1_000)  # under the A$2/day test cap
    record_cost(
        session,
        category="llm",
        provider="openai",
        model="gpt-4o-mini",
        amount_micros=1_900_000,
        purpose="test",
    )
    session.flush()
    with pytest.raises(BudgetExceeded):
        assert_within_budget(session, 500_000)
    # Shadow pricing never touches the ledger.
    shadow = shadow_cost_micros(100_000, 50_000)
    assert 0 < shadow < 100_000


def test_embedder_keeps_related_signals_close_and_unrelated_far_apart():
    from llm.embeddings import cosine, get_embedder

    embedder = get_embedder()
    invoices, reminders, seo = embedder.embed(
        [
            "chase an unpaid invoice without sounding aggressive",
            "overdue invoice reminder email wording",
            "SEO blog outline for a keyword with commercial intent",
        ]
    )
    assert cosine(invoices, reminders) > cosine(invoices, seo) + 0.2
    assert embedder.name.startswith("hashing-v2") or "google" in embedder.name


def test_prompt_ir_compiles_for_every_family_and_is_deterministic():
    from prompt_lab.templates import get_template

    template = get_template("billing")
    assert template is not None
    schema = template.output_schema()
    ir = PromptIR.model_validate(
        {
            "slug": "billing-reminder-test",
            "title": "Billing reminder (test)",
            "task": {
                "objective": "Draft a courteous invoice reminder and a structured summary.",
                "deliverable": "email plus JSON summary",
                "success_criteria": ["states the amount", "no threats"],
                "exclusions": ["legal advice"],
            },
            "audience": {
                "segment": "smb",
                "expertise": "intermediate",
                "context": "AU small business",
            },
            "constraints": {
                "tone": "firm but cordial",
                "length": "short",
                "formatting": "strict JSON",
                "must_include": ["amount_due"],
                "must_avoid": ["threats"],
            },
            "io": {
                "input_vars": ["request", "source_material"],
                "output_format": "json",
                "output_schema": schema,
            },
            "procedure": {"steps": ["Parse the request", "Draft the email", "Verify the JSON"]},
            "safety": {
                "refusal_rules": ["No legal advice", "Never invent amounts"],
                "disclaimers": [],
                "regulated_domain": "none",
            },
            "examples": [
                {
                    "name": "synthetic",
                    "input": {"request": "remind", "source_material": "Invoice 1, A$10, due 1 Sep"},
                    "output": {},
                }
            ],
            "provenance": {
                "origin": "synthetic",
                "generated_by": "tests",
                "license": "operator-owned",
                "source_signal_ids": [],
            },
            "hints": {"preferred_families": ["openai", "anthropic"], "temperature": 0.2},
        }
    )
    compiled = compile_all(ir)
    assert {"openai", "anthropic"} <= set(compiled)
    for family, prompt in compiled.items():
        assert "```json" in prompt.text, family
        assert prompt.token_estimate > 0
        assert prompt.compiled_hash == compile_ir(ir, family).compiled_hash
    assert compiled["openai"].text != compiled["anthropic"].text


def test_json_schema_validation_reports_errors():
    schema = {
        "type": "object",
        "properties": {"amount": {"type": "number"}},
        "required": ["amount"],
        "additionalProperties": False,
    }
    ok, errors = validate_instance({"amount": 10}, schema)
    assert ok and not errors
    ok, errors = validate_instance({"amount": "ten"}, schema)
    assert not ok and errors
    ok, errors = validate_instance({"amount": 1, "extra": True}, schema)
    assert not ok


def test_settings_defaults_are_safe():
    from core.config import get_settings

    settings = get_settings()
    assert settings.require_human_approval is True
    assert settings.auto_publish_enabled is False
    assert settings.budget_enforced is True
    assert settings.ip_block_threshold >= 0.9
    assert settings.jurisdiction == "AU"
    serialised = json.dumps(settings.any(), default=str)
    assert "sk-" not in serialised and "api_key" not in serialised.lower()
