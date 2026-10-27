"""Build the sellable prompt pack: a deterministic, signed ZIP artifact.

Contents (all generated from the IR + compiled variants — no third-party text):

    README.md                 what it is, who it's for, what's inside
    QUICKSTART.md             5-minute setup for each model family
    prompts/<family>.md       the compiled prompt + variable table + params
    prompts/variables.json    machine-readable variable contract
    examples/examples.json    synthetic few-shot examples (+ rendered markdown)
    TROUBLESHOOTING.md        common failure modes and fixes
    WHEN_NOT_TO_USE.md        explicit scope boundaries (honesty = fewer refunds)
    CHANGELOG.md              version history
    MODEL_COMPATIBILITY.md    tested/expected behaviour per model family
    LICENSE.txt               commercial licence terms
    MANIFEST.json             hashes of every file + IR hash + provenance

The ZIP is content-addressed and HMAC-signed; the signature is stored on the
``artifacts`` row and inside ``MANIFEST.json`` so a buyer (or auditor) can verify that
the pack they received is exactly what the gate approved.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.artifacts import get_store
from core.clock import utcnow
from core.config import get_settings
from core.security import sign_payload
from db.models import PromptProduct, PromptVariant, PromptVersion
from db.repository import record_artifact
from prompt_ir.schemas import PromptIR

ZIP_DATE = (2026, 1, 1, 0, 0, 0)  # fixed timestamp → reproducible archives


@dataclass
class PackResult:
    artifact_uri: str
    sha256: str
    size_bytes: int
    signature: str
    files: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    source: str = "zip"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_uri": self.artifact_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "signature": self.signature,
            "files": self.files,
            "manifest": self.manifest,
        }


def _render_readme(product: PromptProduct, version: PromptVersion, ir: PromptIR | None) -> str:
    title = product.name
    lines = [
        f"# {title}",
        "",
        product.positioning or (ir.task.objective if ir else ""),
        "",
        f"**Version:** {version.version}  |  **Licence:** {product.license}  |  **Currency:** {product.currency}",
        "",
        "## Who this is for",
        "",
        f"{product.buyer.replace('_', ' ').title()} — {ir.audience.context if ir else ''}".strip(),
        "",
        "## What you get",
        "",
        "- A model-agnostic prompt (compiled for OpenAI, Anthropic, Google and a generic baseline)",
        "- A strict output contract so downstream automations do not break on format changes",
        "- Synthetic few-shot examples",
        "- A troubleshooting guide and explicit scope boundaries",
        "- A changelog so you know what changed between versions",
        "",
        "## What it does not do",
        "",
    ]
    if ir:
        for exclusion in ir.task.exclusions:
            lines.append(f"- {exclusion}")
    lines += [
        "- It does not replace professional judgement, and it will not invent facts to fill gaps.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "| --- | --- |",
        "| `QUICKSTART.md` | copy/paste setup per model |",
        "| `prompts/*.md` | the prompt, per model family |",
        "| `prompts/variables.json` | the input contract |",
        "| `examples/examples.json` | synthetic examples |",
        "| `TROUBLESHOOTING.md` | common failure modes |",
        "| `WHEN_NOT_TO_USE.md` | scope boundaries |",
        "| `MODEL_COMPATIBILITY.md` | what to expect per model |",
        "| `MANIFEST.json` | integrity hashes + signature |",
        "",
    ]
    if ir and ir.safety.disclaimers:
        lines += ["## Required disclaimer", ""]
        lines += [f"> {text}" for text in ir.safety.disclaimers]
        lines.append("")
    return "\n".join(lines)


def _render_quickstart(
    product: PromptProduct, ir: PromptIR | None, variants: dict[str, PromptVariant]
) -> str:
    lines = [
        f"# Quickstart — {product.name}",
        "",
        "1. Open `prompts/<family>.md` for your model (or use `prompts/generic.md` for anything else).",
        "2. Paste the whole file as the **system prompt**.",
        "3. Send a user message containing the input tags listed in `prompts/variables.json`.",
        "4. Request JSON output if your platform supports structured/JSON mode.",
        "",
        "## Variable contract",
        "",
        "```json",
        json.dumps(
            {
                "variables": ir.variables() if ir else [],
                "output_format": ir.io.output_format if ir else "json",
                "output_schema": ir.io.output_schema if ir else {},
            },
            indent=2,
        ),
        "```",
        "",
        "## Per-family notes",
        "",
    ]
    notes = {
        "openai": "Use `response_format={'type':'json_object'}`; temperature 0–0.3.",
        "anthropic": "Send the file as `system`, keep `max_tokens` generous; XML tags are meaningful.",
        "google": "Set `responseMimeType='application/json'`; the schema sits in the RESPONSE SCHEMA block.",
        "generic": "Works with local models; enforce JSON with a grammar/validator where available.",
    }
    for family in sorted(variants):
        model = variants[family]
        lines.append(
            f"- **{family}** (compiled hash `{model.compiled_hash[:12]}`, ~{model.token_estimate} tokens): "
            f"{notes.get(family, 'standard chat completion.')}"
        )
    lines.append("")
    return "\n".join(lines)


def _render_prompt_file(family: str, variant: PromptVariant, ir: PromptIR | None) -> str:
    lines = [
        f"<!-- {family} · compiled {variant.compiled_hash[:16]} · do not edit by hand -->",
        "",
        variant.compiled_text,
        "",
        "---",
        "",
        "## Parameters",
        "",
        "```json",
        json.dumps(variant.params, indent=2, sort_keys=True),
        "```",
        "",
    ]
    if ir:
        lines += [
            "## Variables",
            "",
            "| Variable | Required | Description |",
            "| --- | --- | --- |",
        ]
        required = set((ir.io.input_schema or {}).get("required", []))
        properties = (ir.io.input_schema or {}).get("properties", {})
        for name in ir.variables():
            description = properties.get(name, {}).get("description", "")
            lines.append(f"| `{name}` | {'yes' if name in required else 'no'} | {description} |")
        lines.append("")
    return "\n".join(lines)


def _render_troubleshooting(ir: PromptIR | None) -> str:
    base = [
        "# Troubleshooting",
        "",
        "| Symptom | Likely cause | Fix |",
        "| --- | --- | --- |",
        "| Output is prose instead of JSON | model ignored the contract | enable JSON/structured mode; lower temperature |",
        "| Missing fields | input lacked the required facts | the model should list them in `needs_more_information`; supply them |",
        "| Made-up numbers | input had no figures | add the figures to `source_material`; the prompt forbids invention |",
        "| Refuses a legitimate task | over-broad refusal rule | supply more context; check the regulated-domain rules |",
        "| Truncated output | `max_tokens` too low | raise to the recommended value below |",
        "",
    ]
    if ir:
        base += [
            "## Recommended parameters",
            "",
            f"- temperature: `{ir.hints.temperature}`",
            f"- max_tokens: `{ir.hints.max_tokens}`",
            "",
            "## Quality gates baked into the prompt",
            "",
        ]
        base += [f"- {gate}" for gate in ir.procedure.quality_gates]
        base.append("")
    return "\n".join(base)


def _render_when_not_to_use(ir: PromptIR | None) -> str:
    lines = [
        "# When not to use this",
        "",
        "This product is deliberately bounded. Do not use it:",
        "",
    ]
    if ir:
        lines += [f"- {item}" for item in ir.task.exclusions]
        if ir.safety.regulated_domain != "none":
            lines += [
                f"- As a substitute for a licensed professional ({ir.safety.regulated_domain}).",
                "- To make decisions without human review where the outcome affects someone's rights, "
                "health, finances or legal position.",
            ]
        if ir.safety.escalate_when:
            lines += ["", "## Escalate immediately when", ""]
            lines += [f"- {item}" for item in ir.safety.escalate_when]
    lines += [
        "",
        "If your use case needs guarantees about correctness, human review is still required.",
        "",
    ]
    return "\n".join(lines)


def _render_model_compatibility(variants: dict[str, PromptVariant], version: PromptVersion) -> str:
    metrics = version.metrics_summary or {}
    lines = [
        "# Model compatibility",
        "",
        "| Family | Compiled hash | Est. tokens | JSON mode | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family, variant in sorted(variants.items()):
        json_mode = "yes" if (variant.params or {}).get("json_mode") else "no"
        lines.append(
            f"| {family} | `{variant.compiled_hash[:12]}` | {variant.token_estimate} | {json_mode} | "
            f"compiled from the same IR |"
        )
    lines += [
        "",
        "## Evaluation summary at release",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "Scores are from the bundled benchmark suites. They are evidence, not a guarantee: "
        "model behaviour changes between versions and providers — re-run the suite when you upgrade.",
        "",
    ]
    return "\n".join(lines)


def _render_license(product: PromptProduct, owner: str) -> str:
    return f"""PROMPTFOUNDRY COMMERCIAL LICENCE — {product.license}

Product: {product.name}
Version: see CHANGELOG.md
Copyright (c) {utcnow().year} {owner}

1. GRANT. You may use these prompts and templates in your own commercial work,
   including client deliverables, for the number of seats stated on purchase.

2. RESTRICTIONS. You may not resell, redistribute, sublicense or publish the prompt
   files themselves (or substantially similar derivatives) as a prompt product,
   template library, course or dataset.

3. NO WARRANTY. Provided "as is". Model outputs vary; you are responsible for
   reviewing output before use. Nothing here is professional advice.

4. THIRD-PARTY CONTENT. This pack contains no third-party prompt text. Examples are
   synthetic. See MANIFEST.json for provenance and integrity hashes.

5. PRIVACY. Do not paste personal information into prompts unless you have a lawful
   basis to process it. PromptFoundry products are designed to minimise personal data.

Generated by PromptFoundry {utcnow().date().isoformat()}.
"""


def build_prompt_pack(
    session: Session,
    *,
    version: PromptVersion,
    product: PromptProduct,
    actor_id: str = "system",
) -> PackResult:
    settings = get_settings()
    variants_list = list(
        session.scalars(select(PromptVariant).where(PromptVariant.version_id == version.id))
    )
    variants = {variant.model_family: variant for variant in variants_list}
    if not variants:
        raise ValueError("cannot package a version with no compiled variants")
    ir = PromptIR.model_validate(version.ir) if version.ir else None

    files: dict[str, bytes] = {}
    files["README.md"] = _render_readme(product, version, ir).encode()
    files["QUICKSTART.md"] = _render_quickstart(product, ir, variants).encode()
    files["TROUBLESHOOTING.md"] = _render_troubleshooting(ir).encode()
    files["WHEN_NOT_TO_USE.md"] = _render_when_not_to_use(ir).encode()
    files["MODEL_COMPATIBILITY.md"] = _render_model_compatibility(variants, version).encode()
    files["LICENSE.txt"] = _render_license(product, owner=actor_id).encode()
    files["CHANGELOG.md"] = (
        f"# Changelog\n\n## {version.version}\n\n{version.changelog or 'Initial release.'}\n"
    ).encode()
    for family, variant in variants.items():
        files[f"prompts/{family}.md"] = _render_prompt_file(family, variant, ir).encode()
    files["prompts/variables.json"] = json.dumps(
        {
            "variables": ir.variables() if ir else [],
            "input_schema": (ir.io.input_schema if ir else {}) or {},
            "output_schema": (ir.io.output_schema if ir else {}) or {},
            "output_format": ir.io.output_format if ir else "json",
        },
        indent=2,
        sort_keys=True,
    ).encode()
    examples = [example.model_dump(mode="json") for example in (ir.examples if ir else [])]
    files["examples/examples.json"] = json.dumps(examples, indent=2, sort_keys=True).encode()
    if examples:
        files["examples/examples.md"] = "\n".join(
            f"## {example['name']}\n\n```json\n{json.dumps(example['input'], indent=2)}\n```\n\n"
            f"Output:\n\n```json\n"
            f"{json.dumps(example['output'], indent=2) if not isinstance(example['output'], str) else example['output']}"
            f"\n```\n"
            for example in examples
        ).encode()

    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())}
    manifest: dict[str, Any] = {
        "product": {
            "id": product.id,
            "name": product.name,
            "slug": product.slug,
            "niche": product.niche,
            "buyer": product.buyer,
            "license": product.license,
            "price_aud": round(product.price_aud_micros / 1_000_000, 2),
        },
        "version": {
            "id": version.id,
            "semver": version.version,
            "ir_hash": version.ir_hash,
            "created_at": version.created_at.isoformat() if version.created_at else None,
            "approved_by": version.approved_by,
        },
        "provenance": {
            "generator": "PromptFoundry prompt_lab",
            "origin": ir.provenance.origin if ir else "synthetic",
            "method": ir.provenance.method if ir else "",
            "third_party_text": False,
            "examples": "synthetic" if ir else "none",
            "source_signal_ids": ir.provenance.source_signal_ids if ir else [],
        },
        "files": hashes,
        "generated_at": utcnow().isoformat(),
        "jurisdiction": settings.jurisdiction,
    }
    signature = sign_payload(manifest, settings.signing_key)
    manifest["signature"] = signature
    files["MANIFEST.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    payload = buffer.getvalue()

    ref = get_store().put_bytes(
        payload, kind="prompt_pack", ext="zip", content_type="application/zip"
    )
    record_artifact(
        session,
        ref,
        kind="prompt_pack",
        subject_type="version",
        subject_id=version.id,
        created_by=actor_id,
        signature=signature,
    )
    version.artifact_uri = ref.uri
    version.signature = signature
    version.signature_alg = "hmac-sha256"
    version.signed_at = utcnow()
    session.flush()
    return PackResult(
        artifact_uri=ref.uri,
        sha256=ref.sha256,
        size_bytes=ref.size_bytes,
        signature=signature,
        files=sorted(files),
        manifest=manifest,
    )
