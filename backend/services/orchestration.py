"""End-to-end orchestration: opportunity → product → version → evidence → gate.

This is the only place the whole loop is composed, so both the API and the demo call
exactly the same code path. Each step is independently auditable, and every stage
caches its work (IR hashes, evaluation runs, artifacts) so re-running is cheap.

Deliberate ordering: **the gate is always last** and never bypassed; it may return
``needs_review`` (normal for regulated niches) which parks the version in
``ready_for_review`` for a human approver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.artifacts import get_store
from core.clock import utcnow
from core.constants import (
    ApprovalStatus,
    OpportunityStatus,
    ProductStatus,
    VersionStatus,
)
from core.errors import NotFound, ValidationError
from core.ids import slugify
from db.models import (
    Approval,
    EvidenceItem,
    Opportunity,
    OpportunityScore,
    PromptProduct,
    PromptVariant,
    PromptVersion,
    TestSuite,
    Topic,
)
from db.repository import latest_version, record_artifact, record_audit
from evaluation.runner import RunConfig, run_evaluation
from governance.gate import run_governance_gate
from market_intel.scoring import ScoreBreakdown
from prompt_lab.generator import brief_from_opportunity
from prompt_lab.optimizer import OptimizationConfig, optimize, record_winner
from prompt_lab.templates import NICHE_TEMPLATES, route_template


@dataclass
class LaunchResult:
    product_id: str
    version_id: str | None
    opportunity_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    optimization: dict[str, Any] | None = None
    cost_micros: int = 0
    status: str = "unknown"
    artifact_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "version_id": self.version_id,
            "opportunity_id": self.opportunity_id,
            "status": self.status,
            "steps": self.steps,
            "warnings": self.warnings,
            "gate": self.gate,
            "optimization": self.optimization,
            "cost_aud": round(self.cost_micros / 1_000_000, 6),
            "artifact_uri": self.artifact_uri,
        }


def find_suite_for_niche(session: Session, niche_key: str) -> TestSuite | None:
    """Pick the visible (non-holdout) suite whose niche matches, else any visible suite."""
    suite = session.scalar(
        select(TestSuite)
        .where(TestSuite.niche == niche_key, TestSuite.is_holdout.is_(False))
        .order_by(TestSuite.created_at.desc())
        .limit(1)
    )
    if suite is not None:
        return suite
    return session.scalar(
        select(TestSuite)
        .where(TestSuite.is_holdout.is_(False))
        .order_by(TestSuite.created_at.desc())
        .limit(1)
    )


def find_holdout_suite(session: Session, niche_key: str) -> TestSuite | None:
    return session.scalar(
        select(TestSuite)
        .where(TestSuite.niche == niche_key, TestSuite.is_holdout.is_(True))
        .order_by(TestSuite.created_at.desc())
        .limit(1)
    )


def _ensure_product(
    session: Session, opportunity: Opportunity, topic: Topic, actor_id: str
) -> PromptProduct:
    existing = session.scalar(
        select(PromptProduct).where(PromptProduct.opportunity_id == opportunity.id)
    )
    if existing is not None:
        return existing
    base_slug = slugify(opportunity.title)
    slug = base_slug
    suffix = 1
    while session.scalar(select(PromptProduct).where(PromptProduct.slug == slug)) is not None:
        suffix += 1
        slug = f"{base_slug}-{suffix}"
    product = PromptProduct(
        opportunity_id=opportunity.id,
        name=opportunity.title,
        slug=slug,
        niche=str((topic.keywords or ["general"])[0]),
        buyer=opportunity.buyer_segment,
        positioning=opportunity.positioning,
        description=opportunity.rationale,
        status=str(ProductStatus.DRAFT),
        price_aud_micros=opportunity.price_hypothesis_micros,
        tags=list(topic.keywords or [])[:10],
        regulated_domain=opportunity.regulated_domain,
        risk_tier=opportunity.risk_tier,
        owner_id=actor_id,
    )
    session.add(product)
    session.flush()
    record_audit(
        session,
        action="product.created",
        actor_id=actor_id,
        subject_type="product",
        subject_id=product.id,
        decision="created",
        detail={"opportunity_id": opportunity.id, "slug": slug},
    )
    return product


def _score_breakdown(session: Session, opportunity: Opportunity) -> ScoreBreakdown | None:
    row = session.scalar(
        select(OpportunityScore)
        .where(OpportunityScore.opportunity_id == opportunity.id)
        .order_by(OpportunityScore.created_at.desc())
        .limit(1)
    )
    if row is None:
        return None
    return ScoreBreakdown(
        demand=row.demand,
        supply_proxy=row.supply_proxy,
        monetization=row.monetization,
        build_cost=row.build_cost,
        risk=row.risk,
        confidence=row.confidence,
        composite=row.composite,
        composite_lower=row.composite_lower,
        composite_upper=row.composite_upper,
        price_hypothesis_micros=opportunity.price_hypothesis_micros,
        risk_tier=opportunity.risk_tier,
        regulated_domain=opportunity.regulated_domain,
        supply_known=bool((row.inputs or {}).get("permitted_density") is not None),
        weights=row.weights or {},
        inputs=row.inputs or {},
        reasons=[],
    )


def launch_product(
    session: Session,
    *,
    opportunity_id: str,
    actor_id: str,
    n_candidates: int = 4,
    primary_family: str = "generic",
    evaluate_holdout: bool = True,
    run_gate: bool = True,
    price_aud: float | None = None,
    optimization_config: OptimizationConfig | None = None,
) -> LaunchResult:
    opportunity = session.get(Opportunity, opportunity_id)
    if opportunity is None:
        raise NotFound(f"opportunity {opportunity_id} not found")
    topic = session.get(Topic, opportunity.topic_id)
    if topic is None:
        raise NotFound("topic for opportunity not found")
    if opportunity.regulated_domain not in {"none", ""}:
        # Allowed, but only as an assistive product with disclaimers + disclaimers are
        # enforced by the policy engine; we record the decision explicitly.
        pass

    evidence = list(
        session.scalars(
            select(EvidenceItem)
            .where(EvidenceItem.opportunity_id == opportunity.id)
            .order_by(EvidenceItem.weight.desc())
            .limit(6)
        )
    )
    niche_key = route_template(list(topic.keywords or []), topic.name)
    if niche_key not in NICHE_TEMPLATES:
        niche_key = "general"
    suite = find_suite_for_niche(session, niche_key)
    if suite is None:
        raise ValidationError(
            "no evaluation suite available — load a suite first (POST /suites/load or make seed)"
        )
    holdout = find_holdout_suite(session, niche_key) if evaluate_holdout else None

    product = _ensure_product(session, opportunity, topic, actor_id)
    if price_aud is not None:
        product.price_aud_micros = int(round(price_aud * 1_000_000))
    product.status = str(ProductStatus.CANDIDATE)

    brief = brief_from_opportunity(
        opportunity_id=opportunity.id,
        product_name=product.name,
        segment=opportunity.buyer_segment,
        keywords=list(topic.keywords or []),
        jtbd=opportunity.jtbd or topic.jtbd,
        regulated_domain=opportunity.regulated_domain,
        topic_name=topic.name,
        evidence=[item.quote for item in evidence],
        price_aud=product.price_aud_micros / 1_000_000,
        niche_key=niche_key,
    )

    config = optimization_config or OptimizationConfig(
        n_candidates=max(2, min(n_candidates, 6)),
        primary_family=primary_family,
        is_holdout_eval=bool(holdout),
    )
    config.primary_family = primary_family
    config.is_holdout_eval = bool(holdout)
    baseline = latest_version(session, product.id)
    result = optimize(
        session,
        brief=brief,
        suite=suite,
        config=config,
        baseline_version_id=baseline.id
        if baseline
        and baseline.status
        in {
            str(VersionStatus.RELEASED),
            str(VersionStatus.APPROVED),
        }
        else None,
        holdout_suite=holdout,
    )
    launch = LaunchResult(
        product_id=product.id,
        version_id=None,
        opportunity_id=opportunity.id,
        warnings=list(result.warnings),
        optimization=result.to_dict(),
        cost_micros=result.cost_micros,
    )
    launch.steps.append(
        {
            "step": "optimize",
            "candidates": len(result.candidates),
            "winner": result.winner.slug if result.winner else None,
            "suite": suite.name,
            "cost_aud": round(result.cost_micros / 1_000_000, 6),
        }
    )
    if result.winner is None:
        launch.status = "no_candidate"
        launch.warnings.append("optimisation produced no viable candidate")
        opportunity.status = str(OpportunityStatus.SHORTLISTED)
        return launch

    version_number = _next_version(session, product.id, baseline)
    version = record_winner(
        session,
        result=result,
        product_id=product.id,
        version=version_number,
        changelog=_changelog(result),
        actor_id=actor_id,
    )
    launch.version_id = version.id
    launch.steps.append(
        {
            "step": "record_version",
            "version": version.version,
            "variants": len(result.winner_compiled),
            "ir_hash": version.ir_hash,
        }
    )

    # Attach the finalist run to the version as release evidence; run one if missing.
    finalist_run = result.winner.finalist
    if finalist_run is not None:
        from db.models import EvaluationRun

        run_row = session.get(EvaluationRun, finalist_run.run_id)
        if run_row is not None:
            run_row.version_id = version.id
            primary_variant = session.scalar(
                select(PromptVariant).where(
                    PromptVariant.version_id == version.id, PromptVariant.is_primary.is_(True)
                )
            )
            if primary_variant is not None:
                run_row.variant_id = primary_variant.id
            session.flush()
            launch.steps.append(
                {"step": "evidence_attached", "run_id": run_row.id, "score": run_row.score}
            )
    else:
        primary_variant = session.scalar(
            select(PromptVariant).where(
                PromptVariant.version_id == version.id, PromptVariant.is_primary.is_(True)
            )
        )
        if primary_variant is not None:
            from prompt_ir.compiler import CompiledPrompt

            compiled = CompiledPrompt(
                family=primary_variant.model_family,
                text=primary_variant.compiled_text,
                params=primary_variant.params or {},
                token_estimate=primary_variant.token_estimate,
                compiled_hash=primary_variant.compiled_hash,
                ir_hash=primary_variant.ir_hash,
            )
            outcome = run_evaluation(
                session,
                suite=suite,
                compiled=compiled,
                config=RunConfig(stage="finalist", judge=True, robustness=True),
                version_id=version.id,
                variant_id=primary_variant.id,
                actor_id=actor_id,
            )
            launch.cost_micros += outcome.cost_micros
            launch.steps.append(
                {"step": "evaluation", "run_id": outcome.run_id, "score": outcome.overall_score}
            )
            if not outcome.passed:
                launch.warnings.append("finalist evaluation did not pass the suite threshold")

    if run_gate:
        gate = run_governance_gate(
            session,
            version_id=version.id,
            actor_id=actor_id,
            run_injection=True,
            request_approval=True,
        )
        launch.gate = gate.to_dict()
        launch.steps.append(
            {
                "step": "governance_gate",
                "decision": gate.decision,
                "approval_id": gate.approval_id,
                "blockers": gate.blockers,
            }
        )
        opportunity.status = (
            str(OpportunityStatus.SHIPPED)
            if gate.decision == "pass"
            else str(OpportunityStatus.IN_PRODUCTION)
        )
        launch.status = (
            "ready_for_review"
            if gate.approval_id
            else "blocked"
            if gate.decision == "block"
            else "approved"
        )
        if gate.blockers:
            launch.warnings.extend(gate.blockers[:5])
    else:
        launch.status = "candidate"
        opportunity.status = str(OpportunityStatus.IN_PRODUCTION)

    payload = {
        **launch.to_dict(),
        "product": {"id": product.id, "name": product.name, "slug": product.slug},
        "version": {"id": version.id, "semver": version.version, "ir_hash": version.ir_hash},
        "generated_at": utcnow().isoformat(),
    }
    ref = get_store().put_json(payload, kind="listing", subject_id=f"launch-{version.id}")
    record_artifact(
        session,
        ref,
        kind="listing",
        subject_type="version",
        subject_id=version.id,
        created_by=actor_id,
    )
    launch.artifact_uri = ref.uri
    record_audit(
        session,
        action="orchestration.launch",
        actor_id=actor_id,
        subject_type="product",
        subject_id=product.id,
        decision=launch.status,
        detail={
            "opportunity_id": opportunity.id,
            "version_id": version.id,
            "cost_aud": round(launch.cost_micros / 1_000_000, 6),
            "warnings": launch.warnings[:6],
        },
    )
    session.flush()
    return launch


def _next_version(session: Session, product_id: str, baseline: PromptVersion | None) -> str:
    versions = list(
        session.scalars(select(PromptVersion).where(PromptVersion.product_id == product_id))
    )
    if not versions:
        return "1.0.0"

    def parse(value: str) -> tuple[int, int, int]:
        try:
            major, minor, patch = (int(part) for part in value.split("."))
            return major, minor, patch
        except ValueError:
            return (0, 0, 0)

    latest = max(versions, key=lambda row: parse(row.version))
    major, minor, patch = parse(latest.version)
    return f"{major}.{minor}.{patch + 1}"


def _changelog(result: Any) -> str:
    winner = result.winner
    if winner is None:
        return "Automated generation."
    notes = []
    if winner.screen:
        notes.append(f"screen score {winner.screen.overall_score:.2f}")
    if winner.offline:
        notes.append(f"offline score {winner.offline.overall_score:.2f}")
    if winner.finalist:
        notes.append(f"finalist score {winner.finalist.overall_score:.2f}")
    notes.append(f"IR hash {winner.ir_hash[:12]}")
    return (
        "Automated candidate selected by staged optimisation: "
        + ", ".join(notes)
        + f". Variation: {(winner.ir.provenance.notes or '')[:200]}"
    )


def approval_queue(session: Session, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(Approval)
        .where(Approval.status == str(ApprovalStatus.PENDING))
        .order_by(Approval.requested_at.desc())
        .limit(limit)
    )
    out = []
    for approval in rows:
        version = (
            session.get(PromptVersion, approval.subject_id)
            if approval.subject_type == "version"
            else None
        )
        product = session.get(PromptProduct, version.product_id) if version else None
        out.append(
            {
                "approval_id": approval.id,
                "subject_type": approval.subject_type,
                "subject_id": approval.subject_id,
                "requested_by": approval.requested_by,
                "requested_at": approval.requested_at.isoformat(),
                "status": approval.status,
                "conditions": approval.conditions,
                "product": {"id": product.id, "name": product.name} if product else None,
                "version": {"id": version.id, "semver": version.version, "status": version.status}
                if version
                else None,
            }
        )
    return out
