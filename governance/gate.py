"""The governance gate + approval workflow + publish/rollback controls.

Gate order (fail fast, cheapest first):

1. **Policy** — prohibited content, regulated-advice boundaries, safety scaffolding.
2. **IP similarity** — catalog + permitted corpus, cosine *and* shingle overlap.
3. **Evaluation evidence** — a passing run for this version at offline/finalist stage.
4. **Injection suite** — adversarial robustness for this version.
5. **Risk tiering + approval** — human approval by default; auto-approval only when
   three independent switches are set and the product is low risk and unregulated.

Nothing can be published without an ``approved`` Approval row (or a documented,
audited automatic approval), and publishing can be halted instantly by the
``publishing.kill_switch`` runtime flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import utcnow
from core.config import get_settings
from core.constants import (
    ApprovalStatus,
    GovernanceDecision,
    ProductStatus,
    RegulatedDomain,
    RiskTier,
    SubjectType,
    VersionStatus,
)
from core.errors import Conflict, NotFound, PublishBlocked
from core.security import canonical_json, sign_payload
from db.models import (
    Approval,
    EvaluationRun,
    IpSimilarityCheck,
    PromptProduct,
    PromptVariant,
    PromptVersion,
    SafetyCheck,
    TestSuite,
)
from db.repository import (
    artifacts_for,
    flag_enabled,
    get_flag,
    raise_alert,
    record_artifact,
    record_audit,
)
from governance.injection import InjectionReport, run_injection_gate
from governance.policy import POLICY_VERSION, PolicyResult, check_policy, policy_snapshot
from governance.similarity import SimilarityResult, check_similarity, load_permitted_corpus
from observability.metrics import GOVERNANCE_DECISIONS


@dataclass
class GateOutcome:
    decision: str
    version_id: str
    risk_tier: str
    policy: dict[str, Any] = field(default_factory=dict)
    similarity: dict[str, Any] = field(default_factory=dict)
    injection: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    approval_id: str | None = None
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    next_step: str = ""
    report_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "version_id": self.version_id,
            "risk_tier": self.risk_tier,
            "policy": self.policy,
            "similarity": self.similarity,
            "injection": self.injection,
            "evaluation": self.evaluation,
            "approval_id": self.approval_id,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "next_step": self.next_step,
            "report_uri": self.report_uri,
        }


def _composite_decision(*decisions: str) -> str:
    order = {
        str(GovernanceDecision.BLOCK): 3,
        str(GovernanceDecision.NEEDS_REVIEW): 2,
        str(GovernanceDecision.WARN): 1,
        str(GovernanceDecision.PASS): 0,
    }
    return max(decisions, key=lambda value: order.get(value, 0))


def _latest_evaluation(session: Session, version_id: str) -> EvaluationRun | None:
    return session.scalar(
        select(EvaluationRun)
        .where(EvaluationRun.version_id == version_id)
        .where(EvaluationRun.status.in_(["succeeded", "partial"]))
        .order_by(EvaluationRun.finished_at.desc())
        .limit(1)
    )


def run_governance_gate(
    session: Session,
    *,
    version_id: str,
    actor_id: str = "system",
    run_injection: bool = True,
    request_approval: bool = True,
    eval_run_id: str | None = None,
    model: str | None = None,
) -> GateOutcome:
    settings = get_settings()
    version = session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    product = session.get(PromptProduct, version.product_id)
    variants = list(
        session.scalars(select(PromptVariant).where(PromptVariant.version_id == version.id))
    )
    if not variants:
        raise NotFound(f"version {version_id} has no compiled variants")
    primary = next((variant for variant in variants if variant.is_primary), variants[0])
    from prompt_ir.schemas import PromptIR

    ir = PromptIR.model_validate(version.ir) if version.ir else None
    texts = {variant.model_family: variant.compiled_text for variant in variants}
    blockers: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------ 1. policy
    policy_result: PolicyResult = check_policy(
        ir=ir,
        texts=texts,
        product_meta={
            "title": product.name if product else "",
            "description": product.description if product else "",
        },
    )
    for finding in policy_result.findings:
        if finding.severity == "block":
            blockers.append(f"policy[{finding.category}]: {finding.message}")
        elif finding.severity == "warn":
            warnings.append(f"policy[{finding.category}]: {finding.message}")
    session.add(
        SafetyCheck(
            subject_type=str(SubjectType.VERSION),
            subject_id=version.id,
            check="policy",
            decision=policy_result.decision,
            score=float(policy_result.scores.get("blockers", 0)),
            policy_version=POLICY_VERSION,
            findings=[finding.to_dict() for finding in policy_result.findings],
            detail={"regulated_domain": policy_result.regulated_domain},
        )
    )

    # -------------------------------------------------------- 2. similarity
    own_refs = {f"catalog:{variant.id}:{variant.model_family}" for variant in variants}
    corpus, skipped_corpus = load_permitted_corpus()
    similarity_result: SimilarityResult = check_similarity(
        session,
        text=primary.compiled_text,
        exclude_refs=own_refs,
        corpus=corpus,
    )
    for finding in similarity_result.findings:
        if finding.severity == "block":
            blockers.append(f"ip[{finding.category}]: {finding.message}")
        else:
            warnings.append(f"ip[{finding.category}]: {finding.message}")
    if skipped_corpus:
        warnings.append(f"corpus entries skipped: {skipped_corpus[:3]}")
    session.add(
        IpSimilarityCheck(
            subject_type=str(SubjectType.VERSION),
            subject_id=version.id,
            compared_against=similarity_result.compared_against,
            max_similarity=similarity_result.max_similarity,
            shingle_overlap=similarity_result.shingle_overlap,
            nearest_ref=similarity_result.nearest_ref,
            threshold=similarity_result.threshold,
            decision=similarity_result.decision,
            detail={"skipped_corpus": skipped_corpus},
        )
    )

    # -------------------------------------------------------- 3. evaluation
    evaluation = (
        session.get(EvaluationRun, eval_run_id)
        if eval_run_id
        else _latest_evaluation(session, version.id)
    )
    eval_summary: dict[str, Any] = {"found": evaluation is not None}
    eval_decision = str(GovernanceDecision.PASS)
    if evaluation is None:
        blockers.append("evaluation: no successful evaluation run for this version")
        eval_decision = str(GovernanceDecision.BLOCK)
        eval_summary["reason"] = "no evaluation run"
    else:
        suite = session.get(TestSuite, evaluation.suite_id) if evaluation.suite_id else None
        min_score = suite.min_score if suite else 0.7
        eval_summary.update(
            {
                "run_id": evaluation.id,
                "stage": evaluation.stage,
                "score": evaluation.score,
                "status": evaluation.status,
                "min_score": min_score,
                "suite_id": evaluation.suite_id,
                "report_uri": evaluation.report_uri,
            }
        )
        if evaluation.score < min_score:
            blockers.append(
                f"evaluation: score {evaluation.score:.3f} below minimum {min_score:.2f}"
            )
            eval_decision = str(GovernanceDecision.BLOCK)
        if evaluation.status == "failed":
            blockers.append("evaluation: run status is failed")
            eval_decision = str(GovernanceDecision.BLOCK)
        warnings.extend((evaluation.summary or {}).get("warnings", [])[:4])

    # --------------------------------------------------------- 4. injection
    injection_summary_dict: dict[str, Any] = {"skipped": True}
    injection_decision = str(GovernanceDecision.PASS)
    if run_injection:
        report: InjectionReport = run_injection_gate(
            session,
            version_id=version.id,
            variant=primary,
            suite_id=evaluation.suite_id if evaluation else None,
            actor_id=actor_id,
            model=model,
        )
        injection_summary_dict = report.to_dict()
        injection_decision = report.decision
        for finding in report.findings:
            if finding.severity == "block":
                blockers.append(f"injection[{finding.category}]: {finding.message}")
            else:
                warnings.append(f"injection[{finding.category}]: {finding.message}")
        session.add(
            SafetyCheck(
                subject_type=str(SubjectType.VERSION),
                subject_id=version.id,
                check="prompt_injection",
                decision=report.decision,
                score=report.pass_rate,
                policy_version=POLICY_VERSION,
                findings=[finding.to_dict() for finding in report.findings],
                detail={"critical_failures": report.critical_failures},
            )
        )

    decision = _composite_decision(
        policy_result.decision, similarity_result.decision, eval_decision, injection_decision
    )
    if blockers:
        decision = str(GovernanceDecision.BLOCK)

    regulated = policy_result.regulated_domain != str(RegulatedDomain.NONE)
    risk_tier = policy_result.risk_tier
    if regulated and risk_tier == str(RiskTier.LOW):
        risk_tier = str(RiskTier.MEDIUM)

    # ----------------------------------------------------- 5. approval flow
    approval_id: str | None = None
    if decision == str(GovernanceDecision.BLOCK) or blockers:
        version.status = str(VersionStatus.QUARANTINED)
        if product:
            product.status = str(ProductStatus.CANDIDATE)
        next_step = "fix blockers, re-run evaluation and the gate"
        approval = session.scalar(
            select(Approval).where(
                Approval.subject_type == str(SubjectType.VERSION),
                Approval.subject_id == version.id,
                Approval.status == str(ApprovalStatus.PENDING),
            )
        )
        if approval is not None:
            approval.status = str(ApprovalStatus.REJECTED)
            approval.decided_at = utcnow()
            approval.reason = "superseded by failing governance gate"
        raise_alert(
            session,
            kind="governance.blocked",
            severity="warning",
            message=f"Version {version.id} blocked by governance gate",
            subject_type=str(SubjectType.VERSION),
            subject_id=version.id,
            detail={"blockers": blockers},
        )
    else:
        auto = _evaluate_auto_approval(
            session, settings=settings, risk_tier=risk_tier, regulated=regulated
        )
        next_step = "human approval required" if not auto.allowed else "auto-approved (audited)"
        if auto.allowed:
            approval = Approval(
                subject_type=str(SubjectType.VERSION),
                subject_id=version.id,
                requested_by="governance",
                status=str(ApprovalStatus.APPROVED),
                decided_by="system:auto_publish",
                decided_at=utcnow(),
                reason=auto.reason,
                policy_snapshot=policy_snapshot(),
                is_automatic=True,
            )
            session.add(approval)
            session.flush()
            approval_id = approval.id
            version.status = str(VersionStatus.APPROVED)
            version.approved_by = "system:auto_publish"
            if product:
                product.status = str(ProductStatus.APPROVED)
        elif request_approval:
            existing = session.scalar(
                select(Approval).where(
                    Approval.subject_type == str(SubjectType.VERSION),
                    Approval.subject_id == version.id,
                    Approval.status == str(ApprovalStatus.PENDING),
                )
            )
            if existing is None:
                existing = Approval(
                    subject_type=str(SubjectType.VERSION),
                    subject_id=version.id,
                    requested_by=actor_id,
                    status=str(ApprovalStatus.PENDING),
                    policy_snapshot=policy_snapshot(),
                    conditions={
                        "risk_tier": risk_tier,
                        "regulated_domain": policy_result.regulated_domain,
                        "warnings": warnings[:10],
                        "evaluation_run_id": evaluation.id if evaluation else None,
                    },
                )
                session.add(existing)
                session.flush()
            approval_id = existing.id
            version.status = str(VersionStatus.STAGED)
            if product:
                product.status = str(ProductStatus.READY_FOR_REVIEW)

    GOVERNANCE_DECISIONS.inc(decision=decision, risk_tier=str(risk_tier))
    outcome = GateOutcome(
        decision=decision,
        version_id=version.id,
        risk_tier=str(risk_tier),
        policy=policy_result.to_dict(),
        similarity=similarity_result.to_dict(),
        injection=injection_summary_dict,
        evaluation=eval_summary,
        approval_id=approval_id,
        blockers=blockers,
        warnings=warnings,
        next_step=next_step,
    )
    report_payload = {
        **outcome.to_dict(),
        "policy_version": POLICY_VERSION,
        "generated_at": utcnow().isoformat(),
        "product": {"id": product.id, "name": product.name} if product else None,
        "signature": sign_payload(outcome.to_dict(), settings.signing_key),
    }
    from core.artifacts import get_store

    ref = get_store().put_json(report_payload, kind="governance_report", subject_id=version.id)
    record_artifact(
        session,
        ref,
        kind="governance_report",
        subject_type=str(SubjectType.VERSION),
        subject_id=version.id,
        created_by=actor_id,
        signature=report_payload["signature"],
    )
    outcome.report_uri = ref.uri
    session.flush()
    record_audit(
        session,
        action="governance.gate",
        actor_id=actor_id,
        subject_type=str(SubjectType.VERSION),
        subject_id=version.id,
        decision=outcome.decision,
        reason="; ".join(blockers[:3]),
        detail={
            "risk_tier": str(risk_tier),
            "approval_id": approval_id,
            "policy": policy_result.decision,
            "similarity": similarity_result.decision,
            "injection": injection_decision,
            "evaluation": eval_decision,
        },
    )
    return outcome


@dataclass
class AutoApprovalDecision:
    allowed: bool
    reason: str


def _evaluate_auto_approval(
    session: Session, *, settings: Any, risk_tier: str, regulated: bool
) -> AutoApprovalDecision:
    """Auto-approval requires *three* independent switches plus low risk."""
    if settings.require_human_approval:
        return AutoApprovalDecision(False, "human approval required by configuration")
    if not settings.auto_publish_enabled:
        return AutoApprovalDecision(False, "auto-publish disabled in settings")
    if not flag_enabled(session, "publishing.auto"):
        return AutoApprovalDecision(False, "publishing.auto runtime flag is off")
    if flag_enabled(session, "publishing.kill_switch", "engaged"):
        return AutoApprovalDecision(False, "publishing kill switch engaged")
    if regulated:
        return AutoApprovalDecision(False, "regulated domain always needs a human")
    if risk_tier not in {str(RiskTier.LOW)}:
        return AutoApprovalDecision(False, f"risk tier {risk_tier} too high for auto-approval")
    return AutoApprovalDecision(True, "auto-approved: low risk, unregulated, all switches enabled")


def decide_approval(
    session: Session,
    *,
    approval_id: str,
    approve: bool,
    actor_id: str,
    reason: str = "",
    conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    approval = session.get(Approval, approval_id)
    if approval is None:
        raise NotFound(f"approval {approval_id} not found")
    if approval.status != str(ApprovalStatus.PENDING):
        raise Conflict(f"approval already {approval.status}")
    if approve and not approval.is_automatic and approval.requested_by == actor_id:
        raise Conflict(
            "separation of duties: the actor who requested this approval cannot approve it"
        )
    approval.status = str(ApprovalStatus.APPROVED if approve else ApprovalStatus.REJECTED)
    approval.decided_by = actor_id
    approval.decided_at = utcnow()
    approval.reason = reason
    if conditions:
        approval.conditions = {**approval.conditions, **conditions}

    version = (
        session.get(PromptVersion, approval.subject_id)
        if approval.subject_type == "version"
        else None
    )
    product = session.get(PromptProduct, version.product_id) if version else None
    if approve:
        if version:
            version.status = str(VersionStatus.APPROVED)
            version.approved_by = actor_id
        if product:
            product.status = str(ProductStatus.APPROVED)
    else:
        if version:
            version.status = str(VersionStatus.DRAFT)
        if product:
            product.status = str(ProductStatus.CANDIDATE)
        raise_alert(
            session,
            kind="approval.rejected",
            severity="info",
            message=f"Approval {approval.id} rejected: {reason[:200]}",
            subject_type=approval.subject_type,
            subject_id=approval.subject_id,
        )
    session.flush()
    record_audit(
        session,
        action="approval.decide",
        actor_id=actor_id,
        subject_type=approval.subject_type,
        subject_id=approval.subject_id,
        decision="approved" if approve else "rejected",
        reason=reason,
        detail={"approval_id": approval.id, "conditions": conditions or {}},
    )
    return {
        "approval_id": approval.id,
        "status": approval.status,
        "version_id": version.id if version else None,
        "product_id": product.id if product else None,
    }


def publish_version(
    session: Session,
    *,
    version_id: str,
    actor_id: str,
    price_aud: float | None = None,
) -> dict[str, Any]:
    """Publish an approved version to the storefront (official APIs only)."""
    from publishing.listings import build_listing
    from publishing.packager import build_prompt_pack
    from publishing.storefront import get_storefront

    if flag_enabled(session, "publishing.kill_switch", "engaged"):
        raise PublishBlocked("publishing kill switch is engaged")
    version = session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    product = session.get(PromptProduct, version.product_id)
    if product is None:
        raise NotFound("product not found for version")

    approval = session.scalar(
        select(Approval)
        .where(
            Approval.subject_type == str(SubjectType.VERSION),
            Approval.subject_id == version.id,
            Approval.status == str(ApprovalStatus.APPROVED),
        )
        .order_by(Approval.decided_at.desc())
        .limit(1)
    )
    if approval is None:
        raise PublishBlocked("no approved approval record for this version")
    if version.status not in {str(VersionStatus.APPROVED), str(VersionStatus.RELEASED)}:
        raise PublishBlocked(f"version status '{version.status}' is not publishable")

    gate_checks = list(
        session.scalars(
            select(SafetyCheck)
            .where(SafetyCheck.subject_id == version.id)
            .order_by(SafetyCheck.created_at.desc())
        )
    )
    if not gate_checks:
        raise PublishBlocked("governance gate has not been run for this version")
    if gate_checks[0].decision == str(GovernanceDecision.BLOCK):
        raise PublishBlocked("latest governance check is a block")

    if price_aud is not None:
        product.price_aud_micros = int(round(price_aud * 1_000_000))

    pack = build_prompt_pack(session, version=version, product=product, actor_id=actor_id)
    listing = build_listing(
        session, product=product, version=version, price_aud=product.price_aud_micros / 1_000_000
    )
    product.listing = listing

    adapter = get_storefront()
    result = adapter.publish(product=product, version=version, listing=listing, pack=pack)
    if not result.ok:
        raise PublishBlocked(f"storefront rejected publish: {result.error}")

    version.status = str(VersionStatus.RELEASED)
    version.released_at = utcnow()
    product.status = str(ProductStatus.PUBLISHED)
    product.current_version_id = version.id
    session.flush()
    record_audit(
        session,
        action="publishing.published",
        actor_id=actor_id,
        subject_type=str(SubjectType.VERSION),
        subject_id=version.id,
        decision="published",
        detail={
            "approval_id": approval.id,
            "storefront": adapter.name,
            "external_id": result.external_id,
            "url": result.url,
            "pack_uri": pack.artifact_uri,
            "listing_fingerprint": result.fingerprint,
            "signature": pack.signature,
        },
    )
    return {
        "status": "published",
        "version_id": version.id,
        "product_id": product.id,
        "storefront": adapter.name,
        "external_id": result.external_id,
        "url": result.url,
        "pack_uri": pack.artifact_uri,
        "listing": listing,
        "pack_signature": pack.signature,
        "approval_id": approval.id,
        "fingerprint": result.fingerprint,
    }


def rollback_version(
    session: Session,
    *,
    product_id: str,
    to_version_id: str,
    actor_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """Revert a product to a previously released version and unpublish the bad one."""
    from publishing.storefront import get_storefront

    product = session.get(PromptProduct, product_id)
    if product is None:
        raise NotFound(f"product {product_id} not found")
    target = session.get(PromptVersion, to_version_id)
    if target is None or target.product_id != product.id:
        raise NotFound("target version not found for this product")
    previous_version_id = product.current_version_id
    if previous_version_id and previous_version_id != target.id:
        previous = session.get(PromptVersion, previous_version_id)
        if previous is not None:
            previous.status = str(VersionStatus.ROLLED_BACK)
            get_storefront().unpublish(product=product, version=previous)
    target.status = str(VersionStatus.APPROVED)
    product.current_version_id = target.id
    product.status = str(ProductStatus.APPROVED)
    session.flush()
    record_audit(
        session,
        action="publishing.rollback",
        actor_id=actor_id,
        subject_type=str(SubjectType.PRODUCT),
        subject_id=product.id,
        decision="rolled_back",
        reason=reason,
        detail={"from_version": previous_version_id, "to_version": target.id},
    )
    raise_alert(
        session,
        kind="publishing.rollback",
        severity="warning",
        message=f"Product {product.id} rolled back to version {target.version}",
        subject_type=str(SubjectType.PRODUCT),
        subject_id=product.id,
        detail={"reason": reason, "from_version": previous_version_id},
    )
    return {
        "product_id": product.id,
        "current_version_id": target.id,
        "previous_version_id": previous_version_id,
        "status": "rolled_back",
        "audit": "recorded",
        "signature": sign_payload(
            {"product_id": product.id, "to": target.id}, get_settings().signing_key
        ),
    }


def gate_history(session: Session, version_id: str) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(SafetyCheck)
        .where(SafetyCheck.subject_id == version_id)
        .order_by(SafetyCheck.created_at.desc())
    )
    out = []
    for row in rows:
        out.append(
            {
                "check": row.check,
                "decision": row.decision,
                "score": row.score,
                "policy_version": row.policy_version,
                "findings": row.findings,
                "created_at": row.created_at.isoformat(),
            }
        )
    return out


def artifacts_summary(session: Session, version_id: str) -> dict[str, Any]:
    rows = artifacts_for(session, version_id)
    return {
        "count": len(rows),
        "kinds": sorted({row.kind for row in rows}),
        "latest": rows[0].uri if rows else None,
        "fingerprint": canonical_json([{"uri": row.uri, "sha256": row.sha256} for row in rows]),
    }


__all__ = [
    "GateOutcome",
    "artifacts_summary",
    "decide_approval",
    "gate_history",
    "get_flag",
    "publish_version",
    "rollback_version",
    "run_governance_gate",
]
