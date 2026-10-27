"""Governance gate, approvals and policy inspection."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import ApprovalDecisionRequest, GateRequest
from backend.services.orchestration import approval_queue
from core.errors import NotFound
from core.rbac import Permission
from db.models import Approval, IpSimilarityCheck, PromptVersion, SafetyCheck
from governance.gate import decide_approval, gate_history, run_governance_gate
from governance.policy import policy_snapshot

router = APIRouter(tags=["governance"])


@router.get("/policy")
def policy(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    return policy_snapshot()


@router.post("/versions/{version_id}/gate")
def gate(
    version_id: str,
    payload: GateRequest,
    context: Annotated[RequestContext, Depends(require(Permission.APPROVE))],
) -> dict[str, Any]:
    outcome = run_governance_gate(
        context.session,
        version_id=version_id,
        actor_id=context.actor_id,
        run_injection=payload.run_injection,
        request_approval=payload.request_approval,
        eval_run_id=payload.eval_run_id,
    )
    return outcome.to_dict()


@router.get("/versions/{version_id}/gate")
def gate_status(
    version_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
) -> dict[str, Any]:
    version = context.session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    checks = gate_history(context.session, version.id)
    similarity = context.session.scalar(
        select(IpSimilarityCheck)
        .where(IpSimilarityCheck.subject_id == version.id)
        .order_by(IpSimilarityCheck.created_at.desc())
        .limit(1)
    )
    return {
        "version_id": version.id,
        "status": version.status,
        "checks": checks,
        "similarity": {
            "decision": similarity.decision,
            "max_similarity": similarity.max_similarity,
            "shingle_overlap": similarity.shingle_overlap,
            "nearest_ref": similarity.nearest_ref,
        }
        if similarity
        else None,
    }


@router.get("/versions/{version_id}/gate/history")
def history(
    version_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_AUDIT))],
) -> dict[str, Any]:
    checks = gate_history(context.session, version_id)
    return {"items": checks, "count": len(checks)}


@router.get("/approvals")
def approvals(
    context: Annotated[RequestContext, Depends(require(Permission.APPROVE))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    items = approval_queue(context.session, limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/approvals/{approval_id}")
def approval_detail(
    approval_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.APPROVE))],
) -> dict[str, Any]:
    approval = context.session.get(Approval, approval_id)
    if approval is None:
        raise NotFound(f"approval {approval_id} not found")
    checks = list(
        context.session.scalars(
            select(SafetyCheck)
            .where(SafetyCheck.subject_id == approval.subject_id)
            .order_by(SafetyCheck.created_at.desc())
        )
    )
    return {
        "approval_id": approval.id,
        "status": approval.status,
        "subject_type": approval.subject_type,
        "subject_id": approval.subject_id,
        "requested_by": approval.requested_by,
        "requested_at": approval.requested_at.isoformat(),
        "conditions": approval.conditions,
        "policy_snapshot": approval.policy_snapshot,
        "decided_by": approval.decided_by,
        "decided_at": approval.decided_at.isoformat() if approval.decided_at else None,
        "reason": approval.reason,
        "checks": [
            {
                "check": row.check,
                "decision": row.decision,
                "score": row.score,
                "findings": row.findings,
            }
            for row in checks
        ],
    }


@router.post("/approvals/{approval_id}/approve")
def approve(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    context: Annotated[RequestContext, Depends(require(Permission.APPROVE))],
) -> dict[str, Any]:
    return decide_approval(
        context.session,
        approval_id=approval_id,
        approve=True,
        actor_id=context.actor_id,
        reason=payload.reason,
        conditions=payload.conditions,
    )


@router.post("/approvals/{approval_id}/reject")
def reject(
    approval_id: str,
    payload: ApprovalDecisionRequest,
    context: Annotated[RequestContext, Depends(require(Permission.APPROVE))],
) -> dict[str, Any]:
    return decide_approval(
        context.session,
        approval_id=approval_id,
        approve=False,
        actor_id=context.actor_id,
        reason=payload.reason,
        conditions=payload.conditions,
    )
