"""Background tasks. Thin wrappers around the same service functions the API calls."""

from __future__ import annotations

from typing import Any

from celery.utils.log import get_task_logger
from sqlalchemy import select

from backend.services.ingestion import run_source
from core.budget import budget_status
from core.clock import utcnow
from core.config import get_settings
from core.errors import BudgetExceeded, DependencyUnavailable
from db.models import Approval, DataSource, Opportunity, PromptVersion
from db.repository import raise_alert, record_audit
from db.session import session_scope
from governance.gate import publish_version
from market_intel.pipeline import run_discovery
from workers.celery_app import app

log = get_task_logger("workers.tasks")


@app.task(bind=True, name="workers.tasks.ingest_all_sources", max_retries=2)
def ingest_all_sources(self: Any, limit: int = 300) -> dict[str, Any]:
    """Pull every enabled source that is *ready* (credentials present)."""
    results: dict[str, Any] = {}
    with session_scope() as session:
        sources = list(session.scalars(select(DataSource).where(DataSource.enabled.is_(True))))
    for source in sources:
        with session_scope() as session:
            try:
                outcome = run_source(session, source_id=source.id, limit=limit, actor_id="worker")
                results[source.id] = {"status": outcome.status, "new": outcome.rows_new}
            except DependencyUnavailable as exc:
                # Missing credentials is an expected, non-fatal state.
                results[source.id] = {"status": "not_ready", "error": str(exc)[:160]}
            except BudgetExceeded as exc:
                results[source.id] = {"status": "budget_blocked", "error": str(exc)[:160]}
            except Exception as exc:
                log.warning("ingest failed for %s: %s", source.id, exc)
                results[source.id] = {"status": "failed", "error": str(exc)[:160]}
    return {"sources": len(results), "results": results}


@app.task(name="workers.tasks.discover_opportunities")
def discover_opportunities(max_signals: int = 500) -> dict[str, Any]:
    with session_scope() as session:
        outcome = run_discovery(session, actor_id="worker", max_signals=max_signals)
        return outcome.to_dict()


@app.task(name="workers.tasks.launch_product_task")
def launch_product_task(opportunity_id: str, actor_id: str = "worker") -> dict[str, Any]:
    """Build + optimise + evaluate + gate a product for one opportunity."""
    from backend.services.orchestration import launch_product

    with session_scope() as session:
        return launch_product(session, opportunity_id=opportunity_id, actor_id=actor_id).to_dict()


@app.task(name="workers.tasks.evaluate_version")
def evaluate_version(
    version_id: str, suite_id: str | None = None, stage: str = "offline"
) -> dict[str, Any]:
    from evaluation.runner import RunConfig, run_evaluation

    with session_scope() as session:
        version = session.get(PromptVersion, version_id)
        if version is None:
            raise ValueError(f"version {version_id} not found")
        outcome = run_evaluation(
            session,
            version_id=version.id,
            suite_id=suite_id,
            config=RunConfig(stage=stage),
            actor_id="worker",
        )
        return outcome.to_dict()


@app.task(bind=True, name="workers.tasks.publish_version_task", max_retries=1)
def publish_version_task(self: Any, version_id: str, actor_id: str = "worker") -> dict[str, Any]:
    """Publish *only* if an approval exists — the gate re-checks this, always."""
    with session_scope() as session:
        return publish_version(session, version_id=version_id, actor_id=actor_id)


@app.task(name="workers.tasks.nightly_maintenance")
def nightly_maintenance() -> dict[str, Any]:
    settings = get_settings()
    summary: dict[str, Any] = {"ran_at": utcnow().isoformat()}
    with session_scope() as session:
        budget = budget_status(session)
        summary["budget"] = budget.to_dict()
        if budget.utilisation >= 0.9:
            raise_alert(
                session,
                kind="budget",
                severity="warning",
                message=f"LLM spend at {budget.utilisation:.0%} of the {settings.budget_window} budget",
                subject_type="budget",
                detail=budget.to_dict(),
            )
        stale = [
            approval
            for approval in session.scalars(select(Approval).where(Approval.status == "pending"))
            if (utcnow() - approval.created_at).days >= 2
        ]
        summary["stale_approvals"] = len(stale)
        if stale:
            raise_alert(
                session,
                kind="approval_backlog",
                severity="info",
                message=f"{len(stale)} approval(s) pending for over 48 hours",
                subject_type="approval",
                detail={"ids": [row.id for row in stale][:10]},
            )
        unreleased = list(
            session.scalars(
                select(Opportunity).where(Opportunity.status.in_(["scored", "approved"]))
            )
        )
        summary["opportunities_waiting"] = len(unreleased)
        record_audit(
            session,
            actor_type="system",
            actor_id="worker",
            action="ops.nightly_maintenance",
            subject_type="system",
            subject_id="maintenance",
            decision="completed",
            detail=summary,
        )
    return summary


@app.task(name="workers.tasks.system_health")
def system_health() -> dict[str, Any]:
    from workers.celery_app import task_health

    with session_scope() as session:
        budget = budget_status(session).to_dict()
    return {"tasks": task_health(), "budget": budget}
