"""Experimentation service: create → assign → convert → analyse → ship/rollback.

Stage 4 of the optimisation loop. Deliberately conservative:

* Traffic is capped by ``traffic_fraction``; subjects outside the rollout get
  ``None`` (no exposure) so the control population stays clean.
* Assignment is stable per subject (hash-based), so a returning buyer does not flip
  arms and pollute attribution.
* Conversions must reference an exposure; the sale/refund ledger is separate from
  conversions so revenue reporting survives tracking bugs.
* A ``rollback`` recommendation can call ``governance.gate.rollback_version`` directly,
  and every state change is audited.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import utcnow
from core.constants import EventType, ExperimentKind, ExperimentStatus, ProductStatus, SubjectType
from core.errors import Conflict, NotFound, ValidationError
from db.models import (
    Conversion,
    Experiment,
    ExperimentArm,
    Exposure,
    PromptProduct,
    PromptVersion,
    Refund,
    SaleEvent,
    SupportTicket,
)
from db.repository import raise_alert, record_audit
from experimentation.bandits import allocate, bandit_state_from_rows
from experimentation.stats import ExperimentAnalysis, analyze_experiment
from observability.metrics import EXPERIMENT_EXPOSURES


@dataclass
class Assignment:
    exposure_id: str
    arm_id: str
    arm_name: str
    version_id: str | None
    propensity: float
    reason: str
    experiment_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "exposure_id": self.exposure_id,
            "experiment_id": self.experiment_id,
            "arm_id": self.arm_id,
            "arm_name": self.arm_name,
            "version_id": self.version_id,
            "propensity": round(self.propensity, 4),
            "reason": self.reason,
        }


def create_experiment(
    session: Session,
    *,
    product_id: str,
    name: str,
    arms: list[dict[str, Any]],
    hypothesis: str = "",
    kind: str = str(ExperimentKind.AB_TEST),
    metric_primary: str = "conversion",
    traffic_fraction: float = 0.1,
    min_sample: int = 200,
    alpha: float = 0.05,
    guardrails: dict[str, Any] | None = None,
    actor_id: str = "system",
) -> dict[str, Any]:
    product = session.get(PromptProduct, product_id)
    if product is None:
        raise NotFound(f"product {product_id} not found")
    if len(arms) < 2:
        raise ValidationError("an experiment needs at least two arms")
    if not 0 < traffic_fraction <= 1:
        raise ValidationError("traffic_fraction must be in (0, 1]")
    controls = [arm for arm in arms if arm.get("is_control")]
    if len(controls) != 1:
        raise ValidationError("exactly one arm must be the control")

    experiment = Experiment(
        product_id=product.id,
        name=name,
        hypothesis=hypothesis,
        kind=kind,
        status=str(ExperimentStatus.RUNNING),
        metric_primary=metric_primary,
        guardrails=guardrails or {"max_refund_rate": 0.12, "max_support_tickets": 50},
        alpha=alpha,
        min_sample=min_sample,
        traffic_fraction=traffic_fraction,
        started_at=utcnow(),
        created_by=actor_id,
    )
    session.add(experiment)
    session.flush()
    for arm in arms:
        session.add(
            ExperimentArm(
                experiment_id=experiment.id,
                name=str(arm["name"]),
                version_id=arm.get("version_id"),
                variant_id=arm.get("variant_id"),
                weight=float(arm.get("weight", 1.0 / len(arms))),
                is_control=bool(arm.get("is_control", False)),
                bandit_params={},
            )
        )
    session.flush()
    if product.status == str(ProductStatus.PUBLISHED):
        product.status = str(ProductStatus.PUBLISHED)  # unchanged; experiment runs on live traffic
    record_audit(
        session,
        action="experiment.created",
        actor_id=actor_id,
        subject_type=str(SubjectType.EXPERIMENT),
        subject_id=experiment.id,
        decision="running",
        detail={"product_id": product.id, "arms": [arm["name"] for arm in arms]},
    )
    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "kind": experiment.kind,
        "status": experiment.status,
        "arms": [
            {"name": arm["name"], "is_control": bool(arm.get("is_control", False))} for arm in arms
        ],
        "traffic_fraction": experiment.traffic_fraction,
        "min_sample": experiment.min_sample,
    }


def _in_rollout(experiment: Experiment, subject_key: str) -> bool:
    digest = hashlib.sha256(f"{experiment.id}|rollout|{subject_key}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return bucket < experiment.traffic_fraction


def assign(
    session: Session,
    *,
    experiment_id: str,
    subject_key: str,
    channel: str = "direct",
    force_arm: str | None = None,
    actor_id: str = "system",
) -> Assignment | None:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    if experiment.status != str(ExperimentStatus.RUNNING):
        raise Conflict(f"experiment is {experiment.status}, not running")
    if not subject_key:
        raise ValidationError("subject_key is required")

    existing = session.scalar(
        select(Exposure).where(
            Exposure.experiment_id == experiment.id, Exposure.subject_key == subject_key
        )
    )
    if existing is not None:
        arm = session.get(ExperimentArm, existing.arm_id)
        return Assignment(
            exposure_id=existing.id,
            arm_id=existing.arm_id,
            arm_name=arm.name if arm else "",
            version_id=arm.version_id if arm else None,
            propensity=existing.assignment_probability,
            reason="existing_assignment",
            experiment_id=experiment.id,
        )

    if not _in_rollout(experiment, subject_key):
        return None

    arms = list(
        session.scalars(select(ExperimentArm).where(ExperimentArm.experiment_id == experiment.id))
    )
    if not arms:
        raise ValidationError("experiment has no arms")
    exposures = list(
        session.scalars(select(Exposure).where(Exposure.experiment_id == experiment.id))
    )
    conversions = list(
        session.scalars(select(Conversion).where(Conversion.experiment_id == experiment.id))
    )
    states = bandit_state_from_rows(arms, exposures, conversions)

    if force_arm:
        arm = next((arm for arm in arms if arm.id == force_arm or arm.name == force_arm), None)
        if arm is None:
            raise ValidationError(f"unknown arm '{force_arm}'")
        arm_id, propensity, reason = arm.id, 1.0, "forced_assignment"
    else:
        strategy = "thompson" if experiment.kind == str(ExperimentKind.BANDIT) else "fixed"
        arm_id, propensity, reason = allocate(
            states, strategy=strategy, subject_key=subject_key, salt=experiment.id
        )
        arm = next(arm for arm in arms if arm.id == arm_id)

    exposure = Exposure(
        experiment_id=experiment.id,
        arm_id=arm_id,
        subject_key=subject_key,
        channel=channel,
        assignment_probability=propensity,
        meta={"reason": reason, "strategy": experiment.kind},
    )
    session.add(exposure)
    session.flush()
    EXPERIMENT_EXPOSURES.inc(arm=arm.name, experiment=experiment.id)
    record_audit(
        session,
        action="experiment.exposed",
        actor_id=actor_id,
        subject_type=str(SubjectType.EXPERIMENT),
        subject_id=experiment.id,
        decision=arm.name,
        detail={"subject_key": subject_key, "propensity": propensity, "reason": reason},
    )
    return Assignment(
        exposure_id=exposure.id,
        arm_id=arm_id,
        arm_name=arm.name,
        version_id=arm.version_id,
        propensity=propensity,
        reason=reason,
        experiment_id=experiment.id,
    )


def record_conversion(
    session: Session,
    *,
    exposure_id: str,
    event_type: str = str(EventType.PURCHASE),
    value_micros: int = 0,
    order_ref: str | None = None,
    meta: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
) -> dict[str, Any]:
    exposure = session.get(Exposure, exposure_id)
    if exposure is None:
        raise NotFound(f"exposure {exposure_id} not found")
    experiment = session.get(Experiment, exposure.experiment_id)
    conversion = Conversion(
        experiment_id=exposure.experiment_id,
        exposure_id=exposure.id,
        arm_id=exposure.arm_id,
        event_type=event_type,
        value_micros=value_micros,
        occurred_at=occurred_at or utcnow(),
        meta=meta or {},
    )
    session.add(conversion)
    session.flush()

    sale_id = None
    if event_type in {str(EventType.PURCHASE), "checkout_completed"}:
        product = (
            session.get(PromptProduct, experiment.product_id)
            if experiment and experiment.product_id
            else None
        )
        version_id = None
        if experiment and experiment.product_id:
            arm = session.get(ExperimentArm, exposure.arm_id)
            version_id = arm.version_id if arm else None
        sale = SaleEvent(
            product_id=product.id if product else None,
            version_id=version_id,
            exposure_id=exposure.id,
            order_ref=order_ref or f"ord_{exposure.id[-10:]}_{conversion.id[-6:]}",
            provider=str((meta or {}).get("provider", "mock")),
            amount_micros=value_micros,
            fee_micros=int((meta or {}).get("fee_micros", 0)),
            currency=str((meta or {}).get("currency", "AUD")),
            channel=exposure.channel,
            country=str((meta or {}).get("country", "AU")),
            occurred_at=conversion.occurred_at,
            meta=meta or {},
        )
        session.add(sale)
        session.flush()
        sale_id = sale.id
    record_audit(
        session,
        action="experiment.conversion",
        actor_id="system",
        subject_type=str(SubjectType.EXPERIMENT),
        subject_id=exposure.experiment_id,
        decision=event_type,
        detail={"exposure_id": exposure.id, "value_micros": value_micros, "sale_id": sale_id},
    )
    return {"conversion_id": conversion.id, "sale_id": sale_id, "exposure_id": exposure.id}


def record_refund(
    session: Session,
    *,
    sale_id: str,
    amount_micros: int,
    reason_code: str = "other",
    note: str = "",
) -> dict[str, Any]:
    sale = session.get(SaleEvent, sale_id)
    if sale is None:
        raise NotFound(f"sale {sale_id} not found")
    refund = Refund(
        sale_id=sale.id,
        amount_micros=min(amount_micros, sale.amount_micros),
        reason_code=reason_code,
        reason_note=note,
    )
    session.add(refund)
    if sale.exposure_id:
        exposure = session.get(Exposure, sale.exposure_id)
        if exposure is not None:
            session.add(
                Conversion(
                    experiment_id=exposure.experiment_id,
                    exposure_id=exposure.id,
                    arm_id=exposure.arm_id,
                    event_type=str(EventType.REFUND),
                    value_micros=-abs(amount_micros),
                    meta={"sale_id": sale.id, "reason_code": reason_code},
                )
            )
    session.flush()
    record_audit(
        session,
        action="commerce.refund",
        actor_id="system",
        subject_type="sale",
        subject_id=sale.id,
        decision=reason_code,
        detail={"amount_micros": amount_micros, "note": note[:200]},
    )
    return {"refund_id": refund.id, "sale_id": sale.id, "amount_micros": refund.amount_micros}


def record_support_ticket(
    session: Session,
    *,
    product_id: str,
    subject: str,
    body: str = "",
    minutes_spent: int = 0,
) -> dict[str, Any]:
    ticket = SupportTicket(
        product_id=product_id,
        subject=subject[:240],
        body=body[:4000],
        minutes_spent=minutes_spent,
    )
    session.add(ticket)
    session.flush()
    return {"ticket_id": ticket.id}


def analyse(session: Session, experiment_id: str, *, actor_id: str = "system") -> dict[str, Any]:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    analysis = analyze_experiment(session, experiment)
    payload = analysis.to_dict()

    # Persist the analysis as an artifact so decisions have a stable reference.
    from core.artifacts import get_store
    from db.repository import record_artifact

    ref = get_store().put_json(payload, kind="experiment_report", subject_id=experiment.id)
    record_artifact(
        session,
        ref,
        kind="experiment_report",
        subject_type=str(SubjectType.EXPERIMENT),
        subject_id=experiment.id,
        created_by=actor_id,
    )
    experiment.report_uri = ref.uri
    if analysis.recommendation in {"ship_treatment", "rollback", "stop_and_review"}:
        experiment.decision = analysis.recommendation
        experiment.decision_reason = analysis.reason
    session.flush()
    record_audit(
        session,
        action="experiment.analyzed",
        actor_id=actor_id,
        subject_type=str(SubjectType.EXPERIMENT),
        subject_id=experiment.id,
        decision=analysis.recommendation,
        detail={"reason": analysis.reason, "p_value": analysis.p_value},
    )
    return {**payload, "report_uri": ref.uri}


def stop_experiment(
    session: Session, *, experiment_id: str, reason: str, actor_id: str
) -> dict[str, Any]:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    experiment.status = str(ExperimentStatus.STOPPED)
    experiment.stopped_at = utcnow()
    experiment.decision = experiment.decision or "stopped"
    experiment.decision_reason = reason
    session.flush()
    record_audit(
        session,
        action="experiment.stopped",
        actor_id=actor_id,
        subject_type=str(SubjectType.EXPERIMENT),
        subject_id=experiment.id,
        decision="stopped",
        reason=reason,
    )
    return {"experiment_id": experiment.id, "status": experiment.status, "reason": reason}


def apply_decision(
    session: Session,
    *,
    experiment_id: str,
    actor_id: str,
    promote_winning_arm: bool = False,
) -> dict[str, Any]:
    """Act on the analysis: promote the winning version or roll back to control."""
    from governance.gate import rollback_version

    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    analysis: ExperimentAnalysis = analyze_experiment(session, experiment)
    if analysis.recommendation == "fix_attribution":
        raise Conflict("refusing to act while attribution issues are unresolved")
    if not experiment.product_id:
        raise ValidationError("experiment has no product")

    if (
        analysis.recommendation == "ship_treatment"
        and promote_winning_arm
        and analysis.winning_arm_id
    ):
        arm = session.get(ExperimentArm, analysis.winning_arm_id)
        if arm and arm.version_id:
            version = session.get(PromptVersion, arm.version_id)
            if version is None:
                raise NotFound("winning arm version missing")
            product = session.get(PromptProduct, experiment.product_id)
            if product is not None:
                product.current_version_id = version.id
                session.flush()
            stop_experiment(
                session,
                experiment_id=experiment.id,
                reason="shipped winning arm",
                actor_id=actor_id,
            )
            record_audit(
                session,
                action="experiment.shipped",
                actor_id=actor_id,
                subject_type=str(SubjectType.PRODUCT),
                subject_id=product.id if product else "",
                decision="shipped",
                detail={"version_id": version.id, "arm": arm.name},
            )
            return {"action": "shipped", "version_id": version.id, "analysis": analysis.to_dict()}
        raise ValidationError("winning arm has no version to promote")

    if analysis.recommendation == "rollback":
        control = next((arm for arm in analysis.arms if arm.is_control), None)
        if control and control.version_id:
            result = rollback_version(
                session,
                product_id=experiment.product_id,
                to_version_id=control.version_id,
                actor_id=actor_id,
                reason=f"experiment {experiment.id}: {analysis.reason}",
            )
            stop_experiment(
                session,
                experiment_id=experiment.id,
                reason="rolled back to control",
                actor_id=actor_id,
            )
            return {"action": "rolled_back", "rollback": result, "analysis": analysis.to_dict()}

    if analysis.recommendation == "stop_and_review":
        stop_experiment(
            session, experiment_id=experiment.id, reason=analysis.reason, actor_id=actor_id
        )
        raise_alert(
            session,
            kind="experiment.guardrail_breach",
            severity="warning",
            message=f"Experiment {experiment.id} stopped: {analysis.reason}",
            subject_type=str(SubjectType.EXPERIMENT),
            subject_id=experiment.id,
        )
        return {"action": "stopped", "analysis": analysis.to_dict()}

    return {"action": "no_change", "analysis": analysis.to_dict()}


def experiment_summary(session: Session, experiment_id: str) -> dict[str, Any]:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFound(f"experiment {experiment_id} not found")
    arms = list(
        session.scalars(select(ExperimentArm).where(ExperimentArm.experiment_id == experiment_id))
    )
    exposures = session.query(Exposure).filter(Exposure.experiment_id == experiment_id).count()
    conversions = (
        session.query(Conversion).filter(Conversion.experiment_id == experiment_id).count()
    )
    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "kind": experiment.kind,
        "status": experiment.status,
        "product_id": experiment.product_id,
        "metric_primary": experiment.metric_primary,
        "alpha": experiment.alpha,
        "min_sample": experiment.min_sample,
        "traffic_fraction": experiment.traffic_fraction,
        "armed": len(arms),
        "exposures": exposures,
        "conversions": conversions,
        "decision": experiment.decision,
        "decision_reason": experiment.decision_reason,
        "report_uri": experiment.report_uri,
        "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
        "stopped_at": experiment.stopped_at.isoformat() if experiment.stopped_at else None,
    }
