"""Online experiment analysis with guardrails and attribution sanity checks.

Reports, per arm: exposures, conversions, conversion rate with Wilson CI, revenue,
refund rate, and (vs control) uplift, z-test p-value and a bootstrap CI on the
difference. Guardrails (refund rate, support burden) can veto a "ship" decision even
when conversion improves — protecting revenue quality, not just clicks.

Attribution sanity checks exist because broken tracking produces confident nonsense:

* conversions without a recorded exposure;
* conversions timestamped before their exposure;
* duplicate exposures for a subject (allocation instability);
* more purchases than exposures for an arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import AUD_MICROS
from db.models import (
    Conversion,
    Experiment,
    ExperimentArm,
    Exposure,
    Refund,
    SaleEvent,
    SupportTicket,
)
from evaluation.stats import (
    Interval,
    independent_mean_difference,
    two_proportion_z,
    wilson_interval,
)


@dataclass
class ArmReport:
    arm_id: str
    name: str
    version_id: str | None
    is_control: bool
    exposures: int
    conversions: int
    conversion_rate: Interval | None
    revenue_aud: float
    refund_rate: float
    revenue_per_exposure_aud: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "name": self.name,
            "version_id": self.version_id,
            "is_control": self.is_control,
            "exposures": self.exposures,
            "conversions": self.conversions,
            "conversion_rate": self.conversion_rate.to_dict() if self.conversion_rate else None,
            "revenue_aud": round(self.revenue_aud, 2),
            "refund_rate": round(self.refund_rate, 4),
            "revenue_per_exposure_aud": round(self.revenue_per_exposure_aud, 4),
        }


@dataclass
class ExperimentAnalysis:
    experiment_id: str
    status: str
    arms: list[ArmReport] = field(default_factory=list)
    uplift: float | None = None
    uplift_ci: Interval | None = None
    p_value: float | None = None
    guardrail_breaches: list[str] = field(default_factory=list)
    product_support_tickets: int = 0
    attribution_issues: list[str] = field(default_factory=list)
    recommendation: str = "continue"
    reason: str = ""
    sample_ok: bool = False
    winning_arm_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "arms": [arm.to_dict() for arm in self.arms],
            "uplift": None if self.uplift is None else round(self.uplift, 4),
            "uplift_ci": self.uplift_ci.to_dict() if self.uplift_ci else None,
            "p_value": None if self.p_value is None else round(self.p_value, 6),
            "guardrail_breaches": self.guardrail_breaches,
            "product_support_tickets": self.product_support_tickets,
            "attribution_issues": self.attribution_issues,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "sample_ok": self.sample_ok,
            "winning_arm_id": self.winning_arm_id,
        }


def _arm_rows(session: Session, experiment_id: str) -> list[ExperimentArm]:
    return list(
        session.scalars(select(ExperimentArm).where(ExperimentArm.experiment_id == experiment_id))
    )


def _exposures(session: Session, experiment_id: str) -> list[Exposure]:
    return list(session.scalars(select(Exposure).where(Exposure.experiment_id == experiment_id)))


def _conversions(session: Session, experiment_id: str) -> list[Conversion]:
    return list(
        session.scalars(select(Conversion).where(Conversion.experiment_id == experiment_id))
    )


def analyze_experiment(session: Session, experiment: Experiment) -> ExperimentAnalysis:
    arms = _arm_rows(session, experiment.id)
    exposures = _exposures(session, experiment.id)
    conversions = _conversions(session, experiment.id)

    # ------------------------------------------------- attribution sanity
    issues: list[str] = []
    exposure_by_id = {exposure.id: exposure for exposure in exposures}
    subject_counts: dict[str, int] = {}
    for exposure in exposures:
        subject_counts[exposure.subject_key] = subject_counts.get(exposure.subject_key, 0) + 1
    duplicated = [key for key, count in subject_counts.items() if count > 1]
    if duplicated:
        issues.append(f"{len(duplicated)} subject(s) exposed more than once")
    orphan_conversions = [
        c for c in conversions if c.exposure_id and c.exposure_id not in exposure_by_id
    ]
    if orphan_conversions:
        issues.append(f"{len(orphan_conversions)} conversion(s) reference a missing exposure")
    unexposed = [c for c in conversions if not c.exposure_id]
    if unexposed:
        issues.append(f"{len(unexposed)} conversion(s) have no exposure (untracked traffic)")
    time_travel = [
        c
        for c in conversions
        if c.exposure_id
        and c.exposure_id in exposure_by_id
        and c.occurred_at < exposure_by_id[c.exposure_id].occurred_at
    ]
    if time_travel:
        issues.append(f"{len(time_travel)} conversion(s) timestamped before their exposure")

    sale_rows = list(
        session.scalars(
            select(SaleEvent).where(
                SaleEvent.exposure_id.in_([exposure.id for exposure in exposures] or [""])
            )
        )
    )
    refunds_by_sale = {refund.sale_id: refund for refund in session.scalars(select(Refund))}
    # Support burden is measured at product level — tickets are not attributable to an
    # arm without an agent-side field, and pretending otherwise would be misleading.
    product_support_tickets = 0
    if experiment.product_id:
        product_support_tickets = len(
            list(
                session.scalars(
                    select(SupportTicket).where(SupportTicket.product_id == experiment.product_id)
                )
            )
        )

    # ------------------------------------------------------------- per arm
    reports: list[ArmReport] = []
    for arm in arms:
        arm_exposures = [exposure for exposure in exposures if exposure.arm_id == arm.id]
        arm_conversions = [conversion for conversion in conversions if conversion.arm_id == arm.id]
        purchases = [
            c for c in arm_conversions if c.event_type in {"purchase", "checkout_completed"}
        ]
        if len(purchases) > len(arm_exposures):
            issues.append(
                f"arm {arm.name}: more purchases ({len(purchases)}) than exposures ({len(arm_exposures)})"
            )
        revenue = sum(int(c.value_micros or 0) for c in arm_conversions)
        arm_sales = [
            sale
            for sale in sale_rows
            if exposure_by_id.get(sale.exposure_id or "")
            and exposure_by_id[sale.exposure_id].arm_id == arm.id
        ]
        arm_refunds = [refunds_by_sale[sale.id] for sale in arm_sales if sale.id in refunds_by_sale]
        refund_rate = (
            sum(refund.amount_micros for refund in arm_refunds) / revenue if revenue else 0.0
        )
        rate_ci = wilson_interval(len(purchases), len(arm_exposures)) if arm_exposures else None
        reports.append(
            ArmReport(
                arm_id=arm.id,
                name=arm.name,
                version_id=arm.version_id,
                is_control=arm.is_control,
                exposures=len(arm_exposures),
                conversions=len(purchases),
                conversion_rate=rate_ci,
                revenue_aud=revenue / AUD_MICROS,
                refund_rate=refund_rate,
                revenue_per_exposure_aud=(revenue / AUD_MICROS) / len(arm_exposures)
                if arm_exposures
                else 0.0,
            )
        )

    # -------------------------------------------------------- comparison
    control = next((report for report in reports if report.is_control), None)
    treatment = max(
        (report for report in reports if not report.is_control and report.exposures > 0),
        key=lambda report: report.revenue_per_exposure_aud,
        default=None,
    )
    uplift: float | None = None
    uplift_ci: Interval | None = None
    p_value: float | None = None
    if control and treatment and control.exposures and treatment.exposures:
        control_rate = control.conversions / control.exposures
        treatment_rate = treatment.conversions / treatment.exposures
        uplift = (treatment_rate - control_rate) / control_rate if control_rate else None
        _, p_value = two_proportion_z(
            control.conversions, control.exposures, treatment.conversions, treatment.exposures
        )
        purchased_ids = {
            c.exposure_id for c in conversions if c.event_type in {"purchase", "checkout_completed"}
        }
        control_flags = [
            1.0 if exposure.id in purchased_ids else 0.0
            for exposure in exposures
            if exposure.arm_id == control.arm_id
        ]
        treatment_flags = [
            1.0 if exposure.id in purchased_ids else 0.0
            for exposure in exposures
            if exposure.arm_id == treatment.arm_id
        ]
        uplift_ci = independent_mean_difference(control_flags, treatment_flags)

    # --------------------------------------------------------- guardrails
    guardrails = experiment.guardrails or {}
    max_refund = float(guardrails.get("max_refund_rate", 0.12))
    max_support = int(guardrails.get("max_support_tickets", 50))
    breaches: list[str] = []
    for report in reports:
        if report.refund_rate > max_refund and report.revenue_aud > 0:
            breaches.append(
                f"arm {report.name} refund rate {report.refund_rate:.1%} > {max_refund:.0%}"
            )
    if product_support_tickets > max_support:
        breaches.append(
            f"product-level support tickets {product_support_tickets} > {max_support} "
            "(not attributable per arm)"
        )
    if (
        treatment
        and uplift is not None
        and uplift < 0
        and p_value is not None
        and p_value < experiment.alpha
    ):
        breaches.append("treatment conversion is significantly worse than control")

    min_sample = max(int(experiment.min_sample), 1)
    sample_ok = bool(reports) and all(report.exposures >= min_sample for report in reports)

    # ---------------------------------------------------------- decision
    recommendation = "continue"
    reason = "collecting data"
    winning_arm_id: str | None = None
    if issues:
        recommendation = "fix_attribution"
        reason = "attribution issues present; results are not trustworthy yet"
    elif breaches:
        recommendation = "stop_and_review"
        reason = "; ".join(breaches[:3])
    elif not sample_ok:
        recommendation = "continue"
        needed = min_sample - min((report.exposures for report in reports), default=0)
        reason = f"need ~{max(needed, 0)} more exposures per arm (min {min_sample})"
    elif treatment and p_value is not None and p_value < experiment.alpha:
        if uplift is not None and uplift > 0:
            recommendation = "ship_treatment"
            reason = (
                f"treatment uplift {uplift:+.1%} with p={p_value:.4f} < alpha={experiment.alpha}"
            )
            winning_arm_id = treatment.arm_id
        else:
            recommendation = "rollback"
            reason = f"treatment underperforms ({uplift:+.1%}) with p={p_value:.4f}"
            winning_arm_id = control.arm_id if control else None
    else:
        recommendation = "inconclusive"
        reason = (
            f"p-value {p_value if p_value is None else round(p_value, 4)} above alpha; keep or stop"
        )

    return ExperimentAnalysis(
        experiment_id=experiment.id,
        status=experiment.status,
        arms=reports,
        uplift=uplift,
        uplift_ci=uplift_ci,
        p_value=p_value,
        guardrail_breaches=breaches,
        attribution_issues=issues,
        product_support_tickets=product_support_tickets,
        recommendation=recommendation,
        reason=reason,
        sample_ok=sample_ok,
        winning_arm_id=winning_arm_id,
    )
