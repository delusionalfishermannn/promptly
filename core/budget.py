"""Cost accounting + hard budget circuit breakers.

All spend is recorded as ``cost_events`` in **AUD micros**. Before any LLM call the
``LLMClient`` asks :func:`assert_within_budget`; if the daily or weekly cap would be
exceeded the call is refused with ``BudgetExceeded`` (HTTP 402) and an alert fires.

The ledger is the single source of truth: spend is recomputed from ``cost_events``
(never cached in memory) so restarts cannot reset the breaker, and tests can inject
costs to prove the breaker trips.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.clock import utcnow
from core.config import AUD_MICROS, get_settings
from core.errors import BudgetExceeded
from db.models import CostEvent
from observability.metrics import ALERTS_FIRED, BUDGET_USED


@dataclass
class BudgetStatus:
    daily_spent_micros: int
    weekly_spent_micros: int
    daily_limit_micros: int
    weekly_limit_micros: int
    ratio: float
    breached: bool
    reason: str | None = None

    @property
    def daily_remaining_aud(self) -> float:
        return max(0, self.daily_limit_micros - self.daily_spent_micros) / AUD_MICROS

    def to_dict(self) -> dict[str, object]:
        return {
            "daily_spent_aud": round(self.daily_spent_micros / AUD_MICROS, 4),
            "weekly_spent_aud": round(self.weekly_spent_micros / AUD_MICROS, 4),
            "daily_limit_aud": round(self.daily_limit_micros / AUD_MICROS, 2),
            "weekly_limit_aud": round(self.weekly_limit_micros / AUD_MICROS, 2),
            "utilisation": round(self.ratio, 4),
            "breached": self.breached,
            "reason": self.reason,
        }


def _window_sums(session: Session) -> tuple[int, int]:
    now = utcnow()
    day_start = now - timedelta(days=1)
    week_start = now - timedelta(days=7)
    stmt = select(CostEvent.created_at, CostEvent.amount_micros).where(
        CostEvent.created_at >= week_start
    )
    daily = weekly = 0
    for created_at, amount in session.execute(stmt):
        weekly += int(amount)
        created = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
        if created >= day_start:
            daily += int(amount)
    return daily, weekly


def budget_status(session: Session) -> BudgetStatus:
    settings = get_settings()
    daily, weekly = _window_sums(session)
    ratio = max(
        daily / settings.budget_daily_llm_micros if settings.budget_daily_llm_micros else 0.0,
        weekly / settings.budget_weekly_llm_micros if settings.budget_weekly_llm_micros else 0.0,
    )
    breached = bool(settings.budget_enforced) and (
        daily >= settings.budget_daily_llm_micros or weekly >= settings.budget_weekly_llm_micros
    )
    reason = None
    if settings.budget_enforced:
        if daily >= settings.budget_daily_llm_micros:
            reason = "daily_llm_cap"
        elif weekly >= settings.budget_weekly_llm_micros:
            reason = "weekly_llm_cap"
    BUDGET_USED.set(ratio)
    if ratio >= settings.budget_alert_ratio and not breached:
        ALERTS_FIRED.inc(alert="budget_warning")
    return BudgetStatus(
        daily_spent_micros=daily,
        weekly_spent_micros=weekly,
        daily_limit_micros=settings.budget_daily_llm_micros,
        weekly_limit_micros=settings.budget_weekly_llm_micros,
        ratio=round(ratio, 4),
        breached=breached,
        reason=reason,
    )


def assert_within_budget(session: Session, estimated_micros: int = 0) -> BudgetStatus:
    status = budget_status(session)
    if status.breached:
        ALERTS_FIRED.inc(alert="budget_circuit_breaker")
        raise BudgetExceeded(
            "LLM budget exhausted; circuit breaker open. Raise caps or wait for the window to roll.",
            details={**status.to_dict(), "estimated_next_call_aud": estimated_micros / AUD_MICROS},
        )
    if (
        estimated_micros
        and status.daily_spent_micros + estimated_micros > status.daily_limit_micros
    ):
        ALERTS_FIRED.inc(alert="budget_preemptive_block")
        raise BudgetExceeded(
            "Next call would exceed the daily LLM cap.",
            details={
                **status.to_dict(),
                "estimated_next_call_aud": estimated_micros / AUD_MICROS,
            },
        )
    return status


def record_cost(
    session: Session,
    *,
    category: str,
    amount_micros: int,
    provider: str | None = None,
    model: str | None = None,
    purpose: str | None = None,
    subject_type: str | None = None,
    subject_id: str | None = None,
    detail: dict[str, object] | None = None,
) -> CostEvent:
    """Append an immutable spend record (never mutates existing rows)."""
    event = CostEvent(
        category=category,
        provider=provider,
        model=model,
        purpose=purpose,
        amount_micros=int(amount_micros),
        currency="AUD",
        subject_type=subject_type,
        subject_id=subject_id,
        detail=detail or {},
    )
    session.add(event)
    session.flush()
    return event


def spend_summary(session: Session, days: int = 7) -> list[dict[str, object]]:
    since = utcnow() - timedelta(days=days)
    stmt = (
        select(
            CostEvent.category,
            func.sum(CostEvent.amount_micros).label("total"),
            func.count(CostEvent.id).label("events"),
        )
        .where(CostEvent.created_at >= since)
        .group_by(CostEvent.category)
        .order_by(func.sum(CostEvent.amount_micros).desc())
    )
    return [
        {
            "category": row.category,
            "total_aud": round(int(row.total) / AUD_MICROS, 4),
            "events": row.events,
        }
        for row in session.execute(stmt)
    ]


def spend_by_day(session: Session, days: int = 14) -> list[dict[str, object]]:
    since = utcnow() - timedelta(days=days)
    stmt = (
        select(CostEvent.created_at, CostEvent.amount_micros)
        .where(CostEvent.created_at >= since)
        .order_by(CostEvent.created_at)
    )
    buckets: dict[str, int] = {}
    for created_at, amount in session.execute(stmt):
        day = (created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)).strftime(
            "%Y-%m-%d"
        )
        buckets[day] = buckets.get(day, 0) + int(amount)
    return [
        {"date": day, "spend_aud": round(total / AUD_MICROS, 4)}
        for day, total in sorted(buckets.items())
    ]


def estimate_micros(prompt_tokens: int, completion_tokens: int, model: str) -> int:
    """Delegate to the model catalogue (kept here so budget logic is self-contained)."""
    from llm.catalog import estimate_cost_micros

    return estimate_cost_micros(prompt_tokens, completion_tokens, model)


__all__ = [
    "BudgetStatus",
    "assert_within_budget",
    "budget_status",
    "datetime",
    "estimate_micros",
    "record_cost",
    "spend_by_day",
    "spend_summary",
    "utcnow",
]


# --------------------------------------------------------------------------- #
# Shadow pricing
# --------------------------------------------------------------------------- #
# The mock simulator is free by design (no keys, no spend). To keep cost *discipline*
# testable during a zero-key demo we also report a "shadow cost": what the same tokens
# would have cost at a reference mid-tier model price. This never touches the ledger.
REFERENCE_USD_PER_MTOK_IN = 0.15
REFERENCE_USD_PER_MTOK_OUT = 0.60


def shadow_cost_micros(tokens_in: int, tokens_out: int) -> int:
    settings = get_settings()
    usd = (
        tokens_in / 1_000_000 * REFERENCE_USD_PER_MTOK_IN
        + tokens_out / 1_000_000 * REFERENCE_USD_PER_MTOK_OUT
    )
    return int(round(usd * settings.aud_per_usd * AUD_MICROS))
