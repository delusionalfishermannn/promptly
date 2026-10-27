"""Deterministic seed data.

Run ``python -m db.seed`` (or ``make demo``) to make a fresh database *useful* without
any API keys or internet access:

* role definitions + runtime flags + demo operator/admin accounts
* test suites from ``demo/data/tests`` (visible + holdout)
* permitted corpora from ``demo/data/corpora`` (originality-check references)
* two data sources: a manual CSV upload and the mock official-API connector
* a demo ingestion run + discovery pass so opportunities exist immediately
* judge calibration from ``demo/data/calibration.json``

Everything is idempotent: re-running only fills in what is missing. Every object records
the actor ``seed`` and an audit entry so the provenance of demo data is obvious.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from backend.services.auth import bootstrap, register_user
from backend.services.ingestion import create_source, ingest_upload, run_source
from core.clock import utcnow
from core.config import get_settings
from core.constants import PermissionBasis
from core.errors import Conflict
from db.models import (
    Benchmark,
    DataSource,
    JudgeCalibration,
    NormalizedSignal,
    Opportunity,
    PromptProduct,
)
from db.repository import assign_role, ensure_default_flags, get_user_by_email, record_audit
from db.session import init_db, session_scope
from evaluation.calibration import CalibrationItem, calibrate_judge
from evaluation.suites import load_suite, parse_suite_file, suite_status_counts
from governance.similarity import load_permitted_corpus

SEED_ACTOR = "seed"

DEMO_USERS = [
    {
        "email": "owner@promptfoundry.local",
        "password": "owner-demo-password",
        "display_name": "Founder (owner)",
        "roles": ["admin", "approver"],
    },
    {
        "email": "analyst@promptfoundry.local",
        "password": "analyst-demo-password",
        "display_name": "Demand analyst",
        "roles": ["analyst"],
    },
    {
        "email": "approver@promptfoundry.local",
        "password": "approver-demo-password",
        "display_name": "Release approver",
        "roles": ["approver"],
    },
]


@dataclass
class SeedReport:
    created: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, kind: str, *, created: bool = True, count: int = 1) -> None:
        bucket = self.created if created else self.skipped
        bucket[kind] = bucket.get(kind, 0) + count

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "skipped": self.skipped,
            "notes": self.notes,
            "seeded_at": utcnow().isoformat(),
        }


def _demo_dir() -> Path:
    return get_settings().demo_dir


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# users
# --------------------------------------------------------------------------- #
def seed_users(session: Any, report: SeedReport) -> None:
    for spec in DEMO_USERS:
        if get_user_by_email(session, spec["email"]) is not None:
            report.add("user", created=False)
            continue
        user = register_user(
            session,
            email=spec["email"],
            password=spec["password"],
            display_name=spec["display_name"],
            role=spec["roles"][0],
            actor_id=SEED_ACTOR,
        )
        for role_name in spec["roles"][1:]:
            assign_role(session, user.id, role_name, granted_by=SEED_ACTOR)
        record_audit(
            session,
            actor_type="system",
            actor_id=SEED_ACTOR,
            action="seed.user_created",
            subject_type="user",
            subject_id=user.id,
            decision="created",
            detail={"email": user.email, "roles": spec["roles"]},
        )
        report.add("user")
    report.notes.append(
        "demo credentials are printed by `make demo`; change them before any real deployment"
    )


# --------------------------------------------------------------------------- #
# suites + corpora
# --------------------------------------------------------------------------- #
def seed_suites(session: Any, report: SeedReport) -> None:
    directory = _demo_dir() / "data" / "tests"
    if not directory.exists():
        report.notes.append(f"no suite directory at {directory}")
        return
    for path in sorted(directory.glob("*.suite.yaml")):
        definition = parse_suite_file(path)
        existing = session.scalar(select(Benchmark).where(Benchmark.key == definition.key))
        if existing is not None:
            report.add("suite", created=False)
            continue
        benchmark, suite = load_suite(session, definition)
        record_audit(
            session,
            actor_type="system",
            actor_id=SEED_ACTOR,
            action="seed.suite_loaded",
            subject_type="test_suite",
            subject_id=suite.id,
            decision="created",
            detail={
                "benchmark": benchmark.key,
                "cases": len(definition.cases),
                "holdout": definition.is_holdout,
            },
        )
        report.add("suite")
        report.notes.append(
            f"suite {benchmark.key}: {len(definition.cases)} cases "
            f"(holdout={definition.is_holdout})"
        )


def seed_corpora(session: Any, report: SeedReport) -> None:
    directory = _demo_dir() / "data" / "corpora"
    if not directory.exists():
        report.notes.append(f"no corpora directory at {directory}")
        return
    documents, skipped = load_permitted_corpus(directory)
    record_audit(
        session,
        actor_type="system",
        actor_id=SEED_ACTOR,
        action="seed.corpus_loaded",
        subject_type="corpus",
        subject_id=str(directory),
        decision="created",
        detail={
            "documents": [doc.ref for doc in documents],
            "licenses": [doc.license for doc in documents],
            "skipped": skipped,
        },
    )
    report.add("corpus_document", count=len(documents))
    if skipped:
        report.notes.append(f"corpus documents skipped (no licence/permission): {skipped}")


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def seed_sources(session: Any, report: SeedReport) -> list[str]:
    """Create the two day-1 sources and ingest demo data into them."""
    source_ids: list[str] = []
    demo = _demo_dir() / "data"

    csv_path = demo / "signals.csv"
    try:
        source = create_source(
            session,
            name="Demo upload — demand questions (CSV)",
            source_type="manual_csv",
            permission_basis=str(PermissionBasis.OPERATOR_PROVIDED),
            permission_notes=(
                "Operator-authored synthetic dataset. No third-party content, no scraping; "
                "safe to keep in the repository."
            ),
            config={"description": "Synthetic demand questions across eleven niches"},
            license="CC0-1.0",
            contains_personal_information=False,
            actor_id=SEED_ACTOR,
        )
        report.add("source")
        source_ids.append(source.id)
        if csv_path.exists() and session.scalar(select(func.count(NormalizedSignal.id))) == 0:
            result = ingest_upload(
                session,
                source_id=source.id,
                content=csv_path.read_bytes(),
                filename=csv_path.name,
                actor_id=SEED_ACTOR,
            )
            report.add("signal", count=result.rows_new)
            report.notes.append(
                f"CSV ingestion: {result.rows_new} new, {result.rows_duplicate} duplicate, "
                f"{result.rows_rejected} rejected, status={result.status}"
            )
    except Conflict:
        existing = session.scalar(select(DataSource).where(DataSource.name.like("Demo upload%")))
        if existing is not None:
            source_ids.append(existing.id)
        report.add("source", created=False)

    try:
        mock = create_source(
            session,
            name="Demo mock API — community questions",
            source_type="mock",
            permission_basis=str(PermissionBasis.SYNTHETIC),
            permission_notes=(
                "Fully mocked connector with the same interface as an official API connector. "
                "Fixture is synthetic; no live network calls and no scraping."
            ),
            config={"fixture_path": str(demo / "mock_api_fixture.json"), "limit": 12},
            license="synthetic",
            contains_personal_information=False,
            actor_id=SEED_ACTOR,
        )
        report.add("source")
        source_ids.append(mock.id)
        from db.models import RawSignal

        if (
            session.scalar(select(func.count(RawSignal.id)).where(RawSignal.source_id == mock.id))
            == 0
        ):
            run = run_source(session, source_id=mock.id, actor_id=SEED_ACTOR, limit=50)
            report.add("signal", count=run.rows_new)
            report.notes.append(
                f"mock API ingestion: {run.rows_new} new, {run.rows_duplicate} duplicate, "
                f"status={run.status}"
            )
    except Conflict:
        existing = session.scalar(select(DataSource).where(DataSource.name.like("Demo mock API%")))
        if existing is not None:
            source_ids.append(existing.id)
        report.add("source", created=False)

    return source_ids


# --------------------------------------------------------------------------- #
# opportunities
# --------------------------------------------------------------------------- #
def seed_discovery(session: Any, report: SeedReport) -> int:
    """Run one discovery pass if no opportunities exist yet."""
    existing = session.scalar(select(func.count(Opportunity.id))) or 0
    if existing:
        report.add("opportunity", created=False, count=existing)
        return existing
    from market_intel.pipeline import run_discovery

    result = run_discovery(session, actor_id=SEED_ACTOR, max_signals=400, min_cluster_size=3)
    created = len(result.opportunities)
    report.add("opportunity", count=created)
    report.notes.append(
        f"discovery: {result.signals_considered} signals → {result.clusters_found} clusters → "
        f"{created} opportunities"
    )
    return created


# --------------------------------------------------------------------------- #
# judge calibration
# --------------------------------------------------------------------------- #
def seed_calibration(session: Any, report: SeedReport) -> None:
    path = _demo_dir() / "data" / "calibration.json"
    payload = _load_json(path)
    if not payload:
        report.notes.append("no calibration set found; judge trust will stay at the 0.5 default")
        return
    if session.scalar(select(func.count(JudgeCalibration.id))):
        report.add("calibration", created=False)
        return
    items = [
        CalibrationItem(
            case_input=row.get("case_input", {}),
            candidate_output=row.get("candidate_output", ""),
            human_score=int(row.get("human_score", 3)),
            criterion=row.get("criterion", "composite"),
        )
        for row in payload.get("items", [])
    ]
    if not items:
        return
    benchmark = session.scalar(select(Benchmark).where(Benchmark.key == "billing-assistant-v1"))
    report_obj = calibrate_judge(
        session,
        items=items,
        judge_model=None,
        benchmark_id=benchmark.id if benchmark else None,
        actor_id=SEED_ACTOR,
    )
    report.add("calibration")
    report.notes.append(
        f"judge calibration: kappa={report_obj.kappa:.3f}, agreement={report_obj.agreement:.2f}, "
        f"usable={report_obj.usable}"
    )


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def seed_all(*, run_discovery_pass: bool = True) -> dict[str, Any]:
    report = SeedReport()
    init_db()
    with session_scope() as session:
        bootstrap(session)
        ensure_default_flags(session)
        seed_users(session, report)
        seed_suites(session, report)
        seed_corpora(session, report)
        seed_sources(session, report)
        if run_discovery_pass:
            seed_discovery(session, report)
        seed_calibration(session, report)
        record_audit(
            session,
            actor_type="system",
            actor_id=SEED_ACTOR,
            action="seed.completed",
            subject_type="seed",
            subject_id="demo",
            decision="created",
            detail=report.to_dict(),
        )
        counts = suite_status_counts(session)
        products = session.scalar(select(func.count(PromptProduct.id))) or 0
    report.notes.append(f"suites: {counts}; products: {products}")
    return report.to_dict()


def main() -> None:
    settings = get_settings()
    print(f"PromptFoundry seed → env={settings.env} db={settings.database_url}")
    if os.environ.get("PF_SEED_SKIP_DISCOVERY"):
        result = seed_all(run_discovery_pass=False)
    else:
        result = seed_all()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
