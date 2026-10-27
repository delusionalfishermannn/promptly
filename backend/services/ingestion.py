"""Ingestion service: sources → connector runs → raw signals → normalized signals.

Guarantees
----------
* **Provenance first.** A source cannot be created without a permitted
  ``permission_basis``, and the connector must accept that basis.
* **Idempotent.** ``(source_id, payload_hash)`` dedupes raw rows; global text hashes
  dedupe normalised rows so the same question from two exports is not double-counted
  (which would inflate demand).
* **Bounded.** Uploads are row- and size-capped; API runs are paged and rate-limited.
* **Auditable.** Every run writes a ``connector_runs`` row, an artifact of the raw
  batch, and an audit entry with counts and rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from connectors.base import SignalRecord, check_permission_basis
from connectors.registry import get_connector, list_connectors
from core.artifacts import get_store
from core.clock import utcnow
from core.config import get_settings
from core.constants import (
    ALLOWED_PERMISSION_BASES,
    PermissionBasis,
    RunStatus,
    SignalKind,
)
from core.errors import Conflict, NotFound, ValidationError
from core.ids import content_hash
from db.models import ConnectorRun, DataSource, NormalizedSignal, RawSignal
from db.repository import raise_alert, record_artifact, record_audit
from llm.embeddings import get_embedder
from market_intel.normalize import normalize_text
from observability.metrics import INGESTED_SIGNALS, SIGNALS_REJECTED


@dataclass
class IngestResult:
    source_id: str
    run_id: str
    status: str
    rows_in: int = 0
    rows_new: int = 0
    rows_duplicate: int = 0
    rows_rejected: int = 0
    normalized: int = 0
    embedded: int = 0
    rejects: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifact_uri: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "run_id": self.run_id,
            "status": self.status,
            "rows_in": self.rows_in,
            "rows_new": self.rows_new,
            "rows_duplicate": self.rows_duplicate,
            "rows_rejected": self.rows_rejected,
            "normalized": self.normalized,
            "embedded": self.embedded,
            "rejects": self.rejects[:25],
            "warnings": self.warnings,
            "artifact_uri": self.artifact_uri,
        }


def create_source(
    session: Session,
    *,
    name: str,
    source_type: str,
    permission_basis: str,
    permission_notes: str = "",
    config: dict[str, Any] | None = None,
    license: str = "",
    contains_personal_information: bool = False,
    retention_days: int = 365,
    schedule_cron: str | None = None,
    enabled: bool = True,
    actor_id: str = "system",
) -> DataSource:
    settings = get_settings()
    if not name.strip():
        raise ValidationError("source name is required")
    if permission_basis not in ALLOWED_PERMISSION_BASES:
        raise ValidationError(f"permission_basis must be one of {sorted(ALLOWED_PERMISSION_BASES)}")
    if session.scalar(select(DataSource).where(DataSource.name == name)) is not None:
        raise Conflict(f"source '{name}' already exists")
    try:
        connector = get_connector(source_type, config or {})
    except KeyError as exc:
        raise ValidationError(str(exc)) from exc
    try:
        check_permission_basis(connector, permission_basis)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if contains_personal_information and permission_basis not in {
        str(PermissionBasis.OPERATOR_PROVIDED),
        str(PermissionBasis.FIRST_PARTY_ANALYTICS),
    }:
        raise ValidationError(
            "sources containing personal information must be operator-provided or first-party"
        )
    source = DataSource(
        name=name.strip(),
        source_type=source_type,
        connector_kind=connector.connector_kind,
        permission_basis=permission_basis,
        permission_notes=permission_notes,
        license=license,
        jurisdiction=settings.jurisdiction,
        contains_personal_information=contains_personal_information,
        retention_days=retention_days,
        config=config or {},
        schedule_cron=schedule_cron,
        enabled=enabled,
        created_by=actor_id,
    )
    session.add(source)
    session.flush()
    record_audit(
        session,
        action="source.created",
        actor_id=actor_id,
        subject_type="data_source",
        subject_id=source.id,
        decision="created",
        detail={
            "name": source.name,
            "source_type": source.source_type,
            "permission_basis": source.permission_basis,
            "contains_personal_information": contains_personal_information,
        },
    )
    return source


def _persist_records(
    session: Session,
    *,
    source: DataSource,
    run: ConnectorRun,
    records: list[SignalRecord],
    rejects: list[dict[str, Any]],
    result: IngestResult,
    embed_on_ingest: bool = True,
) -> None:
    settings = get_settings()
    embedder = get_embedder() if embed_on_ingest else None
    texts_to_embed: list[NormalizedSignal] = []

    for record in records:
        payload = {
            "text": record.text,
            "kind": record.kind,
            "occurred_at": record.occurred_at.isoformat(),
            "volume": record.volume,
            "url": record.url,
            "metadata": record.metadata,
            "external_id": record.external_id,
        }
        payload_hash = content_hash(source.id, json_safe(payload))
        exists = session.scalar(
            select(RawSignal).where(
                RawSignal.source_id == source.id, RawSignal.payload_hash == payload_hash
            )
        )
        if exists is not None:
            result.rows_duplicate += 1
            continue
        raw = RawSignal(
            source_id=source.id,
            run_id=run.id,
            external_id=record.external_id,
            payload=payload,
            payload_hash=payload_hash,
            collected_at=record.occurred_at,
        )
        session.add(raw)
        session.flush()
        result.rows_new += 1

        draft = normalize_text(
            record.text,
            declared_kind=record.kind if record.kind != str(SignalKind.OTHER) else None,
            occurred_at=record.occurred_at,
            volume=record.volume,
            is_synthetic=(source.permission_basis == str(PermissionBasis.SYNTHETIC)),
            segment_hint=record.segment_hint,
        )
        duplicate_text = session.scalar(
            select(NormalizedSignal).where(NormalizedSignal.text_hash == draft.text_hash)
        )
        if duplicate_text is not None:
            result.rows_duplicate += 1
            continue
        if draft.lang != "en":
            result.warnings.append(f"non-English signal skipped ({draft.lang})")
            SIGNALS_REJECTED.inc(reason="language")
            continue
        signal = NormalizedSignal(
            raw_signal_id=raw.id,
            source_id=source.id,
            text=draft.text,
            text_hash=draft.text_hash,
            lang=draft.lang,
            kind=draft.kind,
            features=draft.features,
            intent_strength=draft.intent_strength,
            buyer_segment=draft.buyer_segment,
            urgency=draft.urgency,
            complexity=draft.complexity,
            sentiment=draft.sentiment,
            pain=draft.pain,
            volume=draft.volume,
            velocity=draft.velocity,
            recency_decay=draft.recency_decay,
            quality=draft.quality,
            weight=draft.weight,
            is_synthetic=draft.is_synthetic,
            collected_at=draft.collected_at,
        )
        session.add(signal)
        session.flush()
        result.normalized += 1
        if embedder is not None:
            texts_to_embed.append(signal)
        INGESTED_SIGNALS.inc(source_type=source.source_type)

    if embedder is not None and texts_to_embed:
        vectors = embedder.embed([signal.text for signal in texts_to_embed])
        for signal, vector in zip(texts_to_embed, vectors, strict=True):
            signal.embedding = vector
            signal.embedding_model = embedder.name
        result.embedded = len(texts_to_embed)

    for reject in rejects:
        result.rows_rejected += 1
        SIGNALS_REJECTED.inc(reason=str(reject.get("reason", "unknown"))[:40])
    result.rejects = rejects
    if embedder is not None and len(texts_to_embed) > 200 and settings.llm_configured:
        result.warnings.append(
            f"embedded {len(texts_to_embed)} signals with a paid provider — check /ops/costs"
        )


def json_safe(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, default=str)


def run_source(
    session: Session,
    *,
    source_id: str,
    actor_id: str = "system",
    limit: int = 500,
    since: datetime | None = None,
    embed_on_ingest: bool = True,
) -> IngestResult:
    source = session.get(DataSource, source_id)
    if source is None:
        raise NotFound(f"source {source_id} not found")
    if not source.enabled:
        raise Conflict(f"source '{source.name}' is disabled")
    connector = get_connector(source.source_type, source.config or {})
    check_permission_basis(connector, source.permission_basis)
    if not connector.is_ready():
        readiness = connector.readiness()
        result = IngestResult(
            source_id=source.id,
            run_id="",
            status=str(RunStatus.FAILED),
            warnings=[
                f"connector not ready: missing {readiness.get('missing_secret_env')} "
                f"{readiness.get('config_issues')}"
            ],
        )
        raise_alert(
            session,
            kind="ingestion.not_ready",
            severity="info",
            message=f"Source '{source.name}' is missing credentials or config",
            subject_type="data_source",
            subject_id=source.id,
            detail=readiness,
        )
        return result

    run = ConnectorRun(
        source_id=source.id,
        status=str(RunStatus.RUNNING),
        started_at=utcnow(),
        triggered_by=actor_id,
        cursor=source.config.get("cursor", {}) if isinstance(source.config, dict) else {},
    )
    session.add(run)
    session.flush()
    result = IngestResult(source_id=source.id, run_id=run.id, status=str(RunStatus.RUNNING))

    try:
        connector_result = connector.run(since=since, limit=limit, cursor=run.cursor)
    except Exception as exc:
        run.status = str(RunStatus.FAILED)
        run.finished_at = utcnow()
        run.error = f"{type(exc).__name__}: {exc}"
        session.flush()
        record_audit(
            session,
            action="ingestion.run",
            actor_id=actor_id,
            subject_type="data_source",
            subject_id=source.id,
            decision="failed",
            reason=run.error[:500],
            detail={"run_id": run.id},
        )
        raise_alert(
            session,
            kind="ingestion.failed",
            severity="warning",
            message=f"Connector run failed for '{source.name}': {exc}",
            subject_type="data_source",
            subject_id=source.id,
        )
        result.status = str(RunStatus.FAILED)
        result.warnings.append(run.error)
        return result

    if connector_result.error:
        run.status = str(RunStatus.FAILED)
        run.finished_at = utcnow()
        run.error = connector_result.error
        session.flush()
        result.status = str(RunStatus.FAILED)
        result.warnings.append(connector_result.error)
        return result

    result.rows_in = len(connector_result.records) + len(connector_result.rejects)
    _persist_records(
        session,
        source=source,
        run=run,
        records=connector_result.records,
        rejects=connector_result.rejects,
        result=result,
        embed_on_ingest=embed_on_ingest,
    )

    artifact_payload = {
        "source": {"id": source.id, "name": source.name, "type": source.source_type},
        "permission_basis": source.permission_basis,
        "run_id": run.id,
        "connector_stats": connector_result.stats,
        "rejects": connector_result.rejects[:100],
        "records_sample": [
            {"text": record.text[:400], "kind": record.kind, "external_id": record.external_id}
            for record in connector_result.records[:50]
        ],
        "counts": {
            "rows_in": result.rows_in,
            "rows_new": result.rows_new,
            "normalized": result.normalized,
            "rejected": result.rows_rejected,
        },
        "generated_at": utcnow().isoformat(),
    }
    ref = get_store().put_json(artifact_payload, kind="raw_signal_batch", subject_id=run.id)
    record_artifact(
        session,
        ref,
        kind="raw_signal_batch",
        subject_type="connector_run",
        subject_id=run.id,
        created_by=actor_id,
    )
    result.artifact_uri = ref.uri

    run.status = str(RunStatus.PARTIAL if connector_result.partial else RunStatus.SUCCEEDED)
    run.finished_at = utcnow()
    run.rows_in = result.rows_in
    run.rows_out = result.normalized
    run.rows_rejected = result.rows_rejected
    run.stats = {**connector_result.stats, "duplicates": result.rows_duplicate}
    run.cursor = connector_result.cursor
    source.last_run_at = utcnow()
    if isinstance(source.config, dict):
        source.config = {**source.config, "cursor": connector_result.cursor}
    result.status = run.status
    session.flush()
    record_audit(
        session,
        action="ingestion.run",
        actor_id=actor_id,
        subject_type="data_source",
        subject_id=source.id,
        decision=run.status,
        detail={
            "run_id": run.id,
            "rows_in": result.rows_in,
            "rows_new": result.rows_new,
            "duplicates": result.rows_duplicate,
            "normalized": result.normalized,
            "rejected": result.rows_rejected,
            "artifact_uri": ref.uri,
        },
    )
    return result


def ingest_upload(
    session: Session,
    *,
    source_id: str,
    content: bytes,
    actor_id: str = "system",
    filename: str = "upload.csv",
) -> IngestResult:
    """Parse an uploaded CSV/JSON export and persist it through the same pipeline."""
    source = session.get(DataSource, source_id)
    if source is None:
        raise NotFound(f"source {source_id} not found")
    if source.source_type not in {"manual_csv", "manual_json", "storefront", "support_tickets"}:
        raise ValidationError(f"source type '{source.source_type}' does not accept uploads")
    settings = get_settings()
    if len(content) > 25 * 1024 * 1024:
        raise ValidationError("upload exceeds 25 MB")
    connector = get_connector(source.source_type, source.config or {})
    if hasattr(connector, "parse_bytes"):
        parse_result = connector.parse_bytes(content)  # type: ignore[attr-defined]
    else:
        raise ValidationError(f"connector '{source.source_type}' cannot parse uploads")
    if parse_result.error:
        raise ValidationError(parse_result.error)

    run = ConnectorRun(
        source_id=source.id,
        status=str(RunStatus.RUNNING),
        started_at=utcnow(),
        triggered_by=actor_id,
        stats={"filename": filename, "bytes": len(content)},
    )
    session.add(run)
    session.flush()
    result = IngestResult(source_id=source.id, run_id=run.id, status=str(RunStatus.RUNNING))
    result.rows_in = len(parse_result.records) + len(parse_result.rejects)
    _persist_records(
        session,
        source=source,
        run=run,
        records=parse_result.records,
        rejects=parse_result.rejects,
        result=result,
    )
    raw_batch = {
        "filename": filename,
        "bytes": len(content),
        "sha256": content_hash(filename, str(len(content))),
        "rows_in": result.rows_in,
        "rows_new": result.rows_new,
        "normalized": result.normalized,
        "rejected": result.rows_rejected,
        "rejects": result.rejects[:100],
        "permission_basis": source.permission_basis,
        "uploaded_by": actor_id,
        "uploaded_at": utcnow().isoformat(),
    }
    ref = get_store().put_json(raw_batch, kind="raw_signal_batch", subject_id=run.id)
    record_artifact(
        session,
        ref,
        kind="raw_signal_batch",
        subject_type="connector_run",
        subject_id=run.id,
        created_by=actor_id,
        signature=None,
    )
    result.artifact_uri = ref.uri
    run.status = str(RunStatus.SUCCEEDED)
    run.finished_at = utcnow()
    run.rows_in = result.rows_in
    run.rows_out = result.normalized
    run.rows_rejected = result.rows_rejected
    source.last_run_at = utcnow()
    result.status = run.status
    session.flush()
    record_audit(
        session,
        action="ingestion.upload",
        actor_id=actor_id,
        subject_type="data_source",
        subject_id=source.id,
        decision="succeeded",
        detail={
            "run_id": run.id,
            "filename": filename,
            "rows_new": result.rows_new,
            "normalized": result.normalized,
            "rejected": result.rows_rejected,
            "max_rows": settings.max_upload_rows,
        },
    )
    return result


def connector_catalog() -> list[dict[str, Any]]:
    return list_connectors()


def source_summary(session: Session, source: DataSource) -> dict[str, Any]:
    runs = session.scalars(
        select(ConnectorRun)
        .where(ConnectorRun.source_id == source.id)
        .order_by(ConnectorRun.created_at.desc())
        .limit(5)
    )
    connector = get_connector(source.source_type, source.config or {})
    from db.models import NormalizedSignal, RawSignal

    counts = {
        "raw": session.scalar(
            select(func.count(RawSignal.id)).where(RawSignal.source_id == source.id)
        )
        or 0,
        "signals": session.scalar(
            select(func.count(NormalizedSignal.id)).where(NormalizedSignal.source_id == source.id)
        )
        or 0,
    }
    return {
        "id": source.id,
        "name": source.name,
        "source_type": source.source_type,
        "connector_kind": source.connector_kind,
        "permission_basis": source.permission_basis,
        "permission_notes": source.permission_notes,
        "license": source.license,
        "contains_personal_information": source.contains_personal_information,
        "retention_days": source.retention_days,
        "enabled": source.enabled,
        "schedule_cron": source.schedule_cron,
        "last_run_at": source.last_run_at.isoformat() if source.last_run_at else None,
        "readiness": connector.readiness(),
        "stats": counts,
        "recent_runs": [
            {
                "run_id": run.id,
                "status": run.status,
                "rows_out": run.rows_out,
                "rows_rejected": run.rows_rejected,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
            for run in runs
        ],
    }
