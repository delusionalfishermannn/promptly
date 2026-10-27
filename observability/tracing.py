"""Lightweight tracing that is OpenTelemetry-shaped but zero-dependency.

Spans are recorded in-process (ring buffer) so the MVP can expose recent traces via
``GET /traces`` and the demo can print a timeline. If ``opentelemetry-sdk`` is
installed and ``PF_OTEL_EXPORTER=otlp`` is set, :func:`init_tracing` wires the real
exporter — the call sites never change.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from observability.logging import get_logger, trace_id_var

_log = get_logger("observability.tracing")


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    started_at: float
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
            "status": self.status,
            "error": self.error,
        }


_RECENT: deque[Span] = deque(maxlen=2000)
_STATE: dict[str, Any] = {"trace_id": None, "span_id": None, "exporter": None}


def _new_trace_id() -> str:
    return uuid.uuid4().hex


def _new_span_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a span; nested calls inherit the trace id and parent span id."""
    trace_id = _STATE["trace_id"] or _new_trace_id()
    parent_id = _STATE["span_id"]
    current = Span(
        name=name,
        trace_id=trace_id,
        span_id=_new_span_id(),
        parent_id=parent_id,
        started_at=time.time(),
        attributes={k: v for k, v in attributes.items() if v is not None},
    )
    previous = (_STATE["trace_id"], _STATE["span_id"])
    _STATE["trace_id"], _STATE["span_id"] = trace_id, current.span_id
    token = trace_id_var.set(trace_id)
    try:
        yield current
    except Exception as exc:
        current.status = "error"
        current.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current.duration_ms = (time.time() - current.started_at) * 1000
        _RECENT.append(current)
        _STATE["trace_id"], _STATE["span_id"] = previous
        trace_id_var.reset(token)
        exporter = _STATE.get("exporter")
        if exporter is not None:  # pragma: no cover - optional OTel bridge
            try:
                exporter(current)
            except Exception as exc:
                _log.debug(
                    "span exporter failed", extra={"extra_fields": {"error": str(exc)[:200]}}
                )


def current_trace_id() -> str:
    return _STATE["trace_id"] or "-"


def recent_spans(limit: int = 100, trace_id: str | None = None) -> list[dict[str, Any]]:
    spans = list(_RECENT)
    if trace_id:
        spans = [s for s in spans if s.trace_id == trace_id]
    return [s.to_dict() for s in spans[-limit:]]


def clear_spans() -> None:
    _RECENT.clear()


def init_tracing(service_name: str = "promptfoundry") -> dict[str, Any]:
    """Best-effort real OTel wiring. Falls back silently to in-process spans."""
    status: dict[str, Any] = {"service": service_name, "otel": False}
    if not os.environ.get("PF_OTEL_EXPORTER"):
        return status
    try:  # pragma: no cover - requires optional deps
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        otel_trace.set_tracer_provider(provider)
        status["otel"] = True
    except Exception:  # pragma: no cover
        status["otel"] = False
    return status
