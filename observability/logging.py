"""Structured JSON logging with request/trace correlation.

Deliberately dependency-free so every entry point (API, worker, CLI demo) shares
one log shape. ``observability.tracing`` propagates the current trace id through a
contextvar, which this formatter picks up automatically.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")
actor_var: ContextVar[str] = ContextVar("actor", default="-")

_SECRET_KEYS = {
    "password",
    "api_key",
    "authorization",
    "token",
    "secret",
    "openai_api_key",
    "anthropic_api_key",
    "google_api_key",
    "signing_key",
    "access_token",
}


def _redact(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {
            key: ("***" if key.lower() in _SECRET_KEYS else _redact(val, depth + 1))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value[:20]]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
            "actor": actor_var.get(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload["fields"] = _redact(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    for noisy in ("uvicorn.access", "httpx", "httpcore", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})
