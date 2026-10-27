"""Celery application.

Design notes
------------
* **Eager by default.** With no broker configured, tasks run inline so ``make demo`` and
  the test suite need nothing but Python. Set ``CELERY_BROKER_URL`` (Redis) for a real
  worker.
* **Idempotent tasks.** Every task takes ids, never payloads, and re-reads state from the
  database, so retries are safe.
* **Budget-aware.** Each task that can spend money checks the budget breaker first and
  fails closed.
* **No hidden loop.** Beat schedules ingestion/discovery/evaluation; publishing is never
  scheduled — it always needs a human approval.
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.schedules import crontab

from core.config import get_settings
from observability.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("workers.celery")

app = Celery("promptfoundry", broker=settings.celery_broker_url, backend=settings.celery_broker_url)

app.conf.update(
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Australia/Melbourne",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    broker_connection_retry_on_startup=True,
    task_default_queue="promptfoundry",
    beat_schedule={
        # Hourly: pull any enabled API-backed source. Upload-only installs will no-op.
        "ingest-enabled-sources": {
            "task": "workers.tasks.ingest_all_sources",
            "schedule": crontab(minute=7),
        },
        # Every six hours: re-cluster and re-score demand.
        "discover-opportunities": {
            "task": "workers.tasks.discover_opportunities",
            "schedule": crontab(minute=23, hour="*/6"),
            "kwargs": {"max_signals": 800},
        },
        # Nightly: housekeeping, budget snapshot, stale-approval nudges.
        "nightly-maintenance": {
            "task": "workers.tasks.nightly_maintenance",
            "schedule": crontab(minute=12, hour=3),
        },
    },
)


def task_health() -> dict[str, Any]:
    return {
        "app": app.main,
        "eager": bool(app.conf.task_always_eager),
        "broker_configured": bool(settings.celery_broker_url),
        "queues": [app.conf.task_default_queue],
        "scheduled": sorted(app.conf.beat_schedule),
    }
