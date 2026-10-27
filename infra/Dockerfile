# PromptFoundry API / worker image.
# One image, two entrypoints (uvicorn vs celery) so behaviour cannot drift.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# psycopg[binary] ships wheels; build-essential is only needed for optional extras.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY connectors ./connectors
COPY core ./core
COPY db ./db
COPY evaluation ./evaluation
COPY experimentation ./experimentation
COPY governance ./governance
COPY llm ./llm
COPY market_intel ./market_intel
COPY observability ./observability
COPY prompt_ir ./prompt_ir
COPY prompt_lab ./prompt_lab
COPY publishing ./publishing
COPY workers ./workers
COPY alembic ./alembic
COPY alembic.ini ./
COPY demo ./demo

RUN pip install --upgrade pip && pip install -e ".[postgres]"

RUN useradd --create-home --uid 10001 foundry \
    && mkdir -p /app/var/artifacts \
    && chown -R foundry:foundry /app/var
USER foundry

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
