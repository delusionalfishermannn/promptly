"""PromptFoundry FastAPI application.

Composition root: middleware (request id, structured logs, metrics, security headers,
body-size guard, CORS), error mapping, and routers. Nothing domain-specific lives here
— see ``backend/routers``, ``backend/services`` and the domain packages.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text

from backend.routers import (
    auth,
    experiments,
    governance,
    ops,
    pipeline,
    products,
    publishing,
    sources,
)
from backend.services.auth import bootstrap
from core.config import get_settings
from core.errors import PromptFoundryError
from db.session import get_engine, init_db, session_scope
from observability.logging import configure_logging, get_logger, request_id_var
from observability.metrics import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    render_metrics,
)
from observability.tracing import init_tracing, recent_spans

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger("backend.main")

MAX_BODY_BYTES = 30 * 1024 * 1024  # uploads capped at 25 MB in the service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.ensure_dirs()
    init_tracing("promptfoundry-api")
    init_db()
    with session_scope() as session:
        info = bootstrap(session)
        log.info(
            "bootstrap complete",
            extra={
                "extra_fields": {
                    **info,
                    "env": settings.env,
                    "db": settings.database_url.split(":")[0],
                }
            },
        )
    yield
    log.info("shutdown", extra={"extra_fields": {"app": settings.app_name}})


def create_app() -> FastAPI:
    app = FastAPI(
        title="PromptFoundry",
        version=settings.version,
        description=(
            "Continuous, evidence-driven discovery and production of original prompt products.\n\n"
            "Every pipeline stage is auditable: demand signals with provenance → scored "
            "opportunities → Prompt-IR → staged evaluation with confidence intervals → "
            "governance gate → human approval → publishing with rollback."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["http://localhost:3000", "http://localhost:8000"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-Id"],
        expose_headers=["X-Request-Id", "X-Response-Time-Ms"],
    )

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {"code": "payload_too_large", "message": "request body too large"}
                },
            )
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            request_id_var.reset(token)
            raise
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path_template = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.inc(
            method=request.method, path=path_template, status=str(response.status_code)
        )
        HTTP_LATENCY.observe(elapsed, path=path_template)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.1f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        request_id_var.reset(token)
        return response

    # --------------------------------------------------------- error mapping
    @app.exception_handler(PromptFoundryError)
    async def typed_error_handler(request: Request, exc: PromptFoundryError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"extra_fields": {"path": request.url.path}})
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "internal error",
                    "details": {"request_id": getattr(request.state, "request_id", "-")},
                }
            },
        )

    # -------------------------------------------------------------- routers
    for router in (
        auth.router,
        auth.admin_router,
        sources.router,
        pipeline.router,
        products.router,
        governance.router,
        publishing.router,
        experiments.router,
        ops.router,
    ):
        app.include_router(router)

    # ----------------------------------------------------------- meta routes
    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, Any]:
        healthy = True
        detail: dict[str, Any] = {}
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            detail["database"] = "ok"
        except Exception as exc:
            healthy = False
            detail["database"] = f"error: {exc}"
        return {
            "status": "ok" if healthy else "degraded",
            "version": settings.version,
            "env": settings.env,
            "config": settings.any(),
            "checks": detail,
        }

    @app.get("/readyz", tags=["meta"])
    def readyz() -> JSONResponse:
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
            return JSONResponse({"status": "ready"})
        except Exception:
            return JSONResponse({"status": "not_ready"}, status_code=503)

    @app.get("/metrics", tags=["meta"], include_in_schema=False)
    def metrics() -> Response:
        return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/traces", tags=["meta"])
    def traces(limit: int = 100) -> dict[str, Any]:
        spans = recent_spans(limit=min(limit, 500))
        return {"items": spans, "count": len(spans)}

    @app.get("/", tags=["meta"])
    def root() -> dict[str, Any]:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "env": settings.env,
            "docs": "/docs",
            "health": "/healthz",
            "principles": [
                "no scraping — official APIs, uploads, first-party analytics and synthetic data only",
                "original prompts by construction — IR synthesis, similarity-gated",
                "evidence-first — deterministic evaluation with confidence intervals",
                "human approval by default, with audited rollback and kill switches",
            ],
        }

    return app


app = create_app()
