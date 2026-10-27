"""Products, versions, variants, evaluation suites and evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from backend.deps import RequestContext, require
from backend.schemas import CalibrationRequest, EvaluateRequest, ProductUpdate, SuiteLoadRequest
from core.config import REPO_ROOT, get_settings
from core.errors import NotFound, ValidationError
from core.rbac import Permission
from db.models import (
    EvalMetric,
    EvaluationRun,
    PromptProduct,
    PromptVariant,
    PromptVersion,
    TestCase,
    TestSuite,
)
from db.repository import record_audit
from evaluation.calibration import CalibrationItem, calibrate_judge
from evaluation.runner import RunConfig, run_evaluation
from evaluation.suites import load_suite, load_suite_directory, parse_suite_file, suite_summary
from prompt_ir.compiler import CompiledPrompt

router = APIRouter(tags=["catalog"])


def _product_payload(session: Any, product: PromptProduct) -> dict[str, Any]:
    versions = list(
        session.scalars(
            select(PromptVersion)
            .where(PromptVersion.product_id == product.id)
            .order_by(PromptVersion.created_at.desc())
        )
    )
    return {
        "id": product.id,
        "name": product.name,
        "slug": product.slug,
        "niche": product.niche,
        "buyer": product.buyer,
        "positioning": product.positioning,
        "description": product.description,
        "status": product.status,
        "currency": product.currency,
        "price_aud": round(product.price_aud_micros / 1_000_000, 2),
        "license": product.license,
        "tags": product.tags,
        "risk_tier": product.risk_tier,
        "regulated_domain": product.regulated_domain,
        "current_version_id": product.current_version_id,
        "opportunity_id": product.opportunity_id,
        "versions": [
            {
                "id": version.id,
                "semver": version.version,
                "status": version.status,
                "ir_hash": version.ir_hash,
                "created_at": version.created_at.isoformat(),
                "released_at": version.released_at.isoformat() if version.released_at else None,
                "signature": version.signature,
                "metrics": version.metrics_summary,
            }
            for version in versions
        ],
    }


@router.get("/products")
def list_products(
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    rows = list(
        context.session.scalars(
            select(PromptProduct).order_by(PromptProduct.created_at.desc()).limit(limit)
        )
    )
    items = [
        {
            "id": row.id,
            "name": row.name,
            "slug": row.slug,
            "status": row.status,
            "niche": row.niche,
            "buyer": row.buyer,
            "price_aud": round(row.price_aud_micros / 1_000_000, 2),
            "current_version_id": row.current_version_id,
            "risk_tier": row.risk_tier,
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.get("/products/{product_id}")
def get_product(
    product_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    product = context.session.get(PromptProduct, product_id)
    if product is None:
        raise NotFound(f"product {product_id} not found")
    return _product_payload(context.session, product)


@router.patch("/products/{product_id}")
def update_product(
    product_id: str,
    payload: ProductUpdate,
    context: Annotated[RequestContext, Depends(require(Permission.MANAGE_PRODUCTS))],
) -> dict[str, Any]:
    product = context.session.get(PromptProduct, product_id)
    if product is None:
        raise NotFound(f"product {product_id} not found")
    changes: dict[str, Any] = {}
    if payload.name:
        product.name = payload.name
        changes["name"] = payload.name
    if payload.positioning is not None:
        product.positioning = payload.positioning
        changes["positioning"] = True
    if payload.description is not None:
        product.description = payload.description
        changes["description"] = True
    if payload.price_aud is not None:
        product.price_aud_micros = int(round(payload.price_aud * 1_000_000))
        changes["price_aud"] = payload.price_aud
    if payload.tags is not None:
        product.tags = payload.tags
        changes["tags"] = payload.tags
    if payload.license:
        product.license = payload.license
        changes["license"] = payload.license
    context.audit(
        "product.updated",
        subject_type="product",
        subject_id=product.id,
        decision="updated",
        detail=changes,
    )
    return _product_payload(context.session, product)


@router.get("/versions/{version_id}")
def get_version(
    version_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_CATALOG))],
) -> dict[str, Any]:
    version = context.session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    variants = list(
        context.session.scalars(select(PromptVariant).where(PromptVariant.version_id == version.id))
    )
    runs = list(
        context.session.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.version_id == version.id)
            .order_by(EvaluationRun.created_at.desc())
            .limit(10)
        )
    )
    return {
        "id": version.id,
        "product_id": version.product_id,
        "semver": version.version,
        "status": version.status,
        "changelog": version.changelog,
        "ir": version.ir,
        "ir_hash": version.ir_hash,
        "model_matrix": version.model_matrix,
        "metrics": version.metrics_summary,
        "signature": version.signature,
        "signed_at": version.signed_at.isoformat() if version.signed_at else None,
        "artifact_uri": version.artifact_uri,
        "approved_by": version.approved_by,
        "variants": [
            {
                "id": variant.id,
                "family": variant.model_family,
                "compiled_hash": variant.compiled_hash,
                "token_estimate": variant.token_estimate,
                "params": variant.params,
                "is_primary": variant.is_primary,
                "compiled_text": variant.compiled_text,
            }
            for variant in variants
        ],
        "evaluations": [
            {
                "run_id": run.id,
                "stage": run.stage,
                "status": run.status,
                "score": run.score,
                "report_uri": run.report_uri,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
            for run in runs
        ],
    }


@router.post("/versions/{version_id}/evaluate")
def evaluate_version(
    version_id: str,
    payload: EvaluateRequest,
    context: Annotated[RequestContext, Depends(require(Permission.RUN_EVALUATIONS))],
) -> dict[str, Any]:
    version = context.session.get(PromptVersion, version_id)
    if version is None:
        raise NotFound(f"version {version_id} not found")
    variant = context.session.scalar(
        select(PromptVariant)
        .where(PromptVariant.version_id == version.id)
        .order_by(PromptVariant.is_primary.desc())
        .limit(1)
    )
    if variant is None:
        raise ValidationError("version has no compiled variants")
    if payload.suite_id:
        suite = context.session.get(TestSuite, payload.suite_id)
    else:
        suite = context.session.scalar(
            select(TestSuite)
            .where(TestSuite.is_holdout.is_(False))
            .order_by(TestSuite.created_at.desc())
            .limit(1)
        )
    if suite is None:
        raise NotFound("no test suite available; load suites first")
    compiled = CompiledPrompt(
        family=variant.model_family,
        text=variant.compiled_text,
        params=variant.params or {},
        token_estimate=variant.token_estimate,
        compiled_hash=variant.compiled_hash,
        ir_hash=variant.ir_hash,
    )
    outcome = run_evaluation(
        context.session,
        suite=suite,
        compiled=compiled,
        config=RunConfig(
            stage=payload.stage,
            model=payload.model,
            judge=payload.judge,
            votes=payload.votes,
            robustness=payload.robustness,
            max_cases=payload.max_cases,
            budget_cap_micros=int(payload.budget_cap_aud * 1_000_000),
        ),
        version_id=version.id,
        variant_id=variant.id,
        actor_id=context.actor_id,
    )
    return outcome.to_dict()


@router.get("/evaluations")
def list_evaluations(
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    rows = list(
        context.session.scalars(
            select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit)
        )
    )
    items = [
        {
            "run_id": row.id,
            "version_id": row.version_id,
            "suite_id": row.suite_id,
            "stage": row.stage,
            "status": row.status,
            "score": row.score,
            "n_cases": row.n_cases,
            "n_passed": row.n_passed,
            "cost_aud": round(row.cost_micros / 1_000_000, 6),
            "model": row.model_under_test,
            "report_uri": row.report_uri,
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        }
        for row in rows
    ]
    return {"items": items, "count": len(items)}


@router.get("/evaluations/{run_id}")
def get_evaluation(
    run_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
) -> dict[str, Any]:
    run = context.session.get(EvaluationRun, run_id)
    if run is None:
        raise NotFound(f"evaluation {run_id} not found")
    metrics = list(context.session.scalars(select(EvalMetric).where(EvalMetric.run_id == run.id)))
    return {
        "run_id": run.id,
        "version_id": run.version_id,
        "suite_id": run.suite_id,
        "stage": run.stage,
        "status": run.status,
        "score": run.score,
        "report_uri": run.report_uri,
        "model": run.model_under_test,
        "summary": run.summary,
        "metrics": [
            {
                "metric": row.metric,
                "value": row.value,
                "ci": [row.ci_low, row.ci_high],
                "n": row.n,
            }
            for row in metrics
        ],
    }


@router.get("/suites")
def list_suites(
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
) -> dict[str, Any]:
    rows = list(context.session.scalars(select(TestSuite)))
    items = [suite_summary(context.session, row) for row in rows]
    return {"items": items, "count": len(items)}


@router.post("/suites/load")
def load_suites(
    payload: SuiteLoadRequest,
    context: Annotated[RequestContext, Depends(require(Permission.RUN_EVALUATIONS))],
) -> dict[str, Any]:
    settings = get_settings()
    loaded: list[dict[str, Any]] = []
    if payload.path:
        definition = parse_suite_file(Path(payload.path).expanduser())
        _, suite = load_suite(context.session, definition)
        loaded.append({**definition.to_dict(), "suite_id": suite.id})
    directory = (
        Path(payload.directory).expanduser()
        if payload.directory
        else REPO_ROOT / "demo" / "data" / "tests"
    )
    if not payload.path:
        loaded.extend(load_suite_directory(context.session, directory))
    if not loaded:
        raise NotFound(f"no suite files found in {directory}")
    record_audit(
        context.session,
        action="suite.loaded",
        actor_id=context.actor_id,
        subject_type="suite",
        decision="loaded",
        detail={"count": len(loaded), "max_upload_rows": settings.max_upload_rows},
    )
    return {"loaded": loaded, "count": len(loaded)}


@router.get("/suites/{suite_id}/cases")
def list_cases(
    suite_id: str,
    context: Annotated[RequestContext, Depends(require(Permission.READ_EVALS))],
) -> dict[str, Any]:
    suite = context.session.get(TestSuite, suite_id)
    if suite is None:
        raise NotFound(f"suite {suite_id} not found")
    rows = list(context.session.scalars(select(TestCase).where(TestCase.suite_id == suite.id)))
    return {
        "suite": suite_summary(context.session, suite),
        "items": [
            {
                "case_key": row.case_key,
                "input": row.input,
                "assertions": row.assertions,
                "tags": row.tags,
                "is_adversarial": row.is_adversarial,
                "has_gold": row.gold_output is not None,
            }
            for row in rows
        ],
        "count": len(rows),
    }


@router.post("/evaluations/calibrate")
def calibrate(
    payload: CalibrationRequest,
    context: Annotated[RequestContext, Depends(require(Permission.RUN_EVALUATIONS))],
) -> dict[str, Any]:
    items = [
        CalibrationItem(
            case_input=dict(item.get("case_input") or item.get("input") or {}),
            candidate_output=str(item.get("candidate_output") or ""),
            human_score=int(item.get("human_score", 3)),
            criterion=str(item.get("criterion", "composite")),
        )
        for item in payload.items
    ]
    report = calibrate_judge(
        context.session,
        items=items,
        judge_model=payload.judge_model,
        benchmark_id=payload.benchmark_id,
        actor_id=context.actor_id,
    )
    return report.to_dict()
