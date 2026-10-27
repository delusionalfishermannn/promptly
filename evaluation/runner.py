"""Evaluation runner: staged, cost-aware, auditable, regression-gated.

One call = one ``evaluation_runs`` row + ``eval_metrics`` + ``judge_votes`` +
``injection_tests`` + a signed JSON report artifact.

Design decisions that matter
----------------------------
* **Deterministic first.** ``deterministic_score`` (schema + assertions) is the primary
  metric; judging is a weighted secondary signal whose influence shrinks with measured
  unreliability (kappa) and vote disagreement.
* **Holdout discipline.** Suites flagged ``is_holdout`` are excluded from candidate
  selection in the optimiser; they are used by the gate only.
* **Cost ceiling per run.** The runner stops early if the run exceeds
  ``budget_cap_micros`` and reports ``partial`` instead of silently overspending.
* **Regression gate.** With a baseline run, a paired bootstrap on per-case scores
  produces a CI for the difference; a drop beyond ``max_allowed_drop`` fails the gate.
* **No silent passes.** Any critical injection failure caps the overall score and marks
  the run ``blocked`` so packaging/approval cannot proceed.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from core.artifacts import get_store
from core.budget import shadow_cost_micros
from core.clock import utcnow
from core.config import AUD_MICROS, get_settings
from core.constants import EvalStage, RunStatus
from db.models import EvalMetric, EvaluationRun, JudgeVote, TestCase, TestSuite
from db.repository import raise_alert, record_artifact, record_audit
from evaluation.judges import DEFAULT_RUBRIC, JudgeResult, RubricJudge
from evaluation.robustness import RobustnessOutcome, injection_summary, run_robustness
from evaluation.stats import Interval, mean_interval, paired_difference, wilson_interval
from evaluation.validators import validate_output
from llm.catalog import resolve_model
from llm.client import LLMClient, LLMRequest
from observability.metrics import EVAL_RUNS
from observability.tracing import span
from prompt_ir.compiler import CompiledPrompt

DEFAULT_BUDGET_CAP_MICROS = 750_000  # A$0.75 per run — well inside the weekly cap


@dataclass
class RunConfig:
    stage: str = str(EvalStage.OFFLINE)
    model: str | None = None
    judge: bool = True
    judge_model: str | None = None
    votes: int = 2
    robustness: bool = True
    max_cases: int | None = None
    include_adversarial: bool = True
    budget_cap_micros: int = DEFAULT_BUDGET_CAP_MICROS
    is_regression_gate: bool = False
    baseline_run_id: str | None = None
    max_allowed_drop: float = 0.03
    persist: bool = True
    mock_mode: str = "generate"


@dataclass
class CaseResult:
    case_key: str
    passed: bool
    score: float
    schema_ok: bool
    failures: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    is_adversarial: bool = False
    judge_score: float | None = None
    judge_trust: float | None = None
    cost_micros: int = 0
    latency_ms: float = 0.0
    output_excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key,
            "passed": self.passed,
            "score": round(self.score, 4),
            "schema_ok": self.schema_ok,
            "failures": self.failures[:6],
            "tags": self.tags,
            "is_adversarial": self.is_adversarial,
            "judge_score": None if self.judge_score is None else round(self.judge_score, 4),
            "judge_trust": self.judge_trust,
            "cost_aud": round(self.cost_micros / AUD_MICROS, 6),
            "latency_ms": round(self.latency_ms, 2),
            "output_excerpt": self.output_excerpt[:600],
        }


@dataclass
class RunOutcome:
    run_id: str
    status: str
    overall_score: float
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    case_results: list[CaseResult] = field(default_factory=list)
    judge: JudgeResult | None = None
    robustness: list[RobustnessOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    report_uri: str | None = None
    cost_micros: int = 0
    regression: dict[str, Any] | None = None
    model_under_test: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "overall_score": round(self.overall_score, 4),
            "passed": self.passed,
            "metrics": self.metrics,
            "model_under_test": self.model_under_test,
            "judge": self.judge.to_dict() if self.judge else None,
            "robustness": injection_summary(self.robustness) if self.robustness else None,
            "warnings": self.warnings,
            "report_uri": self.report_uri,
            "cost_aud": round(self.cost_micros / AUD_MICROS, 6),
            "regression": self.regression,
            "cases": [case.to_dict() for case in self.case_results],
        }


def _weights_for(
    suite: TestSuite, *, judge_ran: bool, robustness_ran: bool, judge_trust: float
) -> dict[str, float]:
    configured = {
        **{"deterministic": 0.6, "judge": 0.25, "robustness": 0.15},
        **(suite.weights or {}),
    }
    weights = {
        "deterministic": configured.get("deterministic", 0.6),
        "judge": configured.get("judge", 0.25) if judge_ran else 0.0,
        "robustness": configured.get("robustness", 0.15) if robustness_ran else 0.0,
    }
    if judge_ran and judge_trust < 0.4:
        # Anti-Goodhart: a poorly calibrated judge cannot carry a decision.
        weights["judge"] *= 0.5
        weights["deterministic"] += configured.get("judge", 0.25) * 0.25
    total = sum(weights.values())
    return {key: round(value / total, 4) for key, value in weights.items()} if total else weights


def _case_schema(case: TestCase, compiled: CompiledPrompt) -> dict[str, Any] | None:
    if case.expected_schema:
        return case.expected_schema
    from llm.mock import _extract_schema

    return _extract_schema(compiled.text)


def run_evaluation(
    session: Session,
    *,
    suite: TestSuite,
    compiled: CompiledPrompt,
    config: RunConfig | None = None,
    version_id: str | None = None,
    variant_id: str | None = None,
    cases: list[TestCase] | None = None,
    actor_id: str = "system",
) -> RunOutcome:
    config = config or RunConfig()
    settings = get_settings()
    started = time.perf_counter()
    warnings: list[str] = []
    model = config.model or resolve_model(
        "cheap" if config.stage == str(EvalStage.SCREEN) else "balanced"
    )
    client = LLMClient(session, actor_id=actor_id, actor_label="evaluator")

    if cases is None:
        from evaluation.suites import cases_for_suite

        cases = cases_for_suite(session, suite.id, include_adversarial=config.include_adversarial)
    if config.max_cases:
        cases = cases[: config.max_cases]
    if not cases:
        raise ValueError("no test cases selected for evaluation")

    run = EvaluationRun(
        version_id=version_id,
        variant_id=variant_id,
        suite_id=suite.id,
        stage=config.stage,
        status=str(RunStatus.RUNNING),
        n_cases=len(cases),
        model_under_test=model,
        is_regression_gate=config.is_regression_gate,
        started_at=utcnow(),
        created_by=actor_id,
    )
    session.add(run)
    session.flush()

    outcome = RunOutcome(
        run_id=run.id,
        status=str(RunStatus.RUNNING),
        overall_score=0.0,
        passed=False,
        model_under_test=model,
    )

    deterministic_scores: list[float] = []
    schema_flags: list[float] = []
    assertion_flags: list[float] = []
    judge_scores: list[float] = []
    judge_trusts: list[float] = []
    latencies: list[float] = []
    tokens_in_total = 0
    tokens_out_total = 0
    judge_rationales: list[str] = []
    judge_caveats: list[str] = []
    judge_model_used = ""

    with span("evaluation.run", suite=suite.name, stage=config.stage, model=model, run_id=run.id):
        for case in cases:
            if outcome.cost_micros >= config.budget_cap_micros:
                warnings.append(
                    f"run budget cap reached ({config.budget_cap_micros / AUD_MICROS:.2f} AUD) — "
                    f"{len(outcome.case_results)}/{len(cases)} cases evaluated"
                )
                break
            truth_inputs = dict(case.input or {})
            response = client.complete(
                LLMRequest(
                    messages=[
                        {"role": "system", "content": compiled.text},
                        {"role": "user", "content": compiled.user_message(truth_inputs)},
                    ],
                    model=model,
                    temperature=float(compiled.params.get("temperature", 0.2)),
                    max_tokens=int(compiled.params.get("max_tokens", 1000)),
                    json_mode=bool(compiled.params.get("json_mode")),
                    purpose=f"eval:{config.stage}",
                    subject_type="evaluation_run",
                    subject_id=run.id,
                    mock_mode=config.mock_mode,
                )
            )
            outcome.cost_micros += response.cost_micros
            tokens_in_total += response.tokens_in
            tokens_out_total += response.tokens_out
            latencies.append(response.latency_ms)
            schema = _case_schema(case, compiled)
            report = validate_output(
                text=response.text,
                schema=schema,
                assertions=list(case.assertions or []),
                inputs=truth_inputs,
                require_schema=not case.is_adversarial,
            )
            deterministic_scores.append(report.score)
            # Schema validity is only meaningful for cases that ask for structured output.
            # Adversarial cases deliberately expect a refusal (prose), which the assertion
            # set — not the JSON schema — judges.
            if not case.is_adversarial:
                schema_flags.append(1.0 if report.schema_ok else 0.0)
            assertion_flags.append(1.0 if report.passed else 0.0)

            case_result = CaseResult(
                case_key=case.case_key,
                passed=report.passed,
                score=report.score,
                schema_ok=report.schema_ok,
                failures=report.failures,
                tags=list(case.tags or []),
                is_adversarial=case.is_adversarial,
                cost_micros=response.cost_micros,
                latency_ms=response.latency_ms,
                output_excerpt=response.text,
            )

            judge_allowed = (
                config.judge
                and not case.is_adversarial
                and "no_judge" not in (case.tags or [])
                and outcome.cost_micros < config.budget_cap_micros
            )
            if judge_allowed:
                judge = RubricJudge(
                    session,
                    judge_model=config.judge_model,
                    candidate_model=model,
                    votes=config.votes,
                    rubric=suite.judge_rubric or DEFAULT_RUBRIC,
                    actor_id=actor_id,
                    stage=config.stage,
                    benchmark_id=suite.benchmark_id,
                )
                judged = judge.grade(
                    run_id=run.id,
                    case_id=case.id,
                    case_input=truth_inputs,
                    candidate_output=response.text,
                )
                outcome.cost_micros += judged.cost_micros
                judge_model_used = judged.judge_model
                case_result.judge_score = judged.adjusted_mean
                case_result.judge_trust = judged.trust
                judge_scores.append(judged.adjusted_mean)
                judge_trusts.append(judged.trust)
                judge_rationales.extend(judged.rationales[:1])
                judge_caveats.extend(judged.caveats)

            outcome.case_results.append(case_result)

        robustness_outcomes: list[RobustnessOutcome] = []
        if config.robustness:
            base_case = next((case for case in cases if not case.is_adversarial), cases[0])
            if outcome.cost_micros < config.budget_cap_micros:
                robustness_outcomes = run_robustness(
                    session,
                    run_id=run.id,
                    compiled=compiled,
                    model=model,
                    base_inputs=dict(base_case.input or {}),
                    client=client,
                    persist=config.persist,
                    subject_type="version" if version_id else "evaluation_run",
                    subject_id=version_id or run.id,
                )
                outcome.cost_micros += sum(
                    0 for _ in robustness_outcomes
                )  # costs already recorded on cost_events

        # ------------------------------------------------------------- metrics
        det = mean_interval(deterministic_scores)
        schema_ci = wilson_interval(int(sum(schema_flags)), len(schema_flags))
        assertion_ci = wilson_interval(int(sum(assertion_flags)), len(assertion_flags))
        judge_final: JudgeResult | None = None
        if judge_scores:
            judge_ci = mean_interval(judge_scores)
            judge_final = JudgeResult(
                judge_model=judge_model_used or "unknown",
                votes=config.votes,
                criterion_scores={},
                raw_mean=round(judge_ci.estimate, 4),
                adjusted_mean=round(judge_ci.estimate, 4),
                variance=0.0,
                trust=round(statistics.fmean(judge_trusts), 3) if judge_trusts else 0.5,
                rationales=judge_rationales[:5],
                cost_micros=0,
                caveats=list(dict.fromkeys(judge_caveats))[:6],
            )
        rob_summary = injection_summary(robustness_outcomes) if robustness_outcomes else None
        rob_rate = float(rob_summary["pass_rate"]) if rob_summary else 0.0

        judge_ran = judge_final is not None
        rob_ran = bool(robustness_outcomes)
        weights = _weights_for(
            suite,
            judge_ran=judge_ran,
            robustness_ran=rob_ran,
            judge_trust=judge_final.trust if judge_final else 0.0,
        )
        overall = weights["deterministic"] * det.estimate
        if judge_ran and judge_final is not None:
            overall += weights["judge"] * judge_final.adjusted_mean
        if rob_ran:
            overall += weights["robustness"] * rob_rate
        overall = round(max(0.0, min(1.0, overall)), 4)

        critical_failures = rob_summary["critical_failures"] if rob_summary else []
        if critical_failures:
            overall = min(overall, 0.4)
            warnings.append(f"critical injection failures: {critical_failures}")
        hard_fail = bool(critical_failures) or (schema_ci.estimate < 0.5)
        if schema_ci.estimate < 0.5:
            warnings.append(f"schema validity rate {schema_ci.estimate:.0%} below 50%")

        # ------------------------------------------------------ regression gate
        regression: dict[str, Any] | None = None
        if config.is_regression_gate and config.baseline_run_id:
            baseline = session.get(EvaluationRun, config.baseline_run_id)
            if baseline is None:
                warnings.append("baseline run not found; regression gate skipped")
            else:
                baseline_scores = {
                    item["case_key"]: item["score"]
                    for item in (baseline.summary or {}).get("case_results", [])
                }
                paired_base = [
                    baseline_scores[case.case_key]
                    for case in outcome.case_results
                    if case.case_key in baseline_scores
                ]
                paired_new = [
                    case.score for case in outcome.case_results if case.case_key in baseline_scores
                ]
                if paired_base:
                    diff = paired_difference(paired_base, paired_new)
                    regressed = diff.low < -config.max_allowed_drop
                    regression = {
                        "baseline_run_id": baseline.id,
                        "baseline_score": baseline.score,
                        "paired_mean_diff": round(diff.estimate, 4),
                        "ci_low": round(diff.low, 4),
                        "ci_high": round(diff.high, 4),
                        "max_allowed_drop": config.max_allowed_drop,
                        "regressed": regressed,
                        "n_paired": diff.n,
                    }
                    if regressed:
                        warnings.append(
                            f"regression detected: paired diff {diff.estimate:+.3f} "
                            f"(CI {diff.low:+.3f}, {diff.high:+.3f})"
                        )
                else:
                    warnings.append("no overlapping cases with baseline; gate not evaluable")

        passed = (
            overall >= float(suite.min_score)
            and not hard_fail
            and not (regression and regression.get("regressed"))
        )
        status = (
            str(RunStatus.PARTIAL)
            if warnings and len(outcome.case_results) < len(cases)
            else str(RunStatus.SUCCEEDED)
        )
        if hard_fail:
            status = str(RunStatus.FAILED)

        metrics: dict[str, Any] = {
            "overall": overall,
            "overall_passed": passed,
            "deterministic_score": det.to_dict(),
            "schema_valid_rate": schema_ci.to_dict(),
            "assertion_pass_rate": assertion_ci.to_dict(),
            "judge_score": (mean_interval(judge_scores).to_dict() if judge_scores else None),
            "robustness": rob_summary,
            "weights": weights,
            "n_cases": len(outcome.case_results),
            "n_adversarial": len([case for case in outcome.case_results if case.is_adversarial]),
            "cost_aud": round(outcome.cost_micros / AUD_MICROS, 6),
            "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
            "latency_p95_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2)
            if latencies
            else 0.0,
            "model": model,
            "judge_model": judge_model_used or None,
            "min_score": suite.min_score,
            "warnings": warnings,
        }

        _persist_metrics(
            session, run.id, metrics, det, schema_ci, assertion_ci, judge_scores, rob_rate
        )

        report = {
            "run_id": run.id,
            "suite": {
                "id": suite.id,
                "name": suite.name,
                "version": suite.version,
                "niche": suite.niche,
                "is_holdout": suite.is_holdout,
                "min_score": suite.min_score,
            },
            "config": {
                "stage": config.stage,
                "model": model,
                "judge": config.judge,
                "votes": config.votes,
                "robustness": config.robustness,
                "max_cases": config.max_cases,
                "budget_cap_aud": config.budget_cap_micros / AUD_MICROS,
                "is_regression_gate": config.is_regression_gate,
                "baseline_run_id": config.baseline_run_id,
            },
            "metrics": metrics,
            "judge": judge_final.to_dict() if judge_final else None,
            "robustness": [item.to_dict() for item in robustness_outcomes],
            "regression": regression,
            "cases": [case.to_dict() for case in outcome.case_results],
            "generated_at": utcnow().isoformat(),
            "compiled_hash": compiled.compiled_hash,
            "ir_hash": compiled.ir_hash,
        }
        ref = get_store().put_json(report, kind="eval_report", subject_id=run.id)
        record_artifact(
            session,
            ref,
            kind="eval_report",
            subject_type="evaluation_run",
            subject_id=run.id,
            created_by=actor_id,
        )

        run.status = status
        run.finished_at = utcnow()
        run.n_passed = len([case for case in outcome.case_results if case.passed])
        run.score = overall
        run.cost_micros = outcome.cost_micros
        run.latency_ms = metrics["latency_p50_ms"]
        run.report_uri = ref.uri
        run.summary = {
            "passed": passed,
            "status": status,
            "min_score": suite.min_score,
            "overall_score": overall,
            "cost_aud": round(outcome.cost_micros / AUD_MICROS, 6),
            "shadow_cost_aud": round(
                shadow_cost_micros(tokens_in_total, tokens_out_total) / AUD_MICROS,
                6,
            ),
            "metrics": metrics,
            "case_results": [case.to_dict() for case in outcome.case_results],
            "judge": judge_final.to_dict() if judge_final else None,
            "robustness": rob_summary,
            "regression": regression,
            "warnings": warnings,
        }
        session.flush()

        outcome.status = status
        outcome.overall_score = overall
        outcome.passed = passed
        outcome.metrics = metrics
        outcome.judge = judge_final
        outcome.robustness = robustness_outcomes
        outcome.warnings = warnings
        outcome.report_uri = ref.uri
        outcome.regression = regression

        if critical_failures:
            raise_alert(
                session,
                kind="evaluation.injection_failure",
                severity="critical",
                message=f"Run {run.id} failed critical injection tests: {critical_failures}",
                subject_type="evaluation_run",
                subject_id=run.id,
            )
        if regression and regression.get("regressed"):
            raise_alert(
                session,
                kind="evaluation.regression",
                severity="warning",
                message=f"Run {run.id} regressed against baseline {config.baseline_run_id}",
                subject_type="evaluation_run",
                subject_id=run.id,
                detail=regression,
            )
        record_audit(
            session,
            action="evaluation.run",
            actor_id=actor_id,
            subject_type="version" if version_id else "evaluation_run",
            subject_id=version_id or run.id,
            decision="pass" if passed else "fail",
            detail={
                "run_id": run.id,
                "score": overall,
                "status": status,
                "stage": config.stage,
                "model": model,
                "warnings": warnings,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "aud_per_usd": settings.aud_per_usd,
            },
        )
    EVAL_RUNS.inc(stage=config.stage, outcome="pass" if outcome.passed else "fail")
    return outcome


def _persist_metrics(
    session: Session,
    run_id: str,
    metrics: dict[str, Any],
    det: Interval,
    schema_ci: Interval,
    assertion_ci: Interval,
    judge_scores: list[float],
    rob_rate: float,
) -> None:
    rows = [
        EvalMetric(metric="overall", value=metrics["overall"], n=metrics["n_cases"], detail={}),
        EvalMetric(
            metric="deterministic_score",
            value=det.estimate,
            ci_low=det.low,
            ci_high=det.high,
            n=det.n,
        ),
        EvalMetric(
            metric="schema_valid_rate",
            value=schema_ci.estimate,
            ci_low=schema_ci.low,
            ci_high=schema_ci.high,
            n=schema_ci.n,
        ),
        EvalMetric(
            metric="assertion_pass_rate",
            value=assertion_ci.estimate,
            ci_low=assertion_ci.low,
            ci_high=assertion_ci.high,
            n=assertion_ci.n,
        ),
        EvalMetric(metric="cost_aud", value=metrics["cost_aud"]),
        EvalMetric(metric="latency_p50_ms", value=metrics["latency_p50_ms"]),
    ]
    if judge_scores:
        judge_ci = mean_interval(judge_scores)
        rows.append(
            EvalMetric(
                metric="judge_score",
                value=judge_ci.estimate,
                ci_low=judge_ci.low,
                ci_high=judge_ci.high,
                n=judge_ci.n,
            )
        )
    if metrics.get("robustness"):
        rows.append(
            EvalMetric(
                metric="robustness_pass_rate",
                value=rob_rate,
                n=int(metrics["robustness"].get("total", 0)),
            )
        )
    for row in rows:
        row.run_id = run_id
        session.add(row)
    session.flush()


def latest_run_for_version(
    session: Session, version_id: str, *, stage: str | None = None
) -> EvaluationRun | None:
    from sqlalchemy import select

    stmt = select(EvaluationRun).where(EvaluationRun.version_id == version_id)
    if stage:
        stmt = stmt.where(EvaluationRun.stage == stage)
    stmt = stmt.where(EvaluationRun.status.in_([str(RunStatus.SUCCEEDED), str(RunStatus.PARTIAL)]))
    return session.scalar(stmt.order_by(EvaluationRun.finished_at.desc()).limit(1))


def compare_runs(session: Session, run_a_id: str, run_b_id: str) -> dict[str, Any]:
    run_a = session.get(EvaluationRun, run_a_id)
    run_b = session.get(EvaluationRun, run_b_id)
    if run_a is None or run_b is None:
        raise ValueError("both runs must exist")
    a_scores = {
        item["case_key"]: item["score"] for item in (run_a.summary or {}).get("case_results", [])
    }
    b_scores = {
        item["case_key"]: item["score"] for item in (run_b.summary or {}).get("case_results", [])
    }
    shared = sorted(set(a_scores) & set(b_scores))
    diff = (
        paired_difference([a_scores[key] for key in shared], [b_scores[key] for key in shared])
        if shared
        else None
    )
    votes_a = int(session.query(JudgeVote).filter(JudgeVote.run_id == run_a.id).count())
    votes_b = int(session.query(JudgeVote).filter(JudgeVote.run_id == run_b.id).count())
    return {
        "run_a": {
            "id": run_a.id,
            "score": run_a.score,
            "stage": run_a.stage,
            "judge_votes": votes_a,
        },
        "run_b": {
            "id": run_b.id,
            "score": run_b.score,
            "stage": run_b.stage,
            "judge_votes": votes_b,
        },
        "paired": diff.to_dict() if diff else None,
        "n_shared_cases": len(shared),
        "better": "b" if run_b.score > run_a.score else "a",
    }
