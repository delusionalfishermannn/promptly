"""Judge calibration: measure a judge before trusting it.

Process
-------
1. The operator (or the demo) provides a small calibration set of
   ``(case_input, candidate_output, human_score)`` triples for a niche.
2. We run the judge on the same items and compute **quadratic-weighted Cohen's kappa**
   plus tolerance-1 agreement and mean bias.
3. The result is persisted on ``judge_calibrations`` and applied automatically by
   ``RubricJudge`` (shrinkage towards neutral by trust = f(kappa)).

Interpretation thresholds (documented in DESIGN.md):
* kappa ≥ 0.6 → usable as a secondary signal
* 0.4 ≤ kappa < 0.6 → use with caution; weight halved in the composite
* kappa < 0.4 → treat judging as advisory only; deterministic metrics decide
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from core.artifacts import get_store
from core.clock import utcnow
from db.models import EvaluationRun, JudgeCalibration
from db.repository import record_artifact, record_audit
from evaluation.judges import DEFAULT_RUBRIC, RubricJudge
from evaluation.stats import agreement, cohens_kappa_quadratic


@dataclass
class CalibrationItem:
    case_input: dict[str, Any]
    candidate_output: str
    human_score: int  # 1..5
    criterion: str = "composite"


@dataclass
class CalibrationReport:
    judge_model: str
    kappa: float
    agreement: float
    bias: float
    n_items: int
    usable: bool
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "judge_model": self.judge_model,
            "kappa": self.kappa,
            "agreement_within_1": self.agreement,
            "bias": self.bias,
            "n_items": self.n_items,
            "usable": self.usable,
            "recommendation": self.recommendation,
        }


def recommend(kappa: float) -> tuple[bool, str]:
    if kappa >= 0.6:
        return True, "judge usable as a secondary signal (kappa >= 0.6)"
    if kappa >= 0.4:
        return True, "use with caution; judge weight halved (0.4 <= kappa < 0.6)"
    return False, "judge not reliable (kappa < 0.4); rely on deterministic metrics"


def calibrate_judge(
    session: Session,
    *,
    items: list[CalibrationItem],
    judge_model: str | None = None,
    candidate_model: str | None = None,
    benchmark_id: str | None = None,
    votes: int = 1,
    persist: bool = True,
    actor_id: str = "system",
) -> CalibrationReport:
    if not items:
        raise ValueError("calibration requires at least one item")
    judge = RubricJudge(
        session,
        judge_model=judge_model,
        candidate_model=candidate_model,
        votes=votes,
        rubric=DEFAULT_RUBRIC,
        actor_id=actor_id,
        stage="calibration",
        benchmark_id=benchmark_id,
    )
    # Votes carry a FK to ``evaluation_runs``; calibration needs a real (staged) run row.
    run = EvaluationRun(
        stage="calibration",
        status="running",
        n_cases=len(items),
        model_under_test=judge_model or "auto",
        created_by=actor_id,
        started_at=utcnow(),
        summary={"kind": "judge_calibration", "benchmark_id": benchmark_id},
    )
    if benchmark_id:
        run.suite_id = None
    session.add(run)
    session.flush()

    judge_scores: list[int] = []
    human_scores: list[int] = []
    raw_scores: list[float] = []
    for index, item in enumerate(items):
        result = judge.grade(
            run_id=run.id,
            case_id=f"cal-{index}",
            case_input=item.case_input,
            candidate_output=item.candidate_output,
        )
        scaled = int(round(result.raw_mean * 5))
        judge_scores.append(max(1, min(5, scaled)))
        raw_scores.append(result.raw_mean * 5)
        human_scores.append(max(1, min(5, item.human_score)))

    kappa = cohens_kappa_quadratic(judge_scores, human_scores, max_rating=5)
    agree = agreement(judge_scores, human_scores, tolerance=1)
    bias = round(
        sum(raw - human for raw, human in zip(raw_scores, human_scores, strict=True)) / len(items),
        4,
    )
    usable, note = recommend(kappa)
    run.status = "succeeded"
    run.score = round(agree, 4)
    run.finished_at = utcnow()
    run.summary = {
        "kind": "judge_calibration",
        "benchmark_id": benchmark_id,
        "kappa": kappa,
        "agreement": agree,
        "bias": bias,
        "usable": usable,
        "judge_model": judge.judge_model,
        "n_items": len(items),
    }
    session.flush()
    report = CalibrationReport(
        judge_model=judge.judge_model,
        kappa=kappa,
        agreement=agree,
        bias=bias,
        n_items=len(items),
        usable=usable,
        recommendation=note,
    )

    if persist:
        payload = {
            **report.to_dict(),
            "items": [
                {
                    "case_input": item.case_input,
                    "human_score": item.human_score,
                    "judge_score": judge_scores[index],
                    "candidate_output": item.candidate_output[:1200],
                }
                for index, item in enumerate(items)
            ],
            "caveats": judge.caveats,
        }
        ref = get_store().put_json(
            payload, kind="judge_calibration", subject_id=benchmark_id or judge.judge_model
        )
        record_artifact(
            session,
            ref,
            kind="judge_calibration",
            subject_type="judge",
            subject_id=judge.judge_model,
        )
        # Deactivate previous calibrations for this judge/benchmark.
        for row in session.query(JudgeCalibration).filter(
            JudgeCalibration.judge_model == judge.judge_model,
            JudgeCalibration.benchmark_id == benchmark_id,
        ):
            row.is_active = False
        session.add(
            JudgeCalibration(
                benchmark_id=benchmark_id,
                judge_model=judge.judge_model,
                kappa=kappa,
                agreement=agree,
                n_items=len(items),
                bias_correction={"bias": bias, "usable": usable},
                calib_set=[
                    {
                        "human_score": item.human_score,
                        "judge_score": judge_scores[index],
                        "input": item.case_input,
                    }
                    for index, item in enumerate(items)
                ],
                notes=note,
                artifact_uri=ref.uri,
                is_active=True,
            )
        )
        record_audit(
            session,
            action="judge.calibrated",
            actor_id=actor_id,
            subject_type="judge",
            subject_id=judge.judge_model,
            decision="pass" if usable else "warn",
            detail=report.to_dict(),
        )
        session.flush()
    return report
