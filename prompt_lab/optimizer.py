"""Staged, anti-Goodhart optimisation loop.

    Stage 1  screen   : cheap model, few cases, no judge, no robustness → kill obviously
                        broken candidates for cents.
    Stage 2  offline  : balanced model, full visible suite, calibrated judge (2 votes),
                        deterministic validators dominate.
    Stage 3  finalist : 3 judge votes, full robustness/injection suite, optional frontier
                        model, regression gate against the currently released version.
    Stage 4  online   : handled by ``experimentation`` (bandit rollout, revenue/refunds).

Anti-overfitting measures wired into this module
------------------------------------------------
* Holdout suites are excluded from *selection* here; only the gate in
  ``governance`` consumes them.
* Selection uses the **lower** bound of the metric interval when available, not the
  point estimate (prefer reliably-good over lucky).
* Mutation refinement is only accepted if it beats the incumbent on a *fresh* run.
* Regression gates compare per-case paired differences with a bootstrap CI.
* Every stage is budget-capped; exceeding the cap stops the stage cleanly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from core.config import AUD_MICROS
from core.constants import EvalStage
from core.errors import BudgetExceeded
from db.models import PromptVersion, TestSuite
from db.repository import record_audit
from evaluation.runner import RunConfig, RunOutcome, latest_run_for_version, run_evaluation
from llm.catalog import available_models, resolve_model
from observability.logging import get_logger
from prompt_ir.compiler import compile_all
from prompt_ir.schemas import PromptIR
from prompt_lab.generator import SynthesisBrief, generate_candidates
from prompt_lab.mutation import evolve

log = get_logger("prompt_lab.optimizer")


@dataclass
class OptimizationConfig:
    n_candidates: int = 5
    screen_cases: int = 4
    offline_cases: int = 10
    finalist_cases: int | None = None
    screen_keep: int = 3
    finalist_keep: int = 2
    votes: int = 2
    finalist_votes: int = 3
    primary_family: str = "generic"
    families: tuple[str, ...] = ("openai", "anthropic", "google", "generic")
    seed: int = 11
    screen_budget_micros: int = 200_000  # A$0.20
    offline_budget_micros: int = 900_000  # A$0.90
    finalist_budget_micros: int = 1_500_000  # A$1.50
    mutate_winner: bool = True
    regression_gate: bool = True
    is_holdout_eval: bool = False


@dataclass
class CandidateOutcome:
    ir: PromptIR
    ir_hash: str
    slug: str
    screen: RunOutcome | None = None
    offline: RunOutcome | None = None
    finalist: RunOutcome | None = None
    eliminated_at: str = ""
    elimination_reason: str = ""
    selection_score: float = 0.0
    lower_bound: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "ir_hash": self.ir_hash,
            "title": self.ir.title,
            "variation": (self.ir.provenance.notes or "").split("variation=")[-1].split(";")[0],
            "selection_score": round(self.selection_score, 4),
            "lower_bound": round(self.lower_bound, 4),
            "screen_score": round(self.screen.overall_score, 4) if self.screen else None,
            "offline_score": round(self.offline.overall_score, 4) if self.offline else None,
            "finalist_score": round(self.finalist.overall_score, 4) if self.finalist else None,
            "finalist_run_id": self.finalist.run_id if self.finalist else None,
            "eliminated_at": self.eliminated_at,
            "elimination_reason": self.elimination_reason,
            "warnings": (self.finalist or self.offline or self.screen).warnings
            if (self.finalist or self.offline or self.screen)
            else [],
        }


@dataclass
class OptimizationResult:
    candidates: list[CandidateOutcome] = field(default_factory=list)
    winner: CandidateOutcome | None = None
    winner_compiled: dict[str, Any] = field(default_factory=dict)
    cost_micros: int = 0
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)
    stage_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "winner": self.winner.to_dict() if self.winner else None,
            "cost_aud": round(self.cost_micros / AUD_MICROS, 6),
            "duration_ms": round(self.duration_ms, 2),
            "warnings": self.warnings,
            "lineage": self.lineage,
            "stage_summary": self.stage_summary,
        }


def _selection_key(outcome: RunOutcome) -> tuple[float, float]:
    """Conservative ordering: interval lower bound first, then point estimate."""
    metric = outcome.metrics.get("deterministic_score") or {}
    low = float(metric.get("low", outcome.overall_score))
    return (low * 0.6 + outcome.overall_score * 0.4, outcome.overall_score)


def _run_stage(
    session: Session,
    *,
    suite: TestSuite,
    candidates: list[CandidateOutcome],
    stage: str,
    compiled_lookup: Any,
    config: RunConfig,
    executor: str,
) -> int:
    """Run one stage for all candidates. Returns total cost in micros."""
    total_cost = 0
    for candidate in candidates:
        compiled = compiled_lookup(candidate)
        try:
            outcome = run_evaluation(
                session,
                suite=suite,
                compiled=compiled,
                config=config,
                version_id=None,
                actor_id="optimizer",
            )
        except BudgetExceeded as exc:
            candidate.eliminated_at = stage
            candidate.elimination_reason = f"budget_exceeded: {exc.message}"
            continue
        setattr(candidate, stage, outcome)
        total_cost += outcome.cost_micros
        lower = float((outcome.metrics.get("deterministic_score") or {}).get("low", 0.0))
        score, _ = _selection_key(outcome)
        candidate.selection_score = round(score, 4)
        candidate.lower_bound = round(lower, 4)
    record_audit(
        session,
        action=f"optimizer.stage.{stage}",
        actor_id="optimizer",
        subject_type="optimization",
        decision="completed",
        detail={
            "executor": executor,
            "candidates": [candidate.slug for candidate in candidates],
            "cost_micros": total_cost,
            "model": config.model,
        },
    )
    return total_cost


def optimize(
    session: Session,
    *,
    brief: SynthesisBrief,
    suite: TestSuite,
    config: OptimizationConfig | None = None,
    baseline_version_id: str | None = None,
    holdout_suite: TestSuite | None = None,
) -> OptimizationResult:
    config = config or OptimizationConfig()
    started = time.perf_counter()
    result = OptimizationResult()

    if suite.is_holdout:
        raise ValueError("refusing to optimise against a holdout suite — holdouts are gate-only")

    irs = generate_candidates(brief, n=config.n_candidates, seed=config.seed)
    compiled_map = {
        ir.content_hash(): {
            family: compiled for family, compiled in compile_all(ir, list(config.families)).items()
        }
        for ir in irs
    }
    candidates = [CandidateOutcome(ir=ir, ir_hash=ir.content_hash(), slug=ir.slug) for ir in irs]

    def compiled_for(candidate: CandidateOutcome) -> Any:
        family = config.primary_family
        mapping = compiled_map[candidate.ir_hash]
        return mapping.get(family) or next(iter(mapping.values()))

    # ------------------------------------------------------------- stage 1
    screen_model = resolve_model("cheap")
    screen_config = RunConfig(
        stage=str(EvalStage.SCREEN),
        model=screen_model,
        judge=False,
        robustness=False,
        max_cases=config.screen_cases,
        budget_cap_micros=config.screen_budget_micros,
        include_adversarial=True,
    )
    result.cost_micros += _run_stage(
        session,
        suite=suite,
        candidates=candidates,
        stage="screen",
        compiled_lookup=compiled_for,
        config=screen_config,
        executor="cheap",
    )
    evaluated = [candidate for candidate in candidates if candidate.screen is not None]
    evaluated.sort(key=lambda candidate: candidate.selection_score, reverse=True)
    shortlist = evaluated[: config.screen_keep]
    for candidate in candidates:
        if candidate not in shortlist and not candidate.eliminated_at:
            candidate.eliminated_at = "screen"
            candidate.elimination_reason = "below top-k on deterministic screen"

    # ------------------------------------------------------------- stage 2
    offline_model = resolve_model("balanced")
    offline_config = RunConfig(
        stage=str(EvalStage.OFFLINE),
        model=offline_model,
        judge=True,
        votes=config.votes,
        robustness=False,
        max_cases=config.offline_cases,
        budget_cap_micros=config.offline_budget_micros,
        include_adversarial=True,
    )
    result.cost_micros += _run_stage(
        session,
        suite=suite,
        candidates=shortlist,
        stage="offline",
        compiled_lookup=compiled_for,
        config=offline_config,
        executor="balanced+judge",
    )
    scored = [candidate for candidate in shortlist if candidate.offline is not None]
    scored.sort(key=lambda candidate: candidate.selection_score, reverse=True)
    finalists = scored[: config.finalist_keep]
    for candidate in shortlist:
        if candidate not in finalists and not candidate.eliminated_at:
            candidate.eliminated_at = "offline"
            candidate.elimination_reason = "below top-k after offline evaluation"

    # ------------------------------------------------------------- stage 3
    frontier = [model.name for model in available_models("balanced") if model.judge_capable]
    finalist_model = frontier[0] if frontier else offline_model
    baseline_run = (
        latest_run_for_version(session, baseline_version_id) if baseline_version_id else None
    )
    finalist_config = RunConfig(
        stage=str(EvalStage.FINALIST),
        model=finalist_model,
        judge=True,
        votes=config.finalist_votes,
        robustness=True,
        max_cases=config.finalist_cases,
        budget_cap_micros=config.finalist_budget_micros,
        include_adversarial=True,
        is_regression_gate=config.regression_gate and baseline_run is not None,
        baseline_run_id=baseline_run.id if baseline_run else None,
    )
    result.cost_micros += _run_stage(
        session,
        suite=suite,
        candidates=finalists,
        stage="finalist",
        compiled_lookup=compiled_for,
        config=finalist_config,
        executor="balanced+judge+robustness",
    )
    ranked = [candidate for candidate in finalists if candidate.finalist is not None]
    ranked.sort(key=lambda candidate: candidate.selection_score, reverse=True)

    # -------------------------------------------------- mutation refinement
    if config.mutate_winner and ranked:
        incumbent = ranked[0]
        mutant_irs = evolve(incumbent.ir, generations=1, seed=config.seed + 5)
        if mutant_irs:
            mutant = CandidateOutcome(
                ir=mutant_irs[0], ir_hash=mutant_irs[0].content_hash(), slug=mutant_irs[0].slug
            )
            compiled_map[mutant.ir_hash] = compile_all(mutant.ir, list(config.families))
            result.cost_micros += _run_stage(
                session,
                suite=suite,
                candidates=[mutant],
                stage="offline",
                compiled_lookup=lambda candidate: (
                    compiled_map[candidate.ir_hash].get(config.primary_family)
                    or next(iter(compiled_map[candidate.ir_hash].values()))
                ),
                config=RunConfig(
                    stage=str(EvalStage.OFFLINE),
                    model=offline_model,
                    judge=True,
                    votes=config.votes,
                    robustness=False,
                    max_cases=config.offline_cases,
                    budget_cap_micros=config.offline_budget_micros,
                ),
                executor="mutation-refinement",
            )
            if mutant.offline and incumbent.offline:
                mutant.selection_score = round(_selection_key(mutant.offline)[0], 4)
                incumbent.selection_score = round(_selection_key(incumbent.offline)[0], 4)
                if mutant.selection_score > incumbent.selection_score + 0.01:
                    result.warnings.append(
                        f"mutation '{mutant.slug}' beat the incumbent on offline score "
                        f"({mutant.selection_score:.3f} vs {incumbent.selection_score:.3f}); "
                        "promoted to winner (finalist robustness not re-run)"
                    )
                    result.lineage.append(
                        {
                            "from": incumbent.slug,
                            "to": mutant.slug,
                            "operator_notes": mutant.ir.provenance.notes,
                            "reason": "offline score improvement",
                        }
                    )
                    ranked.insert(0, mutant)
                    result.candidates.append(mutant)
                else:
                    mutant.eliminated_at = "mutation-refinement"
                    mutant.elimination_reason = "did not beat incumbent"
                    result.candidates.append(mutant)

    result.candidates.extend(candidates)
    result.winner = ranked[0] if ranked else (shortlist[0] if shortlist else None)
    if result.winner is not None:
        result.winner_compiled = {
            family: compiled.to_dict()
            for family, compiled in compiled_map[result.winner.ir_hash].items()
        }
        if result.winner.finalist is None:
            result.warnings.append(
                "winner has no finalist-stage evidence (budget cap or earlier failure) — "
                "treat status as provisional"
            )
    else:
        result.warnings.append("no candidate survived screening; check the suite and model config")

    # --------------------------------------------------- holdout gate preview
    if holdout_suite is not None and result.winner is not None and config.is_holdout_eval:
        holdout_config = RunConfig(
            stage="holdout",
            model=finalist_model,
            judge=True,
            votes=config.votes,
            robustness=True,
            budget_cap_micros=config.finalist_budget_micros,
        )
        try:
            holdout_outcome = run_evaluation(
                session,
                suite=holdout_suite,
                compiled=compiled_for(result.winner),
                config=holdout_config,
                actor_id="optimizer",
            )
            result.cost_micros += holdout_outcome.cost_micros
            result.stage_summary["holdout"] = {
                "run_id": holdout_outcome.run_id,
                "score": holdout_outcome.overall_score,
                "passed": holdout_outcome.passed,
            }
            if not holdout_outcome.passed:
                result.warnings.append("holdout evaluation failed — do not promote this version")
        except BudgetExceeded as exc:
            result.warnings.append(f"holdout evaluation skipped: {exc.message}")

    result.stage_summary.update(
        {
            "generated": len(candidates),
            "screened": len(evaluated),
            "offline": len(scored),
            "finalists": len(ranked),
            "screen_model": screen_model,
            "offline_model": offline_model,
            "finalist_model": finalist_model,
            "baseline_run_id": baseline_run.id if baseline_run else None,
        }
    )
    result.duration_ms = (time.perf_counter() - started) * 1000
    record_audit(
        session,
        action="optimizer.completed",
        actor_id="optimizer",
        subject_type="optimization",
        decision="winner_selected" if result.winner else "no_winner",
        detail={
            "winner": result.winner.slug if result.winner else None,
            "cost_aud": round(result.cost_micros / AUD_MICROS, 6),
            "warnings": result.warnings,
        },
    )
    return result


def record_winner(
    session: Session,
    *,
    result: OptimizationResult,
    product_id: str,
    version: str,
    changelog: str,
    actor_id: str,
) -> PromptVersion:
    """Persist the winning IR + all compiled variants as a new prompt version (draft)."""
    from db.models import PromptVariant

    if result.winner is None:
        raise ValueError("no winner to record")
    ir = result.winner.ir
    version_row = PromptVersion(
        product_id=product_id,
        version=version,
        status="candidate",
        changelog=changelog,
        ir=ir.model_dump(mode="json"),
        ir_hash=ir.content_hash(),
        model_matrix={
            family: {
                "compiled_hash": compiled["compiled_hash"],
                "token_estimate": compiled["token_estimate"],
                "params": compiled["params"],
            }
            for family, compiled in result.winner_compiled.items()
        },
        metrics_summary={
            "screen": result.winner.screen.overall_score if result.winner.screen else None,
            "offline": result.winner.offline.overall_score if result.winner.offline else None,
            "finalist": result.winner.finalist.overall_score if result.winner.finalist else None,
            "selection_score": result.winner.selection_score,
        },
        created_by=actor_id,
    )
    session.add(version_row)
    session.flush()
    for family, compiled in result.winner_compiled.items():
        session.add(
            PromptVariant(
                version_id=version_row.id,
                model_family=family,
                model_name=compiled["params"].get("family", family),
                compiled_text=compiled["text"],
                compiled_hash=compiled["compiled_hash"],
                params=compiled["params"],
                ir_hash=compiled["ir_hash"],
                token_estimate=compiled["token_estimate"],
                is_primary=family == "generic",
            )
        )
    session.flush()
    return version_row
