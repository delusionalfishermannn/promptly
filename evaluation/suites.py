"""Benchmark/test-suite definitions: load from files, upsert into the registry.

Suite files are JSON or YAML (``*.suite.yaml`` / ``*.suite.json``) so domain experts can
author tests without touching Python. Holdout suites are flagged and are *never* used by
the optimiser for selection — only for the final gate — which is the main defence
against overfitting to the visible test set.

Example (see ``demo/data/tests/billing_assistant.suite.yaml``)::

    key: billing-assistant-v1
    name: Billing assistant evaluation
    niche: billing
    is_holdout: false
    min_score: 0.7
    weights: {deterministic: 0.6, judge: 0.25, robustness: 0.15}
    judge_rubric: {criteria: {...}}       # optional: overrides DEFAULT_RUBRIC
    cases:
      - key: std-1
        input: {request: ..., source_material: ...}
        assertions: [{type: json_parses}, {type: contains, value: "1042"}]
        tags: [core]
        is_adversarial: false
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.constants import RunStatus
from db.models import Benchmark, TestCase, TestSuite

DEFAULT_WEIGHTS = {"deterministic": 0.6, "judge": 0.25, "robustness": 0.15}


@dataclass
class SuiteCaseDefinition:
    key: str
    input: dict[str, Any]
    assertions: list[dict[str, Any]] = field(default_factory=list)
    expected_schema: dict[str, Any] | None = None
    gold_output: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    difficulty: str = "normal"
    is_adversarial: bool = False
    name: str = ""
    source: str = "synthetic"


@dataclass
class SuiteDefinition:
    key: str
    name: str
    niche: str
    version: str = "1.0.0"
    description: str = ""
    is_holdout: bool = False
    min_score: float = 0.7
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    judge_rubric: dict[str, Any] = field(default_factory=dict)
    cases: list[SuiteCaseDefinition] = field(default_factory=list)
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "niche": self.niche,
            "version": self.version,
            "description": self.description,
            "is_holdout": self.is_holdout,
            "min_score": self.min_score,
            "weights": self.weights,
            "judge_rubric": self.judge_rubric,
            "n_cases": len(self.cases),
            "cases": [
                {
                    "key": case.key,
                    "tags": case.tags,
                    "is_adversarial": case.is_adversarial,
                    "assertions": len(case.assertions),
                }
                for case in self.cases
            ],
            "source_path": self.source_path,
        }


def parse_suite_dict(payload: dict[str, Any], *, source_path: str = "") -> SuiteDefinition:
    cases = []
    for index, raw in enumerate(payload.get("cases", [])):
        case = SuiteCaseDefinition(
            key=str(raw.get("key") or f"case-{index}"),
            input=raw.get("input") or {},
            assertions=raw.get("assertions") or [],
            expected_schema=raw.get("expected_schema"),
            gold_output=raw.get("gold_output"),
            tags=raw.get("tags") or [],
            difficulty=raw.get("difficulty", "normal"),
            is_adversarial=bool(raw.get("is_adversarial", False)),
            name=raw.get("name", ""),
            source=raw.get("source", "synthetic"),
        )
        cases.append(case)
    if not cases:
        raise ValueError(f"suite '{payload.get('key')}' has no cases")
    return SuiteDefinition(
        key=str(payload["key"]),
        name=str(payload.get("name") or payload["key"]),
        niche=str(payload.get("niche") or ""),
        version=str(payload.get("version") or "1.0.0"),
        description=str(payload.get("description") or ""),
        is_holdout=bool(payload.get("is_holdout", False)),
        min_score=float(payload.get("min_score", 0.7)),
        weights={**DEFAULT_WEIGHTS, **(payload.get("weights") or {})},
        judge_rubric=payload.get("judge_rubric") or {},
        cases=cases,
        source_path=source_path,
    )


def parse_suite_file(path: Path) -> SuiteDefinition:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "YAML suite files need PyYAML (pip install pyyaml) — or use JSON"
            ) from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"suite file {path} must contain an object")
    return parse_suite_dict(payload, source_path=str(path))


def load_suite(session: Session, definition: SuiteDefinition) -> tuple[Benchmark, TestSuite]:
    """Idempotent upsert of a benchmark + suite + cases."""
    benchmark = session.scalar(select(Benchmark).where(Benchmark.key == definition.key))
    if benchmark is None:
        benchmark = Benchmark(
            key=definition.key,
            name=definition.name,
            description=definition.description or f"Benchmark for {definition.niche}",
            niche=definition.niche,
            version=definition.version,
            is_holdout=definition.is_holdout,
            config={"weights": definition.weights, "min_score": definition.min_score},
        )
        session.add(benchmark)
        session.flush()
    else:
        benchmark.is_holdout = definition.is_holdout
        benchmark.config = {
            "weights": definition.weights,
            "min_score": definition.min_score,
        }

    suite = session.scalar(
        select(TestSuite).where(
            TestSuite.benchmark_id == benchmark.id, TestSuite.version == definition.version
        )
    )
    if suite is None:
        suite = TestSuite(
            benchmark_id=benchmark.id,
            name=definition.name,
            niche=definition.niche,
            version=definition.version,
            description=definition.description,
            judge_rubric=definition.judge_rubric,
            weights=definition.weights,
            is_holdout=definition.is_holdout,
            min_score=definition.min_score,
        )
        session.add(suite)
        session.flush()
    else:
        suite.judge_rubric = definition.judge_rubric or suite.judge_rubric
        suite.weights = definition.weights
        suite.min_score = definition.min_score

    existing = {
        case.case_key: case
        for case in session.scalars(select(TestCase).where(TestCase.suite_id == suite.id))
    }
    for case in definition.cases:
        row = existing.get(case.key)
        payload = {
            "name": case.name or case.key,
            "input": case.input,
            "expected_schema": case.expected_schema,
            "gold_output": case.gold_output,
            "assertions": case.assertions,
            "tags": case.tags,
            "difficulty": case.difficulty,
            "is_adversarial": case.is_adversarial,
            "source": case.source,
        }
        if row is None:
            session.add(TestCase(suite_id=suite.id, case_key=case.key, **payload))
        else:
            for key, value in payload.items():
                setattr(row, key, value)
    session.flush()
    return benchmark, suite


def load_suite_directory(session: Session, directory: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    if not directory.exists():
        return loaded
    paths = sorted(
        list(directory.glob("*.suite.yaml"))
        + list(directory.glob("*.suite.yml"))
        + list(directory.glob("*.suite.json"))
    )
    for path in paths:
        definition = parse_suite_file(path)
        _, suite = load_suite(session, definition)
        loaded.append({**definition.to_dict(), "suite_id": suite.id})
    return loaded


def cases_for_suite(
    session: Session, suite_id: str, *, include_adversarial: bool = True
) -> list[TestCase]:
    stmt = select(TestCase).where(TestCase.suite_id == suite_id).order_by(TestCase.case_key)
    rows = list(session.scalars(stmt))
    if not include_adversarial:
        rows = [row for row in rows if not row.is_adversarial]
    return rows


def suite_summary(session: Session, suite: TestSuite) -> dict[str, Any]:
    cases = cases_for_suite(session, suite.id)
    return {
        "suite_id": suite.id,
        "benchmark_id": suite.benchmark_id,
        "name": suite.name,
        "niche": suite.niche,
        "version": suite.version,
        "is_holdout": suite.is_holdout,
        "min_score": suite.min_score,
        "weights": suite.weights,
        "n_cases": len(cases),
        "n_adversarial": len([case for case in cases if case.is_adversarial]),
        "tags": sorted({tag for case in cases for tag in (case.tags or [])}),
    }


def suite_status_counts(session: Session) -> dict[str, int]:
    suites = list(session.scalars(select(TestSuite)))
    return {
        "suites": len(suites),
        "holdout": len([suite for suite in suites if suite.is_holdout]),
        "benchmarks": len(list(session.scalars(select(Benchmark)))),
        "cases": len(list(session.scalars(select(TestCase)))),
    }


__all__ = [
    "DEFAULT_WEIGHTS",
    "RunStatus",
    "SuiteCaseDefinition",
    "SuiteDefinition",
    "cases_for_suite",
    "load_suite",
    "load_suite_directory",
    "parse_suite_dict",
    "parse_suite_file",
    "suite_status_counts",
    "suite_summary",
]
