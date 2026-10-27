"""Prometheus-compatible metrics without a hard dependency.

The registry is in-process and thread-safe: counters/gauges/histograms are stored
with label sets and rendered in the standard text exposition format, so
``GET /metrics`` works with or without ``prometheus_client`` installed. Tests read
the snapshots directly (``snapshot()``) instead of scraping.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager

HIST_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def _label_key(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


class Counter:
    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._values: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._values[_label_key(labels)] += amount

    def get(self, **labels: str) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def items(self) -> list[tuple[str, float]]:
        return sorted(self._values.items())


class Gauge:
    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help = help_text
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        with self._lock:
            self._values[_label_key(labels)] = float(value)

    def get(self, **labels: str) -> float:
        return self._values.get(_label_key(labels), 0.0)

    def items(self) -> list[tuple[str, float]]:
        return sorted(self._values.items())


class Histogram:
    def __init__(
        self, name: str, help_text: str, buckets: tuple[float, ...] = HIST_BUCKETS
    ) -> None:
        self.name = name
        self.help = help_text
        self.buckets = buckets
        self._counts: dict[str, list[int]] = {}
        self._sums: dict[str, float] = defaultdict(float)
        self._totals: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = _label_key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self.buckets))
            for idx, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[idx] += 1
            self._sums[key] += float(value)
            self._totals[key] += 1

    def snapshot(self, **labels: str) -> dict[str, float]:
        key = _label_key(labels)
        return {
            "count": self._totals.get(key, 0),
            "sum": self._sums.get(key, 0.0),
            "buckets": dict(
                zip(
                    (str(b) for b in self.buckets),
                    self._counts.get(key, [0] * len(self.buckets)),
                    strict=True,
                )
            ),
        }

    def percentile(self, q: float, **labels: str) -> float:
        key = _label_key(labels)
        total = self._totals.get(key, 0)
        if not total:
            return 0.0
        counts = self._counts.get(key, [])
        target = q * total
        for idx, bound in enumerate(self.buckets):
            if counts[idx] >= target:
                return bound
        return self.buckets[-1] if self.buckets else 0.0


class Registry:
    def __init__(self) -> None:
        self.counters: dict[str, Counter] = {}
        self.gauges: dict[str, Gauge] = {}
        self.histograms: dict[str, Histogram] = {}

    def counter(self, name: str, help_text: str) -> Counter:
        self.counters.setdefault(name, Counter(name, help_text))
        return self.counters[name]

    def gauge(self, name: str, help_text: str) -> Gauge:
        self.gauges.setdefault(name, Gauge(name, help_text))
        return self.gauges[name]

    def histogram(self, name: str, help_text: str) -> Histogram:
        self.histograms.setdefault(name, Histogram(name, help_text))
        return self.histograms[name]

    def render(self) -> str:
        lines: list[str] = []
        for name, counter in sorted(self.counters.items()):
            lines.append(f"# HELP {name} {counter.help}")
            lines.append(f"# TYPE {name} counter")
            for labels, value in counter.items():
                lines.append(f"{name}{_format_labels(labels)} {value:g}")
        for name, gauge in sorted(self.gauges.items()):
            lines.append(f"# HELP {name} {gauge.help}")
            lines.append(f"# TYPE {name} gauge")
            for labels, value in gauge.items():
                lines.append(f"{name}{_format_labels(labels)} {value:g}")
        for name, hist in sorted(self.histograms.items()):
            lines.append(f"# HELP {name} {hist.help}")
            lines.append(f"# TYPE {name} histogram")
            for labels, counts in sorted(hist._counts.items()):
                for bound, count in zip(hist.buckets, counts, strict=True):
                    combined = _merge_labels(labels, f'le="{bound}"')
                    lines.append(f"{name}_bucket{combined} {count}")
                combined = _merge_labels(labels, 'le="+Inf"')
                lines.append(f"{name}_bucket{combined} {hist._totals.get(labels, 0)}")
                lines.append(f"{name}_sum{_wrap(labels)} {hist._sums.get(labels, 0):g}")
                lines.append(f"{name}_count{_wrap(labels)} {hist._totals.get(labels, 0)}")
        return "\n".join(lines) + "\n"


def _format_labels(key: str) -> str:
    return f"{{{key}}}" if key else ""


def _wrap(labels: str) -> str:
    return f"{{{labels}}}" if labels else ""


def _merge_labels(labels: str, extra: str) -> str:
    combined = ",".join(part for part in (labels, extra) if part)
    return f"{{{combined}}}"


REGISTRY = Registry()

# ---- PromptFoundry metric catalogue --------------------------------------
HTTP_REQUESTS = REGISTRY.counter(
    "promptfoundry_http_requests_total", "HTTP requests by method, path template, status"
)
HTTP_LATENCY = REGISTRY.histogram("promptfoundry_http_request_seconds", "HTTP request latency")
LLM_CALLS = REGISTRY.counter(
    "promptfoundry_llm_calls_total", "LLM calls by provider, model, purpose"
)
LLM_TOKENS = REGISTRY.counter("promptfoundry_llm_tokens_total", "LLM tokens by direction")
LLM_COST = REGISTRY.counter("promptfoundry_llm_cost_aud_micros_total", "LLM spend in AUD micros")
LLM_LATENCY = REGISTRY.histogram("promptfoundry_llm_latency_seconds", "LLM latency")
INGESTED_SIGNALS = REGISTRY.counter(
    "promptfoundry_signals_ingested_total", "Signals ingested by source type"
)
SIGNALS_REJECTED = REGISTRY.counter(
    "promptfoundry_signals_rejected_total", "Signals rejected by reason"
)
EVAL_RUNS = REGISTRY.counter(
    "promptfoundry_eval_runs_total", "Evaluation runs by stage and outcome"
)
GOVERNANCE_DECISIONS = REGISTRY.counter(
    "promptfoundry_governance_decisions_total", "Governance gate decisions"
)
BUDGET_USED = REGISTRY.gauge(
    "promptfoundry_budget_used_ratio", "Fraction of the active budget window consumed"
)
EXPERIMENT_EXPOSURES = REGISTRY.counter(
    "promptfoundry_experiment_exposures_total", "Experiment exposures by arm"
)
ALERTS_FIRED = REGISTRY.counter("promptfoundry_alerts_total", "Operational alerts fired")


@contextmanager
def timed(histogram: Histogram, **labels: str) -> Iterator[None]:
    import time

    started = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - started, **labels)


def render_metrics() -> str:
    return REGISTRY.render()


def reset_metrics() -> None:
    """Test helper."""
    REGISTRY.counters.clear()
    REGISTRY.gauges.clear()
    REGISTRY.histograms.clear()
    for metric in (
        HTTP_REQUESTS,
        LLM_CALLS,
        LLM_TOKENS,
        LLM_COST,
        INGESTED_SIGNALS,
        SIGNALS_REJECTED,
        EVAL_RUNS,
        GOVERNANCE_DECISIONS,
        EXPERIMENT_EXPOSURES,
        ALERTS_FIRED,
    ):
        REGISTRY.counters[metric.name] = metric
        metric._values.clear()
    for hist in (HTTP_LATENCY, LLM_LATENCY):
        REGISTRY.histograms[hist.name] = hist
        hist._counts.clear()
        hist._sums.clear()
        hist._totals.clear()
    REGISTRY.gauges[BUDGET_USED.name] = BUDGET_USED
    BUDGET_USED._values.clear()


def _unused_math_guard() -> float:  # pragma: no cover - keeps linters honest
    return math.nan
