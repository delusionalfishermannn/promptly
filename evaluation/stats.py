"""Small-sample statistics: Wilson intervals, bootstrap CIs, paired tests.

Every headline metric in an evaluation report carries an interval, because with 12-40
test cases point estimates are noisy and "X% better" without uncertainty is how
Goodhart starts.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

Z_95 = NormalDist().inv_cdf(0.975)


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    n: int
    method: str

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": round(self.estimate, 4),
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "n": self.n,
            "method": self.method,
        }


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion (stable for small n)."""
    if n <= 0:
        return Interval(0.0, 0.0, 0.0, 0, "wilson")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), n, "wilson")


def mean_interval(values: list[float], *, iterations: int = 400, seed: int = 13) -> Interval:
    """Percentile bootstrap CI for a mean (non-parametric, no normality assumption)."""
    if not values:
        return Interval(0.0, 0.0, 0.0, 0, "bootstrap")
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return Interval(
        sum(values) / n,
        means[int(0.05 * (len(means) - 1))],
        means[int(0.95 * (len(means) - 1))],
        n,
        "bootstrap",
    )


def paired_difference(
    baseline: list[float], candidate: list[float], *, iterations: int = 600, seed: int = 29
) -> Interval:
    """Bootstrap CI of the paired mean difference (candidate − baseline)."""
    n = min(len(baseline), len(candidate))
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0, "paired-bootstrap")
    diffs = [candidate[i] - baseline[i] for i in range(n)]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    return Interval(
        sum(diffs) / n,
        draws[int(0.05 * (len(draws) - 1))],
        draws[int(0.95 * (len(draws) - 1))],
        n,
        "paired-bootstrap",
    )


def independent_mean_difference(
    group_a: list[float], group_b: list[float], *, iterations: int = 600, seed: int = 31
) -> Interval:
    """Bootstrap CI for mean(b) − mean(a) with independent resampling within groups."""
    if not group_a or not group_b:
        return Interval(0.0, 0.0, 0.0, 0, "independent-bootstrap")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(iterations):
        sample_a = [group_a[rng.randrange(len(group_a))] for _ in range(len(group_a))]
        sample_b = [group_b[rng.randrange(len(group_b))] for _ in range(len(group_b))]
        draws.append(sum(sample_b) / len(sample_b) - sum(sample_a) / len(sample_a))
    draws.sort()
    estimate = sum(group_b) / len(group_b) - sum(group_a) / len(group_a)
    return Interval(
        estimate,
        draws[int(0.05 * (len(draws) - 1))],
        draws[int(0.95 * (len(draws) - 1))],
        len(group_a) + len(group_b),
        "independent-bootstrap",
    )


def two_proportion_z(success_a: int, n_a: int, success_b: int, n_b: int) -> tuple[float, float]:
    """Return (z, two-sided p-value) for a difference in conversion rates."""
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    if pooled in (0.0, 1.0):
        return 0.0, 1.0
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if standard_error == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / standard_error
    p_value = 2 * (1 - NormalDist().cdf(abs(z)))
    return z, max(0.0, min(1.0, p_value))


def required_sample_size(
    baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Per-arm sample size for a two-proportion test (normal approximation)."""
    baseline_rate = min(max(baseline_rate, 0.001), 0.999)
    target = min(max(baseline_rate + mde, 0.001), 0.999)
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    p_bar = (baseline_rate + target) / 2
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(baseline_rate * (1 - baseline_rate) + target * (1 - target))
    ) ** 2
    denominator = (target - baseline_rate) ** 2
    return int(math.ceil(numerator / denominator)) if denominator else 0


def cohens_kappa_quadratic(judge: list[int], human: list[int], max_rating: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal rubric ratings."""
    if len(judge) != len(human) or not judge:
        return 0.0
    n = len(judge)
    observed = [[0] * (max_rating + 1) for _ in range(max_rating + 1)]
    for j, h in zip(judge, human, strict=True):
        observed[j][h] += 1
    total = n * n
    weights_sum = 0.0
    weighted_expected = 0.0
    for i in range(max_rating + 1):
        for j in range(max_rating + 1):
            weight = ((i - j) ** 2) / (max_rating**2) if max_rating else 0.0
            weights_sum += weight * observed[i][j] / n
            row = sum(observed[i])
            col = sum(observed[k][j] for k in range(max_rating + 1))
            weighted_expected += weight * row * col / total
    if weighted_expected == 0:
        return 0.0
    kappa = 1 - (weights_sum / weighted_expected)
    return round(max(-1.0, min(1.0, kappa)), 4)


def agreement(judge: list[int], human: list[int], tolerance: int = 1) -> float:
    if not judge or len(judge) != len(human):
        return 0.0
    matches = sum(1 for j, h in zip(judge, human, strict=True) if abs(j - h) <= tolerance)
    return round(matches / len(judge), 4)
