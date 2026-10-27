"""Allocation strategies: Thompson sampling, epsilon-greedy and fixed splits.

All strategies return ``(arm_id, propensity, reason)`` where ``propensity`` is the
probability with which that arm was selected. Propensities are persisted on
``allocations`` so off-policy analysis stays possible later — a detail most bandit
implementations skip and then regret.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ArmState:
    arm_id: str
    weight: float = 0.5
    is_control: bool = False
    exposures: int = 0
    conversions: int = 0
    revenue_micros: int = 0
    refunds: int = 0

    @property
    def successes(self) -> float:
        return float(self.conversions) + 1.0  # Beta(1,1) prior

    @property
    def failures(self) -> float:
        return float(max(self.exposures - self.conversions, 0)) + 1.0

    @property
    def rate(self) -> float:
        return self.conversions / self.exposures if self.exposures else 0.0


def _seed_from(subject_key: str, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{salt}|{subject_key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def thompson(arms: list[ArmState], *, rng: random.Random | None = None) -> tuple[str, float, str]:
    """Sample once from each Beta posterior; pick the argmax. Propensity via MC."""
    rng = rng or random.Random()
    if not arms:
        raise ValueError("no arms")
    draws = {arm.arm_id: rng.betavariate(arm.successes, arm.failures) for arm in arms}
    chosen = max(draws.items(), key=lambda kv: kv[1])[0]
    # Estimate propensity by Monte Carlo (cheap at this scale, honest by construction).
    wins = 0
    simulations = 200
    for _ in range(simulations):
        sample = {arm.arm_id: rng.betavariate(arm.successes, arm.failures) for arm in arms}
        if max(sample.items(), key=lambda kv: kv[1])[0] == chosen:
            wins += 1
    propensity = max(0.02, min(1.0, wins / simulations))
    return chosen, propensity, "thompson_beta_posterior"


def epsilon_greedy(
    arms: list[ArmState], *, epsilon: float = 0.1, rng: random.Random | None = None
) -> tuple[str, float, str]:
    rng = rng or random.Random()
    if rng.random() < epsilon:
        arm = rng.choice(arms)
        return arm.arm_id, epsilon / len(arms), f"explore(epsilon={epsilon})"
    best_rate = max(arm.rate for arm in arms)
    best = [arm for arm in arms if arm.rate == best_rate]
    arm = rng.choice(best)
    return arm.arm_id, (1 - epsilon) / len(best), "exploit_best_rate"


def fixed_split(
    arms: list[ArmState], *, subject_key: str, salt: str = ""
) -> tuple[str, float, str]:
    """Deterministic assignment by hashing the subject key (stable A/B assignment)."""
    total = sum(max(arm.weight, 0.0) for arm in arms) or 1.0
    rng = _seed_from(subject_key, salt)
    point = rng.random() * total
    cumulative = 0.0
    for arm in arms:
        cumulative += max(arm.weight, 0.0)
        if point <= cumulative:
            return arm.arm_id, max(arm.weight, 0.0) / total, "fixed_hash_split"
    return arms[-1].arm_id, max(arms[-1].weight, 0.0) / total, "fixed_hash_split"


def allocate(
    arms: list[ArmState],
    *,
    strategy: str,
    subject_key: str,
    salt: str = "",
) -> tuple[str, float, str]:
    if strategy == "thompson":
        # Deterministic per subject: same subject always sees the same arm in a run.
        return thompson(arms, rng=_seed_from(subject_key, salt + "|thompson"))
    if strategy == "epsilon_greedy":
        return epsilon_greedy(arms, rng=_seed_from(subject_key, salt + "|eg"))
    return fixed_split(arms, subject_key=subject_key, salt=salt)


def bandit_state_from_rows(
    arms: list[Any], exposures: list[Any], conversions: list[Any]
) -> list[ArmState]:
    states: dict[str, ArmState] = {
        arm.id: ArmState(
            arm_id=arm.id,
            weight=arm.weight,
            is_control=arm.is_control,
        )
        for arm in arms
    }
    for exposure in exposures:
        state = states.get(exposure.arm_id)
        if state is not None:
            state.exposures += 1
    for conversion in conversions:
        state = states.get(conversion.arm_id)
        if state is not None:
            if conversion.event_type in {"purchase", "checkout_completed"}:
                state.conversions += 1
            state.revenue_micros += int(conversion.value_micros or 0)
    return list(states.values())


__all__ = [
    "ArmState",
    "allocate",
    "bandit_state_from_rows",
    "epsilon_greedy",
    "fixed_split",
    "thompson",
]
