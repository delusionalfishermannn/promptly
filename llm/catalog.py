"""Model catalogue: tiers, capabilities and per-token prices (as of 2026-10).

Prices are **priors** in USD per 1M tokens and are overridable via env
(``PF_PRICE_OVERRIDE_<MODEL>`` / config edits) — always confirm against your
provider's current price sheet. All spend is converted to AUD micros before being
written to ``cost_events`` so budget maths is integer-exact.

Tiering drives the staged optimisation loop:
  * ``cheap``     → Stage 1 screening (many candidates, low stakes)
  * ``balanced``  → Stage 2 offline evaluation / judges
  * ``frontier``  → Stage 3 finalists only (few trials, expensive)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from core.config import AUD_MICROS, get_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str  # openai | anthropic | google | mock
    tier: str  # cheap | balanced | frontier
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    context_tokens: int
    family: str  # used to group compiled prompt variants
    judge_capable: bool = False
    strengths: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"


CATALOG: dict[str, ModelSpec] = {
    # --- mock simulator (no network, zero cost) -----------------------------
    "mock-cheap": ModelSpec("mock-cheap", "mock", "cheap", 0.0, 0.0, 32_000, "mock"),
    "mock-balanced": ModelSpec("mock-balanced", "mock", "balanced", 0.0, 0.0, 64_000, "mock", True),
    "mock-frontier": ModelSpec(
        "mock-frontier", "mock", "frontier", 0.0, 0.0, 128_000, "mock", True
    ),
    # --- OpenAI ------------------------------------------------------------
    "gpt-4o-mini": ModelSpec("gpt-4o-mini", "openai", "cheap", 0.15, 0.60, 128_000, "openai"),
    "gpt-4o": ModelSpec("gpt-4o", "openai", "balanced", 2.50, 10.00, 128_000, "openai", True),
    "o4-mini": ModelSpec("o4-mini", "openai", "balanced", 1.10, 4.40, 200_000, "openai", True),
    # --- Anthropic ---------------------------------------------------------
    "claude-3-5-haiku": ModelSpec(
        "claude-3-5-haiku", "anthropic", "cheap", 0.80, 4.00, 200_000, "anthropic"
    ),
    "claude-sonnet-4": ModelSpec(
        "claude-sonnet-4", "anthropic", "balanced", 3.00, 15.00, 200_000, "anthropic", True
    ),
    "claude-opus-4": ModelSpec(
        "claude-opus-4", "anthropic", "frontier", 15.00, 75.00, 200_000, "anthropic", True
    ),
    # --- Google ------------------------------------------------------------
    "gemini-2.0-flash": ModelSpec(
        "gemini-2.0-flash", "google", "cheap", 0.10, 0.40, 1_000_000, "google"
    ),
    "gemini-1.5-pro": ModelSpec(
        "gemini-1.5-pro", "google", "balanced", 1.25, 5.00, 2_000_000, "google", True
    ),
}

PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def get_spec(model: str) -> ModelSpec:
    if model in CATALOG:
        return CATALOG[model]
    # Unknown/self-hosted model: charge nothing locally, but keep it usable.
    if model.startswith("local"):
        return ModelSpec(model, "local", "cheap", 0.0, 0.0, 32_000, "local")
    raise KeyError(f"unknown model '{model}' — add it to llm/catalog.py")


def provider_configured(provider: str) -> bool:
    if provider == "mock":
        return True
    env_key = PROVIDER_ENV_KEYS.get(provider)
    if not env_key:
        return provider == "local"
    return bool(os.environ.get(env_key) or getattr(get_settings(), f"{provider}_api_key", ""))


def available_models(tier: str | None = None, judge_only: bool = False) -> list[ModelSpec]:
    configured_providers = {p for p in PROVIDER_ENV_KEYS if provider_configured(p)}
    has_any_key = bool(configured_providers)
    models = []
    for spec in CATALOG.values():
        if tier and spec.tier != tier:
            continue
        if judge_only and not spec.judge_capable:
            continue
        # Mock stays available only when no real provider is configured (keeps demos honest).
        if spec.provider == "mock" and has_any_key:
            continue
        if spec.provider not in {"mock", "local"} and spec.provider not in configured_providers:
            continue
        models.append(spec)
    return sorted(models, key=lambda s: (s.tier, s.name))


def resolve_model(
    tier: str | None = None, *, judge: bool = False, exclude: set[str] | None = None
) -> str:
    """Pick a concrete model for a tier, preferring a *different* vendor for judges."""
    exclude = exclude or set()
    candidates = [m for m in available_models(tier, judge_only=judge) if m.name not in exclude]
    if not candidates:
        candidates = [m for m in available_models(None, judge_only=judge) if m.name not in exclude]
    if not candidates:
        return "mock-balanced"
    if judge:
        # Prefer a different provider than the candidate when both are configured.
        return candidates[-1].name if len(candidates) > 1 else candidates[0].name
    return candidates[0].name


def estimate_cost_micros(prompt_tokens: int, completion_tokens: int, model: str) -> int:
    try:
        spec = get_spec(model)
    except KeyError:
        return 0
    usd = (prompt_tokens / 1_000_000) * spec.input_usd_per_mtok + (
        completion_tokens / 1_000_000
    ) * spec.output_usd_per_mtok
    return int(round(usd * get_settings().aud_per_usd * AUD_MICROS))


def approx_tokens(text: str) -> int:
    """Cheap, provider-agnostic token estimate (~4 chars/token, +word overhead)."""
    return max(1, int(len(text) / 4) + text.count(" ") // 8)
