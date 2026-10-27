"""Unified LLM client: budget-safe, cached, traced, provider-agnostic.

Guarantees
----------
1. **Budget first** — every call is preceded by a hard budget check; the circuit
   breaker refuses (HTTP 402) rather than overspending. Spend is written to
   ``cost_events`` immediately after the call.
2. **Official endpoints only** — OpenAI ``/v1/chat/completions``, Anthropic
   ``/v1/messages``, Google ``generativelanguage`` ``generateContent``. No scraping,
   no unofficial wrappers. Base URLs are configurable for regional endpoints/proxies.
3. **Deterministic when unkeyed** — falls back to the mock simulator, so the whole
   pipeline runs with zero API keys and zero network access.
4. **Observable** — trace span + Prometheus counters + structured logs per call.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session

from core.budget import assert_within_budget, record_cost
from core.config import get_settings
from core.errors import ProviderError
from llm.catalog import (
    approx_tokens,
    estimate_cost_micros,
    get_spec,
    provider_configured,
    resolve_model,
)
from llm.mock import simulate_judge, simulate_update_prompt
from observability.logging import get_logger
from observability.metrics import LLM_CALLS, LLM_COST, LLM_LATENCY, LLM_TOKENS
from observability.tracing import span

log = get_logger("llm.client")

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com",
}


@dataclass
class LLMRequest:
    messages: list[dict[str, str]]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1200
    json_mode: bool = False
    purpose: str = "general"
    seed: int | None = None
    subject_type: str = ""
    subject_id: str = ""
    cache: bool = True
    # Mock-only hint, ignored by real providers:
    mock_mode: str = "generate"  # generate | judge


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_micros: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    ok: bool = True
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def usage(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_aud": round(self.cost_micros / 1_000_000, 6),
            "latency_ms": round(self.latency_ms, 2),
            "cached": self.cached,
        }


def _cache_path(key: str) -> Path:
    root = get_settings().data_dir / "cache" / "llm"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key[:2]}" / f"{key}.json"


class LLMClient:
    def __init__(
        self, session: Session, *, actor_id: str | None = None, actor_label: str = "llm"
    ) -> None:
        self.session = session
        self.actor_id = actor_id
        self.actor_label = actor_label
        self.settings = get_settings()

    # ------------------------------------------------------------------ public
    def complete(self, request: LLMRequest) -> LLMResponse:
        model = request.model or resolve_model()
        try:
            spec = get_spec(model)
        except KeyError as exc:
            raise ProviderError(str(exc)) from exc

        cache_key = self._cache_key(request, model)
        if request.cache:
            cached = self._read_cache(cache_key)
            if cached is not None:
                LLM_CALLS.inc(
                    provider=spec.provider, model=model, purpose=request.purpose, cached="true"
                )
                return cached

        prompt_text = "\n".join(m.get("content", "") for m in request.messages)
        est_in = approx_tokens(prompt_text)
        est_micros = estimate_cost_micros(est_in, request.max_tokens // 2, model)
        assert_within_budget(self.session, est_micros)

        started = time.perf_counter()
        with span(
            "llm.complete",
            provider=spec.provider,
            model=model,
            purpose=request.purpose,
            subject_id=request.subject_id or None,
        ) as sp:
            if spec.provider == "mock" or not provider_configured(spec.provider):
                response = self._call_mock(request, model)
            else:
                response = self._dispatch(request, spec.provider, model)
            response.latency_ms = (time.perf_counter() - started) * 1000
            sp.attributes.update(
                {
                    "tokens_in": response.tokens_in,
                    "tokens_out": response.tokens_out,
                    "cost_micros": response.cost_micros,
                    "cached": response.cached,
                }
            )

        LLM_CALLS.inc(
            provider=response.provider, model=model, purpose=request.purpose, cached="false"
        )
        LLM_TOKENS.inc(response.tokens_in, direction="input")
        LLM_TOKENS.inc(response.tokens_out, direction="output")
        LLM_LATENCY.observe(response.latency_ms / 1000, model=model)
        if response.cost_micros:
            LLM_COST.inc(response.cost_micros, provider=response.provider, model=model)
            record_cost(
                self.session,
                category="llm",
                amount_micros=response.cost_micros,
                provider=response.provider,
                model=model,
                purpose=request.purpose,
                subject_type=request.subject_type or None,
                subject_id=request.subject_id or None,
                detail={"tokens_in": response.tokens_in, "tokens_out": response.tokens_out},
            )
        if request.cache and response.ok:
            self._write_cache(cache_key, response)
        return response

    def complete_json(self, request: LLMRequest) -> tuple[dict[str, Any], LLMResponse]:
        """Complete + parse JSON, tolerating fenced output. Never raises on bad JSON."""
        request.json_mode = True
        response = self.complete(request)
        return parse_json_loose(response.text), response

    # -------------------------------------------------------------- providers
    def _dispatch(self, request: LLMRequest, provider: str, model: str) -> LLMResponse:
        if provider == "openai":
            return self._call_openai(request, model)
        if provider == "anthropic":
            return self._call_anthropic(request, model)
        if provider == "google":
            return self._call_google(request, model)
        raise ProviderError(f"unsupported provider '{provider}'")

    def _post(
        self, url: str, *, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        attempts = max(1, self.settings.llm_max_retries + 1)
        last_error = ""
        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < attempts - 1:
                    last_error = f"HTTP {resp.status_code}"
                    time.sleep(min(2**attempt * 0.5, 4.0))
                    continue
                if resp.status_code >= 400:
                    raise ProviderError(
                        f"provider returned HTTP {resp.status_code}",
                        details={"body": resp.text[:400]},
                    )
                return resp.json()
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < attempts - 1:
                    time.sleep(min(2**attempt * 0.5, 4.0))
                    continue
        raise ProviderError(f"provider call failed after {attempts} attempts ({last_error})")

    def _call_openai(self, request: LLMRequest, model: str) -> LLMResponse:
        base = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URLS["openai"]).rstrip("/")
        payload: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if request.seed is not None:
            payload["seed"] = request.seed
        data = self._post(
            f"{base}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage", {}) or {}
        tin = int(usage.get("prompt_tokens", approx_tokens(str(request.messages))))
        tout = int(usage.get("completion_tokens", approx_tokens(text)))
        return LLMResponse(
            text=text,
            model=model,
            provider="openai",
            tokens_in=tin,
            tokens_out=tout,
            cost_micros=estimate_cost_micros(tin, tout, model),
        )

    def _call_anthropic(self, request: LLMRequest, model: str) -> LLMResponse:
        base = os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URLS["anthropic"]).rstrip("/")
        system_parts = [m["content"] for m in request.messages if m.get("role") == "system"]
        chat = [m for m in request.messages if m.get("role") != "system"]
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": chat,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.json_mode:
            payload["system"] = (
                payload.get("system", "") + "\n\nRespond with valid JSON only."
            ).strip()
        data = self._post(
            f"{base}/v1/messages",
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        blocks = data.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        usage = data.get("usage", {}) or {}
        tin = int(usage.get("input_tokens", approx_tokens(str(request.messages))))
        tout = int(usage.get("output_tokens", approx_tokens(text)))
        return LLMResponse(
            text=text,
            model=model,
            provider="anthropic",
            tokens_in=tin,
            tokens_out=tout,
            cost_micros=estimate_cost_micros(tin, tout, model),
        )

    def _call_google(self, request: LLMRequest, model: str) -> LLMResponse:
        base = os.environ.get("GOOGLE_BASE_URL", DEFAULT_BASE_URLS["google"]).rstrip("/")
        system_parts = [m["content"] for m in request.messages if m.get("role") == "system"]
        chat = [m for m in request.messages if m.get("role") != "system"]
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": m.get("content", "")}]} for m in chat],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
            },
        }
        if request.json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        data = self._post(
            f"{base}/v1beta/models/{model}:generateContent?key={self.settings.google_api_key}",
            headers={"Content-Type": "application/json"},
            payload=payload,
        )
        candidates = data.get("candidates") or [{}]
        parts = candidates[0].get("content", {}).get("parts", []) or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        usage = data.get("usageMetadata", {}) or {}
        tin = int(usage.get("promptTokenCount", approx_tokens(str(request.messages))))
        tout = int(usage.get("candidatesTokenCount", approx_tokens(text)))
        return LLMResponse(
            text=text,
            model=model,
            provider="google",
            tokens_in=tin,
            tokens_out=tout,
            cost_micros=estimate_cost_micros(tin, tout, model),
        )

    def _call_mock(self, request: LLMRequest, model: str) -> LLMResponse:
        system_prompt = "\n".join(
            m.get("content", "") for m in request.messages if m.get("role") == "system"
        )
        user_input = "\n".join(
            m.get("content", "") for m in request.messages if m.get("role") == "user"
        )
        if request.mock_mode == "judge":
            text = simulate_judge(system_prompt, user_input, criterion=request.purpose)
        else:
            text = simulate_update_prompt(
                system_prompt, user_input, temperature=request.temperature
            )
        return LLMResponse(
            text=text,
            model=model,
            provider="mock",
            tokens_in=approx_tokens(system_prompt + user_input),
            tokens_out=approx_tokens(text),
            cost_micros=0,
        )

    # ------------------------------------------------------------------ cache
    def _cache_key(self, request: LLMRequest, model: str) -> str:
        payload = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "json_mode": request.json_mode,
            "mode": request.mock_mode,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _read_cache(self, key: str) -> LLMResponse | None:
        path = _cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        data["cached"] = True
        return LLMResponse(**data)

    def _write_cache(self, key: str, response: LLMResponse) -> None:
        payload = {
            "text": response.text,
            "model": response.model,
            "provider": response.provider,
            "tokens_in": response.tokens_in,
            "tokens_out": response.tokens_out,
            "cost_micros": 0,  # a cache hit costs nothing
            "latency_ms": response.latency_ms,
            "ok": response.ok,
        }
        path = _cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def parse_json_loose(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating fences and leading prose."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate
        candidate = candidate.removeprefix("json").strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {}
    return {}
