"""Embeddings with a zero-key baseline.

Providers
---------
* ``hashing`` (default): a **concept-assisted lexical** vector. Two blocks are
  concatenated and L2-normalised:

  1. a 16-dimension *job-to-be-done concept* block scored from a curated taxonomy
     (billing, email, seo, support, career, ads, onboarding, product, data,
     meetings, legal, health, finance, research, sales, ops). This block carries the
     semantics for clustering: "chase an unpaid invoice" and "overdue invoice
     reminder" land on the same axis even with no shared phrasing.
  2. a hashed lexical block over unigrams/bigrams/char-4-grams for near-duplicate
     detection (used by the originality/IP check).

  Deterministic, offline, free. It is *not* a semantic transformer: with real usage,
  set ``OPENAI_API_KEY`` (or ``PF_EMBEDDING_PROVIDER``) and the vectors are replaced
  transparently — everything downstream only needs ``(name, dim, embed)``.
* ``openai`` / ``google``: official embedding endpoints, used automatically when a
  key is present (or when ``PF_EMBEDDING_PROVIDER`` says so).
* ``local``: sentence-transformers if installed (never required).

The embedder is versioned by name so stored vectors can be invalidated on upgrade
(``normalized_signals.embedding_model``).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Protocol

import httpx
import numpy as np

from core.config import get_settings
from core.errors import ProviderError
from observability.logging import get_logger

log = get_logger("llm.embeddings")

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]+")
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "we",
    "you",
    "they",
    "he",
    "she",
    "my",
    "our",
    "your",
    "their",
    "as",
    "at",
    "by",
    "from",
    "how",
    "what",
    "when",
    "where",
    "why",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "need",
    "want",
    "please",
    "help",
    "me",
    "some",
    "any",
    "about",
    "into",
    "than",
    "then",
    "so",
    "just",
    "like",
    "get",
}

# Concept normalisation keeps the lexical baseline competitive on realistic demand
# text. Real embedding providers make this unnecessary; it is harmless with them.
_CONCEPTS: dict[str, str] = {
    "invoice": "billing",
    "invoices": "billing",
    "billing": "billing",
    "receipt": "billing",
    "receipts": "billing",
    "payment": "billing",
    "payments": "billing",
    "quote": "billing",
    "quoting": "billing",
    "quote_request": "billing",
    "remittance": "billing",
    "email": "email",
    "emails": "email",
    "inbox": "email",
    "outreach": "email",
    "newsletter": "email",
    "nurture": "email",
    "sequence": "email",
    "followup": "email",
    "reply": "email",
    "replies": "email",
    "cold": "email",
    "blog": "seo",
    "seo": "seo",
    "article": "seo",
    "articles": "seo",
    "keyword": "seo",
    "keywords": "seo",
    "ranking": "seo",
    "organic": "seo",
    "content": "seo",
    "support": "support",
    "ticket": "support",
    "tickets": "support",
    "complaint": "support",
    "complaints": "support",
    "helpdesk": "support",
    "escalation": "support",
    "refund": "support",
    "resume": "career",
    "cv": "career",
    "interview": "career",
    "job": "career",
    "jobs": "career",
    "linkedin": "career",
    "hiring": "career",
    "recruiter": "career",
    "ad": "ads",
    "ads": "ads",
    "advert": "ads",
    "copy": "ads",
    "headline": "ads",
    "facebook": "ads",
    "meta": "ads",
    "google_ads": "ads",
    "creative": "ads",
    "onboarding": "onboarding",
    "welcome": "onboarding",
    "training": "onboarding",
    "course": "onboarding",
    "lesson": "onboarding",
    "curriculum": "onboarding",
    "product": "product",
    "roadmap": "product",
    "prd": "product",
    "feature": "product",
    "spec": "product",
    "backlog": "product",
    "userstory": "product",
    "stories": "product",
    "data": "data",
    "sql": "data",
    "dashboard": "data",
    "report": "data",
    "analysis": "data",
    "spreadsheet": "data",
    "excel": "data",
    "csv": "data",
    "meeting": "meetings",
    "standup": "meetings",
    "minutes": "meetings",
    "agenda": "meetings",
    "transcript": "meetings",
    "notes": "meetings",
    "legal": "legal",
    "contract": "legal",
    "contracts": "legal",
    "policy": "legal",
    "term": "legal",
    "terms": "legal",
    "compliance": "legal",
    "nda": "legal",
    "health": "health",
    "medical": "health",
    "clinical": "health",
    "symptom": "health",
    "finance": "finance",
    "budget": "finance",
    "forecast": "finance",
    "cashflow": "finance",
    "tax": "finance",
    "bookkeeping": "finance",
    "accounting": "finance",
}


# Job-to-be-done taxonomy for the concept block. Words are compared *after*
# ``_normalise`` so aliases ("invoice" → "billing") collapse automatically.
CONCEPT_TAXONOMY: dict[str, tuple[str, ...]] = {
    "billing": ("billing", "overdue", "remittance", "debt", "unpaid", "arrears", "instalment"),
    "email": (
        "email",
        "outreach",
        "newsletter",
        "cold",
        "subject",
        "sequence",
        "welcome",
        "winback",
    ),
    "seo": ("seo", "blog", "article", "keyword", "outline", "serp", "ranking", "meta"),
    "support": ("support", "ticket", "complaint", "helpdesk", "escalation", "sla"),
    "career": ("career", "interview", "linkedin", "coverletter"),
    "ads": ("ads", "advert", "headline", "creative", "campaign", "ctr", "audience"),
    "onboarding": ("onboarding", "training", "lesson", "course", "curriculum", "rubric"),
    "product": ("product", "roadmap", "prd", "spec", "feature", "backlog", "userstory"),
    "data": ("data", "sql", "dashboard", "spreadsheet", "csv", "metric", "kpi", "cohort"),
    "meetings": ("meeting", "minutes", "standup", "agenda", "transcript", "decisions"),
    "legal": ("legal", "contract", "clause", "terms", "compliance", "nda", "solicitor"),
    "health": ("health", "clinical", "symptom", "patient", "clinic", "diagnosis"),
    "finance": ("finance", "budget", "forecast", "tax", "gst", "accountant", "bookkeeping"),
    "research": ("research", "survey", "synthesise", "evidence", "insight", "benchmark"),
    "sales": ("sales", "demo", "discovery", "proposal", "crm", "deal", "objection"),
    "ops": ("ops", "process", "sop", "workflow", "automation", "handover"),
}
CONCEPT_DIM = len(CONCEPT_TAXONOMY)
_CONCEPT_TERM_INDEX: dict[str, int] = {
    term: index for index, (_label, terms) in enumerate(CONCEPT_TAXONOMY.items()) for term in terms
}


def _normalise(text: str) -> list[str]:
    lowered = text.lower().replace("_", " ")
    tokens: list[str] = []
    for raw in _WORD_RE.findall(lowered):
        stem = raw.rstrip("'s") if len(raw) > 4 else raw
        if raw in _STOP or stem in _STOP:
            continue
        tokens.append(_CONCEPTS.get(raw, _CONCEPTS.get(stem, stem)))
    return tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Concept-assisted hashed lexical embedder (see module docstring)."""

    def __init__(self, dim: int | None = None, *, concept_weight: float = 1.7) -> None:
        self.dim = dim or get_settings().embedding_dim
        self.concept_weight = concept_weight
        self.name = f"hashing-v2-{self.dim}"
        self.vector_dim = self.dim + CONCEPT_DIM

    def _features(self, text: str) -> Counter[str]:
        tokens = _normalise(text)
        features: Counter[str] = Counter()
        for tok in tokens:
            features[f"w:{tok}"] += 1.0
        for first, second in zip(tokens, tokens[1:], strict=False):
            features[f"b:{first}_{second}"] += 1.2
        compact = " " + " ".join(tokens) + " "
        for i in range(len(compact) - 4):
            features[f"c:{compact[i : i + 4]}"] += 0.25
        return features

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % dim, 1.0 if (value >> 63) & 1 else -1.0

    def _concept_vector(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(CONCEPT_DIM, dtype=np.float64)
        for token in tokens:
            index = _CONCEPT_TERM_INDEX.get(token)
            if index is not None:
                vec[index] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = _normalise(text)
            lexical = np.zeros(self.dim, dtype=np.float64)
            for feature, count in self._features(text).items():
                idx, sign = self._bucket(feature, self.dim)
                lexical[idx] += sign * (1.0 + math.log1p(count))
            lex_norm = np.linalg.norm(lexical)
            if lex_norm > 0:
                lexical = lexical / lex_norm
            combined = np.concatenate([self.concept_weight * self._concept_vector(tokens), lexical])
            norm = np.linalg.norm(combined)
            vectors.append((combined / norm).tolist() if norm > 0 else combined.tolist())
        return vectors


class OpenAIEmbedder:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or get_settings().embedding_model
        self.name = f"openai-{self.model}"
        self.dim = get_settings().embedding_dim if "3-small" in self.model else 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            resp = client.post(
                f"{base}/v1/embeddings",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={"model": self.model, "input": texts},
            )
        if resp.status_code >= 400:
            raise ProviderError(f"embedding provider HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json().get("data", [])
        return [item["embedding"] for item in data]


class GoogleEmbedder:
    def __init__(self, model: str = "text-embedding-004") -> None:
        self.model = model
        self.name = f"google-{model}"
        self.dim = 768

    def embed(self, texts: list[str]) -> list[list[float]]:
        settings = get_settings()
        base = os.environ.get(
            "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com"
        ).rstrip("/")
        out: list[list[float]] = []
        with httpx.Client(timeout=settings.llm_timeout_seconds) as client:
            for text in texts:
                resp = client.post(
                    f"{base}/v1beta/models/{self.model}:embedContent?key={settings.google_api_key}",
                    json={"model": f"models/{self.model}", "content": {"parts": [{"text": text}]}},
                )
                if resp.status_code >= 400:
                    raise ProviderError(
                        f"embedding provider HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                out.append(resp.json()["embedding"]["values"])
        return out


def get_embedder() -> Embedder:
    settings = get_settings()
    choice = (settings.embedding_provider or "auto").lower()
    if choice in {"", "auto"}:
        if settings.openai_api_key:
            return OpenAIEmbedder()
        if settings.google_api_key:
            return GoogleEmbedder()
        return HashingEmbedder()
    if choice == "openai":
        return OpenAIEmbedder()
    if choice == "google":
        return GoogleEmbedder()
    if choice in {"hashing", "local-hash"}:
        return HashingEmbedder()
    log.warning(
        "unknown embedding provider; falling back to hashing",
        extra={"extra_fields": {"choice": choice}},
    )
    return HashingEmbedder()


def cosine(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_pairwise(matrix: np.ndarray) -> np.ndarray:
    """Full cosine similarity matrix for an (n, d) array of already-normalised rows."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = matrix / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)
